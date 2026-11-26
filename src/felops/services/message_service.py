"""Submit-message use case (§2.2, §7.1, §8.2 step 1).

Responsibilities:
  1. Check in-memory idempotency ledger for duplicate Idempotency-Key.
  2. Check no other job is currently running on the same conversation.
  3. Create a Job with status=QUEUED.
  4. Resolve AgentRef (V1: static customer_support, external).
  5. Build initial AgentState from the request.
  6. Spawn an asyncio background task that:
     a. Sets job status to RUNNING.
     b. Calls graph.ainvoke() with the initial state and thread config.
     c. Sets job status to DONE on success, FAILED on exception.
  7. Return the Job (caller extracts job_id for the 202 response).

Future extension points (documented, not built):
  - Task 1.8: event bus notification inside the background task.
  - Task 2.5: agent_config parsing + ToolRegistry.recompute() after step 5.
  - Task 3.4: resume_job() function for approval decisions.
  - Task 3.5: replace in-memory dicts with Postgres-backed JobRepository.
  - Task 3.10: webhook call at step 6c.
  - Task 5.8: startup recovery scan of RUNNING jobs.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from felops.domain.agent import DEFAULT_AUDIENCE, DEFAULT_SUB_AGENT_ID
from felops.domain.job import Job, JobSender, JobStatus, JobType
from felops.domain.state import AgentConfig, AgentState
from felops.domain.workspace import WorkspaceContext
from felops.orchestrator.graph import make_thread_config

logger = logging.getLogger(__name__)


class DuplicateIdempotencyKey(Exception):
    """Raised when the same Idempotency-Key is reused for a workspace."""

    def __init__(self, existing_job: Job) -> None:
        self.existing_job = existing_job
        super().__init__(f"Duplicate key; existing job_id={existing_job.job_id}")


class ConversationBusy(Exception):
    """Raised when a job is already running on the conversation."""

    def __init__(self, existing_job: Job) -> None:
        self.existing_job = existing_job
        super().__init__(
            f"Conversation {existing_job.conversation_id} already has "
            f"a running job: {existing_job.job_id}"
        )


class MessageService:
    """Stateful service managing job submission and the idempotency ledger.

    Instantiated once per process (in the Container) and shared across requests.

    Attributes:
        _idempotency_ledger: Maps (workspace_id, idempotency_key) → Job.
                             In-memory for P1; replaced by Postgres in task 3.5/7.10.
        _active_jobs:        Maps (workspace_id, conversation_id) → Job for
                             currently non-terminal jobs. Used to enforce
                             one-job-per-conversation.
        _jobs:               Maps job_id → Job for status lookup (task 3.5).
        _tasks:              Strong references to running background tasks so
                             asyncio does not garbage-collect them mid-flight.
    """

    def __init__(self) -> None:
        self._idempotency_ledger: dict[tuple[str, str], Job] = {}
        self._active_jobs: dict[tuple[str, str], Job] = {}
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def get_job(self, job_id: str) -> Job | None:
        """Look up a job by id. Used by GET /jobs/{id} (task 3.5)."""
        return self._jobs.get(job_id)

    async def submit_message(
        self,
        *,
        workspace: WorkspaceContext,
        conversation_id: str,
        idempotency_key: str,
        agent_id: str,
        channel: str,
        content: str,
        locale: str,
        sender: JobSender | None = None,
        agent_config: AgentConfig,
        graph: CompiledStateGraph,
    ) -> Job:
        """Submit a new message for processing.

        Returns the Job (new or existing if idempotency key is a duplicate).
        Raises DuplicateIdempotencyKey if the key was already seen in this workspace.
        Raises ConversationBusy if a non-terminal job is already running
        on the same conversation.
        """
        # 1. Idempotency check
        ledger_key = (workspace.workspace_id, idempotency_key)
        existing = self._idempotency_ledger.get(ledger_key)
        if existing is not None:
            raise DuplicateIdempotencyKey(existing)

        # 2. One-job-per-conversation check
        conv_key = (workspace.workspace_id, conversation_id)
        active = self._active_jobs.get(conv_key)
        if active is not None and active.status in (
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.WAITING_APPROVAL,
        ):
            raise ConversationBusy(active)

        # 3. Create the job
        job = Job(
            job_type=JobType.INTERACTIVE,
            status=JobStatus.QUEUED,
            workspace_id=workspace.workspace_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
            # Stored on Job in P1 just to be ready for P4; unused in P1 graph execution.
            locale=locale,
            # Stored on Job in P1 just to be ready for P4 / downstream attribution; unused in P1.
            sender=sender,
        )

        # 4. Register in ledger and active-jobs index
        self._idempotency_ledger[ledger_key] = job
        self._active_jobs[conv_key] = job
        self._jobs[job.job_id] = job

        # 5. Build initial AgentState
        initial_state = AgentState(
            workspace_id=workspace.workspace_id,
            agent_id=agent_id,
            sub_agent_id=DEFAULT_SUB_AGENT_ID,
            audience=DEFAULT_AUDIENCE,
            channel=channel,
            conversation_id=conversation_id,
            agent_config=agent_config,
            messages=[HumanMessage(content=content)],
        )

        # 6. Build thread config
        thread_config = make_thread_config(workspace.workspace_id, conversation_id)

        # 7. Spawn background task and retain a strong reference until done
        task = asyncio.create_task(
            self._run_graph(job, initial_state, thread_config, graph),
            name=f"job-{job.job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        return job

    async def _run_graph(
        self,
        job: Job,
        initial_state: AgentState,
        thread_config: RunnableConfig,
        graph: CompiledStateGraph,
    ) -> None:
        """Background task that runs the LangGraph graph.

        Updates job.status on completion.

        Future extension points:
          - Task 1.8: emit events to the in-process EventBus here.
          - Task 3.10: call webhook sender for job.completed here.
          - Task 5.8: on startup, scan RUNNING jobs and re-enter this function.
        """
        job.status = JobStatus.RUNNING
        try:
            await graph.ainvoke(
                initial_state,
                config=thread_config,
            )
            job.status = JobStatus.DONE
            job.finished_at = datetime.now(timezone.utc)
            logger.info("Job %s completed successfully.", job.job_id)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            logger.exception("Job %s failed.", job.job_id)
        finally:
            conv_key = (job.workspace_id, job.conversation_id)
            if self._active_jobs.get(conv_key) is job:
                del self._active_jobs[conv_key]
