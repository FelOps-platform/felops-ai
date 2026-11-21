"""Responder node — P0 stub (§6.2, §6.3).

In P0 returns a canned AIMessage so the walking skeleton has an
observable output without a live LLM.

In P1 (task 1.5/1.6) this node calls:
  - PromptStore.resolve("responder", ctx)  to get the persona template
  - LLMProvider.complete(req)              to generate the real reply
  - emits message.delta + message.final events

Import rule: only felops.domain and felops.ports are imported here.
The LLMProvider and PromptStore ports are injected by the container (P0: app/container.py)
and passed through graph config — NOT imported directly.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage

from felops.domain.events import EventType
from felops.domain.state import AgentState, TraceEvent

# P0 canned reply — replaced by a real LLM call in P1.
_CANNED_REPLY = "I received your message. (stub reply — LLM not wired yet)"


def responder(state: AgentState) -> dict[str, Any]:
    """Append a canned AIMessage and record a trace event.

    The ``message.final`` TraceEvent data follows §7.2 of the new
    architecture and ``ai-service-contract.md``: ``message_id`` identifies
    the message, ``text`` contains its rendered content, and ``citations``
    contains citation metadata. Envelope-level identity fields such as
    workspace, agent, and sub-agent IDs are not included in this data.

    Writes to:
      messages         (add_messages reducer appends the new AIMessage)
      execution_trace  (operator.add reducer appends the TraceEvent)
    """
    message_id = str(uuid.uuid4())
    reply = AIMessage(content=_CANNED_REPLY, id=message_id)
    trace = TraceEvent(
        event_type=EventType.MESSAGE_FINAL.value,
        data={
            "message_id": message_id,
            "text": _CANNED_REPLY,
            "citations": [],
        },
    )
    return {
        "messages": [reply],
        "execution_trace": [trace],
    }
