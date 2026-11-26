# Fixtures

Fixed seed data for the demo and the evals. Nothing here is generated at runtime, and nothing here is production content.

Fictional brand: **Dar Nour** (دار نور). Domains use the reserved `.example` TLD, and the phone number is not real. Never point a fixture at a real customer or a live store.

## Groups

| Group | Status | Owner task | Consumed by |
|---|---|---|---|
| `kb/` — bilingual knowledge base | done | 0.10 | 1.3 ingestion, 1.4 recall, 2.7/2.8 parsers, 3.12 chunking, 4.9 Arabic normalization, 5.1 hybrid |
| `qa/qa_seed.jsonl` — 20 labelled Q&A items | done | 0.10 | 1.4, 2.9 (Recall@k, MRR), 4.9, 8.4, 8.11 |
| `catalog/catalog.json` — 6 catalog tool entries | done | 0.10 | 0.8 (catalog endpoints), 1.10 (mock MCP), 1.12 (loader), 2.1, 2.10 (catalog sync), 3.7 and 7.6 (outcome handling) |

`onboarding-guide.docx` (the failing-then-retried source in §8.5) belongs with the parser fixtures in P2·2.7, not here.

## Knowledge base conventions

- **One language per file.** `kb/en/<doc_id>.txt` is English, `kb/ar/<doc_id>.ar.txt` is Arabic. ASCII filenames only; the `.ar.` marker is the locale.
- **`.txt` on purpose.** P1·1.3 ingests text/TXT only. Markdown-style ATX headings (`#`, `##`) are used *inside* plain text so heading-aware chunking (P3·3.12) has section structure to read without needing a Markdown parser.
- **No frontmatter, no HTML, no tables.** Keep chunks clean for embedding and for citation display.
- **`doc_id` is the filename stem and is stable.** `qa/` refers to it as `gold_source_id`, and ingestion maps it to `NameSpace(workspace_id, agent_id, source_id)`.
- **AR mirrors EN section for section and fact for fact.** Same `##` headings in the same order, same numbers, same policy outcomes. This is what makes a bilingual eval meaningful: the same question in Arabic must retrieve the Arabic document and yield the same answer.
- **Western digits for all numerals** (`250`, `14`, `TRK8812`, `#1042`). Arabic-Indic digits (`٠–٩`) are avoided so retrieval and Arabic normalization (P4·4.9) are not confounded by numeral scripts. Small counts `1` and `2` (and `"four digits"` / `"أربعة أرقام"`) may use natural singular/dual/word forms in Arabic (`يوم عمل واحد`, `ساعة واحدة`, `يومَي عمل`, `محاولتَي`) and English (`one`, `two`) where idiomatic, provided both languages express the exact same count (`1` vs `1`, `2` vs `2`).
- **Light Arabic marks are kept where normal in formal writing** (tanween on indefinites, shadda on common verbs). There is no full tashkeel and no artificial diacritization of the corpus. Normalization robustness (P4·4.9) is tested from the query side in `qa/` (`ar_variants`).
- **`manifest.json` and this README are not ingested.** Ingest from `kb/en/**` and `kb/ar/**` only.
- **Seeded through the mock backend, not read from disk.** The demo/reset script uploads these files and the AI repo ingests them from a presigned S3 `ref` (§8.5); the paths here are the fixture source of truth, not what the API receives.

## Q&A conventions

`qa/qa_seed.jsonl` — one JSON object per line, UTF-8, not ingested. Twenty items, four per document, each carrying an English and an Arabic rendering of the same question and answer. That is **20 items and 40 query instances**, which is what 2.9 means by "20 Q&A" and what makes the bilingual comparison in 4.9 and 5.9 possible.

```json
{"id": "qa-001", "doc_id": "shipping-delivery", "section": "Shipping fees",
 "question": {"en": "...", "ar": "..."},
 "answer":   {"en": "...", "ar": "..."},
 "ar_variants": ["..."]}
```

- **`doc_id` + `section` is the gold label**, `doc_id` for document-level recall and `section` for chunk-level MRR/nDCG. `section` must equal a `## ` heading in the English document; the Arabic section is found by position, because the headings mirror.
- **Answers are short and assert only facts present in the gold section.** No advice beyond stated policy, no cross-document synthesis. This keeps groundedness scoring (8.4) unambiguous.
- **Figures must be identical in both languages.** Quantities `>= 3`, amounts, percentages, times, and identifiers are written as identical Western digits in English and Arabic alike (`2,000`, `09:00`, `TRK8812`); counts `1` and `2` may use natural singular/dual wording (`1`/`one`/`واحد`/`واحدة`, `2`/`two`/dual) as long as the underlying count (`1` vs `1`, `2` vs `2`) matches across the pair. Any mismatch in value (including `1` vs `2`) is a content bug, not a translation choice.
- **`ar_variants`** (optional, 12 strings across 9 items) are deliberately noisy renderings of the same Arabic question: full tashkeel, bare alef (`أ`/`إ`/`آ` → `ا`), alef maqsura (`ى` → `ا`), and taa marbuta (`ة` → `ه`). They exist to prove P4·4.9 normalization maps every variant onto the canonical query, not just to the KB.
- **Not covered:** questions that need an action, an order lookup or customer data (§8.2 steps 3-11). Those belong to the tool scenarios in 8.11, not to the retrieval seed.

## Catalog tool conventions

One file, `catalog/catalog.json`, holds the six entries the mock backend pushes to the AI repo as the system catalog. They are not knowledge and are never ingested.

| `tool_id` | `connector_id` | Kind | Preview |
|---|---|---|---|
| `lookup_order` | `shopify` | read-only | runs live |
| `refund_order` | `shopify` | side-effecting | mocked |
| `send_invoice` | `quickbooks` | side-effecting | mocked |
| `get_invoice` | `quickbooks` | read-only | runs live |
| `create_case` | `salesforce` | side-effecting | mocked |
| `get_case` | `salesforce` | read-only | runs live |

- **One snapshot, six entries.** The catalog is delivered as one full `{tools: [...]}` payload, never as per-tool patches (§4.3, §7.1). The mock backend (0.8) serves `tools[]` from `GET /catalog`, and the Tool Registry loader (1.12) loads all six entries. `catalog.json` is the fixture source of truth; there is no per-tool file.
- **`tool_id` is the stable id.** The numeric ids `[0..7]` in §4.1 are illustrative; manifests, policy, events and `agent_config.tools` use these strings. A tool id is never reused: the backend deprecates a tool (`deprecated: true`) instead of deleting it (§4.1 inv. 8).
- **The entry is exactly the §4.3 field set:** `tool_id`, `connector_id`, `description`, `parameters`, `result_summary`, `effect_summary`, `version`, `deprecated`. There is no `mcp_tool`: the MCP Client calls `call_tool(connector_id, tool_id, args, idem_key)`, and the mock MCP server exposes each tool under its `tool_id` (1.10). Display labels are backend-owned; the trace can fall back to `tool_id` and `connector_id`.
- **Read vs. write is inferred from `effect_summary`.** `effect_summary: null` marks a read-only tool; a non-null deterministic template marks a side-effecting tool. Preview (4.5) mocks tools with a non-null `effect_summary` and runs `null` tools live. This is a preview/display rule only: the executor's idempotency ledger keys **every** call, read or write (3.6), and `idem_key` is required by the `Connector` protocol (§5.1).
- **`parameters` is the input JSON Schema** used by `validate_args` (2.1), with `additionalProperties: false` everywhere. This is the field name the catalog, the mock backend and the Tool Registry agree on.
- **`result_summary` renders from the `data` object of the execution result; `effect_summary` renders from the call arguments.** Both are deterministic `{field}` templates, never free LLM text (§8.2 step 6, §8.3). Example: `"Invoice {invoice_number} sent to {recipient_email}"` and `"Refund AED {amount} on order {order_number} to the original payment method"`.
- **There is no verifier and no result schema in the entry.** Verification moved to the backend (§6.5 case 2): a call returns a rich execution result, not a `verifier.failed` event. `result_summary` is the only result-shaped field in the catalog.
- **`requires_approval` is deliberately absent.** It is per (agent, tool), set only by the user, and delivered in `agent_config.tools` (§4.1 inv. 5). The catalog has no approval flag; demo approval flags belong in the mock backend's `agent_config` fixtures (0.8).
- **`examples` and `failure_example` are fixture-only.** They are not part of the pushed catalog. `examples` pins the §8.2 demo values and doubles as the contract-test case for 1.10 and 1.12: order `#1042` (total `AED 349.00`, so free delivery), tracking `TRK8812`, invoice `INV-2210` at `349.00` to `sara.ali@example.com`, refund `RFD-3081`, case `CASE-5512`. Each result is the 3.7 execution envelope `{status: "succeeded", data, error: null}`; `failure_example` shows the `{status: "failed", data: null, error: {code, message}}` path.
- **No write entry asserts its own success.** A `tool.completed` result is not business success; the backend verifies and returns the outcome (§6.5 case 2).
- **`version` is per tool** and is emitted as `tool_version` on `tool.started` (§7.2).

## Facts the KB fixes

These are referenced by §8.2 and by the Q&A seed, so never change them in one language only:

| Fact | Value |
|---|---|
| Order (demo) | `#1042` |
| Tracking number format / example | `TRK` + four digits / `TRK8812` |
| Invoice number format / example | `INV` + four digits / `INV-2210` |
| Free-shipping threshold | AED 250 |
| Standard delivery fee / remote surcharge | AED 25 / AED 40 |
| Cash-on-delivery fee / order cap | AED 15 / AED 1,500 |
| Return window / return fee | 14 calendar days / AED 20 |
| Refund lead time | 5 to 7 business days |
| Warranty | 12 months (6 on electrical components) |
| Cancellation window | 1 hour after placing the order |
| Support hours | Mon-Fri 09:00-18:00, Sat 10:00-14:00 GST |
