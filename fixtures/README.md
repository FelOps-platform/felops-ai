# Fixtures

Fixed seed data for the demo and the evals. Nothing here is generated at runtime, and nothing here is production content.

Fictional brand: **Dar Nour** (دار نور). Domains use the reserved `.example` TLD, and the phone number is not real. Never point a fixture at a real customer or a live store.

## Groups

| Group | Status | Owner task | Consumed by |
|---|---|---|---|
| `kb/` — bilingual knowledge base | done | 0.10 | 1.3 ingestion, 1.4 recall, 2.7/2.8 parsers, 3.12 chunking, 4.9 Arabic normalization, 5.1 hybrid |
| `qa/qa_seed.jsonl` — 20 labelled Q&A items | done | 0.10 | 1.4, 2.9 (Recall@k, MRR), 4.9, 8.4, 8.11 |
| `actions/` — 6 action specs | done | 0.10 | 1.10 (mock MCP), 1.12 (loader), 2.1, 3.7 |

`onboarding-guide.docx` (the failing-then-retried source in §8.5) belongs with the parser fixtures in P2·2.7, not here.

## Knowledge base conventions

- **One language per file.** `kb/en/<doc_id>.txt` is English, `kb/ar/<doc_id>.ar.txt` is Arabic. ASCII filenames only; the `.ar.` marker is the locale.
- **`.txt` on purpose.** P1·1.3 ingests text/TXT only. Markdown-style ATX headings (`#`, `##`) are used *inside* plain text so heading-aware chunking (P3·3.12) has section structure to read without needing a Markdown parser.
- **No frontmatter, no HTML, no tables.** Keep chunks clean for embedding and for citation display.
- **`doc_id` is the filename stem and is stable.** `qa/` refers to it as `gold_source_id`, and ingestion maps it to `NameSpace(workspace_id, agent_id, source_id)`.
- **AR mirrors EN section for section and fact for fact.** Same `##` headings in the same order, same numbers, same policy outcomes. This is what makes a bilingual eval meaningful: the same question in Arabic must retrieve the Arabic document and yield the same answer.
- **Western digits everywhere** (`250`, `14`, `TRK8812`, `#1042`). Arabic-Indic digits are deliberately avoided so that retrieval and the future Arabic normalization step (P4·4.9) are not confounded by numeral variance.
- **Light Arabic marks are kept where normal in formal writing** (tanween on indefinites, shadda on common verbs). There is no full tashkeel and no artificial diacritization of the corpus. Normalization robustness (P4·4.9) is tested from the query side in `qa/` (`ar_variants`).
- **`manifest.json` and this README are not ingested.** Ingest from `kb/en/**` and `kb/ar/**` only.

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
- **Figures must be identical in both languages.** They are written as digits in English and Arabic alike (`2,000`, `09:00`, `TRK8812`), so a numeric mismatch between the pair is a content bug, not a translation choice.
- **`ar_variants`** (optional, 12 strings across 9 items) are deliberately noisy renderings of the same Arabic question: full tashkeel, bare alef (`أ`/`إ`/`آ` → `ا`), alef maqsura (`ى` → `ا`), and taa marbuta (`ة` → `ه`). They exist to prove P4·4.9 normalization maps every variant onto the canonical query, not just to the KB.
- **Not covered:** questions that need an action, an order lookup or customer data (§8.2 steps 3-11). Those belong to the tool scenarios in 8.11, not to the retrieval seed.

## Action spec conventions

Six files, one per catalog action. These are not knowledge and are never ingested.

| `action_id` | Integration | Class | Verifier |
|---|---|---|---|
| `lookup_order` | Shopify | read-only | `self_read` |
| `refund_order` | Shopify | side-effecting | `read_back` → `lookup_order` |
| `send_invoice` | QuickBooks | side-effecting | `read_back` → `get_invoice` |
| `get_invoice` | QuickBooks | read-only | `self_read` |
| `create_case` | Salesforce | side-effecting | `read_back` → `get_case` |
| `get_case` | Salesforce | read-only | `self_read` |

- **`action_id` is the filename stem and the stable id.** The numeric ids `[0..7]` in §4.1 are illustrative; manifests, policy and events use these strings.
- **`side_effect_class` is the only input to the executor's idempotency decision** (§5.4): `side_effecting` gets an `idem_key`, `read_only` never does and always runs live.
- **`mcp_tool`** is the tool name the mock MCP server exposes (1.10). **`integration`** is the connector id used in traces and `effective_tools`.
- **`parameters`** is a JSON Schema used by `validate_args` (2.1), with `additionalProperties: false` everywhere. **`result`** is the response contract the mock must return.
- **`verifier`** is a two-kind DSL. `self_read` re-issues the same read and compares; `read_back` calls the paired read action after a side-effecting call. `arguments` and `assert.equals` accept literals or `$args.<name>` / `$result.<name>` placeholders.
- **Read-back verifiers depend on the paired read exposing the changed field.** `refund_order` is checked through `lookup_order.refund_status` and `.refund_amount`; `send_invoice` through `get_invoice`; `create_case` through `get_case`.
- **`requires_approval` is deliberately absent.** It is per (agent, action), resolved by the backend and delivered as a single boolean (§4.1 inv. 5). Demo approval flags belong in the mock backend config fixtures (0.8).
- **No write spec asserts its own success.** A `tool.completed` result is not business success; only the verifier's read is evidence (§6.5 case 2).
- **`example` pins the §8.2 demo values** and doubles as the contract-test case for 1.10 and 1.12: order `#1042` (total `AED 349.00`, so free delivery), tracking `TRK8812`, invoice `INV-2210` at `349.00` to `sara.ali@example.com`, refund `RFD-3081`, case `CASE-5512`.

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
