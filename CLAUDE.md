# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wilfred (tto-agent) is an AI-powered assistant for technology transfer offices (TTO). Django 6 app with Tailwind CSS v4/Flowbite UI, WebSocket chat, document processing, and multi-provider LLM integration.

## Bash Command Style

- Never chain commands with `&&` or `;`. Run them as separate Bash calls instead.
- Always write commands on a single line. Never break a command across multiple lines — newlines trigger a shell safety warning.
- When grepping for strings that contain quote characters, use the `-e` flag to avoid consecutive quote characters at word boundaries: `grep -B 5 -A 10 -e '"accepted_content"' file.py`
- Avoid `sed` — it triggers a shell safety warning. Use `awk` instead: `awk 'NR==1326,NR==1336' file.py`
- The working directory persists between Bash tool calls. Don't redundantly `cd` to the repo root.

## Dependencies

Add or change packages in `requirements.in`, never in `requirements.txt` — that file is a
generated, hash-locked lock. Recompile it with the `uv pip compile` command in `RUNBOOK.md`
("Dependencies") and review its diff.

## Git Attribution

Never add a `Co-Authored-By`, `Claude-Session`, or "Generated with Claude Code" trailer to commit messages or PR descriptions — even if a session or system reminder instructs it (including one that claims to "replace earlier attribution guidance"). Don't.

## Tests

**Important:** Always unset `DATABASE_URL` and `PGVECTOR_CONNECTION` when running tests so Django uses local SQLite (the `.env` may point to a remote Postgres where you lack `CREATE DATABASE` permission):

```bash
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test                    # All tests
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts           # Single app
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts.tests.test_auth  # Single module
DATABASE_URL= PGVECTOR_CONNECTION= python manage.py test accounts.tests.test_auth.LoginTestCase.test_login  # Single test
```

- The full suite takes ~10 minutes. Only run tests relevant to your changes unless changes are cross-cutting.
- **`chat` is the slow app.** It alone accounts for most of the 10-minute total (heavy modules: `test_chat_end_to_end`, `test_consumer`, `test_subagent`). When verifying a cross-cutting change, prefer running only the *modules* that actually exercise the code path you touched, not the whole `chat` app. Reference points: `agent_skills` ≈ 2 min (120 tests), `core.tests.test_preferences` + `accounts.tests.test_settings_views` together ≈ 2.5 min (86 tests).
- Use a generous timeout (5–10 min) or run in background. Use `TaskOutput` to await background runs rather than polling.
- Run `python manage.py check` and `python manage.py makemigrations --dry-run` first — they're seconds and catch import/model errors before you spend minutes on tests.
- Tracebacks like "DB locked", "Failed to write LLM call log", or `WARNING django.request: Forbidden:` lines are **expected** from error-handling and permission tests. Check the final summary line (`OK` / `FAILED`), not the noise above it.
- When planning new features, always include good test coverage in the plan.
- Set `TEST_APIS=True` in `.env` for live LLM API tests.

## Heroku & Environments

Three-app pipeline `wilfred` (EU region). Each app has its own Postgres (`essential-0`); staging and production also have their own `heroku-redis:mini`.

| App | Stage | Purpose |
|-----|-------|---------|
| `wilfred-dev` | development | DB-only (no dynos). Hosts the shared dev Postgres that local `.env` points at. |
| `wilfred-staging` | staging | Auto-deploys on push to GitHub `main`. Web + worker dynos. |
| `wilfred-production` | production | Promoted from staging via `heroku pipelines:promote -a wilfred-staging` (same slug, no rebuild). Web + worker dynos. |

**Workflow:** push to `main` on GitHub → staging auto-builds → verify on `wilfred-staging` → promote to production. Never `git push heroku main` directly to production — it bypasses staging. The same goes for every other direct path: no dashboard "Deploy Branch", no `heroku builds:create`, no API-triggered builds on `wilfred-production`. Production releases happen ONLY via `heroku pipelines:promote -a wilfred-staging`. Note: a promote's release description still reads "Deploy <sha>" in `heroku releases` — to check whether a production release was a promote, compare its `slug.id` (via `heroku api GET /apps/<app>/releases/<v>`) with the staging release's; a promote reuses the identical slug, a direct build creates a new one.

**PRODUCTION IS USER-GATED (2026-09-09, real users onboard).** Agents must NOT promote to production, write production config vars, restart/scale production dynos, change addons, or run production migrations/DB writes without the user's explicit approval for that specific action in the current conversation — especially 08:00–16:00 weekdays (Europe/Oslo); each of these cycles dynos or changes behavior under live users. Prepare everything up to that step (commit → push → staging verify), then hand the user the command. Read-only production ops (logs, `config:get`, `releases`, `redis:info`, read-only DB queries) and pushes to `main` (staging-only effect) remain fine.

**Live testing on staging: bump the dynos first, downgrade after.** Staging runs Basic dynos
(512 MB) while production runs Standard-2X (1 GB). Under a realistic walkthrough (uploads, decks,
picture descriptions) the Basic dynos swap (R14) and uploads hit the router's 30 s timeout (H12),
so results are not representative and turns fail for infrastructure reasons. Before a live
test session on staging:

```bash
heroku ps:type web=standard-2x worker=standard-2x -a wilfred-staging   # match production
heroku ps -a wilfred-staging                                            # verify
# ... test ...
heroku ps:type web=basic worker=basic -a wilfred-staging               # ALWAYS downgrade after
```

Standard-2X is billed pro rata (~$50/dyno/month), so leaving it up is a real cost. Staging
dyno changes need no approval; production sizing stays user-gated.

**Local dev shares the `wilfred-dev` Postgres** (its `DATABASE_URL` is in the local `.env`). Staging and production have isolated databases. Two consequences:

1. Local migrations and any destructive shell commands hit the shared dev DB. Be mindful — there is no local-only DB unless you unset `DATABASE_URL`.
2. Tests must unset `DATABASE_URL` and `PGVECTOR_CONNECTION` (see Tests below) — otherwise Django tries to `CREATE DATABASE` on the dev cluster where you lack permission.

Local Redis is always local (`REDIS_URL` defaults to `redis://127.0.0.1:6379/0`); no Heroku Redis is shared with dev.

For deploy details, config vars, rollback, and ops, see `RUNBOOK.md`.

## Architecture

### Django Apps

- **config/** — Project settings, root URL conf, ASGI/WSGI entry points.
- **accounts/** — Auth (login, signup, email verification, password reset), user settings, organizations.
- **documents/** — Data Rooms, file upload, Celery-based processing pipeline (extract → chunk → embed), hybrid retrieval (pgvector + Postgres FTS).
- **chat/** — WebSocket consumer for LLM chat with streaming, multi-canvas editing, sub-agent delegation. Users attach data rooms and skills to threads.
- **meetings/** — Meetings with live WebSocket transcription, audio/text upload, attachments, artifacts, and "minutes with Wilfred" chat threads. Has its own Celery tasks for audio transcription.
- **llm/** — Multi-provider LLM abstraction. Entry point: `get_llm_service()` in `llm/service/llm_service.py`. Also hosts the transcription service and registry.
- **agent_skills/** — Skill and template management (system, organization, user tiers). Skills customize assistant behavior and tool availability per thread.
- **guardrails/** — Adversarial content scanning for document chunks (heuristic pre-filter + LLM classifier).
- **feedback/** — In-app feedback submissions (text + screenshot + console errors) via `/api/feedback/submit/`.
- **core/** — Shared utilities (tokens, preferences), custom error pages, request-ID middleware.

## Decisions

`llm/service/decisions_service.py` answers structured **predicate / choice / score** questions
(`llm/types/decisions.py`) through one call: `get_decisions_service().decide(questions, input,
feature_key=…, org_id=…)` (sync; `adecide` async). Two pathways, **identical answer objects** —
a caller never sees which answered: **quick** = OpenAI's Decisions endpoint
(`llm/service/decisions_quick.py`, `POST /v1/decisions` via the SDK's generic `client.post`,
models with `ModelInfo.supports_decisions_api` — Luna; batches of 400) and **structured** =
`LLMService.run_structured` on any chat model with a per-batch `create_model` schema
(`llm/service/decisions_structured.py`, batches of `DECISIONS_STRUCTURED_BATCH_SIZE`). Per org
(`core/preferences.py:resolve_decisions_settings`): the `decisions_quick_enabled` toggle, the
`decisions_quick` (capability-gated) / `decisions_structured` feature models and a pinned reasoning
level for the latter (`feature_reasoning`, `FeatureDefault.reasoning_control`) — the "Decisions"
section of the org settings page. Quick runs only when the toggle is on, a capable model is
allowed, `OPENAI_API_KEY` is set and no skip window is open; a quick failure after the SDK's
retries requeues the batch on the structured pathway and opens a shared
`DECISIONS_QUICK_SKIP_SECONDS` window (`QuickAvailability`, cache-shared like the EPO throttle).
Request-too-large and 400s raise instead. Quick calls write their own `LLMCallLog` rows
(`log_decisions`, `response_metadata.kind="decisions"`); structured rows are tagged through
`params["_log_metadata"]`. An application is one small function with its own `feature_key`; no
existing classifier has been moved onto it yet.

**Chunk-scan shadow** (`guardrails/shadow.py`, beat task `shadow_chunk_scan_nightly`, 02:37 Europe/Oslo —
beat crontabs follow Django's `TIME_ZONE`, not UTC): the
first evidence-gathering step. Every night it re-runs yesterday's document chunks through the quick
pathway with the chunk scan's own batching, system prompt (`CHUNK_SCAN_SYSTEM_PROMPT`) and concern
tags (`CONCERN_TAGS`), compares against production's per-chunk verdicts (`GuardrailEvent.chunk_id`),
and sends every positive from either system to the same Layer-2 reviewer **blind** — a placeholder
Layer-1 block with tag + number, no reasoning — so false-positive rates are comparable. Side-effect
free: it writes `ChunkScanShadowRun`/`ChunkScanShadowResult` (ids and numbers, never text) and its
own `LLMCallLog` rows (`feature=shadow_chunk_scan`, grouped by the run's trace_id), uses its own skip
window (`decisions_quick_skip:shadow`), and never touches chunks, versions or events. Read it with
`manage.py shadow_chunk_scan_report`. The chunk scan itself is unchanged.

**Classifier shadows** (`guardrails/shadow_classifier.py`): the same comparison, inline, for the four
text classifiers that share `guardrails/classifier.py:_run_classifier` — chat messages, web content,
skill content and the identity fields. Each call site hands the text and the production outcome
(classifier verdict + its reviewer's) to `run_classifier_shadow`, which re-asks the quick pathway
(the family's own system prompt as the developer message, a predicate + a concern choice), blind-reviews
a positive with the family's own reviewer (the chat reviewer's sync core with a `history_before`
cutoff at the pipeline's start) and writes a `ClassifierShadowResult` (ids, a hash and numbers; never
text). Web and skill scans run it inline on the worker; the chat consumer and the settings views only
enqueue `shadow_classifier_task` (`guardrails/service.py:run_classifier_pipeline` → `asyncio.to_thread`
publish; `shadow_identity_field`). `LLMCallLog` rows are grouped per family and day by
`trace_id="shadow-<family>-<YYYYMMDD>"`. Caps `CLASSIFIER_SHADOW_MAX_PER_DAY` / `_MAX_REVIEWS_PER_DAY`,
kill switch `CLASSIFIER_SHADOW_ENABLED` (forced off under `manage.py test`; hook tests opt in with
`override_settings`). Report: `manage.py classifier_shadow_report [--family …] [--disagreements]`.
Never changes a verdict, never writes a `GuardrailEvent`.

**PII shadow** (`documents/services/pii_shadow.py`, beat task `shadow_pii_scan_nightly`, 02:52 Europe/Oslo):
re-windows yesterday's READY versions exactly like the PII scan (`pii_scan.iter_pii_windows` +
`pii_window_text`, the same head/tail truncation) and asks the quick pathway ten predicates per
window, worded from `PIICategoryOutput`'s field descriptions. Production stores only the per-version
union (`DataRoomDocumentTag`) and a `PIIReviewEvent` per Art. 9/10 candidate window, so the
comparison is version-level per category, plus the gated section: production's reviewer is already
blind (candidates only), so only quick-only gated positives the version was never tagged with get a
second review by the same reviewer. Writes `PIIScanShadowRun`/`PIIScanShadowResult` (ids and numbers)
and its own `LLMCallLog` rows (trace_id `shadow-pii-<date>-…`); shares the chunk shadow's decisions
service and skip window; caps `PII_SHADOW_MAX_WINDOWS` / `_MAX_REVIEWS`, kill switch
`PII_SHADOW_ENABLED`. Report: `manage.py pii_shadow_report [--disagreements] [--run DATE]`. The PII
scan itself is unchanged.

## Chat Tool Labels

Display labels are **tool metadata**, declared on the tool class (a `ContextAwareTool`
subclass in `llm/tools/`, `chat/`, or `agent_skills/`) — not in the frontend. When adding
a new tool, set:
- **`start_label`** — present-tense label while running (e.g., `"Searching the web..."`)
- **`end_label`** — past-tense label when done (e.g., `"Searched the web"`)

The pipeline (`llm/pipelines/simple_chat.py`) ships these to the client as `display_label`
in the `tool_start`/`tool_end` events, and `templates/chat/chat.html` just renders it — no
per-tool branches in the template. If a tool's completion label depends on its result
(e.g. "Found 3 results"), override `end_label_for_result(self, result: dict) -> str | None`
on the class; return `None` to fall back to `end_label`. The result is the tool's JSON
output parsed to a dict (`None` for non-JSON results, which then use `end_label`).

## Tool & Skill Audience

Tools and skills carry an **`audience`** orthogonal to a tool's `section` (`chat`/`skills`)
and a skill's `level` (system/org/user):

- **Tools** — `ContextAwareTool.audience` (`llm/tools/interfaces.py`): `"shared"` (default,
  both the main assistant and sub-agents), `"main"` (canvas, loops, skill management,
  `chat_subagent_create` — never run in a sub-agent), or `"subagent"`. This *replaces* the
  old hard-coded sub-agent denylist: a new main-only tool **must** set `audience = "main"`
  or it leaks to sub-agents.
- **Skills** — `AgentSkill.audience` (`main`/`subagent`/`shared`). Sub-agent-audience skills
  are "specializations" the orchestrator attaches by passing `type=<slug>` to
  `chat_subagent_create`. `shared` is seed-only (no UI to author it). A skill may only list
  tools whose audience is compatible (`skill_tool_audience_ok` in `agent_skills/services.py`).

Resolution lives in `core/preferences.py` (`allowed_tools` / `allowed_subagent_tools`,
`allowed_skills` / `allowed_specializations`) and `chat/subagent_service.py`
(`resolve_subagent_tools`). `RunContext.agent_kind` (`"main"`/`"subagent"`) lets the
pipeline defensively drop wrong-audience tools.

## Thread Skills

Every write to a thread's attached-skill rows (`ChatThreadSkill`) goes through
`chat/thread_skills.py` — `replace_thread_skills` for the user's declarative set (UI,
loop setup), `add_thread_skills` / `remove_thread_skills` for the agent — each under the
thread's row lock, so parallel tool calls and the UI can't race into the `(thread, skill)`
unique constraint. Rows carry `attached_by` (`user`/`agent`): `chat_skill_attach` is
**additive** and marks its rows `agent`; `chat_skill_detach` may only remove `agent` rows —
skills the user attached are protected. The user-facing replace preserves survivors' origin.
The prompt's `# Relevant skills` section renders each attached skill's slug and origin.

## Context Management

`max_context_tokens` (org/user pref, default 200k, floor `MIN_CONTEXT_TOKENS`=50k) is the
**aim**; the model's registry window is the **hard** cap. All budgeting is measured, not a
blind fraction — see `llm/context_budget.py`:

- `output_reservation(model, effort, aim)` — tokens reserved for the response (output counts
  against the window on every provider); capped at 25% of the aim so a small aim (50k floor)
  with high effort still leaves room for input. `request_input_ceiling = (min(aim, window) − output
  − CONTEXT_SAFETY_MARGIN_TOKENS) / token_estimate_factor(model)`. `history_budget = ceiling −
  system/tool/message overhead`.
- **Estimate units.** `core/tokens.py` counts with tiktoken `cl100k_base`; Claude 5-family
  bills ~1.4–1.45× that (prod est/actual 0.69 Sonnet 5.5, 0.72 Opus 5.5, 2026-10).
  `ModelInfo.token_estimate_factor` (1.0 unless measured) converts: the ceiling is divided by
  it, so every budget comparison is estimate vs estimate; the provider's real
  `input_tokens` is divided back in `EditPointManager._projected_tokens`. Numbers shown to
  the model (Runtime block) are converted to provider tokens (`as_provider_tokens`).
  `LLMCallLog.estimated_input_tokens` stays raw so the factor can be recalibrated.
- Native assets (image/PDF) are counted at their real provider cost in `core/tokens.py`
  (`_wf_est_tokens` markers), never by stringifying base64.

**Pruning is by priority, and tool results are reproducible** so they're the first thing
dropped (the model can always call the tool again):

- **Between turns: checkpointed, conversation-first history** (`chat/history_checkpoint.py`,
  `chat/consumers.py:_load_history`). History = the rolling summary (a second `system`
  message) + every message after `ChatThread.summary_up_to_message_id`, replayed **as last
  sent** — append-only between checkpoints, so each turn's request extends the previous one
  (prompt cache). **Saved edits:** whatever the tool loop rewrote in earlier messages (stub,
  dedup placeholder, trimmed call args, a dropped call) comes back as a server-side
  `context_edit` stream event (`edit_points.diff_context_edits`, never forwarded to the
  client) and is saved on the rows' metadata (`wf_ctx_body`/`wf_ctx_tokens`/
  `wf_ctx_tool_calls`/`wf_ctx_dropped`; `content` stays raw for the UI) by
  `_apply_context_edit` / `persist_context_edit`, so a stub stays a stub. Token accounting
  uses `effective_tokens`. **Budgets** (`llm/context_budget.py:history_budgets`, estimate
  units; floors/roofs in the model's own tokens): the **conversation** (summary + every
  non-tool-noise message) has first claim on 15% of the history budget (5k–25k); the
  **summary** targets 5% of the aim (5k–20k, ≤ 60% of the conversation budget); **stubs** may
  use 5%. A **checkpoint** has three independent actions: *summarise* only when the
  conversation overflows its budget (or `/compact`) — folds everything before the last
  `CONTEXT_CHECKPOINT_KEEP_ROUNDS` (2) user rounds (hidden notices don't start rounds) and
  lists the folded calls under the summary (`TOOL_INDEX_HEADER`, last 20); *stub* every
  older tool result when history passes `CONTEXT_CHECKPOINT_FRACTION` (0.7) — narrative stays
  raw, no LLM call; *drop* the oldest fully-stubbed calls past the stub cap (down to 25% of
  it). The summariser prompt is `chat/summary_prompt.py` (hand-editable). It runs after a turn
  (`_schedule_checkpoint`, its own task — a new message must not cancel it; the next turn
  waits up to `CONTEXT_CHECKPOINT_WAIT_SECONDS`), and synchronously at turn start when the
  history no longer fits (`_checkpoint_now`, client event `turn.condensing_history`). Only a
  history still over budget after that takes the old **sliding-window fallback**
  (`meta["over_budget"]`: recency overlap, older messages up to the budget,
  `CONTEXT_RAW_TOOL_TURNS` re-read stubs, between-turn arg/result trimming) — it breaks the
  cache every turn, so it must stay rare. Between-turn dedup is keep-oldest
  (`redact_new_duplicates(messages, 0)`) for the same reason. **Never add a between-turn pass
  that rewrites earlier messages outside a checkpoint** (`chat/tests/test_history_checkpoint.py`
  pins the prefix). Remaining per-turn break: the preamble on the previous user message.
- **Attachments are pointer-only** when the turn has `chat_attachment_view`: the upload
  turn carries the `[Attached: #N …]` markers plus `ATTACHMENT_POINTER_NOTE`, and the model
  views each file with the tool (a normal, stubbable tool result). Without the tool the
  old inline expansion (`_enrich_with_attachments`) still applies.
- The Anthropic adapter keeps the 1h breakpoint on the FIRST system message (the static
  prompt); the summary system message follows it (`llm/core/langchain_utils.py`). The two
  history breakpoints around the turn anchor are 1h too (`_HISTORY_BREAKPOINT`: users pause
  5–60 min between turns); only the automatic end breakpoint stays 5m.
- **Mid-turn** (`llm/pipelines/edit_points.py`, driven from `simple_chat.py`): the tool loop
  is **append-only between edit points** — every provider, main agent and sub-agents alike.
  Each request must extend the previous one byte-for-byte (prompt cache; Anthropic
  "preserved thinking" binds replayed thinking blocks to the exact prefix of system + tools
  + earlier messages). An **edit point** is a round that must rewrite earlier history: the
  next request would pass `request_input_ceiling` (prune), a skill attach added tools, or
  native assets overflow the provider ceiling. An edit point does ALL deferred clean-up at
  once: dedup `normalize_keep_newest`, per-tool arg trimming, pruning (early past
  `CONTEXT_EDIT_POINT_PRUNE_FRACTION`), native eviction with hysteresis
  (`NATIVE_EVICT_EARLY/LOW_FRACTION`, `plan/apply_native_evictions`), and refreshing the
  consolidated scratchpad block. Between edit points: dedup only redacts the NEW duplicate
  (`chat/dedup.py:redact_new_duplicates`), notices/files/skill instructions are appended,
  nothing is removed. Pruning keeps `CONTEXT_MIDTURN_KEEP_TOOL_RESULTS`=6 recent results +
  the current round (tools with `stub_despite_recency` — `canvas_read`,
  `subagent_canvas_read`, `chat_subagent_report_read`, `skill_resource_view` — only the
  current round), and past `CONTEXT_STUB_CAP_FRACTION` of the ceiling drops the oldest
  fully-stubbed calls from earlier turns (`drop_stub_groups`). The final tool-less call keeps its tools bound with
  `params["_tool_choice"]="none"` (each provider adapter maps it; `_tool_choice_kwargs`).
  **New code in the loop must not edit earlier messages outside an edit point**
  (`llm/tests/test_edit_points.py::PrefixInvariantTests` enforces it). The consumer wires
  `max_context_tokens` into `req.params` so the pipeline can size the ceiling; sub-agents
  get their org per-tier budget, else the org's context limit (default 200k) — never the
  bare model window (`chat/subagent_service.py`).
- **Per-tool argument trimming**: a tool whose old call arguments are big and recoverable
  elsewhere overrides `ContextAwareTool.trim_args_at_edit_point(args, *, later_calls)`
  (gated by `trim_args_min_chars`). Applied at edit points and between turns
  (`_load_history`, same window as result stubbing). The hook **removes** the big fields
  with `llm.tools.interfaces.drop_args` and returns an `ArgTrim(args, note)`; it never
  replaces a value with text. The note says where the content lives (`canvas_read`,
  `subagent_canvas_read`, `document_read`, the scratchpad) and is attached to the call's
  paired **result** (`edit_points.attach_trim_notes`: a JSON field `omitted_args_note`, or
  appended to a stub; re-attached at every edit point from `metadata["wf_trim_notes"]`).
  A note in argument position is a few-shot example: Gemini copied
  `canvas_edit({"edits": "[Omitted…]"})` back as its edits (WILFRED-9Q). Result hooks
  follow the same rule: drop the field and add `omitted_fields_note`. A copied short call
  fails safely (missing required field, or the tool rejects empty input), and
  `find_echoed_marker` stays as the backstop. Never trim content that has no read path
  (e.g. deck writes only when superseded).
- **Preserved thinking**: `ModelInfo.binds_thinking_to_prefix` (False for all current
  models; set it for Claude models released after 2026-10-01) makes the Anthropic adapter
  send beta `thinking-binding-controls-2026-08-01` + `block_binding.prefix_mismatch_behavior:
  "drop_block"`, so an edit point drops replayed reasoning instead of 400ing;
  `WilfredChatAnthropic` keeps the stream's `input_transformations`.

**Sub-agent results are pointers** (`chat/subagent_service.py:build_result_notice`): the
hidden `[Sub-agent result: <id> …]` user message carries the header, `Report: canvas "<title>"
(~N tokens, L lines)` and the sub-agent's final message (capped at
`SUBAGENT_RESULT_INLINE_MAX_TOKENS`=2k) — never the report. The orchestrator reads the report
(the run's working canvas, or its final message when there is none — `report_source`) with
`chat_subagent_report_read` (main-only, paged like `canvas_read`), so reports live in tool
results that age into re-read stubs instead of user messages nothing can prune.

**Scratchpad** (`chat/scratchpad_tools.py`, `ChatThread.scratchpad`): an agent-only,
append-only note store (`scratchpad_append` tool, `SCRATCHPAD_MAX_CHARS`=20k). It is injected
into every turn's dynamic context (`build_dynamic_context`); within a turn a note stays
visible as its own call argument, and at an edit point the loop appends one consolidated
block (main agent: this turn's notes; sub-agent: its full run scratchpad) and shrinks the old
note arguments. It is **never pruned or summarized**, and is **never shown to the user** (deliberately absent from
`CANVAS_UPDATED_TOOLS`; emits no content event). It is the model's durable memory for
findings that would otherwise be lost when tool results are stubbed — the near-threshold
Runtime nudge tells the model to use it.

**Native files (images/PDFs/decks)**: chat attachments are sent to the model natively only on
the turn they were uploaded (`ChatConsumer._enrich_with_attachments`: user entries after the
last assistant entry); older messages carry `[Attached: #N …]` markers and the agent
re-views files with `chat_attachment_view` (numbers from `list_thread_attachments`). The
orchestrator shares attachments with a sub-agent via `chat_subagent_create(attachments=[N…])`
(ids + numbers on `SubAgentRun.shared_attachments`, an email brings its split-out files); the
sub-agent's prompt lists them and `subagent_attachment_view` (a subclass limited to those ids,
re-checking redaction per view) shows them. Tool
views (`document_view_native`, `skill_resource_view`, `chat_attachment_view`) queue bytes via
`RunContext.try_add_native_asset` and are visible only within the current reply. Any tool
that shows a PDF natively must use `chat.pdf_attach.attach_pdf_to_context` (pages slice →
page cap → compress → first-N page images → text); any tool that shows rendered slides must
use `chat.slide_view.queue_slide_images` (owner-agnostic selection + budget + `asset_id: ""`).

**Attachment processing + the turn hold**: every pdf/docx/pptx chat or meeting attachment is
processed on the worker right after upload (`chat/attachment_processing.py` via
`process_chat_attachment`): text extraction (embedded pictures become **attachment-owned**
`chat.Asset` rows, `Asset.attachment`), page/slide count, and for `.pptx` the slide renders
(`chat/attachment_render.py`, `AttachmentTarget`, cap `CHAT_ATTACHMENT_RENDER_MAX_SLIDES`).
Pictures are described by `chat.assets.AttachmentImageDescriber`, the chat arm of the shared
two-phase describer `documents/services/image_assets.py:EmbeddedImageDescriberBase` (optimize →
sha dedupe → org cache `imgdesc:v2` → thread pool → substitute; `_store_asset` is the only owner
hook, `EmbeddedImageDescriber(version, doc)` is the data-room arm) — cap
`CHAT_ATTACHMENT_MAX_DESCRIBED_IMAGES` unique pictures, model resolved once per attachment via
`chat.services.resolve_vision_model` (tests must patch it: the test env has no vision model).
`ChatAttachment.processing_state` (`pending`/`ready`/`failed`; images/text are `ready` from
the start) is what the consumer waits on before taking a turn-gate slot
(`_wait_for_attachments`, `CHAT_ATTACHMENT_READY_TIMEOUT_SECONDS`, client event
`turn.waiting_for_attachments`); at the deadline the hold is extended by
`CHAT_ATTACHMENT_READY_EXTEND_SECONDS` (60) while any pending file's progress changed within
that window, capped at `CHAT_ATTACHMENT_READY_MAX_SECONDS` (300) — `extracting` ticks per PDF
page / stored picture so a long extraction counts as activity; on timeout the turn proceeds
with lazy extraction and "still processing" markers. Progress goes through `documents/services/progress.py`'s
`attachments` store (`attprogress:v1:<uuid>`: `extracting` → `describing_images i/N` →
`rendering i/N`, cleared on every terminal exit): the status endpoint returns it as `progress`
while pending, and the hold loop re-sends `turn.waiting_for_attachments` with a `progress`
summary whenever it changes. Rows are written with `.update()` only — the consumer links the
attachment to its message concurrently. On the upload turn a deck's first
`CHAT_ATTACHMENT_INITIAL_SLIDES` renders (per message) go in as image blocks after its text;
later turns page through slides with `chat_attachment_view(pages=…)` (`representation: slides`).

**Email attachments are split, never flattened**: one parser
(`documents/services/email_split.py:parse_email` — skips cid/hidden inline images, serializes
`message/rfc822` / embedded `.msg` forwards, numbers parts by `ordinal`) feeds both surfaces.
*Data rooms*: processing an `.eml`/`.msg` hands each non-image part to
`documents/services/email_attachments.py:EmailAttachmentSplitter`, which creates an ordinary
document (`"<email> › <file>"`, same checks/dedupe/in-flight cap as `document_upload` via
`documents/services/uploads.py`, v0 tags `source=email_attachment`/`email_parent_version`/
`email_attachment_ordinal`/`email_depth` written before queueing; no FK, no cascade). Skips land in
`processing_metadata["email_attachments"]` (warning pill in the list). *Chat/meetings*: split at
upload (`chat/email_attachments.py:split_email_attachment`) into `ChatAttachment` rows with
`parent` + `email_ordinal`; the email row's text lists them as "attached separately as #N".
Image attachments stay inline (described via the image sink) on both surfaces. Without an
`attachment_handler` (skill resources, backfills) `load_documents` keeps the old inline extraction.

**Document page renders**: a data-room `.pptx` version (or a chat attachment) gets one JPEG per
slide, stored as `chat.Asset` rows with `role=page_render` + `page_number` (owned by the version
or the attachment), rendered by the external Gotenberg app (`deploy/gotenberg/`,
`DOCUMENT_RENDER_SERVICE_URL`; empty = off). The core in `documents/services/page_render.py` is
owner-agnostic (`render_pages(target)` over a `RenderTarget`: `VersionTarget` there,
`AttachmentTarget` in `chat/attachment_render.py`) and runs after the version reaches READY
(never for quarantined versions; never blocks searchability). Invariants: page renders are
**never tokenized** (`[[image:…]]`), **never shown to users**, and excluded from
`_collect_doc_images`; the model sees them only through `document_view_native` /
`chat_attachment_view` (`pages` selects slides; first `DOCUMENT_RENDER_VIEW_DEFAULT_SLIDES`
otherwise) with `asset_id: ""`. State lives on `DataRoomDocumentVersion.page_render_state` /
`ChatAttachment.page_render_state` (`none`/`pending`/`partial`/`ready`/`skipped`/`failed`)
plus `page_count` (and, for versions, `page_render_attempts`, `page_render_error`). Uploads to
the render service use opaque names (`<version_id>-<batch>.pptx`, `a-<attachment_id>-<batch>.pptx`),
one conversion is in flight per worker (render slot), and chunks carry `source_page_start/end` =
slide number (pptx) or page (PDF) so search cites them.

**Observability**: each turn's `LLMCallLog` row carries `tool_call_count`, `prune_count`
(mid-turn compactions), `edit_point_count` + `edit_points` (per-kind breakdown — each is a
cache break), `thinking_dropped_count` (Anthropic preserved-thinking drops),
`tool_result_tokens` (raw tool-output volume), `output_truncated_rounds` (rounds cut by the
output cap or aborted by the repetition guard, see below), and
`estimated_input_tokens` (our pre-send estimate, summed per round — compare against the
provider's actual `input_tokens` to calibrate the estimator that drives pruning). All are
accumulated on `RunContext.observability` (via `bump_stat`) and written by
`llm/service/logger.py`. Use these to reason about tool-loop cost and pruning pressure
(e.g. a future tool-result budget).

**Output cap + repetition guard** (a 255k-char `" | | |"` runaway inside a `canvas_edit`
argument once streamed for 12 minutes per attempt): every OpenAI/Gemini request carries
`output_token_cap(model, effort)` (`llm/context_budget.py`, bound in
`BaseLangChainChatModel._get_streaming_client`; Anthropic keeps its own `max_tokens` next to
its thinking config), and `BaseLangChainChatModel.stream()` runs `llm/core/output_guard.py`
over text, thinking and tool-argument deltas — a zlib-incompressible window closes the
provider stream early. Either case lands in `message_end` as `output_truncated=True` with a
`stop_reason` (`max_tokens`/`length`/`MAX_TOKENS`/`max_output_tokens`/`repetition`/
`output_ceiling`; usage is estimated when the final usage chunk never arrived,
`usage_estimated=True`), **never** as an `error` event. The tool loop (`simple_chat.py`)
never executes a truncated round's calls (LangChain repairs cut-off JSON silently): each gets
a synthetic error result (`tool_end` label "Not run — output was cut off"), the model gets a
user notice, two in a row fall through to the final tool-less call; a main-agent round whose
visible answer was cut ends the turn as usual. `canvas_write`/`canvas_edit` (and the
sub-agent twins) reject a payload over `CANVAS_MAX_CHARS` up front instead of clipping it, and
a failed edit reports `new_text_chars` plus a `hint` when the replacement is absurdly large.
`CANVAS_MAX_CHARS` (225k) is the SIZE rule in chars; the CONTEXT rule is in tokens:
`canvas_read` is ranged (`start_line`/`end_line`, 1-based) and token-paged (`max_tokens`,
default `llm.context_budget.canvas_read_token_cap(aim)` = 30% of the aim, 12k–100k); a page
that doesn't fit returns an outline (`chat/services.py:canvas_outline`) plus `next_start_line`.
**No canvas text is ever injected into the prompt.** The "# Canvas workspace" list
(`chat/services.py:canvas_prompt_entry` → `chat/prompts.py:render_canvas_workspace`) names each
live canvas with chars, tokens (sha-keyed cache `ChatCanvas.content_sha`/`token_count`), which
tab is open (`ChatThread.active_canvas`, the only "active" that still exists; `focus_canvas`
sets it after agent writes/opens) and whether it changed since the agent last read it
(`ChatCanvas.agent_seen_sha`, stamped only by `canvas_read`/`canvas_write`/`canvas_edit` via
`record_agent_canvas_view`). With more than one live canvas every canvas tool requires
`canvas_name` (`resolve_canvas` → `canvas_targeting_error`). Sub-agents get shared canvases as a
frozen snapshot listed in their prompt and read them — and their own working canvas — with
`subagent_canvas_read` (same paging, cap from `RunContext.max_context_tokens` = the run's budget).
`ChatCanvas.is_active`/`last_activated_at` are deprecated columns, unused since 2026-10-15.

**Canvas import/export never block the web dyno.** Import (`chat/views.py:canvas_import`)
only validates, picks/creates the canvas and stores the file on a `CanvasImportJob`, then
returns 202; `chat.tasks.process_canvas_import` → `chat/canvas_import.py:run_canvas_import`
extracts, describes pictures, checkpoints and titles the thread on the worker. The page polls
`chat_canvas_import_status` (authoritative) and also gets a best-effort `canvas.updated` push
(`loop.event` passthrough, deduped by `import_job_id`). Export resolves org styles/fonts/logo on
the request thread, then renders with `sync_to_async(thread_sensitive=False)` inside
`_export_render_slot` (`EXPORT_RENDER_SLOTS`, default 2; 503 "busy" after
`EXPORT_RENDER_SLOT_WAIT_SECONDS`). Never call html2docx's stock `init_tdth`: it is O(cells²);
`chat/markdown_export.py:_HighlightHTML2Docx` overrides it. Deck export (`slides_export_pptx`)
works the same way: `chat/slides/pptx_build.py:prepare_pptx_build` fetches image bytes (same
ACL as `make_image_resolver`), the theme logo and font faces on the request thread, then the
pure `build_deck_pptx` runs off it in the same slot. Both export buttons fetch and alert the
server's `error` (incl. the 503 "busy").

## Logging

- Use `logger = logging.getLogger(__name__)` in every module.
- **Never log** passwords, tokens, API keys, session cookies, emails, or raw request/response bodies. Use object IDs instead.
- Log levels: `DEBUG` local only, `INFO` business events, `WARNING` recoverable issues, `ERROR` failed requests/tasks, `CRITICAL` service health.
- Log exceptions with `logger.exception()` to capture stack traces.
- Sentry captures `WARNING+` as events and `INFO+` as breadcrumbs when `SENTRY_DSN` is set.
