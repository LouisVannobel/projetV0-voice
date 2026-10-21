# Calendar Dialogue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Recognize one clear consent, report the verified booking, and use native closing for clear contextual completion.

**Architecture:** Repair the existing whole-utterance parser and bounded result waiter in calendar.py plus model-facing closing instructions/schema in pipeline.py. Existing session authority, backend irreversible-send state, typed matching booking and owned final playback remain authoritative.

**Tech Stack:** Pinned Python3.13.15, Pipecat1.12.0, current SDK/HTTPX/PostgreSQL contracts, uv.

**Spec:** docs/superpowers/specs/2026-10-21-calendar-dialogue-design.md

## Global Constraints

- Preserve the existing 12-second result bound, permanent-failure/cancellation handling, frozen pre-send retry contract and all consent/ACK/current-scope/financial guards.
- After observing unknown, only read the fixed canonical operation; no later error/missing/waiting response may enable submission or change that identity.
- No App change, new abstraction/dependency/metric/model/voice/provider, actual event/transcript rewrite or paid replay. Separate USD5 caps and original lifetime5/used4 authority remain.
- Root alone owns external actions. Offline synthetic native tests; no provider/SSH/Docker/GitHub/Oracle/subagents by implementer. Unchanged complete hosted Linux is Root's canonical full gate, not a duplicate full Windows run.

## Review Focus

- Sentence punctuation cannot convert a correction/question/condition into consent; whole utterance and native proof remain required.
- A possible-send observation followed by read errors or missing state remains read-only on the same canonical operation.
- A permanent unknown reaches the existing bound without invented success or duplicate creation.
- Delayed confirmation after caller departure remains silent; no stale consent or action revival.
- Contextual closing uses one owned final playback and ACK; generic declines/assistant farewell never acquire deterministic closing authority. Scripted tests cannot prove model semantic choice.

### Task 1: Correct the existing consent/result/closing dialogue

**Files:**
- Modify src/projetv0_voice/calendar.py: is_affirmative/needs_confirmation_clarification and await_calendar_result.
- Modify src/projetv0_voice/pipeline.py: existing French system instructions and end_call schema description only.
- Tests: tests/test_calendar_tools.py, tests/test_calendar_postgres.py and existing integration/test_voice_action_authority.py or a focused native fixture extension. Do not alter lifecycle or App contracts to make tests pass.

**Interfaces:** Consumes existing CalendarSink/CalendarResult, exact native recap playback proof and scoped end_call callback. Produces the same return types, same operation identity and same final-frame path. Keep helper signatures/defaults; no new generic service. At deadline an observed persistent unknown may retain its last typed uncertainty; a pre-unknown pending timeout keeps the existing None behavior. Never speak confirmed without the existing matching verified booking/current scope.

- [ ] Step1 — Meaningful RED. Reproduce internal-positive punctuation through the actual native recap/ACK/caller path and pending -> unknown -> confirmed through the real waiter with one immutable submit. Use synthetic data only; an import/mock error is not RED.
- [ ] Step2 — Minimal parser and waiter correction. Normalize harmless comma/period/exclamation separators between the existing closed clauses; retain question/quote/negation/correction/condition exclusions. Support the explicit c'est bon suffix following existing affirmative prefixes. Latch canonical read-only after unknown; preserve pre-send retries and the total 12-second bound. No backend send-state change.
- [ ] Step3 — Close contextual conversations through existing tools. Update only system/schema semantics for clear current completion after further-help offer and no outstanding request. Keep task completion alone insufficient, generic declines ambiguous, narrow direct shortcuts, exact once-only scope and native final ACK/drain unchanged.
- [ ] Step4 — Native regressions. Cover positive sentence/comma/exclamation/apostrophe forms and negative questions/negation/correction/conditions/quoted or reported speech. Reject early/late/stale ACK. Cover unknown -> confirmed, permanent unknown at a shortened synthetic bound, unknown -> read failure/missing/waiting (zero resubmits), canonical mapping before unknown and fixed identity afterward, and caller departure/hangup/capture loss (no late TTS). Use scripted native tool fixtures for contextual close, one goodbye, callback ordering, no early end and interruption/late-ACK containment; disclose semantic model choice remains untested.
- [ ] Step5 — Verify affected native/calendar/authority/financial checks, Ruff, strict mypy, frozen lock/sync and six exports byte-equal against immutable a47e560b615be86dbdde0adf91e477ac9d6e3e0c without replacing goldens. Root retains full hosted suite/privileged/startup/image/security qualification. Report exact failures/warnings/skips/unperformed gates.
- [ ] Step6 — Self-review, cohesive commit, frozen head/tree and full report task-1-report.md. No push/deploy/provider calls. Root follows with independent full-range review and exact-tree hosted qualification.
