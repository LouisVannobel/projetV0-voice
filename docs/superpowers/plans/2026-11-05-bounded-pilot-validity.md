# Bounded Pilot Validity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add one native, idempotent validity renewal lasting at most60 minutes, retaining all existing financial authority and history.

**Architecture:** Extend the existing operator CLI and serialized SQLite owner. CAS the two policy columns only, optionally coalescing the existing release-binding fields. Reuse strict loaders, observation/affordability, schema15 and freshness/publication ordering; no new model, migration, table, framework or provider API.

**Tech Stack:** Existing pinned Python3.13.17, aiosqlite, Pydantic, pytest/pytest-asyncio, uv.

**Spec:** `docs/superpowers/specs/2026-11-05-bounded-pilot-validity-design.md`.

## Global Constraints

- USD5 per-provider caps and native reserve calculations unchanged; no recharge/reset/refund.
- Preserve identity, accounting start, anchors, floors, consumed calls, unresolved effects and retained history.
- Candidate target is a fixed explicit UTC date; maximum3600 seconds at native first-commit validation, bounded by original key expiry and the existing start envelope.
- A new transition renews an already-expired predecessor only; it cannot roll an active window forward. Exact-target observation remains read-only.
- No production activation before recording prepared, human availability confirmed and fresh native financial eligibility.
- Previous readers/refusals and closed source/CI/image/Oracle campaigns remain untouched.
- Root owns remote mutations; implementers/reviewers use local offline evidence only.

## Review Focus

- A retry after target expiry or later legitimate use observes the fixed target without collecting, extending or restoring count/floors.
- Two operators/candidates racing can commit at most one target; exact target observation is distinct from a new accepted transition.
- Cancellation or late commit cannot be mislabeled as a definite refusal or trigger automatic retry.
- An old-context or mid-commit observation cannot make readiness available after policy epoch changes; actual provider reporting scope covers the target accounting interval, and the existing native freshness boundary remains authoritative.
- New release/profile and renewed policy activate coherently, without an image-only opening or alteration of financial identity.

## Task1 — native consumer increment

**Modify:** `src/projetv0_voice/owned_pilot_budget_cli.py`, `src/projetv0_voice/persistence/writer.py`.

**Tests:** `tests/integration/test_owned_pilot_budgeted_calls.py`, `tests/unit/test_owned_pilot_budget_cli.py`.

**Interfaces:** Add action `renew-validity`; extend `operate` with explicit `target_paid_until: datetime | None`. Consume existing expected hash/count, retirement and predecessor profile/image arguments. Native request returns explicit `observation_required`, `target_observed`, refusal or first-commit success; do not use status-string truthiness. Keep existing actions unchanged.

- [ ] Fresh implementer reads the spec and existing native rebind/bootstrap/transaction/CLI tests; records scope and selected test names.
- [ ] Add focused real-owner/SQLite tests for expired predecessor success, coalesced release and exact row/schema conservation; observe functional RED before production changes.
- [ ] Add the fixed-target replay/race/unknown/freshness and native refusal cases from Review Focus. Use existing fixtures and native queue, no live service.
- [ ] Implement the smallest operator-only preflight/commit branch, reusing both native profile validations and existing-store-only mode. Change only the two policy columns by complete CAS.
- [ ] Run new selectors GREEN, then affected budgeted-calls/CLI/budget-unit files and named admission freshness/unknown regressions. Run normative lock, Ruff and mypy checks; report exact commands and outcomes.
- [ ] Independent spec/quality review plus new-boundary Oracle review; resolve findings. Commit only this lot.

## Task2 — exact source/image qualification and operational packet

- [ ] Root confirms current context and history diagnosis, publishes cause/delta/checks/sole-owner window before remote mutation.
- [ ] Deliver the new source through normal PR/Main CI and the repository's immutable manual release, without bypassing required gates or replaying old lots.
- [ ] Bind the candidate image and strict profile provenance. Prepare/review only the smallest new native activation/conservation packet, with no future policy date or active selection yet.
- [ ] Retain current closure; report readiness of the packet separately from phone/Calendar/Inbox acceptance.

## Task3 — bounded activation and human acceptance

- [ ] Obtain the recording owner's concrete readiness/human-availability handoff; no new human permission is needed beyond the accepted bounded lot.
- [ ] At that gate only, fix one UTC target at most60 minutes ahead and verify unchanged current identity, quiescence and fresh native financial eligibility. Insufficient cap/coverage stays closed, without a raise/recharge proposal.
- [ ] Execute the reviewed one-shot native ownership/CAS/coherent-tuple window with current before/after conservation. Resolve unknown writes by canonical observation, never automatic replay.
- [ ] Verify current native readiness and conservation. Hand off the real phone call to the human/recording owner; no operator call or synthetic event.
- [ ] Close the operational window with truthful source/CI/image/runtime/readiness/live-evidence boundaries and unresolved gates, if any.
