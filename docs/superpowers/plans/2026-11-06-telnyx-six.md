# Telnyx Six Pilot Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development, with a fresh implementer and independent review for the new monetary delta.

**Goal:** Explicit native Telnyx6, OpenRouter5, coherent release and closed staging; activation withheld until CaptureReady plus PhoneReady and fresh eligibility.

**Architecture:** One optional serialized policy marker and one guarded extension of `renew-validity`; no new table, migration, service, operator or accounting model.

**Spec:** `docs/superpowers/specs/2026-11-06-telnyx-six-design.md`.

## Constraints and review focus

Keep same identity/account/DID/Workspace, accounting start, reserves/forecasts, exact anchors/floors/counter/history. No recharge/reset/refund/exclusion of fees. Legacy missing marker means5 and preserves hashes. Only5→6 or unchanged cap; no OpenRouter increase. No date target before both explicit readiness signals. Pro access limitations and Sonar findings remain evidence limits rather than false passes.

## Task1 — new monetary consumer delta

Modify `owned_pilot_budget.py`, `persistence/writer.py`, and the existing budget unit/integration tests; CLI only if a real new consumer requires it.

- [x] Fresh implementer writes focused default-hash/5→6/native-conservation/directional-refusal tests and observes functional RED.
- [x] Implement optional cap marker and per-provider affordability; extend only native renewal whitelist plus directional guard.
- [x] GREEN new selectors, affected budget policy/renewal and CLI consumers, retained freshness/close cases; normative lock/Ruff/mypy. No whole local suite or new harness.
- [ ] Independent source review and small-diff Oracle standard if Pro actually available; resolve concrete findings before source integration.

## Task2 — qualified release and closed staging

- [ ] Update the existing PR73, verify exact-head native CI, then authorized normal merge, Main CI and manual immutable release. No bypass or old campaign rerun.
- [ ] Verify release subject/source/artifacts; publish concrete sole-owner mutation windows and review the small adaptation of existing operators.
- [ ] Stage image/materials unselected under an owned case; keep active expired tuple and original state. No future date or budget CAS.
- [ ] Hand off exact receipts and remaining conditions; wait for CaptureReady plus PhoneReady.

## Task3 — gated real activation

- [ ] Only after both signals: fresh current context/financial facts, fixed <=60-minute target, native5→6/date/release CAS, coherent runtime, before/after conservation and fresh readiness.
- [ ] If facts are insufficient, remain closed without a cap/recharge proposal. Unknown writes are canonically observed, not retried.
- [ ] Return phone-ready status only after actual runtime/route readiness. Human/recording owner makes the real call; no operator call or fabricated provider/Calendar action.
