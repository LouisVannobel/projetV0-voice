# Explicit Telnyx six-dollar pilot cap

The human authorized Telnyx USD5→6 without recharge, keeping OpenRouter USD5 and all original accounting, identity, reserves, anchors, floors, counter and history. Reuse PR73's fixed-target native validity renewal. No activation or future target date before explicit CaptureReady plus PhoneReady and fresh native eligibility.

## Minimal contract

Add an optional `telnyx_cap_usd: Literal[6] | None` to the existing frozen budget policy. Absence/None means the historical USD5 limit and is excluded from serialization, preserving legacy canonical JSON and hashes. An explicit value can only be6, only in reported budget-only mode. No general cap setter or OpenRouter cap field is introduced.

`affordable` uses USD5 for OpenRouter and the policy's effective USD5/6 for Telnyx. Reserves, forecasts, floors, cash/key checks and accounting scope remain unchanged. Invalid unvalidated cap values must fail closed. `observation_valid` still requires the OpenRouter key limit to equal5; the new field never changes provider reporting scope or the original identity/key expiry.

Extend only the existing `renew-validity` transition whitelist to include this field, with an additional directional guard: unchanged cap is allowed, or implicit5→explicit6. Explicit6→implicit5 and every other cap change refuse. No default silently raises old policies. All non-whitelisted serialized policy fields and every other SQLite column/table remain unchanged.

The existing full CAS, exact expected hash/count/floors, native old/new profile checks, retired predecessor, schema15, exposure checks, fixed UTC target, expired-predecessor guard, maximum3600-second first-commit window, post-COMMIT freshness barrier, replay and unknown-result handling remain authoritative. This is the only mutating entrypoint; editing a candidate file does not grant durable admission.

## Delivery and handoff

Qualify this new small delta, not the closed PR73 campaigns. Use focused real consumer RED/GREEN, independent source review, Oracle standard only if Pro access is actually available, and exact-head CI. Preserve the earlier Oracle/Sonar limitations honestly; no model downgrade, API charges, quota loop or invented verdict.

After authorized normal merge/Main CI/manual immutable release, stage the image and protected native profile/policy materials unselected and closed. Staging may describe cap6 with the old expired date but must not write the budget row or change the active tuple. Do not construct a future target date yet.

At explicit CaptureReady plus PhoneReady only, reobserve current state and collect fresh native facts for the candidate tuple. Freeze one absolute UTC target at most60 minutes ahead, verify original-provider scope and affordability with Telnyx6/OpenRouter5, fence/retire the predecessor, then coalesce cap, image/profile and date in the existing native CAS. Start the coherent tuple and require its own post-COMMIT/startup fresh observation before readiness. No operator phone call, fake Calendar event, replay, recharge, refund, scope reset or removal of another number.

Before each remote mutation, the sole source/delivery owner publishes the actual cause, delta, checks and bounded operation window. An unknown possible write is observed canonically, never automatically retried. Same DID/Workspace/accounting start and historical data remain.

Reported spend remains a pilot circuit breaker: delayed charges can exceed the configured cap. No final-invoice guarantee, live booking, video acceptance or Google approval follows from source/CI/image/readiness qualification.

## Focused proof

- Legacy default policy JSON/hash and USD5 behavior remain identical.
- Reported cap6 can fund a synthetic native case above5 but within6 including retained reserve; exact boundary passes and a value beyond it refuses. OpenRouter remains5 and key limit6 still refuses.
- Native5→6 renewal coalesces only the authorized cap/date/release fields; all rows/schema/counter/floors/identity/anchors conserved. Reverse/other caps or unvalidated model copies refuse.
- Existing target-observed replay never recollects, extends or restores older state; the postcommit/startup barrier and unknown-close fix remain effective.
