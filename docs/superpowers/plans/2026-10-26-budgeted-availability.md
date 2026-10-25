# Budget-governed continuous admission implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute the coupled native budget change as one task with independent review, then deliver through the sole operator.

**Goal:** Open Sparra whenever the original budget can cover an admission, without manual per-call quota extensions.

**Architecture:** Extend the existing policy, affordability predicate and SQLite transition. Reuse observation refresh, permits, session ownership and finalization.

**Tech Stack:** Pinned Python3.13.17, existing Pydantic/aiosqlite/Pipecat graph.

**Spec:** ../specs/2026-10-26-budgeted-availability.md

## Global constraints

- Original independent USD5 caps; same financial identity, anchors, history and paid boundary; no reset/refund/payment.
- New mode total_calls:null only; legacy JSON/hashes/arithmetic remain unchanged.
- New reserve max(R0-m*d,u), same operands in cap/cash/key checks; no negative reserve.
- V2 schema14/N7/used7 -> schema15/null via one explicit native CAS; no startup upgrade.
- No dependency, transport, audio, Calendar, Inbox, worker or schema qualification_runs change.
- Tests run sequentially with fresh >=2GiB physical and virtual RAM; use only this worktree's pinned environment.

## Review focus

- A stale, foreign, missing or regressing funding fact must not create authority.
- Advancing the global count must not terminate the final already admitted permit.
- An uncertain transition must preserve data and refuse automatic replay.
- A restart/missing/restored store must not add allowance or replace history.
- Legacy policies, historical permits and unrelated tables retain their exact meaning.

### Task1: One native budget-only admission transition

**Files:** owned_pilot_budget.py, owned_pilot_budget_cli.py, persistence/schema.py,
persistence/writer.py; existing unit tests test_owned_pilot_budget.py and
test_owned_pilot_budget_cli.py; new tests/integration/test_owned_pilot_budgeted_calls.py.
Only add a small relevant operator-doc update if its actual consumer needs it.

**Interfaces:** total_calls:null, widened permit ordinal, action
enable-budgeted-calls with predecessor-retired and expected-policy-sha256.
Preserve all other public/native interfaces.

- [ ] Refresh RAM and run the three existing focused budget test modules as baseline; record actual failures without broad repair.
- [ ] Write behavioral RED for null policy and positive-reserve native admission at count7, using synthetic anchorOR0.30/Tel3.28 and synthetic complete funding facts. Verify the existing code fails for the intended reason.
- [ ] Implement the smallest new predicate and model support. Test legacy hash/JSON, exact cap/cash/key boundaries, raw/floor/forecast dominance, invalid counts, reserve floor and count8 refusal. Use SQLite maximum2^63-1 for ordinal/count and explicitly refuse increment at that maximum. Include a key balance leaving0.50 after deficit, which passes the old declining reserve but must refuse the new0.75 floor.
- [ ] Write RED for native schema14/N7/used7 ->15/null, then implement only the explicit guarded transition and exact schema recognition. Preserve all existing tables/history and clear observation after commit.
- [ ] Verify native SQLite migration, stale/double CAS, refusal with active/unknown effects, rollback/late commit, restart/missing-store and legacy refusal. Evaluate candidate affordability with total_calls:null at count7; use a fresh observe after transition. Reuse existing integration fixtures.
- [ ] Add CLI behavior with existing factories, fresh funding and unknown-result cleanup; test wrong contract/predecessor/arguments and unknown outcome without real providers.
- [ ] Test admission8, duplicate deliveries and both webhook orders using transport doubles and real SQLite. Demonstrate new-admission closure at global8 while permit8 continues at index7; keep existing pre-effect/expiry/generation gates.
- [ ] Run affected offline tests, ruff, mypy and frozen lock validation. The full offline suite is required on the exact source before merge, using canonical hosted CI. A local full run may substitute only if proportionate; preserve any partial/interrupted local run honestly and never rerun it merely to duplicate the required hosted gate. No container/provider/phone test locally.
- [ ] Self-review diff and commit the scoped implementation. Record exact SHA/tree, RED/GREEN commands/results and any qualification limit.
- [ ] Independent spec/quality review and any focused repair before release.

### Delivery through the sole operator

- [ ] Exact-head hosted CI, reviewed merge and immutable native image/release qualification using existing workflows; no new pipeline or dependency.
- [ ] Operator refreshes original store/provider observations and before-state, retires/joins old owner, preserves a consistent protected backup, performs one native transition and deploys the qualified consumer. Unknown result uses same-operation observation.
- [ ] Verify count/history/floors/identity/paid boundary, new readiness, ordinary restart conservation and public handlers without a synthetic paid call.
- [ ] Signal the owner when the number is actually open. After the owner's real call, correlate automatic closure and exactly one actual Google event/Inbox record. Do not invent/replay a request or manually book.

Oracle session sparra-budget-only-oct26 approved B with the exact amendments
incorporated above. Its transcript is in the root artifact folder
2026-10-26/continuous-budgeted-voice/ORACLE_BUDGET_ONLY_VERDICT.md. It is advisory
source-contract evidence only; implementation and all downstream gates remain open.

Execution ruling: the local full suite was still at23percent after287seconds,
following384passing affected tests plus green lint/typecheck. Use the required
exact-source hosted full suite before merge instead of duplicating a long local
full run. The implementer may stop only its own runner cleanly; preserve partial
output and cleanup evidence. This changes the location of full qualification,
not the requirement, and introduces no live action.
