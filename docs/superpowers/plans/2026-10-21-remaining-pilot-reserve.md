# Remaining pilot reserve implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for the bounded source task and independent review. Steps use checkbox syntax.

**Goal:** Admit the authorized remaining pilot calls using the remaining forecast
reserve, retaining the current monetary caps and original consumption authority.

**Architecture:** Change the existing affordable() calculation only. Existing
writer publication, admission and permit dispatch keep their original count
basis, transactions and pre-effect ownership. Root separately qualifies the new
immutable consumer and preserves the existing stored authority during deployment.

**Tech Stack:** Existing pinned Python 3.13.15/Pipecat 1.12.0/SQLite runtime.

**Spec:** `docs/superpowers/specs/2026-10-21-remaining-pilot-reserve-design.md`.

## Global Constraints

- Keep separate control caps C=5 USD, lifetime N=5 admissions, original accounts/credentials/slot/policy UUID, historical anchors, monotone spending floors, committed counts, receipts, paid intent and paid/key expiry fences.
- New admission uses m=n<5. An existing permit of ordinal k uses m=k-1.
- E=max(H,S,A+m*d), R(m)=R0-m*d; apply the same view/index to cap, cash and OpenRouter key comparisons.
- Retain R0/d/u constants, strict validation and all identity/freshness/generation/unknown-effect fences. No reset, refund, re-enrollment, top-up or expanded allowance.
- No provider/dependency/schema/ledger/framework addition; accepted Calendar/Pipecat/audio/LLM behavior is unchanged.
- Source/native tests, hosted Linux, immutable release, deployed readiness and actual booking acceptance are separate gates. No live services in source tests.

## Review Focus

- Report-dominant facts can loosen with a larger count: native current-permit checks must use k-1 before the paid effect.
- Passing cap arithmetic is insufficient: cash and key remaining need the full E-S adjustment independently.
- A committed/pending mix must not combine E(m) with R(m+1) or favorable funding from another view.
- Fifth-call consumption closes new admission while its correctly indexed permit may finish; restart and duplicate webhooks cannot add allowance.
- Downward restatements retain actual high-water floors; unknown observations/intents and stale generations still deny paid work.

## Task 1: Approved arithmetic, native proof and current documentation

**Files:** modify `src/projetv0_voice/owned_pilot_budget.py`,
`tests/unit/test_owned_pilot_budget.py`,
`tests/integration/test_owned_pilot_budget_admission.py`,
`docs/audio-consumption.md`; add concise supersession notes to the historical
continuous-pilot/restatement specs and continuous-pilot plan. Include this plan
and its spec. Do not rewrite closed evidence or edit unrelated source.

**Interface:** existing affordable(policy,fact,count,reported_floors=...) -> bool,
unchanged callers. No added persisted field or new runtime abstraction.

- [x] Write meaningful revised boundaries at m=0..4 for each provider and each funding resource; exact equality passes and one micro-dollar short refuses. Keep invalid count/floor and m=0 witnesses.
- [x] Observe RED on current source. Include the Oracle ordinal counterexample: after first commitment, A=0,H=S=1.50,cash=key=3.50 must deny ordinal1 at m0 before any paid effect, though an erroneous m1 would pass.
- [x] Implement R0-m*d consistently inside affordable(); preserve existing validation and Decimal context. No caller changes unless a concrete native failure proves one necessary.
- [x] Prove through existing SQLite/native fixtures: five consumed calls, sixth denied, last permit at m4, nonzero-count funding/restatements, restrictive coherent snapshots, duplicate/restart continuity and actual pre-effect denial. Keep previous refusal cases.
- [x] Run both affected test files, Ruff and mypy; document exact RED/GREEN, immutable source and normal inherited warnings. Commit source/docs/tests with the checked-in amendment.
- [ ] Obtain one independent spec/quality review of the bounded range. Resolve material findings through scoped correction/re-review.
- [ ] Root runs unchanged exact-source hosted Linux CI as the complete canonical gate, including full pytest, privileged/native startup, packaging/image and security. Local focused tests avoid a redundant whole Windows suite; any new unexplained failure remains open.

## Root integration and runtime

- [ ] Verify reviewed tree, green exact PR gate, actual-main binding and main-only immutable eight-file release. Parallel main CI and release only after reviewed-tree equality; no runtime switch before all required gates succeed.
- [ ] Prepare fresh count-two financial/store/ownership proof and bounded operator review. Never replay count-one scripts, shadow enroll, reset, restore old data or replace the selected custom agent bundle with the release default.
- [ ] Retire the old consumer, preserve the same SQLite identity/history/floors/counts and rebind image/profile only. Deploy via owned ops01 manager to ops02; observe unknown outcomes instead of blindly retrying.
- [ ] Restore only the original SQL admission and two Telnyx paths after native readiness/funding proof. Request one real human appointment call, then independently verify the Google event and matching Inbox reference.
