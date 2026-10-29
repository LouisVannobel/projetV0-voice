# Reported pilot budget implementation plan

> **For agentic workers:** Use superpowers:executing-plans in this existing Root-owned checkout. One final fresh review; no implementer delegation per task.

**Goal:** Prepare an explicit monitored-accounting option without rewriting financial history or increasing either configured USD5 cap.

**Architecture:** Reuse the native budget policy, authenticated observations and sole SQLite writer. The default lifetime-forecast policy stays unchanged. An explicit native transition enables reported spending with the existing exclusive active-call reserve.

**Tech Stack:** Pinned Python3.13.17, Pydantic2.13.4, aiosqlite0.22.1; frozen dependencies, offline fixtures.

**Spec:** docs/superpowers/specs/2026-10-29-observed-pilot-budget.md

## Global constraints

- Preserve count8, identity, anchors, floors, every historical row, expiry and paid-intent fencing.
- Keep independent USD5 controls, forecasts0.505068/0.12 and minimum reserves0.75/0.50.
- No production/provider mutation, call, key, payment, support message or new ledger while preparing this option.
- Legacy JSON/hash/arithmetic must remain identical. Image-only rebind cannot enable the option.
- A fresh collection after native closure is an observation, not a final invoice receipt.

## Review focus

- Fresh collection started before closure and delivered later must not authorize another call.
- Matching native settlement cannot clear someone else's paid intent; unknown intent remains fenced across restart.
- Every cap, cash and key check must use the same effective/raw amounts and reserve.
- Pending writer publication cannot widen committed permission; old permits retain k-1 semantics.
- Activation/replay/wrong predecessor must preserve native rows, count and floors exactly.

## One coherent task: explicit mode, guarded transition and closure freshness

**Files:** src/projetv0_voice/owned_pilot_budget.py; persistence/writer.py; owned_pilot_budget_cli.py. Extend the existing unit and integration budget fixtures; no new framework.

**Interfaces:** Optional literal policy field admission_accounting=reported, omitted when absent. Existing affordable(policy,fact,count,reported_floors=...) signature stays unchanged. Explicit native/CLI action enable-reported-spend changes only that opt-in after the release bindings have been rotated separately using the unchanged rebind-release action.

- [x] Baseline: uv lock --check; uv sync --all-groups --frozen; affected budget unit/CLI/integration selectors.
- [x] Record Oracle design amendments before changing product behavior.
- [x] RED: extend the existing fixture with explicit reported mode at count8; it admits under the unchanged observations/reserves while the legacy mode still refuses. Legacy JSON/hash stay identical; finite modes reject the opt-in. Parameterize the existing independent cap/cash/key boundary checks for reported/floor dominance.
- [x] GREEN: conditional policy field exclusion, finite-mode validation, effective=max(H,S,A) only in explicit reported mode; preserve the original reserve and raw-S comparisons.
- [x] RED: native fixture activates only against exact canonical predecessor with no live/unknown exposure; conserves the full store and count8, rejects replay/mismatch and release-only mode switching.
- [x] GREEN: add enable-reported-spend as a separate existing-store-only writer transaction and CLI action. It changes only policy JSON/hash, clears observations, and keeps startup/rebind incapable of activation.
- [x] RED: native fixture closes one call then delivers a collection started before closure; new admission remains false through pending/committed views and restart until a post-closure collection. Keep unknown paid intent denied.
- [x] GREEN: one writer helper compares collection_started_at strictly with a volatile local barrier, reset on owner startup/CAS and advanced on effective budgeted terminal transition and matching paid-intent resolution. Observe/publication/consume all enforce it; duplicates and foreign settlement do not advance it. No timeout-based release or rewrite of carrier closed_at.
- [x] Run affected budget selectors, uv run ruff check ., uv run mypy src. Canonical hosted CI is required before any release, not duplicated manually.
- [x] One fresh whole-change source review; fix material issues and rerun only affected checks.
- [ ] Commit reviewable source and record that production policy/admission is unchanged until a separately qualified cutover. Present the exact weaker historical-liability guarantee to Louis; support is not a gate.
