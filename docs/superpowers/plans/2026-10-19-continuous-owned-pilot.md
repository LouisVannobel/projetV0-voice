# Continuous owned-pilot phone implementation

The whole-reserve constraint below is historical and is superseded by the
owner-authorized [21 October remaining-reserve plan](2026-10-21-remaining-pilot-reserve.md).
The original qualification evidence and all other authority remain unchanged.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute task-by-task with fresh implementer and independent review; the user's standing mandate removes human approval pauses.

**Goal:** Keep the owned phone listener available between calls while every new paid call consumes persistent authority under the unchanged original budgets.

**Architecture:** Native strict mode, one small SQLite budget authority, immutable provider observations and one call/generation-bound permit. Existing Pipecat, persistence, native action ownership and Inbox remain the consumers; no recurring opener or second billing engine.

**Tech Stack:** Python3.13.15, existing pinned Pipecat/Pydantic/HTTPX/aiosqlite dependencies, native Telnyx Voice API, owned Dokploy deployment.

**Spec:** `docs/superpowers/specs/2026-10-19-continuous-owned-pilot-design.md`.

## Global Constraints

- Original separate caps USD5 each; whole reserves OR3.780408/Tel1.22; forecasts OR0.505068/Tel0.12; positive uncertainties already in reserves. No reset/refund/topup or silently changed budget scope.
- Fixed original-pilot slot; immutable policy/hash/identity; lifetime total5; no reinterpretation of thirteen historical qualification rows.
- Only strict Sparra requires the new budget consumer; existing candidate/override and generic Voice contracts keep their behavior.
- Observations <=60s, total metadata deadline15s, each response <=2MiB, no redirects/retries, GET only. Secrets/raw provider bodies are never logged or included in reports.
- Preserve native300s call timer, 600s STT, 20000 TTS characters, 13952+2048 completion ceilings, accepted conversation implementation/models/pipeline/tenant result contracts.
- Same process/database/policy restart never creates allowance. Missing/restored/different budget authority cannot auto-enroll.
- Tests offline with provider doubles; do not read live .env or production stores for fixtures. Heavy local qualification only above existing2GiB physical/6GiB virtual floors, one test owner.
- User authorizes needed owned push/merge/release/deploy; never use global Dokploy MCP (wrong server), mutate unrelated services, fabricate live records or perform a financial transaction without its actual terms.

## Review Focus

- Early answered-first path cannot skip financial authority or issue another Answer.
- A late commit/unknown transport/cancelled claim cannot refund, duplicate, reset or revive authority.
- The last consumed allocation remains usable by its live generation although no new permit remains.
- Metadata includes actual account cash, correct scopes, complete pagination and monotone reported floors, not key headroom substituted for credit.
- Busy/stale/exhausted health does not restart a live listener or prevent legitimate finalization; paid-period fence also applies to paid summaries.

### Task 1: Import-safe original-account observation adapter

**Files:** Create `src/projetv0_voice/budget_observation.py`, `tests/unit/test_budget_observation.py`; add concise account-observation contract to `docs/audio-consumption.md`.

**Interfaces:** Define frozen typed `BudgetObservationIdentity` (original credential/account fingerprints, DID/connection, accounting start), `BudgetObservation` (collection start/end UTC, credential/account bindings, per-provider reported spend and available cash, key limit/reset/expiry, active call-control IDs, provenance digest), and constant-safe `BudgetObservationUnavailable`. Expose `async collect_budget_observation(client: httpx.AsyncClient, *, identity, openrouter_key: SecretStr, telnyx_key: SecretStr, utcnow, monotonic) -> BudgetObservation`. The caller owns the client. This adapter issues no grant and writes nothing.

- [ ] Write failing tests through HTTPX MockTransport for valid original accounts with non-management inference key supporting credits; assert only expected GET endpoints and bearer origins, finite exact monetary parsing, acquisition-start freshness, complete pagination, no sensitive repr/error output.
- [ ] Observe RED. Then implement the smallest inert adapter. Reference only public code of `C:/Users/louis/.codex/artifacts/sparra/2026-10-14/per-call-limits/material_r5_financial.py:226` for the original calculation; never import its operator globals/private files.
- [ ] Collect `/api/v1/key` AND `/api/v1/credits`; validate creator-account fingerprint, original key/caps/no-reset/expiry, and actual cash separately. Collect Telnyx seven products (`call-control`, `media-streaming`, `call-control-features`, `recording`, `sip-trunking`, `text-to-speech`, `speech-to-text`), absolute charges, USD wallet, exact active DID/connection and full active-call IDs. Preserve original UTC report-start accounting; no negative entry refunds spend.
- [ ] Cover wrong key/account/DID/connection, non-USD or nonfinite/bool money, reset key, missing cash/403, incomplete pagination, redirects, oversized/duplicate-key JSON, timeout/cancellation and transport failure. Refusal exposes no raw body/key and triggers no additional request after a known identity failure.
- [ ] Run focused pytest and Ruff, mypy on affected source. Commit scoped adapter/tests/doc; report commands/RED/GREEN/commit and remaining integration concerns. No provider calls/deployment by implementer.

### Task 2: Durable budget authority consumed by native strict admission

**Files:** Create `src/projetv0_voice/owned_pilot_budget.py`, `tests/unit/test_owned_pilot_budget.py`, `tests/integration/test_owned_pilot_budget_admission.py`; modify native `persistence/schema.py`, `persistence/writer.py`, `admission.py`, `runtime_config.py`, `production_wiring.py`, `lifecycle.py` and their focused tests. Adapt CLI/operator enrollment and exporter contracts only where actual consumers require it; retain the exact six/eight artifact release boundary.

**Interfaces:** Consume Task1 observations. Produce frozen `OwnedPilotBudgetPolicy` and call-bound `OwnedPilotBudgetPermit`, an explicit operator-only enrollment command, writer budget read/observe/consume operations, and a native strict Sparra budget gate. Keep the permanent slot a source constant, not a caller-selectable UUID. Place grant consumption in the existing first-identity transaction; persist permit with existing call/receipt data, reuse existing single-session claim and action owner. Do not make raw observations bearer authority.

- [ ] Read spec/Oracle amendments and current code before choosing exact private signatures. Write native failing regressions for both webhook orders and financial denial with zero Answer/stream/inference attempts.
- [ ] Add one small owned-pilot budget table with immutable original-slot/policy binding, monotone count and reported-spending high-water marks. Add only the necessary nullable permit data to existing native receipts/leases. Forward migration preserves historical qualification rows and all old logical table contents. Startup never enrolls.
- [ ] Implement exact Decimal affordability from the spec. New calls use current n; current permit k uses k-1. Update authenticated reported floors durably; do not compare lagging reporting with the forecast debit as a regression. Missing/incompatible/expired/exhausted/unknown authority denies all new paid work.
- [ ] Collect facts outside SQLite; revalidate inside real transaction and after preparatory awaits immediately before existing Answer/stream/session construction. Carry one permit through all paths, atomically claim one session, and preserve native dedupe, cancellation, shielding and unknown-outcome handling. Existing admitted calls may finalize when new admission is closed.
- [ ] Wire mandatory policy path/hash for strict Sparra through existing settings/protected readers/client ownership. Keep other modes compatible. Fresh observations may be obtained on demand or by an existing owned refresh task; facts never enroll/regrant or clear unknown ownership.
- [ ] Carry the genuine paid/key boundary into existing lifecycle and paid summary dispatch; never reset original300s deadline. Refuse starts whose call/finalization envelope would cross the boundary. Liveness remains independent of admission readiness; no restart loop on budget refusal.
- [ ] Prove equality/just-over thresholds, separate-provider refusal, once-only uncertainty, reporting lag vs true regression, changed UUID/hash/anchor/total, increased cash/midnight/restart, missing/restored store, old-row preservation, duplicate/concurrent identity, both event orders, WSS races, stale-after-await, paid-fence crossing, cancellation/late commit, unknown dispatch/restart and last-permit continuation. Exercise real writer/SQLite/native ownership; external transport doubles only.
- [ ] Run targeted tests, existing bounded native sequence/result/farewell/session tests, then normative Ruff/mypy/full offline suite. Commit scoped source/tests/docs and provide report for independent review. No live deployment or provider mutation by implementer.

### Task 3: Owned deployment and continuous-availability acceptance (Root)

**Inputs:** Approved Task1/2 source, exact-SHA CI, native release handoff, current actual closed predecessor and private original-account cost/funding/provenance. **Output:** One enrolled original budget authority, strict same-policy listener running, truthful readiness and preserved genuine Inbox evidence.

- [ ] Independent final branch review; push PR, attach it, require hosted gates, merge under standing mandate, run existing main-only immutable release and verify exact source/image/contract/handoff. No new dependency or base-image drift without fresh required scans.
- [ ] Reauthenticate actual predecessor, full native state/all-table contents and old thirteen rows, SQL/routes, no unknown carrier commitment, original credentials/caps/cost scope/high-water facts, actual account cash and paid/resource renewal disposition. Preserve a protected consistent backup and historical artifacts. Do not reuse old consumed/expired runs.
- [ ] Produce one protected actual policy and strict qualified profile for the released image. Perform native offline artifact/config/migration/enrollment/restart/refusal qualification with owned disposable stores; no production data copied into tests.
- [ ] Retire/fence old writer, enroll once into the existing owned store only after actual joins, deploy through own ops01 manager to ops02. Preserve app, PostgreSQL, Stripe worker, Cloudflare and private8443/8444 routes. Use original ingress/tenant binding and explicit unknown-dispatch observation, never a blind retry.
- [ ] Verify real metadata capability through the new adapter, warm live/ready, idle past the old manual-window mechanism, and an ordinary restart with identical policy/counter/floors. Negative session/admission probes cause zero paid effects. Open permanently while funded; do not claim a new positive carrier test until it happens.
- [ ] Save actual deployment/budget/readiness evidence and update the product checkpoint. Keep the accepted real-call/Inbox link. Then proceed to a separately scoped commercial-offer operator tool using existing Sparra agreed terms; no fake offer/payment or automatic pilot-budget replenishment.
