# Closed STT request diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. One cohesive task, then independent full-branch spec and quality review. Steps use checkbox syntax.

**Goal:** Distinguish local paid-request refusal from HTTP/response failure without changing call shutdown or spending authority.

**Architecture:** Add one small segment-scoped closed record shared by existing STT request hooks, the bounded native adapter and its once-only observer. Keep HTTPX send interception as a small pass-through on the existing call-owned production client. Export only validated closed values through the existing counter.

**Tech Stack:** Existing pinned Python3.13.15, Pipecat1.12.0, HTTPX/OpenAI SDK, OpenTelemetry and pytest; no dependencies.

**Spec:** docs/superpowers/specs/2026-10-21-stt-request-diagnostics-design.md

## Global Constraints

- Diagnostic change only; retain terminal stt_failed, batch discard and all paid-work guard decisions.
- Keep USD5 caps, lifetime5/used3, identities, financial/store history, remaining reserve and validity/TTL rules.
- No paid/provider/SSH/Docker operation by implementer; offline real SDK/Pipecat with MockTransport only.
- No raw bodies/headers/audio/transcript/error strings or identifiers in diagnostics; no new endpoint/metric name/store/dependency.
- Preserve original SDK/client options, exactly one actual predicate check and no upload retry.

## Review Focus

1. A local guard refusal must dispatch zero HTTP and remain distinguishable from a provider failure.
2. A200 response followed by a body timeout must preserve both boundary facts.
3. Successful/queued/cancelled segments and concurrent calls must not contaminate later records.
4. No failed or late transcription may authorize a booking from prior caller evidence.
5. Diagnostics must stay closed/bounded and not change bootstrap imports, sanitization or client teardown.

---

### Task 1: Preserve closed STT boundary facts through the native failure path

**Files:**
- Create src/projetv0_voice/inference/stt_diagnostics.py for the focused stdlib-only diagnostic record/binding.
- Modify src/projetv0_voice/production_wiring.py (existing native STT client factory/pass-through send and response observation).
- Modify src/projetv0_voice/session_factory.py (record the actual existing paid check without repeating it).
- Modify src/projetv0_voice/inference/services.py (per-run scope, immutable snapshot before replacement, finally cleanup).
- Modify src/projetv0_voice/pipeline.py and src/projetv0_voice/metrics.py (existing once-only observer, validated closed dimensions).
- Tests: new focused tests/unit/test_stt_request_diagnostics.py plus affected tests/unit/test_runtime_metrics.py, tests/unit/test_pipeline.py and integration/test_process_session_factory.py or an equivalent focused native-path test. Reuse existing test helpers; do not mirror production logic or fabricate the parent ErrorFrame.

**Interfaces:**
- Consumes actual grant paid_work_permitted, native HTTPX request/response hooks, public AsyncClient.send, native run_stt, FatalErrorFrame.metadata and existing first-push observer.
- Produces one immutable snapshot with the spec's exact three fields/values, attached under a fixed project metadata key and exported once through the existing STT failure counter. Keep the record/binding module free of heavy runtime imports to preserve bootstrap order. Internal helper names may be chosen together by this sole task owner; no later task depends on a new API.

- [ ] **Step1 — Write and run the native RED.** `test_stt_guard_denial_preserves_local_cause_before_transport`: false grant, synthetic WAV, actual factory/SDK/native adapter/observer, MockTransport request counter0, terminal stt_failed and exactly one failure observation; new closed paid_check=denied fails before production edits. Missing imports or mocked parent errors are not meaningful RED.
- [ ] **Step2 — Implement the smallest capture.** Scope/reset/finally the closed record per executing segment. Add one actual paid-check observation, one status-only response hook and one public send delegation with typed exception capture. Preserve cancellation and all options. Freeze before error replacement; never inspect arbitrary exception text or provider content.
- [ ] **Step3 — Extend focused assertions.** Check guard exception versus denial, allowed4xx/5xx one dispatch/no retry, timeout/connection/remaining RequestError with no invented response,200 plus body timeout, malformed200, prior-success isolation, queued/concurrent/cancelled invocations, complete teardown and canary privacy. Demonstrate real installed SDK buffered-send behavior rather than assuming it. Observe the counter once and assert the closed cardinality bound; malformed metadata cannot leak or affect policy.
- [ ] **Step4 — Pin native containment and financial coupling.** Exercise failed/late transcription with no completed batch or booking authority; retain existing pending/committed budget tests and prove later valid publication cannot revive terminated caller scope. Do not change financial source to make tests pass.
- [ ] **Step5 — Verify.** Run frozen lock/sync, affected native/unit/integration tests, Ruff and mypy. Check the six generated exports remain byte-compatible without replacing a golden. Root retains the unchanged full hosted Linux workflow as canonical full pytest/privileged/startup/image/security gate; do not duplicate a full Windows run merely to repeat the same closure. Report every actual failure/skip/warning and any unperformed full gate explicitly.
- [ ] **Step6 — Self-review and commit.** Remove speculative functionality and duplicate assertions. Commit only this cohesive source/tests/docs change; freeze head/tree and write full report with RED/GREEN commands/results, native boundary coverage, export proof, privacy/cardinality evidence and unresolved live-cause/acceptance limits. No push/deploy/provider call.

Root follows with independent full-range spec/quality review, hosted qualification, immutable release, a fresh once-only owned deployment preserving the current count-three authority, and genuine appointment verification. No task from prior Calendar/confirmation/reserve plans is reopened by this diagnostic lot.
