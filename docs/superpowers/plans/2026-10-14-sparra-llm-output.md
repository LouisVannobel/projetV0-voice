# Sparra requested LLM output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Bound cumulative completion ceilings submitted by each Sparra call and preserve one actual final result attempt.

**Architecture:** Extend the existing per-call OpenRouter service through public HTTPX and Pipecat hooks. A small final-result application method wraps the existing retained-call consumer; the session uses it after cleanup. Retain all existing orchestration, routing, result validation and HTTP ownership.

**Tech Stack:** Python3.13.15, Pipecat1.12.0, OpenAI2.54.0, HTTPX0.28.1; unchanged frozen dependencies.

**Spec:** [2026-10-14-sparra-llm-output.md](../specs/2026-10-14-sparra-llm-output.md)

## Global Constraints

- Sparra-only opt-in; generic consumers retain behavior.
- Ordinary13952 + one final2048; final serialized body, positive integer ceiling XOR fields; n absent or integer1.
- No refund/reset/retry/redirect for guarded requests; no inference during teardown except actual protected final consumer.
- No new dependencies, providers, real data replay, production writes or budget/grant/clock changes.
- Input250000 and monetary/unattended guarantees remain unqualified.
- Root alone runs native checks in a coordinated Windows slot; all owned processes/HTTP peers must join.

## Review Focus

- Builder versus SDK-final extra_body override must be discriminated by a real wire peer.
- A schema-bearing ordinary call must never spend the protected final allowance.
- Cancellation after received POST must preserve debit; exactly one final attempt survives uncertainty.
- Native error sanitization must preserve explicit fatal disposition and close the actual call owner.
- Cleanup must retain this same service/client for final inference; no new instance resets allowance.

## Task 1: Service guard and actual result consumer

**Files:** Modify src/projetv0_voice/inference/services.py and src/projetv0_voice/session.py; create tests/unit/test_llm_consumption_limits.py. Existing business_result helper remains unchanged.

**Interfaces:** Public application method infer_final_result_once(retained: RetainedCall, provider_callback: str | None) -> MessageResultV1 | None; session fallback to existing infer_partial_result for generic services without this method.

- [ ] Add real SDK/HTTP loopback tests for final body, n/ceiling refusals, ordinary structured scope, cumulative arithmetic and cancellation.
- [ ] Root executes causal RED before implementation, preserving failure artifacts.
- [ ] Implement constructor-local counters, request hook and explicit result invocation, plus typed fatal error override.
- [ ] Root verifies GREEN, unchanged audio/legacy/result cohorts, Ruff and strict mypy. No test mirrors a replaced inference method.
- [ ] Independent source review verifies pins and consumer/error boundaries.

## Task 2: Native owner acceptance and source delivery

**Files:** Extend appropriate tests/integration/test_result_observation.py or test_call_session.py and existing service tests; update docs/audio-consumption.md.

- [ ] Demonstrate ordinary refusal through real native LLMContextFrame/error boundary/worker and existing call-owner teardown; no server POST on refusal.
- [ ] Demonstrate actual post-cleanup retained-call result on same client,2048/nonstream/schema, provenance validation; second/concurrent/uncertain attempts refuse.
- [ ] Test separate call isolation, cleanup closure, erased/no-user-data, malformed final responses, timeout and withheld headers without refunds/retries.
- [ ] Run affected native cohorts and canonical checks; use existing hosted full Linux and image CI for exact source SHA, recording honest skips.
- [ ] Create and attach reviewed PR; merge only after native/source gates. Publishing/deploying and real phone/Inbox acceptance remain distinct, preserving original budget and period.
