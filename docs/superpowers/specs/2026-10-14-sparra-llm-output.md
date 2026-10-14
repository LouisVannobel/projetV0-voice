# Sparra requested LLM output policy

Extend the existing call-owned OpenRouter service through public HTTPX and
Pipecat 1.12 hooks. No new dependency, orchestrator, global ledger or caller
capability protocol. Generic voice consumers retain their existing behavior.

Initialize monotone counters before native construction. Inspect the buffered
SDK-final request body before HTTP dispatch, requiring exactly one positive
integer completion ceiling and one completion (n absent or integer1). Reserve
ordinary ceilings <=256 cumulatively <=13,952, including warmup and tool calls.
Refuse malformed/exhausted requests without clamping, suspension or refund.
Sparra SDK retries, native timeout retry and HTTP redirects remain disabled.

Reserve 2,048 for one actual retained-call result after cleanup, through a small
infer_final_result_once application method. Its once-only entry is claimed
before its first await; a separate invocation scope permits exactly one final
POST. The existing helper builds context from retained turns and preserves
erased/no-caller early return, schema validation and result provenance. Ordinary
structured requests receive no protected allowance. Keep the same service/client
through final inference, then let the existing call owner close it.

Recognize only the typed local refusal in the public native error hook and
preserve explicit fatal disposition through the existing sanitized boundary.
Unrelated errors retain their existing treatment. Ordinary inference closes
during teardown; cancellation, cleanup, timeout and unknown outcomes never
reset counters or permit another final attempt.

Acceptance uses real pinned Pipecat/OpenAI/HTTPX against an owned TCP HTTP peer:
final serialization, strict ceiling/n validation, arithmetic/races, cancellation
after received POST, actual call-owner teardown and same-client final result,
unknown/concurrent final attempts, instance isolation and unchanged legacy use.
Fixtures remain offline; no provider call or production record is acceptance.

The 16,000 total bounds cumulative requested completion ceilings, not native
billed tokens or provider invoices. Native prompt/input250,000 remains unqualified
without exact provider prompt/schema/tokenizer transformations or a sound bound.
The source lot does not change existing live qualification, budgets, quota,
period, deployment identity, recording, relay or number-opening policy.
