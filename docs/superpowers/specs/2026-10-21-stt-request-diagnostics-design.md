# Closed STT request diagnostics

Sparra must complete a genuine telephone appointment in Google Calendar and the Inbox. A controlled call reached availability but ended with `stt_failed` before selection. Existing `reason=transport` combines a local paid-request refusal, provider/HTTP errors and response-processing failures. The initiating incident cause is unknown; later healthy funding and provider200 responses cannot reconstruct it.

Oracle `sparra-stt-cutoff-oct21` reviewed the public source and official Pipecat1.12 interfaces. Adopt its bounded diagnostic change first. Preserve existing terminal behavior; this lot does not claim to fix the historical cutoff or complete appointment acceptance.

## Required behavior

One executing native STT segment owns a small diagnostic record. Freeze only these fields into sanitized failure metadata before the parent error loses its meaning:

| Field | Closed values |
| --- | --- |
| `paid_check` | `not_checked`, `allowed`, `denied`, `check_error` |
| `response_family` | `none`, `2xx`, `3xx`, `4xx`, `5xx`, `other` |
| `httpx_failure` | `unobserved`, `timeout`, `transport_error`, `other_request_error` |

The actual existing paid-request hook records its one check; never evaluate the predicate again for diagnostics. A response hook reads only status_code, before body consumption. A small public AsyncClient.send override delegates once with unchanged arguments, records typed HTTPX TimeoutException/TransportError/remaining RequestError and re-raises. Cancellation propagates. Buffered200-body failure must retain both response family and typed failure; absence remains unknown.

Bind/reset inside each executing run_stt invocation; finally clears the binding on success, failure, cancellation or generator close. Concurrent calls/queued segments cannot inherit another segment's record. Do not retain raw request/response/exception objects or messages, headers, body, audio, text or customer identifiers.

The existing once-only STT observer exports the closed snapshot through the existing `projetv0.voice.stt.failures` counter, retaining legacy reason semantics and one observation per call. No new metric name, public endpoint, logger, persistent content store or telemetry destination. Additional dimensions must be closed, validated and have an explicit finite cardinality bound (at most4x6x4 diagnostic combinations per existing reason). Unknown/malformed metadata must not expose supplied values or authorize work.

Preserve native Pipecat1.12, the current HTTP client lifecycle/options, disabled SDK upload retries, sanitized terminal `stt_failed`, failed-batch discard and prohibition on late transcripts/old booking authority. No provider fallback, audio replay, nonfatal recovery, dependency/model/voice change or fabricated production request/event.

## Financial and operational constraints

Keep separate USD5 caps, lifetime5 admissions with3 already consumed, original financial identity/anchors/floors/history/SQLite identity/paid-until and remaining-reserve formula. No change to affordability, observation validity/freshness, pending/committed intersection or paid-request guard decisions. No additional paid diagnostic calls. Local tests stay offline and use synthetic WAV plus real installed SDK/Pipecat code and httpx.MockTransport.

Source work is separate from release/deployment and genuine phone acceptance. Root alone owns GitHub/provider/SSH operations. An unchanged complete hosted Linux gate qualifies the frozen reviewed tree; verify the immutable release and exact runtime/export compatibility before a later same-authority consumer rebind. Keep own ops01 manager -> ops02 and current custom bundle. No App migration or redeploy is part of this lot.

## Acceptance

Prove meaningful RED through the actual production factory hook -> SDK -> Pipecat -> adapter -> observer with zero HTTP dispatch on guard denial. Test guard check error,4xx/5xx, typed pre-header failures,200 plus body failure, unusable200 content, success then refusal/no-response failure, cancellation/teardown and canary privacy. Failed/late audio cannot authorize a booking. Preserve existing budget tests, including disqualifying pending views and recovery without reviving terminated caller authority. Full canonical source/static/image/security gates remain required; this diagnostic lot alone does not establish live appointment success.
