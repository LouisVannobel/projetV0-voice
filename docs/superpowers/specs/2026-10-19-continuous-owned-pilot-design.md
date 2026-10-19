# Continuous availability for the owned Sparra pilot

The user authorized continuous availability under the existing independent USD5
Telnyx and USD5 OpenRouter pilot caps, followed by Stripe with prices chosen by
the commercial operator. No further human approval is required for the scoped
implementation, qualification, and owned deployment. Missing network access or
actual commercial input is not authorization to invent it.

## Product behavior

Keep the existing Pipecat listener warm between calls. Admit one call at a time
whenever the original budget can cover it. Do not open recurring thirty-minute
windows, reset allowance at midnight/restart, top up, change providers, or change
the accepted conversation/result/Inbox behavior. Exhaustion closes admission,
not webhook finalization or process liveness.

This is a finite prepaid pilot, not unlimited calls. Preserve the original
operational forecast and uncertainty; it is not a guaranteed provider invoice
ceiling. No new caller audio test is required for the unchanged media pipeline.

## Financial authority

Use native strict mode and one permanent original-pilot authority in the existing
SQLite database. Add a small owned-pilot budget table, rather than changing the
meaning of historical qualification rows. Preserve all thirteen old rows.
Enrollment is an explicit once-only operator action after predecessor retirement;
runtime startup never enrolls, repairs, or replaces a missing/different policy.

Bind the singleton `sparra-original-pilot` to one policy UUID/hash and the original
accounts, credential fingerprints, DID, connection, cost scope, historical
spending anchors, paid/key boundary and exact qualified consumer. Keep an immutable
lifetime total of five admissions, bounded by the original six-call forecast.
Reported cash, cheap/failed calls, new UUIDs, new dates, and restart cannot add
allowance. Store monotone reported-spending high-water marks separately from
the larger pessimistic forecast debit. Never refund a committed admission.

For each provider, C=5, R is the unchanged whole reserve, A the authenticated
historical anchor, d the fixed per-admission forecast, S the fresh reported
spending and n the committed count. A distinct new admission requires n<N and
`max(S, A+n*d)+R <= C`, plus actual funding sufficient for the same envelope.
R includes uncertainty exactly once. R_OR=3.780408, R_T=1.22;
d_OR=0.505068, d_T=0.12; uncertainty_OR=0.75, uncertainty_T=0.5.
For the already-consumed permit k=n+1, subsequent dispatch checks use k-1,
not the advanced global count, so the last admitted call can finish.
This deliberately conservative policy can refuse before cash is exhausted.

## Observation, admission and ownership

Create an inert bounded read adapter for the original provider metadata. It must
read actual OpenRouter account credits separately from the key's remaining limit.
The intended inference credential was observed to support both GETs; bind and
qualify this capability on the actual deployed host before activation. Do not
assume documentation or a key limit proves account cash, or expand credentials.

Preserve the original seven Telnyx usage products, absolute charge totals, wallet
balance, DID/connection identity and complete active-call listing. Observations
are facts only. Credentials and raw responses never enter reports/logs. GETs have
no redirects/retries, bounded bodies and a total deadline. Collect outside SQLite.

The native writer atomically commits the first receipt, call-bound permit,
pending lease, reported floors, and one counter increment. The permit binds
native call identity and generation, policy hash and ordinal. Reuse existing
transaction, deduplication, action ownership and session claim machinery.

Cover both webhook orders. In particular, an early answered event must not allow
streaming or session construction without the same committed financial permit.
Do not send an extra Answer or introduce a second hangup owner. Validate permit,
generation, paid boundary and observation age 0..60s after relevant preparatory
awaits, immediately before the existing owned transport/session entry.

Owned finalization is not a new admission. Cancellation/late commit/unknown
dispatch burns consumption and cannot revive a generation or refund it. Reuse
native reconciliation; unresolved older paid effects prevent new admissions.

## Lifespan and recovery

The genuine earliest paid DID/key boundary is absolute. Refuse starts whose
existing call/finalization envelope crosses it; clamp the existing lifecycle and
paid final-summary entry to that fence. Preserve the original 300s call timer
and existing STT/TTS/LLM submission limits. Stale/missing/regressing/wrong-account
facts or incompatible policy fail closed. Reuse observations/automatic read
refresh only as data; it cannot grant an allowance or reopen an expired scope.

Liveness remains independent of admission readiness. A busy, stale, exhausted or
expired readiness response must not cause a restart loop. Same-policy ordinary
restart preserves counter/floors/receipts. A known restored/missing database does
not automatically resume or enroll; do not claim arbitrary rollback detection.

Closing admissions does not cancel number rental. Deployment must verify the
existing resource/renewal disposition against the authorized paid scope; no new
subscription, provider renewal or payment is authorized by this implementation.

## Scope and evidence

Use Python3.13.15, the existing dependency graph, and current Voice main
6f1394ab2de063e418d9d2ee4802ee04cf3cc071. Its runtime matches accepted262af18;
the intervening merge only adds the bounded native sequence test. Keep Pipecat
transport, serializer `auto_hang_up=False`, models, prompts, recording policy,
tenant protocol and the accepted real-call/Inbox evidence unchanged.

Oracle session `sparra-continuous-budget-oct19` approved this small lot with the
above exact amendments. Its answer is advisory, not implementation/live proof.
Use transport doubles in offline tests, real SQLite/native ownership, then exact
SHA hosted CI/release and authenticated deployment/read/refusal/restart checks.
No production records or secrets become test fixtures or Oracle attachments.

Stripe is a separate next lot: consume its existing immutable agreed terms and
native Checkout. Add only the bounded operator entry for a real client/price;
do not create a fake client, Price, payment or runtime sales role.
