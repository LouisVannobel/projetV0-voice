# Continuous admission under the original Sparra budgets

The human explicitly requested reopening the number continuously on 26 October,
within the current budget, to book by phone and find the event in Google Calendar
and the Inbox. Implement only the missing budget-governed admission mode. Reuse
the already continuous listener, native SQLite authority and accepted pipeline.

## Money and availability

Keep separate original USD5 OpenRouter and USD5 Telnyx caps, lifetime admission
count, original UUID/slot/accounts/credentials/anchors/accounting window, monotone
reported floors, unknown paid-intent fence and absolute paid/key boundaries.
There is no reset, refund, re-enrollment, payment, renewal or provider change.

For the new mode alone, retain E=max(H,S,A+m*d), with m=n for new admission and
m=k-1 for an existing permit. Use R=max(R0-m*d,u), where existing constants are:

| Provider | R0 | d | u (minimum pre-call reserve) |
| --- | ---: | ---: | ---: |
| OpenRouter | 3.780408 | 0.505068 | 0.75 |
| Telnyx | 1.22 | 0.12 | 0.50 |

Require E+R<=5 and cash-(E-S)>=R independently for both providers; additionally
require key_remaining-(E-S)>=R for OpenRouter. Use the same index and reserve in
all comparisons and the existing restrictive committed/pending intersection.
Floors remain reported high-water values, never forecasts. Do not release the
pessimistic historical forecast because an observed call was cheap.

This preserves the old reserve through index6 and then holds that last approved
minimum. Since u>d, it covers one forecast call plus residual margin u-d
(OR0.244932/Telnyx0.38). It does not retain a further full u after that call and
is not a guaranteed final invoice ceiling. No number of calls/minutes is promised.
With the present observations only one further admission is forecast affordable;
funding must be observed afresh at cutover and before paid work.

The listener stays available without manual windows while all native gates pass.
Exhaustion refuses new admission; an already admitted permit still uses k-1 and
may finish normally even after the global count closes new admissions. Existing
expiry, one-at-a-time admission, finalization and readiness/liveness remain intact.

## Smallest source and native transition

Use total_calls: Literal[5,6,7] | None =5; null denotes the new budget-only mode.
Do not add a parallel policy/ledger/provider. Legacy serialization, hashes and
arithmetic remain byte-for-byte compatible. Keep the permit format; widen only
its positive strict integer ordinal to the fixed SQLite bound9,223,372,036,854,775,807.
The SQLite count uses the same nonnegative technical bound, not an allowance.
Refuse new admission at that maximum before calculating or storing n+1.

Add one explicit V2-only CLI/native action enable-budgeted-calls. Its sole
predecessor is canonical policyN7/used7/schema14. Require expected exact policy
hash/JSON, unsuspended row, old owner retired, no active lease/carrier/unknown paid
intent, valid complete fresh observation, affordable proposed policy and existing
335-second start envelope. In one native transaction migrate only the budget
table14->15 and CAS that same row to total_calls:null. Only total_calls plus the
truthful qualified profile/image bindings may change; preserve every other
policy value and every existing row/permit/history. Floors may only advance to
max(H,S). Preserve historical schemas10-14 and qualification_runs unchanged.

Recognize exact schema15 for ordinary reopen; old executables must refuse it.
Ordinary startup must never migrate or enroll. After transition clear the cached
observation and require a new observe before readiness. Preserve current unknown
commit/close handling: observe uncertain results, never blind-retry or restore
an earlier allowance-bearing database.

## Evidence and ownership

Root owns this source branch; thread01a16023 is the sole live operator. No audio,
model, turn-taking, prompt, Calendar, Inbox, auth, worker or dependency changes.
Use pinned Python3.13.17, frozen dependencies and offline native SQLite tests.
Use synthetic test data, never copied provider records or existing secrets.
Oracle standard Pro session sparra-budget-only-oct26 returned
APPROVE_B_WITH_EXACT_AMENDMENTS on this financial proposal. Apply the same
reserve in the separate OpenRouter key check; use rawS in every deficit; preserve
per-view restrictive intersection and permit k-1; assess the transition using
the proposed null policy rather than the exhausted predecessor; perform the
schema/CAS atomically; and reject ordinal overflow before increment. These are
implementation/qualification requirements, not executed evidence. User has already authorized
autonomous implementation, review, qualification and owned deployment; no extra
approval loop is required. Source tests do not prove provider/phone acceptance.

## Required security prerequisite discovered during exact-source CI

The first candidate's mandatory repository scan reported HIGH CVE-2026-112940
in existing transitive filelock3.32.4. The maintainer advisory confirms the
affected range and fix4.2.1; current stable PyPI is4.2.4 (Python>=3.11). Narrowly
allow this existing dependency to resolve to4.2.4 in uv.lock, with every other
package record and all direct pins unchanged. This necessary security
prerequisite supersedes only the no-dependency-change sentence above. No scanner
suppression, provider/model change, override patch or general upgrade is allowed.
Verify official provenance, actual consumer compatibility and frozen installation;
the exact final source must pass the unmodified mandatory scan and full hosted CI.

Primary sources:
- https://github.com/tox-dev/filelock/security/advisories/GHSA-v4x8-3wr5-3wq8
- https://pypi.org/project/filelock/4.2.4/
