# Preserve authority across provider spend restatements

The reserve-preserving rule below records this repair's historical contract.
The separately authorized [21 October remaining-reserve amendment](2026-10-21-remaining-pilot-reserve-design.md)
supersedes only that reserve rule after disclosure of reduced delayed-charge
protection. The high-water/raw distinction and other financial authority remain.

This narrow amendment supersedes the 19 October design's refusal of every raw
reported-spend decrease. It preserves the original financial authority and the
accepted Pipecat/Telnyx conversation. No reserve, cap, forecast, identity, count,
scope, lifetime, paid fence or funding grant changes.

Keep authenticated raw observations separate from durable reported high-water
floors H. Persist H'=max(H, raw), never a forecast. For every admission,
transactional dispatch and synchronous current-permit check, independently per
provider compute E=max(H, anchor+m*forecast, raw). Require E+reserve<=5 and
cash-(E-raw)>=reserve; OpenRouter additionally requires key_remaining-(E-raw)
>=reserve. New admissions use m=n; an existing permit ordinal k uses m=k-1.
Never substitute H for raw inside the cash/key deficit subtraction.

Frozen writer snapshots carry their own committed reported floors and raw fact.
Preserve the restrictive committed/pending intersection, immediate pre-paid
checks and all identity/completeness/freshness/unknown-intent/expiry refusals.
Retain refusal of raw below the original policy anchor as a narrow original-
scope fence; ordinary restatements above that anchor are acceptable only under
the full floor-aware funding checks. No tolerance epsilon or grace interval.

For this compatible source repair only, Root may perform one audited offline
CAS of image_digest and qualified_profile_sha256 in the existing policy. This
explicitly amends the otherwise immutable consumer binding, not the grant.
Retire/join the old owner; require no active carrier calls, terminal native
leases, empty outbox, unsuspended row and no unknown paid intent. Verify a new
qualified immutable release and take a protected consistent backup. Compare the
exact old JSON/hash, count, floors and ownership state in the transaction; update
only policy JSON/hash, exactly one row. Preserve policy UUID and every other
field, historical permits/receipts/qualification rows and all other tables.
Stage matching protected policy/config while stopped. Unknown COMMIT is observe-
only, never replay or restore an older allowance-bearing database. No enrollment,
automatic upgrade framework, dependency change or provider mutation is needed.

Oracle standard Pro session sparra-spend-restatemen-oct20 completed with
APPROVE_WITH_EXACT_AMENDMENTS. Reported floors must be explicit trusted inputs,
with no implicit zero default: require exactly two finite, nonnegative Decimals
within the existing numeric bounds before use, including writer-loaded floors;
missing or invalid snapshot floors cannot grant admission or continuation. The
retained raw-below-anchor veto is an explicitly stricter anomaly policy, not proof
against an accounting-window reset. Independent source review has endorsed the
equations and this bounded rebind. Source tests remain offline; no production
data or provider credentials in fixtures, logs or second-model attachments.
