# Remaining pilot reserve amendment

The owner explicitly authorized a reserve calculated over remaining calls on
2026-10-21, after disclosure that it lowers protection against delayed charges.
This replaces the fixed-whole-reserve rule in the 19 October continuous-pilot
and 20 October restatement designs. Their historical qualification remains true;
this is a newly authorized policy, not a correction to what those tasks promised.

## Authorized rule

Keep separate control caps `C=5` USD, lifetime `N=5` admissions, original
accounts/credentials/slot/policy UUID, historical anchors, monotone spending
floors, committed counts, receipts, paid intent and paid/key expiry fences.
There is no refund, top-up, re-enrollment, renewed allowance or midnight reset.
The live pilot had consumed two admissions when this amendment was authorized.

For each provider retain `R0=6*d+u` and the existing constants:

| Provider | R0 | d | u |
| --- | ---: | ---: | ---: |
| OpenRouter | 3.780408 | 0.505068 | 0.75 |
| Telnyx | 1.22 | 0.12 | 0.5 |

New admission uses `m=n<5`. An existing permit of ordinal `k` uses `m=k-1`.
Compute all terms from the same applicable view and index:

`E=max(H,S,A+m*d)` and `R(m)=R0-m*d`.

Require `E+R(m)<=5`, `cash-(E-S)>=R(m)` and, separately for OpenRouter,
`key_remaining-(E-S)>=R(m)`. Retain all strict validation, identity/freshness,
original-anchor, count, active/unknown-effect, generation and paid fences.
The minimum admitted-index reserves at m=4 are 1.760136 USD OpenRouter and
0.74 USD Telnyx. The sixth admission always refuses.

The combined envelope is `A+R0` under forecast dominance and grows for reported
overruns. This is a smaller reserve than the prior rule at m>0, including when
reports dominate. It is an operational forecast, not a guaranteed final invoice
ceiling or proof of accounting finality. No observed cheap call justifies it.

## Native authority and qualification

Money comparisons can loosen as m increases: the largest global count is not
the conservative substitute for a permit's k-1. Preserve coherent operands in
each committed/pending view, their restrictive intersection, publication only
after commit and actual pre-effect rechecks. Missing/invalid/unknown authority
continues to deny paid work, even if this arithmetic would otherwise pass.

Use the current affordable() boundary and existing real SQLite/native tests.
No new provider, dependency, schema, ledger, framework or configuration surface.
Pipecat 1.12.0, Python 3.13.15, audio/LLM ceilings and existing Calendar, end-call
and transfer contracts remain unchanged.

Root must qualify an immutable image and explicitly rebind only its consumer
image/profile identities after old-owner retirement. Preserve original financial
identity and all stored history; the historical count-one Calendar operators
are invalid after the actual new call. Readiness and a real Google booking plus
Inbox reference require fresh runtime/provider/human evidence.
