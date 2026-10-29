# Pilot admission using reported spending and the current call reserve

## Intent and scope

Louis rejected the proposed support detour on 29 October and wants to use the
existing provider dashboard. Its displayed OpenRouter spending is below the
configured lifetime USD5 key limit; the present refusal instead comes from
`max(H,S,A+n*d)+R`. Historical forecasts remain reserved after calls terminate.

Prepare the smallest explicit pilot accounting change. Keep the independent
USD5 control caps, the original key/account/UUID/slot/anchors/accounting start,
monotone reported floors, used_calls8, every lease/permit/history, expiry,
paid-intent fencing and qualified image/profile binding. No support message,
provider setting, payment, new key, reset, call or invitation is authorized.

## Approaches

1. Keep lifetime forecasts and obtain exhaustive final-cost receipts: strongest
   historical coverage, but no usable finality primitive was found and Louis
   does not want the support detour. Park it; it is not a continuation gate.
2. Reported spending plus the existing active-call reserve: recommended for this
   pilot. Reuse the current readers and exclusive native owner, with an explicit
   policy opt-in and fresh observations after operational closure.
3. Dashboard alone and no local admission checks: insufficient. A dashboard does
   not fence concurrent dispatch, stale data, unknown paid intent or expiry.

## Exact protection and its limit

For the opt-in budget-only policy, replace only historical forecast liability:
`E_p=max(H_p,S_p,A_p)`. Keep `R_p=max(R0_p-m*d_p,u_p)` and all three existing
comparisons unchanged: `E_p+R_p<=5`, `cash_p-(E_p-S_p)>=R_p`, and the separate
OpenRouter key-remaining check using the same raw S and reserve. Keep m=n for
new admission and m=k-1 for an existing permit. At the current count8 the
unchanged minimum reserves are USD0.75 OpenRouter and USD0.50 Telnyx.

The native nonterminal lease and durable paid_call_control_id represent the
single active or operationally unresolved call: they block a second admission.
Do not clear them on a timer, an absent receipt, a cheap aggregate or an operator
claim. Preserve the matching accepted-hangup/verified-terminal authority.
Require a provider collection started strictly after both the effective local
terminal transition and the matching resolution of paid intent. A carrier's
closed_at can predate its local processing; keep that durable timestamp truthful.
Use a volatile writer invalidation barrier advanced on these effective native
transitions, owner startup and activation CAS. An earlier collection delivered
late cannot repopulate eligibility. Restart discards observations and resets
the barrier while preserving all durable fences and floors. Repeated/foreign
settlement and terminal notifications cannot advance it. Preserve observations
for a valid active permit: admission closure alone must not erase them.

This deliberately stops reserving worst-case cost forever for operationally
closed calls. Operational closure and a subsequent observation are NOT proof
that all billing is final. Delayed charges, provider work continuing after
cancellation, or costs above the forecast can make final invoices exceed the
configured control caps. The existing margins reduce that exposure; they do
not establish an upper bound on all unreported historical liabilities. Do not
describe this mode as preserving the old pessimistic historical guarantee.

OpenRouter checks configured key limits on paid requests; its separate in-flight
mechanism estimates token prices, excludes several other cost dimensions, and
does not apply to every account. Do not claim it guarantees an exact final
invoice ceiling for speech. Telnyx outbound daily limits do not establish a
lifetime USD5 ceiling for inbound calls, media and number charges. The local
Telnyx cap, cash check and native observations therefore remain mandatory.
The dashboard is an operator view, not an automatic alert or settlement receipt.

## Smallest implementation boundary

Add one optional literal policy field, `admission_accounting="reported"`,
valid only with total_calls:null. Omit the absent field from serialization with
the pinned Pydantic conditional exclusion so every legacy JSON/hash and default
arithmetic remains identical. Existing finite5/6/7 policies remain unchanged.
Do not add a ledger, service, dependency, request-cost store or general budget API.

Add one explicit native CLI/writer action `enable-reported-spend`; ordinary
startup and image-only rebind-release cannot enable it. Use the existing owner
transaction and schema15 with an exact expected predecessor hash/JSON/count,
retired predecessor, no nonterminal lease/carrier/paid intent, valid fresh
matching facts and a sufficient start envelope. Truthful
image/profile bindings must already have been rotated by the separate unchanged
rebind-release action. This CAS changes only the opt-in; preserve count8, the
exact other policy fields, anchors and floors, using expected_used_calls too.
Clear the observation after the CAS, requiring another refresh at runtime.
No store creation or schema migration is allowed for this operator action.
An old executable must refuse the new policy rather than silently reinterpret it.

Centralize the post-invalidation observation check in the writer observe,
snapshot and consume boundary so committed and pending views remain restrictive.
Invalidate on matching operational settlement and terminal lease transition;
preserve permit checks during actual active work. Never lower a
reported floor or use content_departed_generation as a billing marker.

## Qualification and activation

Use pinned Python3.13.17 and frozen dependencies; no production exports in tests.
Extend the existing unit/integration fixtures for the eighth/ninth boundary,
both independent cap/cash/key boundaries, a pre-closure observation delivered
late, unknown paid intent, restart, restrictive pending publication, and atomic
opt-in conservation/refusal. Reuse existing expiry/ordinal/legacy tests.
Run the affected selectors and canonical static checks once; hosted CI remains
required for any release. Oracle reviews the exact protection tradeoff.

Oracle standard Pro review sparra-observed-budget-design-oct29 returned AMEND
with the single temporal interleaving above; it found the proposed arithmetic
and the explicit risk statement coherent. This is a design review, not source
or runtime acceptance. Prepare/review the patch first, then obtain a concrete
human decision before production activation of the weaker historical-liability
protection. Support and exhaustive finality are not preparation gates.

Primary documentation checked on 29 October2026:
- https://openrouter.ai/docs/api-reference/limits
- https://github.com/team-telnyx/telnyx-node/blob/master/src/resources/outbound-voice-profiles.ts
