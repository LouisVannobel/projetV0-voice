# Bounded native pilot validity renewal

The human authorized the phone lot needed for one continuously filmed product booking: qualify today's operational context, add a native renewal, and activate at most 60 minutes only after recording and human availability are confirmed and current financial checks pass. Existing USD5 per-provider caps, identity, anchors, floors, counter, reserves and retained history stay intact. No recharge, reset, refund, extra accounting service or fabricated event.

## Existing boundary

The reported-spend pilot persists one canonical policy and its hash in schema15. `paid_until` is expired. Editing its mounted JSON cannot change durable admission authority. `rebind-release` permits only image/profile differences and requires a currently valid financial observation, so it cannot by itself deliver the new code across an expired budget window.

The qualified deployment profile has no sliding validity window. Its existing strict loader validates the deployment, runtime, bundle, inference and image bindings. The budget policy contains the independent time gate and original key expiry. Current runtime/context qualification remains separate from source, image qualification and a successful real phone booking.

## Smallest native change

Add one explicit operator action, `renew-validity`, through the existing CLI and persistence owner. No schema, model, provider adapter, new table or durable operation-receipt subsystem is needed.

The candidate policy may differ from its exact predecessor only in:

- `paid_until`;
- `image_digest` and `qualified_profile_sha256`, allowing the new release to be activated in the same atomic transition.

All other policy JSON values, including their serialized Decimal representation, remain identical. Keep the same policy ID, `total_calls=null`, `admission_accounting=reported`, financial identity, key expiry, accounting start, anchors, runtime contract, deployment, bundle and inference. Native reserve/forecast/cap calculations remain unchanged. Only `policy_json` and `policy_sha256` may be updated; every other column and table remains intact.

Use a stopped/fenced predecessor and a dedicated existing-store operator writer, not a live runtime reconfiguration. Reuse native profile loading for both old and new tuples, exact schema15/store checks, ownership queue/transaction, and bounded writer shutdown. Missing or legacy stores are never created or migrated by this action.

## Fixed request and idempotence

Inputs are the predecessor's exact policy hash and consumed-call count, the explicitly retired predecessor, its native qualified profile/image, and an explicit timezone-aware UTC `target_paid_until` equal to the candidate policy date. There is no sliding `now + duration` CLI flag.

The policy hash is the durable version/epoch. The requested absolute target never changes during retry or recovery:

1. Native preflight reads and validates the canonical row before any provider collection. An exact canonical candidate already present returns the distinct `target_observed` result, without collecting, invalidating a newer observation, extending time or changing any row. This remains an observation after expiry or later legitimate consumption; it does not restore the old counter/floors or claim readiness/causality.
2. A valid exact predecessor needing current financial facts returns `observation_required`, without mutation. Structural/ownership/hash/count/whitelist failures refuse before collection.
3. The operator collects once using the existing bounded authenticated GET-only collector. It then submits the same fixed request and fresh fact to the native owner.
4. The owner revalidates everything in its existing immediate transaction. Another target is a conflict. The exact target winning a race is `target_observed`; it never causes a second update.
5. A newly committed target returns a distinct accepted result. No caller treats nonempty status strings as boolean success. Existing actions retain their public behavior.

On a possibly committed timeout/cancellation/shutdown failure, retain `owned_budget_transition_unknown`. Observe the exact fixed target before deciding what happened; never blindly retry or manufacture a later date.

## First-commit gates

All existing financial and exposure gates remain. In addition:

- V2 strict runtime, schema15 existing store, canonical valid predecessor/candidate, exact expected policy hash/count and validated old/new native profile bindings.
- Both policies remain reported and budget-only. Compare their serialized JSON excluding only the three whitelisted fields above. Candidate hash must advance; old date must be strictly earlier than the target.
- The predecessor is actually retired and its window is already expired (`predecessor.paid_until <= writer_now`); this action cannot roll an active window forward. Exact-target observation remains read-only and takes precedence over this first-commit condition. No local nonterminal lease, provider active call, suspension or unresolved paid intent; stored floors are finite and nonnegative.
- At the native mutation check, `start_envelope <= target_paid_until - writer_now <= 3600 seconds`; use the existing start envelope, not another copied constant. The policy validator still limits the date to the unchanged original key expiry.
- `observation_valid(candidate, fact, writer_now)` and `affordable(candidate, fact, stored_count, reported_floors=stored_floors)` both pass. Checking the candidate's newly authorized time window is the explicit new behavior; all identity/freshness/cash/key/cap checks are unchanged.
- Full CAS predicate includes slot, predecessor hash/JSON, consumed count, both floors, unsuspended state and null paid intent. Unexpected rowcount is a native conflict.

The absolute target cannot move while waiting to commit, so time spent preparing or shutting down shortens the window. Do not alter provider settings if financial eligibility is insufficient; stay closed and report the observed reason.

## Publication and lifecycle

A first commit invalidates the old reported observation and retains the existing volatile freshness barrier. Publish no admission capacity from the activation fact. Existing precommit restrictive snapshots and postcommit publication ordering remain authoritative. The exact new runtime starts against the exact committed policy/profile/image and requires its own subsequent fresh observation before readiness. A fact collected for the old context or begun before the new native freshness boundary must not reopen admission merely because its timestamp is recent. Verify the actual provider reporting scope still covers the accounting interval and target context after the old window ended; time freshness alone is not sufficient. The old process cannot resume alongside it.

The already-present-target path is read-only and must not clear a newer legitimate observation or reopen admission. This is an isolated operator contract, not an in-place live writer policy switch.

## Delivery and activation ownership

The source/delivery owner prepares only this new lot. Current host qualification must explain representation drift against qualified declarations, retain every previous refusal, and establish a fresh stable current state. Never replace historical hashes just to make an operator pass. Full current state is the conservation baseline for a later mutation; historical retention or other legitimate evolution must be characterized first.

Qualify the new source with focused RED/GREEN, independent review, Oracle, exact new CI and immutable image/profile provenance. Closed prior campaigns remain closed. Before each remote mutation, provide the concrete cause/delta/checks/owner/window and bind one-shot controls to the actual current tuple.

No production policy extension or candidate activation is prepared until the recording owner confirms the recording setup and human availability. At that final gate, fix the target once, at most60 minutes ahead; obtain fresh native eligibility, fence/retire the predecessor, execute the reviewed native CAS once, start the coherent tuple, and verify conservation plus current readiness. A refusal or unknown outcome preserves closure until canonical observation resolves it. No operator phone call, replay or Calendar connector event substitutes for the human's real product flow.

Reported provider charges can arrive late and exceed configured caps; this remains the previously accepted pilot risk, not a final-invoice guarantee. A source or readiness pass never establishes Google approval or a successful booking.

## Focused validation

Use the existing native budget integration harness and CLI tests with offline provider seams. Pin one successful expired-window renewal, coalesced release, complete before/after row conservation, fixed replay across restart/time, concurrent/conflicting targets, precommit failure/unknown result, volatile freshness, and the time/exposure/identity/finance refusal matrix. Extend existing consumers; avoid mirrored helper tests or a new test framework. Run only the named affected regression selectors locally, plus the repository's normative type/lint/lock checks and new release gates.
