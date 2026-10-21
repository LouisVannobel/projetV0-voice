# Bounded Calendar dialogue correction

A genuine telephone call created one Google event, with the matching Inbox backlink. The conversation repeated an acknowledged recap, announced uncertainty after a fast successful booking, and stayed open after a clear contextual ending. This is a correction of the existing Pipecat flow, not a new orchestration layer.

## Proven boundaries

- The whole-utterance consent parser rejects harmless internal sentence punctuation between already allowed positive clauses. Keep exact native recap/ACK/first-current-turn/one-claim authority; normalize separators without accepting questions, negation, corrections, conditions, quotations or reported consent. Explicit supported suffix `c'est bon` may follow an existing affirmative prefix; bare thanks/negation are not booking consent.
- App prepare-create durably publishes `unknown` before provider insert, then verified success becomes `confirmed`. The native waiter exits immediately on `unknown`. A synthetic same-operation sequence reproduces premature uncertainty. Continue canonical result reads within the unchanged 12-second bound. Observing `unknown` latches read-only: later transient/ambiguous read errors or missing/waiting-evidence responses never resubmit. Never create another operation/provider insert. Preserve pre-send frozen retries before a possible-send observation. A persistent unknown remains uncertain at the deadline, with pending identity retained. A permanent failure remains a failure, cancellation propagates, and stale caller scope cannot speak or regain authority.
- Existing end_call instructions exclude natural contextual completion. Permit clear present caller completion after an offer of further help when no request remains, through the existing scoped end_call tool. Mere task success, thanks, declining a proposed date/service/contact, quotation, negation, rhetorical question or another outstanding request is insufficient. Keep context-free direct-command matching narrow; assistant farewell text alone never gains authority. Keep one owned goodbye, final PCM/mark ACK and native EndWorker/End drain unchanged.

The historical exact RPC sample and actual production-model tool proposal were not recorded. These source/probe defects are supported corrections, not a claim of recovered historical traces. Scripted native tests verify execution; they do not evaluate production-model semantic choice.

## Limits and qualification

No App code/migration/redeploy, data repair, new metric/store/framework/dependency/model/voice/provider, timeout extension, money-policy change or paid replay. Preserve original UUID/qualified_at/identity/anchors/floors/SQLite/history, separate USD5 caps and lifetime5 admissions with4 already used. The Google event and original transcript remain untouched. Same original custom bundle and ops01-owned-manager to ops02 deployment.

One cohesive source owner, independent full-range spec/quality review, affected offline/native/static checks, then unchanged complete hosted Linux/native/image/security gate and immutable release compatibility. Root alone owns GitHub/provider/SSH/deploy. Keep genuine phone acceptance separate; no extra paid synthetic diagnostics.
