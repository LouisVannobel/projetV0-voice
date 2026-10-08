# Listening while HTTP transcription is pending

In Pipecat 1.7, segmented HTTP transcription occupied the STT processor's input
dispatcher. Audio arriving while the request was pending could not reach the
downstream VAD. The held-request regression fails on 1.7 and passes on the
official 1.12 release, which includes [upstream PR5630](https://github.com/pipecat-ai/pipecat/pull/5630).

Keep the existing Telnyx input → STT → user aggregator → LLM → TTS → Telnyx
output composition. Pipecat owns the segment audio buffer, one sequential
transcription task, cancellation and graceful queue drainage.

HTTP completion alone does not authorize a response. The bounded adapter
retains only unpublished text and attaches the successful FIFO segment
frontier to native `TranscriptionFrame.metadata`. The native SmartTurn stop
strategy is extended to require that the consumed frontier covers the latest
locally observed VAD segment and that speech has stopped. This also covers an
older transcript that was already delivered before the caller continued.

An empty response still supplies a completion receipt. It adds no user text.
The stop strategy may use an internal frame clone to keep previously observed
text in its endpoint state; that clone never enters the user aggregation.
Joined text preserves the last provider result as that segment's result,
not as provenance for every word in the joined text.

The independent native turn watchdog can force a stop. The strategy's public
`handle_user_turn_stopped` callback latches the existing failure owner when
coverage is missing. A native `FunctionFilter` immediately before the LLM
blocks contexts after failure or admission closure, including terminal flushes.
The user aggregator's deferred timeout event is not a synchronous veto.
Its synchronous before-process hook closes inference on End or Cancel before
the native aggregator flushes retained text, even while the call owner joins it.

The pilot bounds are four pending segments, 32 KiB of retained UTF-8 text,
eight seconds per HTTP request and eight seconds for graceful drainage.
The watchdog is 36 seconds: four request budgets plus four seconds of
delivery allowance. Successful complete turns release immediately; this
watchdog is not a normal response delay or a measured STT p99. Overflow and
timeouts fail explicitly instead of dropping speech or starting parallel work.
These bounds do not establish a total audio-buffer memory or latency guarantee.

Native End drains accepted segments and does not submit an open speech buffer.
If End arrives while the caller is still speaking, retained known text is marked
as terminal partial text and cannot certify the open speech as complete. The
call owner closes admission before terminal drainage. Cancel discards retained
text and joins native cancellation.

The configuration also explicitly sets `empty_user_turn=None`: silence or an
interruption with no transcript must not inject the new default recovery prompt.
Business configuration, models, prompt, voice and audio retention policy remain
unchanged by this correction.

The 1.12 migration additionally updates the public `assert_given` import,
checks native processor usability after setup, and preserves the project's
outbound extension authority: only its validated `TelnyxMarkFrame` can send a
mark. Generic normal or urgent transport messages are rejected.

## Verification boundary

Offline regressions use the actual native worker, HTTP STT, aggregator and
strategy. HTTP, VAD predictions and endpoint predictions are controlled where
specified. Such tests prove scheduling and completion policy, not acoustic
quality or real-provider latency. Deployment additionally requires the hosted
security/image gates and a fresh operator qualification; passing local tests
does not qualify a live phone call.

Official references: [STT latency](https://docs.pipecat.ai/pipecat/fundamentals/stt-latency-tuning),
[interruptions](https://docs.pipecat.ai/pipecat/fundamentals/interruptions),
[Telnyx streaming](https://docs.pipecat.ai/pipecat/telephony/telnyx-websockets).
