# Operator voice delivery rate

Set optional `inference.tts_speed` in the protected candidate/qualified profile
to pass OpenRouter's standard top-level `speed` parameter. The pilot setting
prepared for Sparra is `1.15`, with its selected MAI Voice 2.1 Flash model and
French Harper voice.

The profile accepts finite numbers from 0.5 to 2.0. These are local operating
bounds, not a guarantee that every model supports the range. Omission or null
sends no speed parameter and preserves the existing canonical profile bytes
and digests. Any explicit rate participates in the inference digest and must
be included in fresh qualification.

OpenRouter documents provider-dependent support: an unsupported model can
ignore or reject a non-default rate. Use the setting only with a qualified
model/voice combination. Do not apply a generic default or alter PCM sample
rates to accelerate speech.

An operator comparison used the same non-customer phrase and MAI Flash/French
Harper voice, alternating three default and three 1.15 requests. All returned
PCM successfully; median generated duration was 5.148 versus 4.006 seconds.
This is evidence about that synthesis duration, not listening quality,
telephone acceptance, or end-to-end response latency. No audio or credentials
were retained by the comparison.

References: [OpenRouter TTS](https://openrouter.ai/docs/guides/overview/multimodal/tts),
[Pydantic field exclusion](https://docs.pydantic.dev/latest/concepts/serialization/#field-inclusion-and-exclusion).
