# livekit-plugins-munsit

[LiveKit Agents](https://docs.livekit.io/agents/) plugin for **Munsit** voice AI:
Arabic speech-to-text (batch and live streaming) and Arabic text-to-speech.

> This package consolidates the former `livekit-plugins-faseeh` (TTS) — that name now
> ships a deprecation shim that installs this package.

## Installation

```bash
pip install livekit-plugins-munsit
```

Set your API key (from the [Munsit dashboard](https://app.munsit.com/en/api-keys)):

```bash
export MUNSIT_API_KEY=your-key
```

## Speech-to-Text

```python
from livekit.agents import AgentSession
from livekit.plugins import munsit, silero

session = AgentSession(
    stt=munsit.STT(mode="streaming"),   # live transcription over WS /api/v1/listen
    vad=silero.VAD.load(),
)
```

Two modes:

| | `mode="streaming"` (default) | `mode="batch"` |
|---|---|---|
| Transport | `WS /api/v1/listen` | `POST /api/v1/audio/transcribe` |
| Interim results | yes (`INTERIM_TRANSCRIPT`) | no |
| Turn detection | server-side (endpointing + semantic smart-turn) | client energy VAD |
| Word timestamps + confidence | yes, on finals | yes |
| Models | `munsit`, `munsit-en-ar` (default) | `munsit`, `munsit-en-ar` (default) |

Streaming options:

```python
munsit.STT(
    mode="streaming",
    endpointing_ms=800,          # silence that ends a turn (100–5000; retunable live)
    smart_turn=True,             # semantic end-of-turn gating
    hotwords=["فرحان", "تطبيق منصت"],   # custom vocabulary (≤200 entries, ≤40 chars)
    interim_results=True,
    correlation_id="call-1234",
    on_enrichment=lambda ev: print(ev),  # per-turn Sentiment / Gender events
)
```

Batch extras: `hotwords=[...]` and `return_confidence=True` (per-word confidence on
timestamps). `hotwords` only bias the Arabic `munsit` model — with `munsit-en-ar`
(the default) they are ignored with a warning, in both modes.

End of turn in streaming mode: a `Results` event with `is_final=true` and
`speech_final=true` is a genuine end of turn, so the plugin emits `END_OF_SPEECH` right
after that `FINAL_TRANSCRIPT` instead of waiting for the server's `UtteranceEnd` (which is
then deduplicated). A final with `speech_final=false` is a forced split during long
speech and emits `FINAL_TRANSCRIPT` only.

There is no client-side finalize on `/api/v1/listen`: the only control messages are
`KeepAlive`, `Configure` and `CloseStream`, and `CloseStream` ends the whole session.
Turn boundaries are decided server-side (`endpointing_ms` + `smart_turn`), so the plugin
ignores LiveKit's `flush()` on the STT stream — it cannot ask the server for an early
final without closing the socket.

## Diagnostics

**`MUNSIT_TRACE_FILE`** — set this env var to a file path and the streaming client appends
one JSON line per server event received (`rx`) and per control message sent (`tx`):

```json
{"ts": "2026-09-24T10:15:32.118+00:00", "munsit_session_id": "…", "request_id": "…",
 "direction": "rx", "type": "Results", "turn_id": 3, "is_final": true, "speech_final": true,
 "text_len": 42, "confidence": 0.97}
```

Lines carry only event metadata — never transcript text, the API key or the connection
query string — so the file is safe to share. Use it to see exactly when finals and
`UtteranceEnd` landed relative to each other. Writing is best-effort: an unwritable path
disables tracing instead of failing the stream.

**`filter_caption_artifacts`** (constructor flag, default `True`) — the code-switch model
can emit caption-style hallucinations on interims before any real words, e.g. `[Music]`,
`[BLANK_AUDIO]`, `(music)` or a `>> SPEAKER NAME:` tag. With the flag on, interims that are
nothing but bracketed tags are dropped, and a leading speaker tag is stripped from interims
and finals (an interim left empty is dropped). Finals are never dropped. Each filtered
event is logged at `DEBUG` on the `livekit.plugins.munsit` logger. Pass
`filter_caption_artifacts=False` to receive the raw server text.

## Text-to-Speech

```python
from livekit.plugins import munsit

tts = munsit.TTS(
    voice_id="ar-hijazi-female-1",
    model="faseeh-v1-preview",
)
```

`MUNSIT_API_KEY` is used for both engines (the legacy `FASEEH_API_KEY` env var still works
for TTS).

## Documentation

Full API reference: https://docs.munsit.com — see *Speech-to-Text Streaming* for the
`/api/v1/listen` protocol (events, close codes, billing).

## Development

```bash
uv sync
uv run pytest tests/ -k "not Integration"   # offline suite (fake server)
MUNSIT_API_KEY=... uv run pytest tests/     # includes live integration tests
```

## License

Apache-2.0
