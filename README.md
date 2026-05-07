# LiveKit Plugins Munsit

STT plugin for [Munsit](https://munsit.com), the merged Faseeh + Munsit ASR product. Optimized for Arabic with optional Arabic/English code-switching (`munsit-en-ar` model).

## Install

```bash
pip install livekit-plugins-munsit
```

(While developing inside this monorepo, use `uv sync --all-extras --dev`.)

## Quickstart

```python
from livekit.agents import AgentSession
from livekit.plugins import munsit, silero

session = AgentSession(
    stt=munsit.STT(),                # batch mode is the default
    vad=silero.VAD.load(),
    # ... llm, tts ...
)
```

`MUNSIT_API_KEY` is read from the environment by default. Munsit's batch HTTP endpoint is more accurate and returns word-level timestamps; we use it by default. The `AgentSession`'s VAD signals end-of-speech via `flush()`, which causes the plugin to POST the buffered utterance and emit a single `FINAL_TRANSCRIPT` with per-word timings.

## Modes

| Mode | When to use | Endpoint | Latency | Word timestamps | Interim events |
|---|---|---|---|---|---|
| `"batch"` (default) | Production with `AgentSession` + VAD; transcribing recorded audio. | `POST /api/v1/audio/transcribe` | VAD detection + upload + server processing (~1–2 s for short utterances) | yes — populated on `SpeechData.words` | no |
| `"streaming"` | You need live captions / on-the-fly UI updates while the user is still speaking. | WebSocket `/api/v1/websocket/speech-to-text` | ~700 ms idle threshold by default | no | yes (`INTERIM_TRANSCRIPT` flow as cumulatives arrive) |

```python
munsit.STT()                       # batch (default)
munsit.STT(mode="streaming")       # opt-in for low-latency interims
```

`STT.recognize(audio_buffer)` always uses the batch HTTP endpoint regardless of `mode`.

## Parameters

| Param | Default | Notes |
|---|---|---|
| `mode` | `"batch"` | `"batch"` (HTTP) or `"streaming"` (WS). |
| `model` | `"munsit"` | `"munsit"` (Arabic) or `"munsit-en-ar"` (code-switching). |
| `api_key` | `MUNSIT_API_KEY` env | Required. |
| `base_url` | Munsit prod WSS | Override the streaming WebSocket URL. |
| `batch_base_url` | Munsit prod HTTPS | Override the batch HTTP URL. |
| `auth_method` | `"header"` | `"header"` / `"bearer"` / `"query"` — Munsit accepts all three on both endpoints. |
| `sample_rate` | `16000` | Used for the synthesized WAV header. |
| `num_channels` | `1` | |
| `interim_results` | `True` | Emit `INTERIM_TRANSCRIPT` events. Only meaningful in streaming mode. |
| `endpointing` | `"server_diff"` | Streaming-mode only: `"server_diff"` (idle timer) or `"client_vad"` (energy-based). |
| `finalize_after_silence_ms` | `700` | Streaming-mode `server_diff` idle threshold. |
| `energy_filter` | `False` | Streaming-mode `client_vad` only. |
| `vad_silence_ms` | `1500` | Streaming-mode `client_vad` silence-end duration. |
| `language` | `None` | Label attached to `SpeechData.language`. Defaults to `"ar"`. |
| `http_session` | `None` | Custom `aiohttp.ClientSession`. |
| `extra_query_params` | `None` | Forward-compat for new Munsit query params (streaming WS only). |

## Streaming mode endpointing

When using `mode="streaming"`, the plugin offers two end-of-utterance strategies:

- **`server_diff`** (default) — keep one long-lived WS open, emit `INTERIM_TRANSCRIPT` on each transcript update, emit `FINAL_TRANSCRIPT` + `END_OF_SPEECH` after `finalize_after_silence_ms` of server silence. Best when `AgentSession` already has its own VAD / turn detector.
- **`client_vad`** — open WS on local energy onset, close on silence end. Each utterance gets a fresh WS connection. Slightly higher per-utterance latency but harder utterance boundaries.

## Examples

See [`examples/README.md`](examples/README.md). Three demos cover file-based, mic-based, and full-agent scenarios. The file demo prints per-word timings when batch mode is active.
