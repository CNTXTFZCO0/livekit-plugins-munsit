# LiveKit Plugins Munsit

Streaming STT plugin for [Munsit](https://munsit.com), the merged Faseeh + Munsit ASR product. Optimized for Arabic with optional Arabic/English code-switching (`munsit-en-ar` model).

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
    stt=munsit.STT(model="munsit", interim_results=True),
    vad=silero.VAD.load(),
    # ... llm, tts ...
)
```

`MUNSIT_API_KEY` is read from the environment by default.

## Parameters

| Param | Default | Notes |
|---|---|---|
| `model` | `"munsit"` | `"munsit"` (Arabic) or `"munsit-en-ar"` (code-switching). |
| `api_key` | `MUNSIT_API_KEY` env | Required. |
| `base_url` | Munsit prod WSS | Override for staging or self-hosted. |
| `auth_method` | `"header"` | `"header"` / `"bearer"` / `"query"` — Munsit accepts all three. |
| `sample_rate` | `16000` | Used for the synthesized first-chunk WAV header. |
| `num_channels` | `1` | |
| `interim_results` | `True` | Emit `INTERIM_TRANSCRIPT` for cumulative updates. |
| `endpointing` | `"server_diff"` | Or `"client_vad"` — see below. |
| `finalize_after_silence_ms` | `700` | Idle threshold for `server_diff` finalization. |
| `energy_filter` | `False` | `AudioEnergyFilter` instance or `True` (defaults). Used in `client_vad`. |
| `vad_silence_ms` | `1500` | Silence duration that ends an utterance in `client_vad`. |
| `language` | `None` | Label only — Munsit doesn't return language. Defaults to `"ar"`. |
| `http_session` | `None` | Custom `aiohttp.ClientSession`. |
| `extra_query_params` | `None` | Forward-compat for new Munsit query params. |

## Endpointing modes

The plugin offers two strategies for deciding when an utterance has ended:

- **`server_diff`** (default) — keep one long-lived WS open, emit `INTERIM_TRANSCRIPT` on each transcript update, emit `FINAL_TRANSCRIPT` + `END_OF_SPEECH` after `finalize_after_silence_ms` of server silence. Best when `AgentSession` already has its own VAD / turn detector.
- **`client_vad`** — open WS on local energy onset, close on silence end. Each utterance gets a fresh WS connection. Slightly higher per-utterance latency but harder utterance boundaries.

## Examples

See [`examples/README.md`](examples/README.md). Three demos cover file-based, mic-based, and full-agent scenarios.
