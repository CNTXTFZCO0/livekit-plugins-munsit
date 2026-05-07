---
title: LiveKit STT Plugin
description: Integrate Munsit speech-to-text into a LiveKit Agents voice agent with one pip install.
---

# LiveKit STT Plugin

Build production Arabic and Arabic/English voice agents with [LiveKit Agents](https://docs.livekit.io/agents/) using Munsit's speech-to-text API. The `livekit-plugins-munsit` plugin is a drop-in `STT` provider for LiveKit's `AgentSession`, with both an HTTP batch mode (default, returns word-level timestamps) and a low-latency WebSocket streaming mode.

## Integrating Munsit STT with LiveKit Agents

LiveKit Agents is an open-source framework for real-time, multimodal AI agents. By plugging Munsit STT into an `AgentSession`, your agent can transcribe Arabic (or Arabic/English code-switched) speech with first-class support and surface the results to any LLM, TTS, or downstream tool.

The plugin handles audio framing, WAV-header construction, end-of-utterance detection, retry/backoff, and result delivery to LiveKit's `STT.recognize()` and `STT.stream()` interfaces — so you can focus on conversation design, not transport plumbing.

## Prerequisites

* Python 3.10 or newer
* A Munsit API key — see [API Key](#api-key) below
* A LiveKit project (or use `console` mode for local mic testing without a server)
* Optional: an LLM provider key (OpenAI, Anthropic, etc.) and a TTS provider key (Cartesia, ElevenLabs, Faseeh) for a full voice agent

## API Key

Sign in to the [Munsit Dashboard](https://dashboard.munsit.com), open **API Keys**, and create a new key.

<Warning>
  Your API key is only shown once. Save it securely.
</Warning>

Set it in your environment as `MUNSIT_API_KEY`. The plugin reads it automatically — you don't need to pass it as a constructor argument.

## Installation

Install the plugin from PyPI:

```bash theme={null}
pip install livekit-plugins-munsit
```

`livekit-agents` is pulled in as a transitive dependency. If you also want the standard agent stack (Silero VAD, OpenAI LLM, Cartesia TTS), install those plugins too:

```bash theme={null}
pip install livekit-plugins-silero livekit-plugins-openai livekit-plugins-cartesia python-dotenv
```

## Step 1: Set Up Your Environment

Create a `.env` file in your project root:

```bash theme={null}
# Munsit Configuration
MUNSIT_API_KEY=your_MUNSIT_API_KEY_here

# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# LLM Configuration (for the agent's brain)
OPENAI_API_KEY=your_openai_api_key

# Optional: Cartesia for Arabic TTS
CARTESIA_API_KEY=your_cartesia_api_key
```

## Step 2: Create Your First Arabic Voice Agent

Create `agent.py`:

```python theme={null}
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.plugins import cartesia, munsit, openai, silero

load_dotenv()

server = AgentServer()


class ArabicAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly Arabic assistant. Reply in the language "
                "the user speaks. Keep responses brief."
            )
        )


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        # Batch mode is the default — accurate, returns word timestamps.
        stt=munsit.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(activation_threshold=0.6, min_speech_duration=0.3),
        turn_handling={"interruption": {"mode": "vad"}},
    )

    await session.start(room=ctx.room, agent=ArabicAssistant())
    await session.generate_reply(instructions="Greet the user warmly in Arabic.")


if __name__ == "__main__":
    agents.cli.run_app(server)
```

## Step 3: Run Your Agent

Use LiveKit's `console` mode to test locally with your microphone — no LiveKit server required:

```bash theme={null}
python agent.py console
```

For production deployment to a LiveKit room, use `dev` (with hot reload) or `start`:

```bash theme={null}
python agent.py dev      # development with hot reload
python agent.py start    # production
```

## Step 4: Test Your Agent

Speak in Arabic into your microphone. You should see logs like:

```
INFO  livekit.agents  received user transcript {"user_transcript": "مرحبا، كيف حالك؟", "language": "ar", "transcript_delay": 1.4}
```

The agent will:

1. Capture your audio via Silero VAD
2. POST it to Munsit's `/api/v1/audio/transcribe` endpoint
3. Receive a transcript with word-level timestamps
4. Pass the transcript to the LLM
5. Speak the LLM's reply through Cartesia TTS

## Configuration Options

The `munsit.STT(...)` constructor accepts:

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"batch"` | `"batch"` (HTTP, default) or `"streaming"` (WebSocket). |
| `model` | `"munsit"` | `"munsit"` (Arabic) or `"munsit-en-ar"` (Arabic/English code-switching). |
| `api_key` | `MUNSIT_API_KEY` env var | Your Munsit API key. |
| `auth_method` | `"header"` | `"header"` (`x-api-key`), `"bearer"`, or `"query"`. |
| `sample_rate` | `16000` | Sample rate of the audio sent to Munsit. |
| `num_channels` | `1` | Audio channel count (mono is recommended). |
| `language` | `None` | Optional language label attached to the transcript. Defaults to `"ar"`. |
| `interim_results` | `True` | Streaming mode only — emit `INTERIM_TRANSCRIPT` events as cumulatives arrive. |
| `endpointing` | `"server_diff"` | Streaming mode only — `"server_diff"` (idle timer) or `"client_vad"` (energy-based). |
| `finalize_after_silence_ms` | `700` | Streaming `server_diff` idle threshold in milliseconds. |
| `vad_silence_ms` | `1500` | Streaming `client_vad` end-of-speech silence duration. |
| `base_url` | Munsit production WSS | Override the streaming WebSocket URL. |
| `batch_base_url` | Munsit production HTTPS | Override the batch HTTP URL. |
| `http_session` | `None` | Pass a shared `aiohttp.ClientSession`. |

## Modes

| Mode | When to use | Endpoint | Latency | Word timestamps | Interim events |
|---|---|---|---|---|---|
| `"batch"` (default) | Production agents with VAD; transcribing recorded audio. | `POST /api/v1/audio/transcribe` | ~1–2 s for short utterances | ✅ yes — populated on `SpeechData.words` | ❌ no |
| `"streaming"` | Live captions / on-the-fly UI updates while the user is still speaking. | WebSocket `/api/v1/websocket/speech-to-text` | ~700 ms idle threshold by default | ❌ no | ✅ yes (cumulatives) |

```python theme={null}
munsit.STT()                       # batch (default, recommended)
munsit.STT(mode="streaming")       # opt-in low-latency interims
```

`STT.recognize(audio_buffer)` always uses the batch HTTP endpoint regardless of `mode` — useful for transcribing pre-recorded audio.

## Advanced Features

### Choosing a model

* **`munsit`** — Arabic-only, the most accurate option for monolingual Arabic conversations.
* **`munsit-en-ar`** — Arabic/English code-switching. Use when callers naturally mix English technical terms or names into Arabic speech.

```python theme={null}
munsit.STT(model="munsit-en-ar")
```

### Authentication methods

Munsit accepts your API key in three places. Choose what fits your environment:

```python theme={null}
# Default — x-api-key header
munsit.STT(auth_method="header")

# Authorization: Bearer <key>
munsit.STT(auth_method="bearer")

# Query parameter (?token=<key>) — useful for proxies that strip headers
munsit.STT(auth_method="query")
```

### Synchronous batch recognition

If you have a recorded audio buffer (e.g., a voicemail file) and don't need streaming, call `recognize` directly:

```python theme={null}
from livekit import rtc

stt = munsit.STT()
frames = [...]  # list of rtc.AudioFrame
result = await stt.recognize(rtc.combine_audio_frames(frames))
print(result.alternatives[0].text)
for word in result.alternatives[0].words:
    print(f"{word.start_time:.2f}s → {word.end_time:.2f}s  {word}")
```

### Streaming endpointing modes

When `mode="streaming"`, choose how end-of-utterance is detected:

* **`server_diff`** (default) — keep one long-lived WS open, emit final after `finalize_after_silence_ms` of server silence. Best when `AgentSession` already has its own VAD/turn detector.
* **`client_vad`** — open WS on local energy onset, close on silence end. Each utterance gets a fresh WS connection.

```python theme={null}
munsit.STT(
    mode="streaming",
    endpointing="client_vad",
    vad_silence_ms=1500,
)
```

## Monitoring and Metrics

The plugin emits standard LiveKit `STTMetrics` for each utterance:

* `audio_duration` — seconds of audio sent to Munsit
* `duration` — wall-clock time from end-of-speech to final transcript
* `streamed` — `True` for streaming mode, `False` for batch
* `request_id` — present in error events for cross-referencing

Subscribe via the conversation-item event for per-turn rollup:

```python theme={null}
@session.on("conversation_item_added")
def _on_item(event):
    msg = event.item
    if msg.role == "user":
        m = msg.metrics or {}
        print(f"STT delay: {m.get('transcription_delay', 0)*1000:.0f}ms")
        print(f"EOU delay: {m.get('end_of_turn_delay', 0)*1000:.0f}ms")
```

## Best Practices

* ✅ **Use batch mode in production agents.** It returns word timestamps and is more accurate than streaming on Arabic.
* ✅ **Tighten Silero VAD for real-world mic input.** `activation_threshold=0.6` and `min_speech_duration=0.3` filter out background noise and AEC residual without hurting real speech.
* ✅ **Use `console` mode early.** It runs locally with your mic and gives you full transcripts in stdout — fastest debugging loop.
* ✅ **Pin a Munsit model in production.** Pass `model="munsit"` or `model="munsit-en-ar"` explicitly so a future plugin default change doesn't surprise you.
* ❌ **Don't forget to set `MUNSIT_API_KEY`.** The plugin reads from the environment by default; missing it is the #1 setup error.
* ❌ **Don't build a feedback loop.** Make sure your TTS audio isn't being captured by the same mic at high volume — combine LiveKit's AEC with a tighter VAD threshold (above).

## Troubleshooting

### `MUNSIT_API_KEY environment variable is not set`

The plugin couldn't find your key. Verify:

```bash theme={null}
echo $MUNSIT_API_KEY
```

If empty, source your `.env` file (`source .env`) or pass the key explicitly: `munsit.STT(api_key="...")`.

### Silence / no transcripts in `console` mode

The plugin needs a VAD to know when you've finished speaking. Ensure your `AgentSession` includes one:

```python theme={null}
session = AgentSession(
    stt=munsit.STT(),
    vad=silero.VAD.load(),  # ← required
    ...
)
```

### Repeated identical transcripts during agent's TTS playback

Your microphone is picking up the agent's own audio output and Munsit is transcribing it. Tighten VAD thresholds (see [Best Practices](#best-practices)) and ensure LiveKit's acoustic echo canceller is enabled (it is by default in `console` mode).

### `Munsit batch HTTP 500: Internal server error`

Transient backend error — the plugin will retry automatically with exponential backoff. If you see it persistently on a specific audio clip, capture the WAV and share it with Munsit support.

### Mid-stream WS disconnects in streaming mode

For multi-turn agents, prefer batch mode — it's more robust. If you must use streaming, the plugin will reconnect with backoff; the only effect is a small gap in the timeline of that one utterance.

## Support

* **Plugin repository:** [github.com/CNTXTFZCO0/livekit-plugins-munsit](https://github.com/CNTXTFZCO0/livekit-plugins-munsit)
* **Issues / feature requests:** [GitHub issues](https://github.com/CNTXTFZCO0/livekit-plugins-munsit/issues)
* **Munsit support:** [support@munsit.com](mailto:support@munsit.com)
* **LiveKit Agents docs:** [docs.livekit.io/agents](https://docs.livekit.io/agents/)

## License

The `livekit-plugins-munsit` package is released under the [Apache 2.0 license](https://github.com/CNTXTFZCO0/livekit-plugins-munsit/blob/main/LICENSE).
