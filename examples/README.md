# Munsit STT examples

Three runnable demos:

| File | Needs mic? | Needs LiveKit room? | Use it for |
|---|---|---|---|
| `stt_file_demo.py` | No | No | CI-friendly smoke test against any 16-bit PCM WAV. |
| `stt_console_demo.py` | Yes | No (uses `agents` console mode) | Quick sanity check of live transcription quality. |
| `voice_agent_demo.py` | Yes | Yes | Full STT → LLM → TTS bilingual voice agent. |

## Setup

```bash
cp .env.example .env
# fill in MUNSIT_API_KEY (and LiveKit / OpenAI / Cartesia for the agent demo)
```

From the repo root:

```bash
uv sync --all-extras --dev
```

## Run

```bash
# 1. File demo — fastest sanity check
uv run python examples/stt_file_demo.py path/to/audio.wav

# 2. Mic demo — speak into your mic, see live transcripts
uv run python examples/stt_console_demo.py console

# 3. Full agent — connects to a LiveKit room
uv run python examples/voice_agent_demo.py dev
```

## What you should see

`stt_file_demo.py` and `stt_console_demo.py` print one line per STT event with timestamps relative to the start of audio:

```
[+0ms]    START_OF_SPEECH      req=abc123
[+340ms]  INTERIM_TRANSCRIPT   "مر"
[+520ms]  INTERIM_TRANSCRIPT   "مرحبا"
[+780ms]  INTERIM_TRANSCRIPT   "مرحبا كيف"
[+1340ms] FINAL_TRANSCRIPT     "مرحبا كيف حالك"
[+1340ms] END_OF_SPEECH

Utterance metrics: ttfi=340 ms, ttf=1340 ms, interim_updates=3, interim_p50_gap=220 ms, audio_sent=1.34 s
```

`voice_agent_demo.py` prints per-turn latency rollups:

```
Turn latency: EOU 240 ms + LLM 510 ms + TTS 380 ms = 1.13s | stt_audio=1.50s
```

## Troubleshooting

- **`ValueError: Munsit API key is required`** — set `MUNSIT_API_KEY` in `.env` or your shell.
- **`APIStatusError(401)`** — wrong API key, or the wrong `auth_method` for your account. Try `STT(auth_method="bearer")`.
- **No interim updates** — check `interim_results=True` (default) and that audio frames are reaching the stream.
