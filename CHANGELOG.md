# Changelog

## 0.4.1

Streaming (`WS /api/v1/listen`) changes from the Modon/Agntix turn-latency investigation
(2026-09-24): finals were reaching the agent 0.5–0.85 s after speech had ended, interims
carried caption-style hallucinations, and there was no way to see what the socket said.

- `END_OF_SPEECH` is emitted immediately after a `FINAL_TRANSCRIPT` whose `Results` event
  has `is_final=true` and `speech_final=true`; the server's later `UtteranceEnd` for the
  same `turn_id` no longer emits a second one. `UtteranceEnd` for a turn without such a
  final still emits `END_OF_SPEECH`; forced splits (`speech_final=false`) still emit
  `FINAL_TRANSCRIPT` only.
- New `filter_caption_artifacts` constructor flag (default on): interims that are only
  bracketed caption tags (`[Music]`, `[BLANK_AUDIO]`, `(music)`) are dropped and a leading
  `>> SPEAKER NAME:` tag is stripped from interims and finals. Finals are never dropped.
- New `MUNSIT_TRACE_FILE` env var: appends one JSON line per server event and per client
  control message (metadata only — never transcript text, the API key or the query string).
- README: "Diagnostics" section; note that `/api/v1/listen` has no client-side finalize
  (`CloseStream` ends the session), so LiveKit's `flush()` is ignored.
- Lint: repository now passes `ruff check` and `ruff format --check`.

## 0.4.0

Rewrite on Munsit's typed `/api/v1/listen` streaming protocol and consolidation of the
former `livekit-plugins-faseeh` TTS. See the README for the full feature set.
