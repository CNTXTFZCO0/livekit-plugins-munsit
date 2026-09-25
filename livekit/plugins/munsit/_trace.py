# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Opt-in wire trace for ``/api/v1/listen``.

When ``MUNSIT_TRACE_FILE`` is set, every server event received (``rx``) and every client
control message sent (``tx``) is appended to that file as one JSON line. It exists because
turn-latency investigations need to know *what the socket actually said and when* (which
Results were interim/final/speech_final, when UtteranceEnd landed relative to the final)
and application logs only show what the plugin chose to emit.

Lines carry only metadata (event type, turn_id, flags, transcript length, confidence) —
never transcript text, the API key or the connection query string — so a trace file is
safe to attach to a support ticket.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, TextIO

TRACE_ENV_VAR = "MUNSIT_TRACE_FILE"

_COPIED_FIELDS = ("turn_id", "is_final", "speech_final", "confidence")


class Tracer:
    """Append-only JSONL writer for one stream. Best-effort: never raises."""

    def __init__(self, *, request_id: str) -> None:
        self.request_id = request_id
        self.session_id: str | None = None
        self._path = os.environ.get(TRACE_ENV_VAR) or None
        self._fh: TextIO | None = None

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def write(self, direction: str, message: dict[str, Any], *, note: str | None = None) -> None:
        if self._path is None:
            return
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "munsit_session_id": self.session_id,
            "request_id": self.request_id,
            "direction": direction,
            "type": message.get("type"),
        }
        for key in _COPIED_FIELDS:
            if key in message:
                record[key] = message[key]
        if "transcript" in message:
            record["text_len"] = len(message.get("transcript") or "")
        if note:
            record["note"] = note
        try:
            if self._fh is None:
                self._fh = open(self._path, "a", encoding="utf-8")
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            # Tracing must never take the audio path down; drop the line and disable.
            self.close()
            self._path = None

    def close(self) -> None:
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
