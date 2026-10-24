"""Opt-in local, non-overlapping wall/thread CPU spans for serving diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import threading
import time
from typing import Iterator

identity: ContextVar[dict] = ContextVar("serving_identity", default={})


class Diagnostics:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, stage: str, **fields: object) -> Iterator[dict]:
        record = {}
        if self.path is None:
            yield record
            return
        wall, cpu = time.perf_counter(), time.thread_time()
        try:
            yield record
        finally:
            record.update(identity.get())
            record.update(fields)
            record.update(stage=stage, wall_seconds=time.perf_counter() - wall,
                          thread_cpu_seconds=time.thread_time() - cpu)
            with self._lock, self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")


OFF = Diagnostics()
