"""Small dependency-free Prometheus counters for the single-node runtime."""

from __future__ import annotations

import threading
from collections import defaultdict


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._sums[name] += value
            self._counts[name] += 1

    def render(self, *, queue_depth: int) -> str:
        with self._lock:
            lines = [f"reed_{name} {value:g}" for name, value in sorted(self._counters.items())]
            for name in sorted(self._counts):
                lines.append(f"reed_{name}_sum {self._sums[name]:g}")
                lines.append(f"reed_{name}_count {self._counts[name]}")
        lines.append(f"reed_ingestion_queue_depth {queue_depth}")
        return "\n".join(lines) + "\n"
