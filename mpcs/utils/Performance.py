"""Opt-in, low-overhead counters for reproducible training profiling."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter_ns
from typing import Iterator, Mapping


@dataclass(slots=True)
class PerformanceProfiler:
    """Thread-safe counters and aggregate timings for one runtime."""

    enabled: bool = False
    _counts: dict[str, int] = field(default_factory=dict)
    _duration_ns: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def count(self, name: str, amount: int = 1) -> None:
        """Add a cheap integer counter when profiling is enabled."""

        if not self.enabled:
            return
        if not isinstance(name, str) or not name:
            raise ValueError("performance counter name must be non-empty")
        if type(amount) is not int or amount < 0:
            raise ValueError("performance counter amount must be non-negative")
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + amount

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Aggregate one duration without retaining per-call samples."""

        if not self.enabled:
            yield
            return
        start_ns = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - start_ns
            with self._lock:
                self._duration_ns[name] = (
                    self._duration_ns.get(name, 0) + elapsed_ns
                )

    def snapshot(self) -> Mapping[str, object]:
        """Return deterministic JSON-safe counters and aggregate timings."""

        with self._lock:
            counts = {
                name: self._counts[name]
                for name in sorted(self._counts)
            }
            timings_ms = {
                name: self._duration_ns[name] / 1_000_000.0
                for name in sorted(self._duration_ns)
            }
        return {
            "enabled": self.enabled,
            "counts": counts,
            "timings_ms": timings_ms,
        }
