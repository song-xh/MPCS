"""Terminal progress and stage events for experiment execution."""

from __future__ import annotations

from contextlib import contextmanager
from time import monotonic
from typing import Callable, Iterator, Mapping, TextIO
import sys

from rich.console import Console
from rich.live import Live
from rich.table import Table


class StageReporter:
    def __init__(
        self,
        emit: Callable[[str, Mapping[str, object]], None],
        display: "TerminalProgress",
    ) -> None:
        self._emit = emit
        self._display = display

    @contextmanager
    def stage(self, stage_id: str, **details: object) -> Iterator[None]:
        started = monotonic()
        self._emit("stage", {"stage": stage_id, "status": "start", **details})
        self._display.stage(stage_id, "start")
        try:
            yield
        except BaseException as error:
            payload = {
                "stage": stage_id,
                "status": "fail",
                "elapsed_s": monotonic() - started,
                "error": type(error).__name__,
                **details,
            }
            self._emit("stage", payload)
            self._display.stage(stage_id, "fail")
            raise
        payload = {
            "stage": stage_id,
            "status": "done",
            "elapsed_s": monotonic() - started,
            **details,
        }
        self._emit("stage", payload)
        self._display.stage(stage_id, "done")


class TerminalProgress:
    """One Rich view for a suite, with a readable non-TTY fallback."""

    def __init__(self, *, enabled: bool = True, stream: TextIO | None = None) -> None:
        self._stream = sys.stderr if stream is None else stream
        self._enabled = enabled
        self._console = Console(file=self._stream, force_terminal=False)
        self._live: Live | None = None
        self._stage = "ready"
        self._rows: dict[str, tuple[int, int, int, float, float]] = {}

    def __enter__(self) -> "TerminalProgress":
        if self._enabled and self._stream.isatty():
            self._live = Live(
                self._table(), console=self._console, refresh_per_second=4
            )
            self._live.start()
        return self

    def __exit__(self, *_error: object) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def stage(self, stage_id: str, status: str) -> None:
        self._stage = f"{stage_id}: {status}"
        if self._live is not None:
            self._live.update(self._table())
        elif self._enabled:
            self._console.print(f"[{status.upper()}] {stage_id}")

    def batch(self, method: str, record: Mapping[str, object]) -> None:
        self._rows[method] = (
            int(record["batch"]),
            int(record["assigned"]),
            int(record["total"]),
            float(record["profit"]),
            float(record["assignment_rate"]),
        )
        if self._live is not None:
            self._live.update(self._table())
        elif self._enabled:
            batch, assigned, total, profit, rate = self._rows[method]
            self._console.print(
                f"{method} batch={batch} assigned={assigned}/{total} "
                f"AR={rate:.3f} OP={profit:.2f}"
            )

    def _table(self) -> Table:
        table = Table(title=f"MPCS · {self._stage}")
        for name in ("Method", "Batch", "Assigned", "AR", "OP"):
            table.add_column(name, justify="right" if name != "Method" else "left")
        for method, (batch, assigned, total, profit, rate) in self._rows.items():
            table.add_row(
                method,
                str(batch),
                f"{assigned}/{total}",
                f"{rate:.3f}",
                f"{profit:.2f}",
            )
        return table
