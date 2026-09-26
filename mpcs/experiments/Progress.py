"""Terminal progress and stage events for experiment execution."""

from __future__ import annotations

from contextlib import contextmanager
from time import monotonic
from typing import Callable, Iterator, Mapping, TextIO
import sys

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class StageReporter:
    def __init__(
        self,
        emit: Callable[[str, Mapping[str, object]], None],
        display: "TerminalProgress",
        *,
        phase: str | None = None,
    ) -> None:
        self._emit = emit
        self._display = display
        self._phase = phase

    @contextmanager
    def stage(self, stage_id: str, **details: object) -> Iterator[dict[str, object]]:
        started = monotonic()
        result: dict[str, object] = {}
        phase = {} if self._phase is None else {"phase": self._phase}
        label = stage_id if self._phase is None else f"{self._phase}/{stage_id}"
        self._emit("stage", {"stage": stage_id, "status": "start", **phase, **details})
        self._display.stage(label, "start", details)
        try:
            yield result
        except BaseException as error:
            payload = {
                "stage": stage_id,
                "status": "fail",
                "elapsed_s": monotonic() - started,
                "error": type(error).__name__,
                **phase,
                **details,
                **result,
            }
            self._emit("stage", payload)
            self._display.stage(label, "fail", payload)
            raise
        payload = {
            "stage": stage_id,
            "status": "done",
            "elapsed_s": monotonic() - started,
            **phase,
            **details,
            **result,
        }
        self._emit("stage", payload)
        self._display.stage(label, "done", payload)


class TerminalProgress:
    """One updating Rich view on a terminal, with concise log output elsewhere."""

    def __init__(self, *, enabled: bool = True, stream: TextIO | None = None) -> None:
        self._stream = sys.stderr if stream is None else stream
        self._enabled = enabled
        self._console = Console(file=self._stream)
        self._live: Live | None = None
        self._stages: list[dict[str, object]] = []
        self._rows: dict[str, tuple[int, int, int, float, float]] = {}

    def __enter__(self) -> "TerminalProgress":
        if self._enabled and self._stream.isatty():
            self._live = Live(
                self._panel(), console=self._console, refresh_per_second=4
            )
            self._live.start()
        return self

    def __exit__(self, *_error: object) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def stage(
        self,
        stage_id: str,
        status: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        record = {"status": status, **(details or {}), "stage": stage_id}
        if status == "start":
            self._stages.append(record)
        else:
            for index in range(len(self._stages) - 1, -1, -1):
                if (
                    self._stages[index]["stage"] == stage_id
                    and self._stages[index]["status"] == "start"
                ):
                    self._stages[index] = record
                    break
            else:
                self._stages.append(record)
        if self._live is not None:
            self._live.update(self._panel(), refresh=True)
        elif self._enabled and status != "start":
            elapsed = record.get("elapsed_s")
            duration = f" ({float(elapsed):.1f}s)" if elapsed is not None else ""
            summary = self._details(record)
            suffix = f"  {summary}" if summary else ""
            self._console.print(f"[{status.upper()}] {stage_id}{duration}{suffix}")

    def batch(self, method: str, record: Mapping[str, object]) -> None:
        self._rows[method] = (
            int(record["batch"]),
            int(record["assigned"]),
            int(record["total"]),
            float(record["profit"]),
            float(record["assignment_rate"]),
        )
        if self._live is not None:
            self._live.update(self._panel())

    @staticmethod
    def _details(record: Mapping[str, object]) -> str:
        ignored = {"stage", "status", "elapsed_s", "phase", "graph_audit", "grid_audit"}
        parts = []
        for key, value in record.items():
            if key in ignored:
                continue
            if isinstance(value, Mapping):
                formatted = ", ".join(f"{name}:{count}" for name, count in value.items())
            else:
                formatted = str(value)
            parts.append(f"{key}={formatted}")
        return "  ".join(parts)

    def _panel(self) -> Panel:
        stages = Table(title="Stages", expand=True)
        stages.add_column("Stage", style="cyan")
        stages.add_column("State", width=8)
        stages.add_column("Result")
        stages.add_column("Time", justify="right", width=8)
        current_phase = str(self._stages[-1]["stage"]).rsplit("/", 1)[0] if self._stages else ""
        milestones = {"scenario_prepare", "training", "evaluation", "algorithm_run"}
        visible = [
            record
            for record in self._stages
            if str(record["stage"]).rsplit("/", 1)[0] == current_phase
            or str(record["stage"]).rsplit("/", 1)[-1] in milestones
        ]
        for record in visible:
            elapsed = record.get("elapsed_s")
            stages.add_row(
                str(record["stage"]),
                str(record["status"]),
                self._details(record),
                f"{float(elapsed):.1f}s" if elapsed is not None else "",
            )
        frames = Table(title="Simulation", expand=True)
        for name in ("Method", "Frame", "Assigned", "AR", "Profit"):
            frames.add_column(name, justify="right" if name != "Method" else "left")
        for method, (batch, assigned, total, profit, rate) in self._rows.items():
            frames.add_row(
                method,
                str(batch),
                f"{assigned}/{total}",
                f"{rate:.3f}",
                f"{profit:.2f}",
            )
        return Panel(Group(stages, frames), title="MPCS · Simulator", border_style="blue")
