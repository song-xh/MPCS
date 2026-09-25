"""Structured run artifacts shared by all algorithms."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping


_CSV_FIELDS = (
    "batch",
    "simulated_time_s",
    "assigned",
    "expired",
    "waiting",
    "cross_pool",
    "total",
    "local_assignments",
    "cross_assignments",
    "assignment_rate",
    "profit",
    "mean_bpt_s",
    "wall_runtime_s",
)


class EventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def close(self) -> None:
        self._handle.close()

    def event(self, kind: str, payload: Mapping[str, object]) -> None:
        self._handle.write(
            json.dumps({"event": kind, **payload}, sort_keys=True) + "\n"
        )
        self._handle.flush()


class ArtifactWriter:
    """Write raw progress as it occurs, then plot the unchanged series."""

    def __init__(self, output_dir: Path, *, tensorboard: bool) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._events = EventLog(self.output_dir / "events.jsonl")
        self._metrics = (self.output_dir / "metrics.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._csv = csv.DictWriter(self._metrics, fieldnames=_CSV_FIELDS)
        self._csv.writeheader()
        self._rows: list[dict[str, float | int]] = []
        self._tensorboard = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._tensorboard = SummaryWriter(str(self.output_dir / "tensorboard"))

    def __enter__(self) -> "ArtifactWriter":
        return self

    def __exit__(self, *_error: object) -> None:
        self._events.close()
        self._metrics.close()
        if self._tensorboard is not None:
            self._tensorboard.close()

    def event(self, kind: str, payload: Mapping[str, object]) -> None:
        self._events.event(kind, payload)

    def batch(self, record: Mapping[str, float | int]) -> None:
        row = {key: record[key] for key in _CSV_FIELDS}
        self._csv.writerow(row)
        self._metrics.flush()
        self._rows.append(row)
        self.event("batch", row)
        progress_path = self.output_dir / "progress.json"
        pending_path = self.output_dir / "progress.pending.json"
        pending_path.write_text(
            json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
        )
        pending_path.replace(progress_path)
        if self._tensorboard is not None:
            step = int(row["batch"])
            for field in ("assignment_rate", "profit", "mean_bpt_s"):
                self._tensorboard.add_scalar(field, float(row[field]), step)

    def finish(self, summary: Mapping[str, object]) -> None:
        (self.output_dir / "summary.json").write_text(
            json.dumps(dict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._plot()

    def _plot(self) -> None:
        if not self._rows:
            return
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        batches = [int(row["batch"]) for row in self._rows]
        axes[0].plot(batches, [float(row["profit"]) for row in self._rows])
        axes[0].set_ylabel("Operating profit")
        axes[1].plot(batches, [float(row["assignment_rate"]) for row in self._rows])
        axes[1].set_ylabel("Assignment rate")
        axes[1].set_xlabel("Physical batch")
        figure.tight_layout()
        figure.savefig(self.output_dir / "metrics.png", dpi=150)
        plt.close(figure)
