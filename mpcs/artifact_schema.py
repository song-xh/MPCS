"""Central behavior and persisted-artifact version identifiers."""

from __future__ import annotations

from pathlib import Path
import subprocess


MECHANISM_LINEAGE = "v14-ppo-alternating-mc"
AUCTION_MECHANISM_VERSION = "versioned-auction-v2"
REWARD_SCHEMA_VERSION = 9
PROFIT_REPORT_SCHEMA_VERSION = 5
AUTHORIZED_FL_FEATURE_SCHEMA_VERSION = 2
DDQN_CHECKPOINT_SCHEMA_VERSION = 13
PPO_CHECKPOINT_SCHEMA_VERSION = 9
DATA_SPLIT_MANIFEST_SCHEMA_VERSION = 3
EXPERIMENT_MANIFEST_SCHEMA_VERSION = 11
TRAINER_MANIFEST_SCHEMA_VERSION = 10

_PROJECT_ROOT = Path(__file__).resolve().parent


def behavior_versions() -> dict[str, str | int]:
    """Return a fresh JSON-safe description of behavior-relevant versions."""

    return {
        "mechanism_lineage": MECHANISM_LINEAGE,
        "auction_mechanism": AUCTION_MECHANISM_VERSION,
        "reward_schema": REWARD_SCHEMA_VERSION,
        "profit_report_schema": PROFIT_REPORT_SCHEMA_VERSION,
        "authorized_fl_feature_schema": (
            AUTHORIZED_FL_FEATURE_SCHEMA_VERSION
        ),
        "ppo_checkpoint_schema": PPO_CHECKPOINT_SCHEMA_VERSION,
        "data_split_manifest_schema": DATA_SPLIT_MANIFEST_SCHEMA_VERSION,
        "experiment_manifest_schema": (
            EXPERIMENT_MANIFEST_SCHEMA_VERSION
        ),
        "trainer_manifest_schema": TRAINER_MANIFEST_SCHEMA_VERSION,
    }


def legacy_checkpoint_rejection_message(
    checkpoint_kind: str,
    observed_lineage: object,
) -> str:
    """Describe the staged-lineage fail-fast boundary for old checkpoints."""

    return (
        f"{checkpoint_kind} checkpoint rejected: "
        f"mechanism lineage={observed_lineage!r}; "
        "checkpoint schema/config predates the current mechanism; "
        "start a new run"
    )


def source_control_provenance() -> dict[str, str | bool | None]:
    """Return bounded Git provenance without paths, diffs, or identities."""

    revision = _run_git(["git", "rev-parse", "HEAD"])
    status = _run_git(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ]
    )
    commit = None
    if revision is not None and revision.returncode == 0:
        candidate = revision.stdout.strip().lower()
        if (
            len(candidate) == 40
            and all(character in "0123456789abcdef" for character in candidate)
        ):
            commit = candidate
    dirty = (
        bool(status.stdout.strip())
        if status is not None and status.returncode == 0
        else None
    )
    return {
        "commit": commit,
        "dirty": dirty,
    }


def _run_git(
    command: list[str],
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
