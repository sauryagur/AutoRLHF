"""Per-run artifact layout, atomic IO, manifests, and exports.

Layout (one directory per run) — see the project README for the tree.

Two invariants worth stating:

* every JSON write is atomic (temp file + ``os.replace``), so a killed process
  can never leave a half-written ``manifest.json`` or ``config.json``;
* a checkpoint is only advertised as resumable when
  :func:`is_checkpoint_complete` agrees, which is why trainers drop a
  ``COMPLETE`` sentinel after ``trainer.save_model`` returns.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from autorlhf.config.character import CharacterSpec
from autorlhf.config.run_config import RunConfig
from autorlhf.core.logging import get_logger
from autorlhf.storage.db import RunStore

logger = get_logger("autorlhf.storage.runs")

#: Marker file written by a trainer once a checkpoint directory is fully
#: written.  Its absence means "do not resume from here".
CHECKPOINT_SENTINEL = "COMPLETE"
#: Files that must exist for a checkpoint to be resumable.
REQUIRED_CHECKPOINT_FILES = ("trainer_state.json",)
#: Files that must exist for an adapter directory to be loadable.
REQUIRED_ADAPTER_FILES = ("adapter_config.json",)


class RunPaths:
    """Resolved artifact paths for a single run."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- files ------------------------------------------------------------- #

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def character_path(self) -> Path:
        return self.root / "character.json"

    @property
    def rubric_path(self) -> Path:
        return self.root / "rubric.json"

    @property
    def dataset_path(self) -> Path:
        return self.root / "dataset.jsonl"

    @property
    def preferences_path(self) -> Path:
        return self.root / "preferences.jsonl"

    @property
    def sft_path(self) -> Path:
        return self.root / "sft.jsonl"

    @property
    def eval_prompts_path(self) -> Path:
        return self.root / "eval_prompts.jsonl"

    @property
    def baseline_eval_path(self) -> Path:
        return self.root / "baseline_eval.json"

    @property
    def final_eval_path(self) -> Path:
        return self.root / "final_eval.json"

    @property
    def comparison_path(self) -> Path:
        return self.root / "comparison.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def reviews_path(self) -> Path:
        return self.root / "reviews.jsonl"

    @property
    def judgments_path(self) -> Path:
        return self.root / "judgments.jsonl"

    @property
    def split_manifest_path(self) -> Path:
        return self.root / "splits.json"

    # -- directories ------------------------------------------------------- #

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def adapter_dir(self) -> Path:
        return self.root / "adapter"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    # -- operations -------------------------------------------------------- #

    def ensure(self) -> RunPaths:
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        return self

    def log_path(self, name: str) -> Path:
        return self.logs_dir / name

    def describe(self) -> list[tuple[str, str]]:
        if not self.root.exists():
            return []
        rows: list[tuple[str, str]] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_dir() and not any(path.iterdir()):
                rows.append(("dir", str(path.relative_to(self.root))))
        for path in sorted(self.root.iterdir()):
            if path.is_file():
                rows.append((_human_size(path.stat().st_size), path.name))
        return rows


def _human_size(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024.0
    return f"{size:.1f} GiB"


# --------------------------------------------------------------------------- #
# atomic IO
# --------------------------------------------------------------------------- #


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write ``payload`` as JSON to ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, records: dict[str, Any] | Iterable[dict[str, Any]]) -> int:
    """Append one or more JSON records; each line is flushed and fsynced.

    Returns the number of records written.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, dict):
        items = [records]
    else:
        items = list(records)
    if not items:
        return 0
    with path.open("a", encoding="utf-8") as handle:
        for record in items:
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return len(items)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines but reporting how many."""

    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def count_jsonl(path: Path) -> int:
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def file_hash(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed.  Returns '' for a missing file."""

    import hashlib

    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:32]


def directory_size(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# config persistence
# --------------------------------------------------------------------------- #


def save_config(paths: RunPaths, config: RunConfig) -> None:
    paths.ensure()
    write_json(paths.config_path, config.model_dump(mode="json"))
    write_json(paths.character_path, config.character.model_dump(mode="json"))
    write_json(paths.rubric_path, config.rubric.model_dump(mode="json"))


def load_config(paths: RunPaths) -> RunConfig:
    """Load a run's config.

    ``character.json`` and ``rubric.json`` are re-read when present because they
    are the artifacts the user edits; the config file is the pinned snapshot.
    """

    from autorlhf.config.rubric import Rubric

    payload = read_json(paths.config_path, default=None)
    if payload is None:
        raise FileNotFoundError(f"no config.json in {paths.root}")
    config = RunConfig.model_validate(payload)
    character = read_json(paths.character_path, default=None)
    if character:
        try:
            config.character = CharacterSpec.model_validate(character)
        except Exception as exc:
            # An unreadable edit is worth knowing about: the pinned config is
            # used instead, and the user is told rather than silently ignored.
            logger.warning(
                "ignoring unreadable %s (%s); using the pinned character from config.json",
                paths.character_path.name,
                exc,
            )
    rubric = read_json(paths.rubric_path, default=None)
    if rubric:
        try:
            config.rubric = Rubric.model_validate(rubric)
        except Exception as exc:
            logger.warning(
                "ignoring unreadable %s (%s); using the pinned rubric from config.json",
                paths.rubric_path.name,
                exc,
            )
    return config


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


def is_checkpoint_complete(checkpoint_dir: Path) -> bool:
    """A checkpoint is complete only when the trainer's sentinel and the
    required state files are present and non-empty."""

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return False
    if not (checkpoint_dir / CHECKPOINT_SENTINEL).is_file():
        return False
    for name in REQUIRED_CHECKPOINT_FILES:
        candidate = checkpoint_dir / name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            return False
    has_weights = (
        any(
            (checkpoint_dir / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors")
        )
        or any(checkpoint_dir.glob("pytorch_model*.bin"))
        or any(checkpoint_dir.glob("model-*.safetensors"))
    )
    return bool(has_weights)


def mark_checkpoint_complete(checkpoint_dir: Path) -> None:
    write_text_atomic(
        Path(checkpoint_dir) / CHECKPOINT_SENTINEL,
        json.dumps({"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n",
    )


def checkpoint_step(checkpoint_dir: Path) -> int:
    """Parse the step from a directory name; ``-1`` when it is not ``checkpoint-N``."""

    name = Path(checkpoint_dir).name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except ValueError:
            return -1
    state = read_json(Path(checkpoint_dir) / "trainer_state.json", default={}) or {}
    try:
        return int(state.get("global_step", -1))
    except (TypeError, ValueError):
        return -1


def list_checkpoint_dirs(checkpoints_dir: Path) -> list[Path]:
    """Every ``checkpoint-*`` directory, newest step first (including incomplete)."""

    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir():
        return []
    dirs = [p for p in checkpoints_dir.glob("checkpoint-*") if p.is_dir()]
    return sorted(dirs, key=checkpoint_step, reverse=True)


def select_resume_checkpoint(checkpoints_dir: Path) -> Path | None:
    """The highest-step *complete* checkpoint, or ``None``.

    Partially written checkpoints are ignored rather than resumed from — that is
    the difference between a resumed run and a corrupt one.
    """

    for path in list_checkpoint_dirs(checkpoints_dir):
        if is_checkpoint_complete(path):
            return path
    return None


def _last_complete_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Highest complete checkpoint at ``checkpoint-*`` *or* ``<objective>/checkpoint-*``.

    Trainers nest their series under ``checkpoints/<objective>/`` so the SFT and
    DPO series never collide; the manifest reports the best complete one across
    both levels.
    """

    candidates = list(list_checkpoint_dirs(checkpoints_dir))
    for objective_root in checkpoints_dir.glob("*"):
        if objective_root.is_dir():
            candidates.extend(list_checkpoint_dirs(objective_root))
    candidates.sort(key=checkpoint_step, reverse=True)
    for path in candidates:
        if is_checkpoint_complete(path):
            return path
    return None


def is_adapter_complete(adapter_dir: Path) -> bool:
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        return False
    if not all((adapter_dir / name).is_file() for name in REQUIRED_ADAPTER_FILES):
        return False
    return any(
        (adapter_dir / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ) or any(adapter_dir.glob("pytorch_model*.bin"))


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def runtime_versions() -> dict[str, str]:
    """Best-effort importable library versions, for the manifest."""

    import platform
    import sys

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "textual",
        "openrouter",
        "wandb",
        "weave",
    ):
        try:
            imported = __import__(module)
            versions[module] = str(getattr(imported, "__version__", "unknown"))
        except Exception:
            versions[module] = "not installed"
    return versions


def build_manifest(
    *,
    run_id: str,
    config: RunConfig,
    paths: RunPaths,
    dataset_counts: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Self-describing record of exactly what this run was.

    Contains no credentials: only model ids, hashes, versions, and paths.
    """

    from autorlhf import __version__

    datasets: dict[str, Any] = {}
    for name, path in (
        ("dataset", paths.dataset_path),
        ("preferences", paths.preferences_path),
        ("sft", paths.sft_path),
        ("eval_prompts", paths.eval_prompts_path),
    ):
        if path.is_file():
            datasets[name] = {
                "path": str(path.relative_to(paths.root)),
                "records": count_jsonl(path),
                "hash": file_hash(path),
                "bytes": path.stat().st_size,
            }
    checkpoint = _last_complete_checkpoint(paths.checkpoints_dir)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "autorlhf_version": __version__,
        "schema_version": config.schema_version,
        "created_at": config.created_at,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "slug": config.slug,
        "feedback_mode": config.feedback_mode.value,
        "base_model": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "torch_dtype": config.model.torch_dtype,
            "attn_implementation": config.model.attn_implementation,
        },
        "teacher_model": config.generation.teacher_model,
        "judge_model": config.generation.judge_model,
        "training": {
            "objective": config.training.objective.value,
            "method": config.training.method.value,
            "preset_id": config.training.preset_id,
            "effective_hyperparameters": config.training.model_dump(mode="json"),
        },
        "rubric": config.rubric.model_dump(mode="json"),
        "fingerprints": {
            "config": config.fingerprint(),
            "dataset": config.dataset_fingerprint(),
            "character": config.character.fingerprint(),
            "rubric": config.rubric.fingerprint(),
            "hyperparameters": config.hyperparameter_fingerprint(),
        },
        "dataset_counts": dataset_counts or {},
        "datasets": datasets,
        "artifacts": {
            "config": "config.json",
            "character": "character.json",
            "rubric": "rubric.json",
            "dataset": "dataset.jsonl",
            "preferences": "preferences.jsonl",
            "eval_prompts": "eval_prompts.jsonl",
            "baseline_eval": "baseline_eval.json",
            "final_eval": "final_eval.json",
            "checkpoints": "checkpoints/",
            "adapter": "adapter/",
            "logs": "logs/",
        },
        "last_complete_checkpoint": str(checkpoint.relative_to(paths.root)) if checkpoint else None,
        "runtime_versions": runtime_versions(),
        "tracking": {
            "wandb_enabled": config.wandb_enabled,
            "wandb_project": config.wandb_project,
            "weave_enabled": config.weave_enabled,
            "weave_project": config.weave_project,
        },
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(paths: RunPaths, manifest: dict[str, Any]) -> None:
    write_json(paths.manifest_path, manifest)


def refresh_manifest(
    store: RunStore,
    run_id: str,
    paths: RunPaths,
    config: RunConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild and persist the manifest from current on-disk state."""

    counts = {d["split"]: d["count"] for d in store.datasets(run_id)}
    manifest = build_manifest(
        run_id=run_id,
        config=config,
        paths=paths,
        dataset_counts=counts,
        extra=extra,
    )
    write_manifest(paths, manifest)
    store.save_artifact(run_id, "manifest", paths.manifest_path, kind="json")
    return manifest


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def export_run(
    *,
    run_id: str,
    paths: RunPaths,
    config: RunConfig,
    destination: Path,
    include_checkpoints: bool = False,
    extra_files: Iterable[Path] = (),
) -> Path:
    """Copy a self-contained, credential-free experiment bundle.

    Includes config/character/rubric, every dataset, both evaluations, the
    comparison report, the manifest, logs, and the adapter.  ``config.env`` and
    anything looking like a secret file is never copied.
    """

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(run_id=run_id, config=config, paths=paths)
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "config.json", config.model_dump(mode="json"))
    write_json(destination / "character.json", config.character.model_dump(mode="json"))
    write_json(destination / "rubric.json", config.rubric.model_dump(mode="json"))

    for name in (
        "dataset.jsonl",
        "preferences.jsonl",
        "sft.jsonl",
        "eval_prompts.jsonl",
        "baseline_eval.json",
        "final_eval.json",
        "comparison.json",
        "judgments.jsonl",
        "splits.json",
        "reviews.jsonl",
    ):
        source = paths.root / name
        if source.is_file():
            _copy_file(source, destination / name)

    if is_adapter_complete(paths.adapter_dir):
        shutil.copytree(paths.adapter_dir, destination / "adapter", dirs_exist_ok=True)

    if paths.logs_dir.is_dir():
        shutil.copytree(paths.logs_dir, destination / "logs", dirs_exist_ok=True)

    if include_checkpoints and paths.checkpoints_dir.is_dir():
        shutil.copytree(paths.checkpoints_dir, destination / "checkpoints", dirs_exist_ok=True)

    for path in extra_files:
        path = Path(path)
        if path.is_file() and not _is_secret_like(path):
            _copy_file(path, destination / path.name)

    return destination


def _is_secret_like(path: Path) -> bool:
    name = path.name
    if name in {"config.env", ".env", ".env.local", "wandb", "cache"}:
        return True
    if name.endswith((".env", ".key", ".pem")):
        return True
    if "config.env" in name or name.startswith(".env"):
        return True
    return False


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


__all__ = [
    "CHECKPOINT_SENTINEL",
    "REQUIRED_ADAPTER_FILES",
    "REQUIRED_CHECKPOINT_FILES",
    "RunPaths",
    "append_jsonl",
    "build_manifest",
    "checkpoint_step",
    "count_jsonl",
    "directory_size",
    "export_run",
    "file_hash",
    "is_adapter_complete",
    "is_checkpoint_complete",
    "iter_jsonl",
    "list_checkpoint_dirs",
    "load_config",
    "mark_checkpoint_complete",
    "read_json",
    "read_jsonl",
    "refresh_manifest",
    "runtime_versions",
    "save_config",
    "select_resume_checkpoint",
    "write_json",
    "write_manifest",
    "write_text_atomic",
]
