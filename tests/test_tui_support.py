"""Shared helpers for the TUI tests (no fixtures live here; see conftest)."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from textual.pilot import Pilot

from autorlhf.config.paths import AppPaths
from autorlhf.config.run_config import RunConfig
from autorlhf.core.enums import RunStatus
from autorlhf.core.events import EventBus
from autorlhf.core.ids import new_run_id, run_folder_name
from autorlhf.core.types import HardwareInfo
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import RunPaths, save_config
from autorlhf.tui.app import AutorlhfApp
from autorlhf.tui.worker import PipelineHooks

CPU_HARDWARE = HardwareInfo(
    device_type="cpu",
    device_name="test-cpu",
    device_count=0,
    total_ram_mib=16 * 1024,
    available_ram_mib=12 * 1024,
    cpu_count=4,
    estimate_notes=["hand-built fixture: nothing here was measured"],
)


def make_run(
    store: RunStore,
    paths: AppPaths,
    config: RunConfig,
    *,
    status: RunStatus | None = None,
    notes: str = "tui test",
) -> tuple[str, RunPaths]:
    """Create a stored run with pinned artifacts; returns ``(run_id, paths)``."""

    run_id = new_run_id()
    folder = run_folder_name(run_id, config.slug or config.character.name)
    run_paths = RunPaths(paths.run_dir(folder)).ensure()
    store.create_run(
        config,
        run_id=run_id,
        folder=folder,
        config_path=run_paths.config_path,
        notes=notes,
    )
    save_config(run_paths, config)
    if status is not None and status is not RunStatus.CREATED:
        # Fabricated fixture row: the status only needs to be *present* for the
        # screen under test, so the validated state machine is bypassed here.
        store.update_status(run_id, status, note="tui test fixture", validate=False)
    return run_id, run_paths


def make_app(
    paths: AppPaths,
    *,
    store: RunStore | None = None,
    bus: EventBus | None = None,
    config: RunConfig | None = None,
    **kwargs: Any,
) -> AutorlhfApp:
    """Build an app whose hardware probes are stubbed (no torch, no network)."""

    app = AutorlhfApp(paths=paths, store=store, bus=bus, config=config, **kwargs)
    app.detect_hardware = lambda: CPU_HARDWARE  # type: ignore[method-assign]
    app.draft_param_count = lambda: 1_000_000  # type: ignore[method-assign]
    return app


@contextlib.asynccontextmanager
async def running(app: AutorlhfApp, *, size: tuple[int, int] = (120, 40)) -> AsyncIterator[Pilot]:
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        yield pilot


def dataset_records(run_paths: RunPaths, count: int = 2) -> list[dict[str, Any]]:
    records = []
    for index in range(count):
        records.append(
            {
                "record_type": "candidate",
                "dataset_version": "1",
                "run_id": run_paths.root.name,
                "prompt_id": f"p{index:04d}",
                "candidate_id": f"p{index:04d}-c0",
                "split": "train" if index == 0 else "eval",
                "category": "ordinary",
                "prompt": f"Question {index}?",
                "prompt_context": [],
                "response": f"Answer {index}.",
                "source": "teacher",
                "model_id": "openai/gpt-4o-mini",
                "status": "accepted",
                "error": None,
                "truncated": False,
                "tokens": 8,
                "latency_s": 0.5,
                "cached": False,
                "seed": 0,
                "temperature": 0.8,
                "edits": [],
                "edited": False,
                "overall": 3.5,
                "criterion_scores": {"character_consistency": 3.5},
                "judgement": {"overall": 3.5},
                "reviewer": "",
                "created_at": "2026-01-01T00:00:00",
            }
        )
    return records


def static_text(widget: Any) -> str:
    """Plain text of a ``Static``/``Label``-style widget."""

    content = getattr(widget, "content", None)
    if content is None:
        content = getattr(widget, "renderable", "")
    return str(content)


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter.

    Import-isolation assertions must not purge ``sys.modules`` in the test
    process: re-importing Textual produces fresh class objects and breaks
    unrelated tests.
    """

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )


class StubContext:
    """Stands in for ``PipelineContext``: real events, no network."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.review_queue = kwargs.get("review_queue")
        self.closed = False

    def emit(self, kind: str, message: str = "", **kw: Any) -> None:
        return None

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def wait_if_paused(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class StubRunner:
    """Stands in for ``PipelineRunner``; records what it was asked to do."""

    def __init__(self, ctx: StubContext, record: dict[str, Any]) -> None:
        self.ctx = ctx
        self.record = record
        self.adapter_path: Any = None

    def prepare_dataset(self, *, resume: bool = True) -> str:
        self.record["prepare_dataset_resume"] = resume
        return "bundle"

    def run_review(self, bundle: Any) -> Any:
        return bundle

    def run_baseline_eval(self) -> str:
        return "baseline-report"

    def build_training_plan(self, bundle: Any, *, adapter_path: Any = None) -> SimpleNamespace:
        from autorlhf.core.enums import TrainingObjective

        return SimpleNamespace(objective=TrainingObjective.DPO, adapter_path=adapter_path)

    def run_final_eval(self, *, adapter_path: Any) -> tuple[str, str]:
        self.adapter_path = adapter_path
        self.record["adapter_path"] = adapter_path
        return "final-report", "comparison-report"


class StubJob:
    """Stands in for ``TrainingJob``."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.resume_from: Any = None
        self.stopped = False
        self.paused = False
        self.status = {"step": 0}
        #: Called at the top of :meth:`run`, so a test can steer the worker
        #: from inside the training step (as a real callback would).
        self.on_run: Callable[[], None] | None = None
        self.adapter_path: str | None = "/tmp/stub-adapter"
        self.stub_status = "completed"

    def run(self, *, resume_from: Any = None) -> SimpleNamespace:
        self.resume_from = resume_from
        if self.on_run is not None:
            self.on_run()
        return SimpleNamespace(adapter_path=self.adapter_path, status=self.stub_status)

    def request_stop(self) -> None:
        self.stopped = True

    def request_pause(self) -> None:
        self.paused = True

    def request_resume(self) -> None:
        self.paused = False


class StubWorker:
    """Stands in for ``PipelineWorker`` when a test only needs the app plumbing."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.run_id = kwargs.get("run_id", "")
        self.paths = kwargs.get("paths")
        self.bus = kwargs.get("bus")
        self.ctx = None
        self.job = None
        self.plan = None
        self.error: BaseException | None = None
        self.calls: list[str] = []
        self.paused = False
        self.stopped = False
        self.on_run: Callable[[], None] | None = None

    def run(self) -> None:
        self.calls.append("run")
        if self.on_run is not None:
            self.on_run()

    def request_stop(self) -> None:
        self.stopped = True

    def request_pause(self) -> None:
        self.paused = True

    def request_resume(self) -> None:
        self.paused = False

    @property
    def status(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "calls": list(self.calls)}


def stub_hooks(record: dict[str, Any] | None = None) -> PipelineHooks:
    """A ``PipelineHooks`` bundle whose four factories build the stubs above."""

    record = record if record is not None else {}
    record.setdefault("resume_from", None)

    def make_context(**kwargs: Any) -> StubContext:
        ctx = StubContext(**kwargs)
        record["ctx"] = ctx
        return ctx

    def make_runner(ctx: StubContext) -> StubRunner:
        return StubRunner(ctx, record)

    def make_job(**kwargs: Any) -> StubJob:
        job = StubJob(**kwargs)
        record["job"] = job
        return job

    def find_resume_checkpoint(paths: Any, *, objective: Any = None) -> None:
        record["checkpoint_objective"] = objective
        return record["resume_from"]

    return PipelineHooks(make_context, make_runner, make_job, find_resume_checkpoint)


def review_item(item_id: str = "p0001") -> dict[str, Any]:
    """A ``ReviewItem``-shaped dict, exactly what the bus carries."""

    return {
        "item_id": item_id,
        "prompt_id": item_id,
        "prompt": "Will the fog lift today?",
        "context": [{"role": "user", "content": "Morning."}],
        "category": "ordinary",
        "candidates": [
            {
                "candidate_id": f"{item_id}-c0",
                "prompt_id": item_id,
                "source": "teacher",
                "model_id": "openai/gpt-4o-mini",
                "text": "Not before the afternoon tide.",
                "status": "accepted",
            },
            {
                "candidate_id": f"{item_id}-c1",
                "prompt_id": item_id,
                "source": "teacher",
                "model_id": "openai/gpt-4o-mini",
                "text": "It will lift by noon, most likely.",
                "status": "accepted",
            },
        ],
        "judgements": {
            f"{item_id}-c0": {
                "overall": 3.1,
                "criterion_scores": {
                    "character_consistency": {
                        "criterion_name": "Character consistency",
                        "score": 3.0,
                        "scale_max": 4,
                        "probabilities": {"3": 0.6, "4": 0.4},
                        "confidence": 0.7,
                    }
                },
            },
            f"{item_id}-c1": {
                "overall": 2.4,
                "criterion_scores": {
                    "character_consistency": {
                        "criterion_name": "Character consistency",
                        "score": 2.0,
                        "scale_max": 4,
                        "probabilities": {"2": 0.9, "3": 0.1},
                        "confidence": 0.8,
                    }
                },
            },
        },
        "comparison": {
            "pair_id": f"{item_id}-a-vs-b",
            "winner": "a",
            "probabilities": {"a": 0.81, "b": 0.19},
            "confidence": 0.67,
            "margin": 0.7,
            "error": None,
        },
        "recommended_candidate_id": f"{item_id}-c0",
        "recommendation_reasons": ["overall margin 0.70 > tie_epsilon 0.35"],
        "status": "pending",
    }


__all__ = [
    "CPU_HARDWARE",
    "StubContext",
    "StubJob",
    "StubRunner",
    "StubWorker",
    "dataset_records",
    "make_app",
    "make_run",
    "review_item",
    "run_python",
    "running",
    "static_text",
    "stub_hooks",
    "write_jsonl",
]
