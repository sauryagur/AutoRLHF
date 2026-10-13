"""The pipeline worker.

The heavy modules (:mod:`autorlhf.pipeline`, :mod:`autorlhf.training`) are
imported lazily through :class:`PipelineHooks`, so importing this module never
pulls in Textual, torch or transformers, and a test can substitute stub hooks to
assert the exact call order without a network or a GPU.

The worker itself is UI-free: it runs on a plain thread and speaks only through
the :class:`~autorlhf.core.events.EventBus`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autorlhf.config.run_config import RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.events import EventBus
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import RunPaths

#: The documented order the worker drives the interfaces in.
CALL_ORDER = (
    "prepare_dataset",
    "run_review",
    "run_baseline_eval",
    "build_training_plan",
    "TrainingJob.run",
    "run_final_eval",
)


@dataclass
class PipelineHooks:
    """Factories for the four frozen interfaces the worker composes."""

    make_context: Callable[..., Any]
    make_runner: Callable[[Any], Any]
    make_job: Callable[..., Any]
    find_resume_checkpoint: Callable[..., Any]


def default_hooks() -> PipelineHooks:
    """Resolve the real pipeline and training objects (imports on demand)."""

    from autorlhf.pipeline.context import PipelineContext
    from autorlhf.pipeline.orchestrate import PipelineRunner
    from autorlhf.training.jobs import TrainingJob, find_resume_checkpoint

    return PipelineHooks(
        make_context=PipelineContext,
        make_runner=PipelineRunner,
        make_job=TrainingJob,
        find_resume_checkpoint=find_resume_checkpoint,
    )


class PipelineWorker:
    """Drives one run from dataset preparation to the final comparison."""

    def __init__(
        self,
        *,
        run_id: str,
        env: Environment,
        store: RunStore,
        paths: RunPaths,
        config: RunConfig,
        bus: EventBus,
        interactive: bool = False,
        verbose: bool = False,
        hooks: PipelineHooks | None = None,
        resume: bool = False,
    ) -> None:
        self.run_id = run_id
        self.env = env
        self.store = store
        self.paths = paths
        self.config = config
        self.bus = bus
        self.interactive = interactive
        self.verbose = verbose
        self.resume = resume
        self.hooks = hooks

        self.ctx: Any = None
        self.runner: Any = None
        self.job: Any = None
        self.bundle: Any = None
        self.baseline: Any = None
        self.plan: Any = None
        self.result: Any = None
        self.final: Any = None
        self.comparison: Any = None
        self.error: BaseException | None = None
        #: Names of the interface methods actually invoked, in order.
        self.calls: list[str] = []

    # -- control ----------------------------------------------------------- #

    @property
    def cancel_event(self):
        return getattr(self.ctx, "cancel_event", None)

    @property
    def pause_event(self):
        return getattr(self.ctx, "pause_event", None)

    def request_stop(self) -> None:
        event = self.cancel_event
        if event is not None:
            event.set()
        if self.job is not None:
            try:
                self.job.request_stop()
            except Exception as exc:  # noqa: BLE001 - the stop path must never wedge
                self.bus.emit("warning", f"could not forward stop to the training job: {exc}")

    def request_pause(self) -> None:
        event = self.pause_event
        if event is not None:
            event.set()
        if self.job is not None:
            try:
                self.job.request_pause()
            except Exception as exc:  # noqa: BLE001
                self.bus.emit("warning", f"could not forward pause to the training job: {exc}")

    def request_resume(self) -> None:
        event = self.pause_event
        if event is not None:
            event.clear()
        if self.job is not None:
            try:
                self.job.request_resume()
            except Exception as exc:  # noqa: BLE001
                self.bus.emit("warning", f"could not forward resume to the training job: {exc}")

    @property
    def paused(self) -> bool:
        event = self.pause_event
        return bool(event is not None and event.is_set())

    @property
    def status(self) -> dict[str, Any]:
        job_status: dict[str, Any] = {}
        if self.job is not None:
            try:
                job_status = dict(self.job.status)
            except Exception as exc:  # noqa: BLE001 - status is advisory only
                job_status = {"error": str(exc)}
        return {
            "run_id": self.run_id,
            "calls": list(self.calls),
            "paused": self.paused,
            "job": job_status,
            "error": str(self.error) if self.error else None,
        }

    # -- the sequence ------------------------------------------------------ #

    def run(self) -> None:
        """Execute the documented sequence; never raises for user stops."""

        hooks = self.hooks or default_hooks()
        self.bus.emit(
            "phase",
            "starting pipeline",
            run_id=self.run_id,
            phase="setup",
            interactive=self.interactive,
            resume=self.resume,
        )
        ctx = hooks.make_context(
            env=self.env,
            store=self.store,
            paths=self.paths,
            config=self.config,
            bus=self.bus,
            interactive=self.interactive,
            verbose=self.verbose,
        )
        self.ctx = ctx
        try:
            runner = hooks.make_runner(ctx)
            self.runner = runner

            self.calls.append("prepare_dataset")
            self.bundle = runner.prepare_dataset(resume=self.resume)

            self.calls.append("run_review")
            self.bundle = runner.run_review(self.bundle)

            self.calls.append("run_baseline_eval")
            self.baseline = runner.run_baseline_eval()

            self.calls.append("build_training_plan")
            self.plan = runner.build_training_plan(self.bundle, adapter_path=None)

            self.calls.append("TrainingJob.run")
            job = hooks.make_job(
                run_id=self.run_id,
                paths=self.paths,
                config=self.config,
                plan=self.plan,
                bus=self.bus,
                cancel_event=ctx.cancel_event,
                pause_event=ctx.pause_event,
                env=self.env,
                store=self.store,
            )
            self.job = job
            resume_from = hooks.find_resume_checkpoint(
                self.paths, objective=getattr(self.plan, "objective", None)
            )
            self.result = job.run(resume_from=resume_from)

            self.calls.append("run_final_eval")
            adapter = (
                Path(self.result.adapter_path)
                if getattr(self.result, "adapter_path", None)
                else None
            )
            self.final, self.comparison = runner.run_final_eval(adapter_path=adapter)
        finally:
            close = getattr(self.ctx, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask the real error
                    self.bus.emit(
                        "warning",
                        f"pipeline context did not close cleanly: {exc}",
                        run_id=self.run_id,
                    )


__all__ = ["CALL_ORDER", "PipelineHooks", "PipelineWorker", "default_hooks"]
