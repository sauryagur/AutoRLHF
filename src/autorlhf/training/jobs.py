"""Training orchestration: turn a :class:`TrainingPlan` into a trained adapter.

Importing this module is deliberately cheap and torch-free: ``torch``, ``trl``
and ``peft`` are imported inside :meth:`TrainingJob.run`, so the TUI can start
without paying for the deep-learning stack (and ``detect_hardware`` imports torch
inside its own function).

``run()`` never raises for a user-requested stop: it returns a
:class:`TrainingResult` with ``status="interrupted"``.  Real failures are
reported the same way (``status="failed"`` plus ``error``), with one exception:
:class:`InsufficientDataError` is raised, because "there is nothing to train on"
is a programming/configuration error the caller must handle, not a run outcome.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from autorlhf.config.run_config import RunConfig
from autorlhf.core.enums import SplitName, TrainingMethod, TrainingObjective
from autorlhf.core.errors import AutorlhfError, ConfigError, InsufficientDataError
from autorlhf.core.events import EventBus
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import RunPaths, is_adapter_complete
from autorlhf.training.plan import TrainingPlan

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module torch-free
    import threading

    from autorlhf.config.settings import Environment

#: Sample previews are truncated so a single bus event stays small.
_PREVIEW_CHARS = 400


class TrainingResult(BaseModel):
    """Outcome of one training job, regardless of how it ended."""

    run_id: str
    objective: TrainingObjective
    method: TrainingMethod
    #: ``completed`` | ``interrupted`` | ``failed``
    status: str
    steps_completed: int = 0
    adapter_path: str | None = None
    checkpoints: list[str] = Field(default_factory=list)
    resumed_from: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    wandb_url: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"completed", "interrupted"}


def find_resume_checkpoint(paths: RunPaths, *, objective: TrainingObjective | str) -> Path | None:
    """Highest-step complete checkpoint for ``objective``, or ``None``.

    Thin re-export of :func:`autorlhf.training.trainers.find_resume_checkpoint`,
    imported lazily so that this module stays free of ``torch``/``trl``.
    """

    from autorlhf.training.trainers import find_resume_checkpoint as _implementation

    return _implementation(paths, objective=objective)


def _truncate(text: str, limit: int = _PREVIEW_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


class TrainingJob:
    """Runs the objective(s) in ``plan`` and reports progress on the bus."""

    def __init__(
        self,
        *,
        run_id: str,
        paths: RunPaths,
        config: RunConfig,
        plan: TrainingPlan,
        bus: EventBus,
        cancel_event: threading.Event,
        pause_event: threading.Event,
        env: Environment,
        store: RunStore | None = None,
    ) -> None:
        self.run_id = run_id
        self.paths = paths
        self.config = config
        self.plan = plan
        self.bus = bus
        self.cancel_event = cancel_event
        self.pause_event = pause_event
        self.env = env
        self.store = store
        self._state = "idle"
        self._result: TrainingResult | None = None
        self._warnings: list[str] = []
        self._tracker_instance: Any | None = None
        self._tracking_started = False

    # -- control ----------------------------------------------------------- #

    def request_pause(self) -> None:
        """Ask the trainer to pause at the next step boundary."""

        self.pause_event.set()
        self._state = "pausing" if self._state == "running" else "paused"
        self._emit("status", "pause requested")

    def request_resume(self) -> None:
        """Clear a pause request."""

        self.pause_event.clear()
        if self._state in {"paused", "pausing"}:
            self._state = "running"
        self._emit("status", "resume requested")

    def request_stop(self) -> None:
        """Ask the trainer to stop gracefully and keep what it has."""

        self.cancel_event.set()
        self.pause_event.clear()
        if self._state in {"running", "paused", "pausing"}:
            self._state = "stopping"
        self._emit("status", "stop requested", level="warning")

    @property
    def status(self) -> dict[str, Any]:
        """Snapshot for the TUI (no side effects)."""

        snapshot: dict[str, Any] = {
            "run_id": self.run_id,
            "state": self._state,
            "objective": self.plan.objective.value,
            "method": self.plan.method.value,
            "pause_requested": self.pause_event.is_set(),
            "stop_requested": self.cancel_event.is_set(),
        }
        if self._result is not None:
            snapshot.update(
                {
                    "result_status": self._result.status,
                    "steps_completed": self._result.steps_completed,
                    "adapter_path": self._result.adapter_path,
                    "error": self._result.error,
                }
            )
        return snapshot

    # -- events ------------------------------------------------------------ #

    def _emit(self, kind: str, message: str, **kwargs: Any) -> None:
        self.bus.emit(
            kind, message, run_id=self.run_id, phase=kwargs.pop("phase", "train"), **kwargs
        )

    def _warn(self, message: str) -> None:
        self._warnings.append(message)
        self._emit("warning", message, level="warning")

    # -- data -------------------------------------------------------------- #

    @staticmethod
    def _split(records: list[dict], split: SplitName) -> list[dict]:
        wanted = split.value
        return [record for record in records if str(record.get("split", "train")) == wanted]

    def _prepare_data(self) -> dict[str, Any]:
        from autorlhf.training.trainers import (
            MIN_PREFERENCE_PAIRS,
            MIN_SFT_EXAMPLES,
            validate_dpo_records,
            validate_sft_records,
        )

        sft = list(self.plan.sft_examples)
        pairs = list(self.plan.preference_pairs)

        sft_train = self._split(sft, SplitName.TRAIN)
        sft_validation = self._split(sft, SplitName.VALIDATION)
        pairs_train = self._split(pairs, SplitName.TRAIN)
        pairs_validation = self._split(pairs, SplitName.VALIDATION)

        for label, records, validator in (
            ("sft", sft_train, validate_sft_records),
            ("preference", pairs_train, validate_dpo_records),
        ):
            problems = validator(records)
            if problems:
                self._warn(
                    f"{len(problems)} unusable {label} record(s): " + "; ".join(problems[:3])
                )

        has_sft = len(sft_train) >= MIN_SFT_EXAMPLES
        has_pairs = len(pairs_train) >= MIN_PREFERENCE_PAIRS
        objective = TrainingObjective(self.plan.objective)
        if not has_sft and not has_pairs:
            raise InsufficientDataError(
                "no training data at all: 0 supervised examples and 0 preference pairs",
                hint=(
                    "Run data preparation and review first; check that the dataset has "
                    "accepted candidates and that review thresholds are not rejecting every pair."
                ),
            )
        if objective is TrainingObjective.SFT and not has_sft:
            raise InsufficientDataError(
                f"SFT needs at least {MIN_SFT_EXAMPLES} supervised example(s); found "
                f"{len(sft_train)}",
                hint="Approve at least one candidate response, or switch the objective to DPO.",
            )
        if objective is TrainingObjective.DPO and not has_pairs:
            raise InsufficientDataError(
                f"DPO needs at least {MIN_PREFERENCE_PAIRS} preference pair(s); found "
                f"{len(pairs_train)}",
                hint=(
                    "Increase candidates_per_prompt or relax the judge thresholds so at least "
                    "one comparison is decisive."
                ),
            )
        return {
            "sft_train": sft_train,
            "sft_validation": sft_validation,
            "pairs_train": pairs_train,
            "pairs_validation": pairs_validation,
            "has_sft": has_sft,
            "has_pairs": has_pairs,
        }

    def _emit_samples(self, records: list[dict], *, objective: TrainingObjective) -> None:
        for record in records[:3]:
            if objective is TrainingObjective.SFT:
                preview = [
                    {"role": m.get("role"), "content": _truncate(m.get("content", ""))}
                    for m in record.get("messages", [])
                ]
                payload = {"record_type": "sft", "messages": preview}
            else:
                payload = {
                    "record_type": "preference",
                    "prompt": _truncate(record.get("prompt", "")),
                    "chosen": _truncate(record.get("chosen_text", "")),
                    "rejected": _truncate(record.get("rejected_text", "")),
                }
            payload["example_id"] = record.get("example_id") or record.get("pair_id") or ""
            payload["objective"] = objective.value
            self._emit(
                "sample",
                f"{objective.value} sample {payload['example_id']}",
                verbose_only=True,
                **payload,
            )

    # -- model loading ----------------------------------------------------- #

    def _store_or_none(self) -> RunStore | None:
        if self.store is not None:
            return self.store
        try:
            store = RunStore(self.env.paths.db_path)
            store.init_schema()
            return store
        except Exception as exc:  # pragma: no cover - storage must not block training
            self._warn(f"training metrics will not be persisted: {exc}")
            return None

    def _load_base(self, hp: Any) -> Any:
        from autorlhf.training.models import load_model

        loaded = load_model(
            self.config.model,
            hp,
            hf_token=self.env.hf_token(),
            for_training=True,
        )
        for warning in loaded.warnings:
            self._warn(warning)
        self._emit(
            "status",
            f"loaded {loaded.model_id} @ {loaded.resolved_revision} "
            f"({loaded.param_count / 1e6:.2f}M parameters)",
            device="cpu",
            param_count=loaded.param_count,
        )
        return loaded

    def _resolve_target_modules(self, hp: Any, model: Any) -> bool:
        """Fill in LoRA targets from the real model; return ``fan_in_fan_out``."""

        from autorlhf.training.lora import (
            discover_target_modules,
            needs_fan_in_fan_out,
            validate_target_modules,
        )

        if hp.method is TrainingMethod.FFT:
            return False
        fan_in_fan_out = needs_fan_in_fan_out(model)
        if not hp.lora_target_modules:
            targets, notes = discover_target_modules(model, preferred=None)
            hp.lora_target_modules = targets
            for note in notes:
                self._emit("log", f"target modules: {note}", verbose_only=True)
            if not targets:
                raise ConfigError(
                    "no LoRA target modules could be discovered for this model",
                    hint="Set lora_target_modules manually to module names that exist.",
                )
        validate_target_modules(model, hp.lora_target_modules)
        return fan_in_fan_out

    # -- stages ------------------------------------------------------------ #

    def _run_sft_stage(
        self,
        hp: Any,
        data: dict[str, Any],
        store: RunStore | None,
        *,
        resume_from: Path | None = None,
    ) -> Any:
        from autorlhf.training.trainers import find_resume_checkpoint, run_sft

        resume = resume_from or find_resume_checkpoint(self.paths, objective=TrainingObjective.SFT)

        loaded = self._load_base(hp)
        fan_in_fan_out = self._resolve_target_modules(hp, loaded.model)
        hp.assert_valid()
        self._emit_samples(data["sft_train"], objective=TrainingObjective.SFT)
        self._emit(
            "phase",
            "SFT warm-up"
            if TrainingObjective(self.plan.objective) is TrainingObjective.DPO
            else "supervised fine-tuning",
            phase="train_sft",
        )
        return run_sft(
            run_id=self.run_id,
            paths=self.paths,
            hp=hp,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            train_records=data["sft_train"],
            eval_records=data["sft_validation"],
            bus=self.bus,
            cancel_event=self.cancel_event,
            pause_event=self.pause_event,
            store=store,
            resume_from=resume,
            report_to=self._report_to(),
            run_name=self._run_name(TrainingObjective.SFT),
            adapter_dir=self._sft_adapter_dir(),
            fan_in_fan_out=fan_in_fan_out,
            extra_warnings=self._warnings,
        )

    def _tracker(self) -> Any | None:
        """Lazily build the optional tracking wrapper; ``None`` when disabled.

        W&B/Weave are never imported unless the run configuration enables them,
        and a missing or broken integration degrades to a warning instead of
        failing the run.
        """

        if not (self.config.wandb_enabled or self.config.weave_enabled):
            return None
        try:
            from autorlhf.integrations.tracking import TrainingTracker
        except ImportError:  # pragma: no cover - integrations ship with the project
            self._warn(
                "tracking is enabled but autorlhf.integrations.tracking is unavailable; "
                "training continues without it"
            )
            return None
        return TrainingTracker(config=self.config, env=self.env, run_id=self.run_id, bus=self.bus)


    def _finish_tracking(self, result: Any | None) -> None:
        """Close the tracking run, whatever the outcome (``result`` may be ``None``)."""

        tracker = self._tracker_instance
        if tracker is None:
            return
        summary = None
        if result is not None:
            summary = {
                "status": getattr(result, "status", "unknown"),
                "steps_completed": getattr(result, "steps_completed", 0),
            }
        try:
            tracker.finish(summary=summary)
        except Exception as exc:  # pragma: no cover - the tracker is meant to never raise
            self._warn(f"closing experiment tracking failed: {exc}")

    def _start_tracking(self) -> None:
        """Initialise tracking once, collecting every warning it reports."""

        if self._tracking_started:
            return
        self._tracker_instance = self._tracker()
        self._tracking_started = True
        tracker = self._tracker_instance
        if tracker is None:
            return
        try:
            state = tracker.start()
        except Exception as exc:  # pragma: no cover - the tracker is meant to never raise
            self._warn(f"initialising experiment tracking failed: {exc}")
            return
        for warning in getattr(state, "warnings", []):
            if warning not in self._warnings:
                self._warnings.append(warning)

    def _report_to(self) -> list[str]:
        """Backends TRL should log to; only those that really initialised."""

        tracker = self._tracker_instance
        if tracker is None:
            return []
        return list(getattr(tracker, "report_to", []) or [])

    def _run_name(self, objective: TrainingObjective) -> str:
        base = self.config.training.run_name or self.config.slug or self.run_id
        return f"{base}-{objective.value}"

    def _sft_adapter_dir(self) -> Path:
        return self.paths.root / "adapter-sft"

    def _reusable_warmup_adapter(self) -> Path | None:
        """An adapter from a previous warm-up that can be reused, or ``None``.

        Prefers the finished ``adapter-sft``; otherwise the highest complete SFT
        checkpoint, whose directory holds adapter weights as well.  Reusing either
        means the warm-up is not re-run on a resumed job.
        """

        from autorlhf.training.trainers import find_resume_checkpoint

        candidate = self._sft_adapter_dir()
        if is_adapter_complete(candidate):
            return candidate
        return find_resume_checkpoint(self.paths, objective=TrainingObjective.SFT)

    def _run_dpo_stage(
        self, hp: Any, data: dict[str, Any], store: RunStore | None, *, resume_from: Path | None
    ) -> Any:
        from autorlhf.training.trainers import run_dpo

        warmup = self._reusable_warmup_adapter()
        # The warm-up stage injects LoRA layers into the base model, so DPO always
        # starts from a freshly loaded base (plus, when available, the warm-up
        # adapter that was just saved).
        loaded = self._load_base(hp)
        model = loaded.model
        apply_new_lora = True
        if warmup is not None:
            from peft import PeftModel

            # `is_trainable=True` is required: PeftModel.from_pretrained defaults to
            # an inference-only adapter (all parameters frozen), which would make
            # the DPO loss carry no gradient.
            model = PeftModel.from_pretrained(model, str(warmup), is_trainable=True)
            apply_new_lora = False
            self._emit(
                "status",
                f"continuing from the SFT warm-up adapter at {warmup.name}",
                phase="train_dpo",
                adapter_path=str(warmup),
            )
        fan_in_fan_out = self._resolve_target_modules(hp, model)
        hp.assert_valid()
        self._emit_samples(data["pairs_train"], objective=TrainingObjective.DPO)
        self._emit("phase", "preference optimisation (DPO)", phase="train_dpo")
        return run_dpo(
            run_id=self.run_id,
            paths=self.paths,
            hp=hp,
            model=model,
            tokenizer=loaded.tokenizer,
            preference_records=data["pairs_train"],
            system_prompt=self.config.character.system_prompt,
            eval_records=data["pairs_validation"],
            bus=self.bus,
            cancel_event=self.cancel_event,
            pause_event=self.pause_event,
            store=store,
            resume_from=resume_from,
            report_to=self._report_to(),
            run_name=self._run_name(TrainingObjective.DPO),
            adapter_dir=self.paths.adapter_dir,
            peft_config=None,
            apply_new_lora=apply_new_lora,
            fan_in_fan_out=fan_in_fan_out,
            extra_warnings=self._warnings,
        )

    # -- wandb ------------------------------------------------------------- #

    def _wandb_url(self) -> str | None:
        """The live W&B run url, or ``None`` when tracking is off or failed."""

        tracker = self._tracker_instance
        if tracker is None:
            return None
        return getattr(getattr(tracker, "state", None), "wandb_url", None)

    # -- the job ----------------------------------------------------------- #

    def run(self, *, resume_from: Path | None = None) -> TrainingResult:
        """Run the plan's objective(s).  See the module docstring for outcomes."""

        hp = self.plan.hyperparameters.model_copy(deep=True)
        objective = TrainingObjective(self.plan.objective)
        self._state = "running"
        self._emit("status", f"training started ({hp.method.value}, {objective.value})")
        self._emit("progress", "preparing training data", fraction=0.0, phase="train")

        try:
            from autorlhf.core.trl_compat import install_cpu_logprob_fallback
            from autorlhf.training.hardware import detect_hardware
        except ImportError as exc:  # pragma: no cover - trl/torch are hard dependencies
            return self._failed(hp, objective, f"training dependencies are missing: {exc}")

        if install_cpu_logprob_fallback():
            self._emit(
                "log",
                "TRL's Triton log-probability kernel is unusable here; using the "
                "pure-PyTorch fallback",
                verbose_only=True,
            )

        data = self._prepare_data()
        store = self._store_or_none()
        result: Any = None

        hardware = detect_hardware()
        for warning in hardware.warnings:
            if "triton" not in warning:
                self._emit("log", warning, verbose_only=True)
        supported, reason = hardware.supports(hp.method)
        if not supported:
            return self._failed(
                hp,
                objective,
                f"{hp.method.value} cannot run on this machine: {reason}",
                warnings=[f"detected device: {hardware.device_type or 'unknown'}"],
            )
        self._start_tracking()

        try:
            if objective is TrainingObjective.SFT:
                result = self._run_sft_stage(hp, data, store, resume_from=resume_from)
                result = result.model_copy(update={"warnings": self._merge(result)})
            else:
                result = self._run_dpo_with_warmup(hp, data, store, resume_from=resume_from)
        except InsufficientDataError:
            raise
        except AutorlhfError as exc:
            return self._failed(hp, objective, self._describe(exc), warnings=list(self._warnings))
        except Exception as exc:  # noqa: BLE001 - reported, never a bare traceback in the UI
            return self._failed(
                hp, objective, f"{type(exc).__name__}: {exc}", warnings=list(self._warnings)
            )
        finally:
            # Close the W&B/Weave run whatever happened, or buffered traces never flush.
            self._finish_tracking(result)

        self._result = result
        self._state = "finished"
        url = self._wandb_url()
        if url:
            result = result.model_copy(update={"wandb_url": url})
            self._result = result
        self._emit(
            "status",
            f"training {result.status}: {result.steps_completed} step(s), adapter "
            f"{result.adapter_path}",
            level="info" if result.ok else "error",
            status=result.status,
            steps=result.steps_completed,
            adapter_path=result.adapter_path,
        )
        self._emit("progress", "training finished", fraction=1.0, phase="train")
        return result

    def _run_dpo_with_warmup(
        self, hp: Any, data: dict[str, Any], store: RunStore | None, *, resume_from: Path | None
    ) -> TrainingResult:
        from autorlhf.training.trainers import find_resume_checkpoint

        warmup_result: Any | None = None
        if hp.sft_warmup and data["has_sft"] and self._reusable_warmup_adapter() is None:
            warmup_result = self._run_sft_stage(hp, data, store)
            warmup_result = warmup_result.model_copy(
                update={"warnings": self._merge(warmup_result)}
            )
            if warmup_result.status == "failed":
                return warmup_result
            if warmup_result.status == "interrupted":
                self._emit(
                    "status",
                    "warm-up stopped before DPO; returning the warm-up result",
                    level="warning",
                )
                return TrainingResult(
                    run_id=self.run_id,
                    objective=TrainingObjective.SFT,
                    method=hp.method,
                    status="interrupted",
                    steps_completed=warmup_result.steps_completed,
                    adapter_path=warmup_result.adapter_path,
                    checkpoints=warmup_result.checkpoints,
                    resumed_from=warmup_result.resumed_from,
                    metrics={f"sft_{key}": value for key, value in warmup_result.metrics.items()},
                    error=None,
                    warnings=warmup_result.warnings,
                )
        elif hp.sft_warmup and data["has_sft"]:
            self._emit(
                "status",
                "reusing the existing SFT warm-up adapter; skipping supervised warm-up",
                phase="train_sft",
            )

        dpo_resume = resume_from
        if dpo_resume is None:
            dpo_resume = find_resume_checkpoint(self.paths, objective=TrainingObjective.DPO)
            if dpo_resume is not None:
                self._emit("status", f"resuming DPO from {dpo_resume.name}", phase="train_dpo")

        result = self._run_dpo_stage(hp, data, store, resume_from=dpo_resume)
        warnings = self._merge(result)
        metrics = dict(result.metrics)
        checkpoints = list(result.checkpoints)
        if warmup_result is not None:
            metrics = {
                **{f"sft_{key}": value for key, value in warmup_result.metrics.items()},
                **metrics,
            }
            checkpoints = [*warmup_result.checkpoints, *checkpoints]
        return result.model_copy(
            update={"warnings": warnings, "metrics": metrics, "checkpoints": checkpoints}
        )

    # -- helpers ----------------------------------------------------------- #

    def _merge(self, result: Any) -> list[str]:
        merged = list(self._warnings)
        for warning in getattr(result, "warnings", []):
            if warning not in merged:
                merged.append(warning)
        return merged

    def _describe(self, exc: BaseException) -> str:
        hint = getattr(exc, "hint", "")
        return f"{exc} ({hint})" if hint else str(exc)

    def _failed(
        self,
        hp: Any,
        objective: TrainingObjective,
        message: str,
        *,
        warnings: list[str] | None = None,
    ) -> TrainingResult:
        self._state = "finished"
        self._emit("error", message, level="error")
        result = TrainingResult(
            run_id=self.run_id,
            objective=objective,
            method=hp.method,
            status="failed",
            error=message,
            warnings=warnings if warnings is not None else list(self._warnings),
        )
        self._result = result
        return result


__all__ = ["TrainingJob", "TrainingResult", "find_resume_checkpoint"]
