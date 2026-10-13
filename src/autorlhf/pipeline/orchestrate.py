"""The pipeline runner: dataset -> review -> baseline -> training plan -> final.

Each method validates its preconditions *before* spending money, advances the
run status through :class:`~autorlhf.storage.db.RunStore` (which validates the
transition), persists artifacts atomically, updates counters, and refreshes the
manifest.  Every method is idempotent on resume: cached prompts and judgements
are reused, and ``dataset.jsonl``/``preferences.jsonl``/``sft.jsonl`` are
rebuilt from the in-memory bundle rather than appended to blindly.

The training stack (``torch``/``transformers``/``trl``) is reached through small
module-level indirections that import lazily, so importing this module pulls in
neither ``textual`` nor ``torch``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autorlhf.config.presets import (
    PresetFeasibility,
    estimate_training_memory,
    feasible_presets,
    preset_choice_options,
    preset_ranking_state,
    recommend_preset,
)
from autorlhf.core.enums import (
    RunStatus,
    SampleStatus,
    SplitName,
    TrainingMethod,
    TrainingObjective,
)
from autorlhf.core.errors import (
    ConfigError,
    CredentialError,
    HardwareError,
    InsufficientDataError,
)
from autorlhf.core.logging import get_logger
from autorlhf.core.types import HardwareInfo
from autorlhf.llm.schemas import (
    CandidateResponse,
    ComparisonReport,
    EvaluationReport,
    PairDecision,
    PreferencePair,
)
from autorlhf.pipeline.candidates import CandidateGenerator
from autorlhf.pipeline.context import PipelineContext
from autorlhf.pipeline.dataset import (
    append_records,
    build_eval_prompts,
    build_sft_records,
    candidate_record,
    compute_counters,
    eval_prompt_record,
    judgement_record,
    load_dataset,
    load_eval_prompts,
    pair_decision_record,
    persist_counters,
    rewrite_records,
    validate_dataset,
    validation_summary,
)
from autorlhf.pipeline.evaluate import (
    compare_reports,
    eval_candidate_id,
    evaluate_model,
    load_eval_model,
)
from autorlhf.pipeline.models import DatasetBundle, ReviewDecision, ReviewItem
from autorlhf.pipeline.preferences import PreferenceBuild, build_preferences, to_records
from autorlhf.pipeline.prompts import generate_prompts
from autorlhf.pipeline.splits import apply_splits, load_or_plan
from autorlhf.storage.runs import file_hash, read_json, read_jsonl, refresh_manifest, write_json

logger = get_logger("autorlhf.pipeline.orchestrate")

#: Below this many usable preference pairs, DPO is not worth running.
MIN_DPO_PAIRS = 2

#: Splits whose prompts may contribute training data.
TRAINING_SPLITS = (SplitName.TRAIN.value, SplitName.VALIDATION.value)


# --------------------------------------------------------------------------- #
# lazy, monkeypatchable indirections into the training stack
# --------------------------------------------------------------------------- #


def _detect_hardware() -> HardwareInfo:
    from autorlhf.training.hardware import detect_hardware

    return detect_hardware()


def _param_count(model_config: Any, hf_token: str | None) -> int | None:
    from autorlhf.training.models import model_param_count_from_metadata

    return model_param_count_from_metadata(model_config, hf_token=hf_token)


def _training_plan_type() -> Any:
    from autorlhf.training.plan import TrainingPlan

    return TrainingPlan


class PipelineRunner:
    """Drives one run through the pipeline, stage by stage."""

    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx
        self.config = ctx.config
        self.paths = ctx.paths
        self.store = ctx.store
        self._hardware: HardwareInfo | None = None
        self._param_count: int | None = None

    # ------------------------------------------------------------------ #
    # preflight
    # ------------------------------------------------------------------ #

    def preflight(self) -> dict[str, Any]:
        """Validate config, credentials, and model metadata before any spend."""

        problems = self.config.validate_config()
        if problems:
            raise ConfigError(
                "run configuration is invalid:\n  - " + "\n  - ".join(problems),
                hint="Fix the listed fields, then run the preflight again.",
            )
        if not self.ctx.has_credentials():
            raise CredentialError(
                "no OpenRouter API key is configured, so generation and judging cannot run",
                hint="Run `autorlhf --setup` to store OPENROUTER_API_KEY, then retry.",
            )

        warnings: list[str] = []
        try:
            self._hardware = _detect_hardware()
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            warnings.append(f"hardware detection failed: {exc}")
            logger.warning("hardware detection failed: %s", exc)
        try:
            self._param_count = _param_count(self.config.model, self.ctx.hf_token)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            warnings.append(f"could not resolve the model's parameter count: {exc}")

        for warning in warnings:
            self.ctx.emit("warning", warning, phase="setup")
        report: dict[str, Any] = {
            "ok": True,
            "run_id": self.ctx.run_id,
            "problems": [],
            "warnings": warnings,
            "config_fingerprint": self.config.fingerprint(),
            "dataset_fingerprint": self.config.dataset_fingerprint(),
            "hardware": self._hardware.model_dump(mode="json") if self._hardware else None,
            "param_count": self._param_count,
            "judge_model": self.config.generation.judge_model,
            "teacher_model": self.config.generation.teacher_model,
        }
        self.ctx.emit(
            "log",
            "preflight complete",
            phase="setup",
            hardware=report["hardware"],
            param_count=self._param_count,
        )
        return report

    # ------------------------------------------------------------------ #
    # dataset preparation
    # ------------------------------------------------------------------ #

    def _judge_candidates(
        self,
        prompts: list[Any],
        candidates: list[CandidateResponse],
        existing: DatasetBundle | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Score every usable candidate, reusing judgements already on disk."""

        from autorlhf.llm.schemas import Judgement

        reusable: dict[str, Judgement] = {}
        if existing is not None:
            reusable = {
                cid: judgement
                for cid, judgement in existing.judgements.items()
                if judgement.error is None
            }
        prompts_by_id = {p.prompt_id: p for p in prompts}
        judgements: dict[str, Judgement] = {}
        new_records: list[dict[str, Any]] = []
        judged = 0
        total = sum(1 for c in candidates if c.status.value != "failed" and c.text.strip())
        for candidate in candidates:
            if self.ctx.cancelled():
                break
            if candidate.status.value == "failed" or not candidate.text.strip():
                continue
            prompt = prompts_by_id.get(candidate.prompt_id)
            if prompt is None:
                continue
            if candidate.candidate_id in reusable:
                judgements[candidate.candidate_id] = reusable[candidate.candidate_id]
                judged += 1
                continue
            judgement = self.ctx.jev.score_response(
                prompt_id=candidate.prompt_id,
                candidate=candidate,
                prompt=prompt.text,
                context=prompt.context,
            )
            judgements[candidate.candidate_id] = judgement
            new_records.append(
                judgement_record(
                    run_id=self.ctx.run_id, judgement=judgement, phase="dataset_review"
                )
            )
            judged += 1
            self.ctx.progress(
                f"judging candidates: {judged}/{total}",
                fraction=judged / max(1, total),
                candidate_id=candidate.candidate_id,
            )
            if judgement.error is None:
                self.ctx.bump("judgements_ok")
            else:
                self.ctx.bump("judgements_failed")
        return judgements, new_records

    @staticmethod
    def _append_new_judgements(path: Path, records: list[dict[str, Any]]) -> int:
        """Append audit records that are not already present (idempotent resume)."""

        if not records:
            return 0
        existing_ids = {str(record.get("judgement_id", "")) for record in read_jsonl(path)}
        fresh = [
            record for record in records if str(record.get("judgement_id", "")) not in existing_ids
        ]
        return append_records(path, fresh)

    @staticmethod
    def _apply_judgement_status(
        candidates: list[CandidateResponse], judgements: dict[str, Any]
    ) -> None:
        """Promote judged candidates: clean ones are accepted, failed ones rejected.

        A candidate whose generation failed stays ``failed``; a truncated response
        stays ``pending`` so it can never be accepted automatically.
        """

        for candidate in candidates:
            if candidate.status is SampleStatus.FAILED:
                continue
            judgement = judgements.get(candidate.candidate_id)
            if judgement is None:
                continue
            if judgement.error is not None or all(
                c.failed for c in judgement.criterion_scores.values()
            ):
                candidate.status = SampleStatus.REJECTED
            elif candidate.truncated:
                candidate.status = SampleStatus.PENDING
            else:
                candidate.status = SampleStatus.ACCEPTED

    def prepare_dataset(self, *, resume: bool = True) -> DatasetBundle:
        """Generate prompts and candidates, judge them, and write the dataset files."""

        self.ctx.set_phase("data")
        self.ctx.set_status(RunStatus.PREPARING_DATA)
        self.ctx.emit("phase", "preparing dataset", phase="data")

        self.ctx.emit(
            "log",
            f"dataset generation started: target={self.config.generation.num_prompts} prompts, "
            f"{self.config.generation.candidates_per_prompt} candidates per prompt, resume={resume}",
            phase="data",
            verbose_only=True,
        )

        existing = load_dataset(self.paths) if resume else None
        prompts = list(existing.prompts) if existing is not None else []
        if prompts:
            self.ctx.emit("log", f"reusing {len(prompts)} prompts from dataset.jsonl", phase="data")
        else:
            prompts = generate_prompts(
                client=self.ctx.openrouter,
                character=self.config.character,
                generation=self.config.generation,
                run_id=self.ctx.run_id,
                cache=self.ctx.cache,
                bus=self.ctx.bus,
                cancel_event=self.ctx.cancel_event,
                pause_event=self.ctx.pause_event,
            )
        self.ctx.emit(
            "log",
            f"prompt generation complete: {len(prompts)} prompts",
            phase="data",
            verbose_only=True,
        )
        if not prompts:
            raise InsufficientDataError(
                "no synthetic prompts could be generated",
                hint="Check the teacher model id and the OpenRouter credential, then retry.",
            )
        dataset_hash = self.config.dataset_fingerprint()
        assignment = load_or_plan(
            self.paths,
            prompts,
            generation=self.config.generation,
            dataset_hash=dataset_hash,
            run_id=self.ctx.run_id,
        )
        apply_splits(prompts, assignment)
        self.ctx.emit(
            "log",
            f"prompt splits assigned: train={len(assignment.prompt_ids(SplitName.TRAIN))}, "
            f"validation={len(assignment.prompt_ids(SplitName.VALIDATION))}, "
            f"eval={len(assignment.prompt_ids(SplitName.EVAL))}",
            phase="data",
            verbose_only=True,
        )

        eval_prompts = build_eval_prompts(
            prompts,
            character=self.config.character,
            include_seed_prompts=self.config.evaluation.include_seed_prompts,
            assignment=assignment,
        )
        if not eval_prompts:
            raise InsufficientDataError(
                "the split produced no held-out evaluation prompts",
                hint=(
                    "Raise eval_fraction or min_eval_prompts on the run config so at least "
                    "one prompt is held out."
                ),
            )
        self.ctx.emit(
            "log",
            f"held-out eval prompts prepared: {len(eval_prompts)}",
            phase="data",
            verbose_only=True,
        )
        rewrite_records(
            self.paths.eval_prompts_path,
            [eval_prompt_record(run_id=self.ctx.run_id, prompt=p) for p in eval_prompts],
        )
        self.ctx.emit(
            "artifact",
            f"{len(eval_prompts)} held-out eval prompts",
            phase="data",
            path=str(self.paths.eval_prompts_path),
        )

        self.ctx.set_status(RunStatus.GENERATING)
        self.ctx.set_phase("generate")
        generator = CandidateGenerator(
            client=self.ctx.openrouter,
            config=self.config,
            run_id=self.ctx.run_id,
            cache=self.ctx.cache,
            bus=self.ctx.bus,
            hf_token=self.ctx.hf_token,
            cancel_event=self.ctx.cancel_event,
            pause_event=self.ctx.pause_event,
        )
        try:
            candidates = generator.generate(
                prompts, existing=existing.candidates if existing is not None else None
            )
        finally:
            generator.close()
        self.ctx.emit(
            "log",
            f"candidate generation complete: {len(candidates)} candidates",
            phase="generate",
            verbose_only=True,
        )
        judgements, new_judgement_records = self._judge_candidates(prompts, candidates, existing)
        self.ctx.emit(
            "log",
            f"candidate judging complete: {len(judgements)} judgements, "
            f"{len(new_judgement_records)} newly scored",
            phase="generate",
            verbose_only=True,
        )
        self._append_new_judgements(self.paths.judgments_path, new_judgement_records)
        self._apply_judgement_status(candidates, judgements)

        prompts_by_id = {p.prompt_id: p for p in prompts}
        records = [
            candidate_record(
                run_id=self.ctx.run_id,
                prompt=prompts_by_id[candidate.prompt_id],
                candidate=candidate,
                judgement=judgements.get(candidate.candidate_id),
            )
            for candidate in candidates
            if candidate.prompt_id in prompts_by_id
        ]
        rewrite_records(self.paths.dataset_path, records)
        self.ctx.emit(
            "log",
            f"dataset records written: {len(records)} candidates",
            phase="data",
            verbose_only=True,
        )
        self.ctx.emit(
            "artifact",
            f"dataset.jsonl ({len(records)} records)",
            phase="data",
            path=str(self.paths.dataset_path),
        )
        problems = validate_dataset(records)
        self.ctx.set_counter("dataset_problems", len(problems))
        for kind, count in validation_summary(problems).items():
            self.ctx.set_counter(f"dataset_problem_{kind}", count)
        for problem in problems:
            self.ctx.emit("warning", f"dataset: {problem}", phase="data")

        decisions: dict[str, ReviewDecision] = {}
        sft_records = build_sft_records(
            run_id=self.ctx.run_id,
            prompts=prompts,
            candidates=candidates,
            character=self.config.character,
            decisions=decisions,
        )
        rewrite_records(self.paths.sft_path, sft_records)
        self.ctx.emit(
            "log",
            f"supervised fine-tuning records written: {len(sft_records)}",
            phase="data",
            verbose_only=True,
        )
        counters = compute_counters(
            prompts=prompts,
            candidates=candidates,
            judgements=judgements,
            pairs=[],
            sft_records=sft_records,
        )
        persist_counters(self.store, self.ctx.run_id, counters)
        self.store.record_dataset(
            self.ctx.run_id,
            "dataset",
            self.paths.dataset_path,
            count=len(records),
            hash_=file_hash(self.paths.dataset_path),
        )

        bundle = DatasetBundle(
            run_id=self.ctx.run_id,
            dataset_hash=dataset_hash,
            split_hash=assignment.split_hash,
            prompts=sorted(prompts, key=lambda p: p.prompt_id),
            candidates=candidates,
            judgements=judgements,
            train_prompt_ids=assignment.prompt_ids(SplitName.TRAIN),
            validation_prompt_ids=assignment.prompt_ids(SplitName.VALIDATION),
            eval_prompt_ids=sorted({p.prompt_id for p in eval_prompts}),
            counters=counters,
        )
        self._refresh_manifest(extra={"stage": "dataset", "dataset": bundle.to_summary()})
        self.ctx.emit("counters", "dataset counters updated", phase="data", counters=counters)
        return bundle

    # ------------------------------------------------------------------ #
    # review / curation
    # ------------------------------------------------------------------ #

    def _review_prompts(self, bundle: DatasetBundle) -> list[Any]:
        return [p for p in bundle.prompts if p.split in TRAINING_SPLITS]

    def _review_items(
        self, prompts: list[Any], bundle: DatasetBundle, build: PreferenceBuild
    ) -> list[ReviewItem]:
        outcomes = {o.prompt_id: o for o in [*build.accepted, *build.rejected]}
        items: list[ReviewItem] = []
        for prompt in prompts:
            usable = bundle.usable_candidates(prompt.prompt_id)
            if len(usable) < 2:
                continue
            outcome = outcomes.get(prompt.prompt_id)
            decision = outcome.decision if outcome is not None else None
            recommended = None
            if outcome is not None and outcome.pair is not None:
                recommended = outcome.pair.chosen_id
            elif decision is not None:
                recommended = decision.winner_id()
            items.append(
                ReviewItem(
                    item_id=prompt.prompt_id,
                    prompt_id=prompt.prompt_id,
                    prompt=prompt.text,
                    context=[dict(turn) for turn in prompt.context],
                    category=prompt.category,
                    candidates=usable,
                    judgements={
                        c.candidate_id: bundle.judgements[c.candidate_id]
                        for c in usable
                        if c.candidate_id in bundle.judgements
                    },
                    comparison=decision,
                    recommended_candidate_id=recommended,
                    recommendation_reasons=list(outcome.reasons) if outcome else [],
                )
            )
        return items

    @staticmethod
    def _apply_decision_to_candidates(
        candidates: list[CandidateResponse], decision: ReviewDecision
    ) -> None:
        """Mirror :meth:`ReviewQueue.apply_to_dataset` on the in-memory candidates."""

        by_id = {c.candidate_id: c for c in candidates if c.prompt_id == decision.item_id}
        for candidate_id, text in decision.edited_texts.items():
            candidate = by_id.get(candidate_id)
            if candidate is not None:
                candidate.text = text
                candidate.status = SampleStatus.EDITED
        if decision.action == "reject_both":
            for candidate in by_id.values():
                if candidate.status is not SampleStatus.FAILED:
                    candidate.status = SampleStatus.REJECTED
        elif decision.chosen_id and decision.rejected_id:
            chosen = by_id.get(decision.chosen_id)
            rejected = by_id.get(decision.rejected_id)
            if chosen is not None:
                chosen.status = SampleStatus.ACCEPTED
            if rejected is not None:
                rejected.status = SampleStatus.REJECTED

    def run_review(self, bundle: DatasetBundle) -> DatasetBundle:
        """Curate the dataset: build preferences, optionally with a human in the loop."""

        self.ctx.set_phase("review")
        self.ctx.set_status(RunStatus.REVIEWING)
        self.ctx.emit("phase", "curating dataset", phase="review")

        prompts = self._review_prompts(bundle)
        prompts_by_id = {p.prompt_id: p for p in prompts}
        candidates_by_id = {c.candidate_id: c for c in bundle.candidates}

        build = build_preferences(
            prompts=prompts,
            candidates=bundle.candidates,
            judgements=bundle.judgements,
            jev=self.ctx.jev,
            rubric=self.config.rubric,
            cancel_event=self.ctx.cancel_event,
        )

        decisions: dict[str, ReviewDecision] = {}
        if self.ctx.interactive:
            items = self._review_items(prompts, bundle, build)
            if items:
                self.ctx.review_queue.submit_items(items)
                for item in items:
                    if self.ctx.cancelled():
                        break
                    decision = self.ctx.review_queue.wait_for_decision(
                        item.item_id, cancel_event=self.ctx.cancel_event
                    )
                    if decision is None:
                        continue
                    decisions[item.prompt_id] = decision
                    self.ctx.review_queue.apply_to_dataset(self.paths, decision)
                    # The queue's write-through only rewrites dataset.jsonl; the
                    # in-memory candidates are what preferences.jsonl and sft.jsonl
                    # are built from, so they must learn the same verdict.
                    self._apply_decision_to_candidates(bundle.candidates, decision)
                build = build_preferences(
                    prompts=prompts,
                    candidates=bundle.candidates,
                    judgements=bundle.judgements,
                    jev=self.ctx.jev,
                    rubric=self.config.rubric,
                    decisions=decisions,
                    cancel_event=self.ctx.cancel_event,
                )

        records = to_records(
            build,
            run_id=self.ctx.run_id,
            prompts_by_id=prompts_by_id,
            candidates_by_id=candidates_by_id,
            rubric=self.config.rubric,
        )
        rewrite_records(self.paths.preferences_path, records)
        self.ctx.emit(
            "artifact",
            f"preferences.jsonl ({len(records)} pairs)",
            phase="review",
            path=str(self.paths.preferences_path),
        )

        decision_records = [
            pair_decision_record(run_id=self.ctx.run_id, decision=outcome.decision, phase="review")
            for outcome in [*build.accepted, *build.rejected]
            if outcome.decision is not None
        ]
        self._append_new_judgements(self.paths.judgments_path, decision_records)

        sft_records = build_sft_records(
            run_id=self.ctx.run_id,
            prompts=bundle.prompts,
            candidates=bundle.candidates,
            character=self.config.character,
            decisions=decisions,
        )
        rewrite_records(self.paths.sft_path, sft_records)

        pairs: list[PreferencePair] = [
            outcome.pair for outcome in build.accepted if outcome.pair is not None
        ]
        bundle.preference_pairs = pairs
        bundle.pair_decisions = [
            outcome.decision
            for outcome in [*build.accepted, *build.rejected]
            if outcome.decision is not None
        ]
        counters = compute_counters(
            prompts=bundle.prompts,
            candidates=bundle.candidates,
            judgements=bundle.judgements,
            pairs=pairs,
            sft_records=sft_records,
            rejections=build.kinds,
        )
        bundle.counters = counters
        persist_counters(self.store, self.ctx.run_id, counters)

        for pair in pairs:
            self.store.save_preference(
                self.ctx.run_id,
                pair.pair_id,
                prompt_id=pair.prompt_id,
                chosen_id=pair.chosen_id,
                rejected_id=pair.rejected_id,
                margin=pair.margin,
                judge_confidence=pair.judge_confidence,
                source=pair.source,
                decision=pair.decision,
                reviewer=pair.reviewer,
                payload=pair.payload,
            )
        if records:
            self.store.record_dataset(
                self.ctx.run_id,
                "preferences",
                self.paths.preferences_path,
                count=len(records),
                hash_=file_hash(self.paths.preferences_path),
            )
        if sft_records:
            self.store.record_dataset(
                self.ctx.run_id,
                "sft",
                self.paths.sft_path,
                count=len(sft_records),
                hash_=file_hash(self.paths.sft_path),
            )
        if self.ctx.interactive:
            bundle.warnings.append(f"human review decisions: {len(decisions)}")
        self._refresh_manifest(extra={"stage": "review", "review": bundle.to_summary()})
        self.ctx.emit(
            "counters",
            "review counters updated",
            phase="review",
            pairs_accepted=len(pairs),
            rejections=build.kinds,
        )
        return bundle

    # ------------------------------------------------------------------ #
    # training plan
    # ------------------------------------------------------------------ #

    def _sft_train_records(self) -> list[dict[str, Any]]:
        return [
            record
            for record in read_jsonl(self.paths.sft_path)
            if record.get("record_type") == "sft" and record.get("split") == SplitName.TRAIN.value
        ]

    def _pair_records(self, bundle: DatasetBundle) -> list[dict[str, Any]]:
        records = [
            record
            for record in read_jsonl(self.paths.preferences_path)
            if record.get("record_type") == "preference"
        ]
        return records or [
            {
                "pair_id": pair.pair_id,
                "prompt_id": pair.prompt_id,
                "chosen_id": pair.chosen_id,
                "rejected_id": pair.rejected_id,
                "split": bundle.split_of(pair.prompt_id),
                "source": pair.source,
            }
            for pair in bundle.preference_pairs
        ]

    def _choose_preset(
        self,
        feasibilities: list[PresetFeasibility],
        *,
        objective: TrainingObjective,
        hardware: HardwareInfo,
    ) -> tuple[PresetFeasibility | None, dict[str, Any]]:
        feasible = [f for f in feasibilities if f.feasible]
        info: dict[str, Any] = {"options": preset_choice_options(feasible), "chosen": None}
        if not feasible:
            return None, info
        preferred = self.config.training.preset_id
        if len(feasible) >= 2 and self.ctx.has_credentials():
            state = preset_ranking_state(
                feasible,
                objective=objective,
                hw=hardware,
                hp=self.config.training,
                model_id=self.config.model.model_id,
            )
            chosen, ranking = self.ctx.jev.rank_presets(
                state=state, options=preset_choice_options(feasible)
            )
            info = ranking
            if chosen:
                match = next((f for f in feasible if f.preset.id.value == chosen), None)
                if match is not None:
                    return match, info
        fallback = recommend_preset(
            hardware, objective=objective, param_count=self._param_count, preferred=preferred
        )
        if fallback is None:
            return feasible[0], info
        info.setdefault("fallback", "recommend_preset")
        return fallback, info

    def build_training_plan(
        self, bundle: DatasetBundle, *, adapter_path: Path | None = None
    ) -> Any:
        """Resolve hyperparameters and objective, asking Jev to rank feasible presets."""

        hardware = self._hardware or _detect_hardware()
        self._hardware = hardware
        sft_train = self._sft_train_records()
        pair_records = self._pair_records(bundle)
        train_pairs = [
            record
            for record in pair_records
            if record.get("split", SplitName.TRAIN.value) == SplitName.TRAIN.value
        ]
        validation_pairs = [
            record for record in pair_records if record.get("split") == SplitName.VALIDATION.value
        ]

        warnings: list[str] = []
        rationale: list[str] = []
        objective = self.config.objective
        if objective is TrainingObjective.DPO and len(train_pairs) < MIN_DPO_PAIRS:
            if sft_train:
                rationale.append(
                    f"only {len(train_pairs)} valid preference pair(s) (need {MIN_DPO_PAIRS}); "
                    "switching to SFT on the supervised examples"
                )
                warnings.append(
                    "DPO was requested but there were too few preference pairs; "
                    "the plan uses SFT instead"
                )
                objective = TrainingObjective.SFT
            else:
                raise InsufficientDataError(
                    f"cannot train: {len(train_pairs)} valid preference pair(s) "
                    f"(need {MIN_DPO_PAIRS}) and 0 supervised examples",
                    hint=(
                        "Generate more candidates, loosen the rubric thresholds, or review "
                        "the dataset so at least one objective has usable data."
                    ),
                )
        if objective is TrainingObjective.SFT and not sft_train:
            raise InsufficientDataError(
                "cannot train: 0 supervised examples and 0 usable preference pairs",
                hint="Re-run dataset preparation so candidates are accepted into sft.jsonl.",
            )

        feasibilities = feasible_presets(
            hardware, objective=objective, param_count=self._param_count
        )
        feasibility, ranking = self._choose_preset(
            feasibilities, objective=objective, hardware=hardware
        )
        if feasibility is None:
            raise HardwareError(
                "no training preset is feasible on this machine",
                hint="Install the optional extras, or lower the model size on the Model screen.",
            )
        preset = feasibility.preset
        hyperparameters = preset.to_hyperparameters(self.config.training)
        hyperparameters.objective = objective
        hyperparameters.run_name = self.config.slug
        if not hyperparameters.lora_target_modules and hyperparameters.method in (
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
        ):
            warnings.append(
                "lora_target_modules is empty; the trainer discovers them from the model"
            )

        estimate = feasibility.estimate or estimate_training_memory(
            hyperparameters,
            param_count=self._param_count,
            gpu_count=max(1, hardware.device_count),
        )
        rationale.extend(
            [
                f"selected preset '{preset.id.value}' ({preset.label})",
                f"objective {objective.value.upper()} from {len(sft_train)} supervised "
                f"example(s) and {len(train_pairs)} preference pair(s)",
            ]
        )
        if ranking.get("chosen"):
            rationale.append(
                f"Jev ranked the feasible presets and preferred '{ranking['chosen']}'"
                + (
                    f" (confidence {ranking['confidence']:.2f})"
                    if isinstance(ranking.get("confidence"), float)
                    else ""
                )
            )
            rationale.append(
                "that distribution is the judge's preference among the listed presets, "
                "not a probability that training will succeed"
            )
        elif ranking.get("error"):
            warnings.append(f"preset ranking unavailable: {ranking['error']}")
            rationale.append("preset chosen deterministically from hardware feasibility")
        else:
            rationale.append("preset chosen deterministically from hardware feasibility")
        if feasibility.tight:
            warnings.append(
                f"{preset.label} is a tight fit on this hardware; expect little headroom"
            )
        if preset.experimental:
            warnings.append(preset.warning or f"{preset.label} is experimental")

        plan_type = _training_plan_type()
        plan = plan_type(
            run_id=self.ctx.run_id,
            objective=objective,
            method=hyperparameters.method,
            hyperparameters=hyperparameters,
            preset_id=preset.id.value,
            rationale=rationale,
            jev_ranking=ranking or None,
            resource_estimate=estimate,
            hardware=hardware,
            sft_examples=sft_train,
            preference_pairs=train_pairs + validation_pairs,
            train_count=len(sft_train) + len(train_pairs),
            validation_count=len(validation_pairs),
            warnings=warnings,
        )
        for warning in warnings:
            self.ctx.emit("warning", warning, phase="train")
        return plan

    def plan_training(self, bundle: DatasetBundle) -> Any:
        """Build, persist, and announce the training plan."""

        self.ctx.set_phase("train")
        plan = self.build_training_plan(bundle)
        plan.save(self.paths.root / "training_plan.json")
        self.ctx.emit(
            "artifact",
            f"training plan: {plan.preset_id} / {plan.objective.value}",
            phase="train",
            path=str(self.paths.root / "training_plan.json"),
            preset_id=plan.preset_id,
            objective=plan.objective.value,
        )
        self.ctx.set_counter("train_count", int(plan.train_count))
        self.ctx.set_counter("validation_count", int(plan.validation_count))
        self._refresh_manifest(
            extra={"stage": "plan", "training_plan": plan.model_dump(mode="json")}
        )
        return plan

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #

    def _eval_prompts(self) -> list[Any]:
        prompts = load_eval_prompts(self.paths)
        cap = int(self.config.evaluation.max_eval_prompts)
        if cap > 0:
            prompts = prompts[:cap]
        return prompts

    def run_baseline_eval(self) -> EvaluationReport:
        """Evaluate the base model on the held-out prompts."""

        self.ctx.set_phase("evaluate")
        self.ctx.set_status(RunStatus.EVALUATING_BASE)
        self.ctx.emit("phase", "baseline evaluation", phase="evaluate")
        self.ctx.emit("log", "loading held-out baseline evaluation prompts", phase="evaluate")
        prompts = self._eval_prompts()
        if not prompts:
            raise InsufficientDataError(
                "no held-out evaluation prompts are available",
                hint="Run dataset preparation first so eval_prompts.jsonl is written.",
            )
        self.ctx.emit(
            "log", f"loaded {len(prompts)} held-out prompts; loading base model", phase="evaluate"
        )
        handle = load_eval_model(
            config=self.config, variant="base", adapter_path=None, hf_token=self.ctx.hf_token
        )
        self.ctx.emit(
            "log", "base model loaded; starting response generation and judging", phase="evaluate"
        )
        report = evaluate_model(
            handle=handle,
            prompts=prompts,
            jev=self.ctx.jev,
            rubric=self.config.rubric,
            run_id=self.ctx.run_id,
            eval_prompts_hash=file_hash(self.paths.eval_prompts_path),
            variant="base",
            judge_model=self.config.generation.judge_model,
            judge_repeats=self.config.evaluation.judge_repeats,
            min_responses_for_claims=self.config.evaluation.min_responses_for_claims,
            max_new_tokens=self.config.generation.max_new_tokens,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            seed=self.config.generation.seed,
            bus=self.ctx.bus,
            cancel_event=self.ctx.cancel_event,
            pause_event=self.ctx.pause_event,
        )
        self._save_report(report, self.paths.baseline_eval_path, "baseline_eval")
        self.ctx.set_counter("baseline_scored", report.scored_responses)
        self.ctx.set_counter("baseline_failed", report.failed_judgements)
        self._refresh_manifest(extra={"stage": "baseline_eval"})
        return report

    def run_final_eval(
        self, *, adapter_path: Path | None
    ) -> tuple[EvaluationReport, ComparisonReport]:
        """Evaluate the fine-tuned model and compare it with the baseline."""

        self.ctx.set_phase("evaluate")
        self.ctx.set_status(RunStatus.EVALUATING_FINAL)
        self.ctx.emit("phase", "final evaluation", phase="evaluate")
        prompts = self._eval_prompts()
        if not prompts:
            raise InsufficientDataError(
                "no held-out evaluation prompts are available",
                hint="Run dataset preparation first so eval_prompts.jsonl is written.",
            )
        baseline_payload = read_json(self.paths.baseline_eval_path, default=None)
        if not isinstance(baseline_payload, dict):
            raise InsufficientDataError(
                "no baseline evaluation is available to compare against",
                hint="Run the baseline evaluation before the final one.",
            )
        baseline = EvaluationReport.model_validate(baseline_payload)

        handle = load_eval_model(
            config=self.config,
            variant="finetuned",
            adapter_path=adapter_path,
            hf_token=self.ctx.hf_token,
        )
        report = evaluate_model(
            handle=handle,
            prompts=prompts,
            jev=self.ctx.jev,
            rubric=self.config.rubric,
            run_id=self.ctx.run_id,
            eval_prompts_hash=file_hash(self.paths.eval_prompts_path),
            variant="finetuned",
            judge_model=self.config.generation.judge_model,
            judge_repeats=self.config.evaluation.judge_repeats,
            min_responses_for_claims=self.config.evaluation.min_responses_for_claims,
            max_new_tokens=self.config.generation.max_new_tokens,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            seed=self.config.generation.seed,
            bus=self.ctx.bus,
            cancel_event=self.ctx.cancel_event,
            pause_event=self.ctx.pause_event,
        )
        self._save_report(report, self.paths.final_eval_path, "final_eval")

        pairwise_pairs = self._pairwise_eval(prompts, baseline, report, adapter_path=adapter_path)
        comparison = compare_reports(
            baseline,
            report,
            pairwise_pairs=pairwise_pairs,
            prompts_by_id={p.prompt_id: p for p in prompts},
            rubric=self.config.rubric,
            min_responses_for_claims=self.config.evaluation.min_responses_for_claims,
            noise_epsilon=self.config.rubric.tie_epsilon,
        )
        write_json(self.paths.comparison_path, comparison.model_dump(mode="json"))
        self.store.save_artifact(
            self.ctx.run_id, "comparison", self.paths.comparison_path, kind="json"
        )
        self.ctx.set_counter("finetuned_scored", report.scored_responses)
        self.ctx.set_counter("comparison_inconclusive", int(comparison.inconclusive))
        self.ctx.emit(
            "artifact",
            "comparison.json written",
            phase="evaluate",
            path=str(self.paths.comparison_path),
            inconclusive=comparison.inconclusive,
        )
        self._refresh_manifest(extra={"stage": "final_eval"})
        return report, comparison

    def _pairwise_eval(
        self,
        prompts: list[Any],
        baseline: EvaluationReport,
        report: EvaluationReport,
        *,
        adapter_path: Path | None,
    ) -> list[PairDecision]:
        if not self.config.evaluation.pairwise:
            return []
        base_responses = {r.prompt_id: r for r in baseline.responses}
        final_responses = {r.prompt_id: r for r in report.responses}
        base_judgements = {j.candidate_id: j for j in baseline.judgements}
        final_judgements = {j.candidate_id: j for j in report.judgements}
        pairs: list[PairDecision] = []
        records: list[dict[str, Any]] = []
        for prompt in prompts:
            if self.ctx.cancelled():
                break
            base_response = base_responses.get(prompt.prompt_id)
            final_response = final_responses.get(prompt.prompt_id)
            if (
                base_response is None
                or final_response is None
                or not base_response.text.strip()
                or not final_response.text.strip()
            ):
                continue
            base_judgement = base_judgements.get(eval_candidate_id("base", prompt.prompt_id))
            final_judgement = final_judgements.get(eval_candidate_id("finetuned", prompt.prompt_id))
            decision = self.ctx.jev.compare_responses(
                prompt_id=prompt.prompt_id,
                option_a=CandidateResponse(
                    candidate_id=eval_candidate_id("base", prompt.prompt_id),
                    prompt_id=prompt.prompt_id,
                    source="local",
                    model_id=baseline.judge_model,
                    text=base_response.text,
                ),
                option_b=CandidateResponse(
                    candidate_id=eval_candidate_id("finetuned", prompt.prompt_id),
                    prompt_id=prompt.prompt_id,
                    source="local",
                    model_id=report.judge_model,
                    text=final_response.text,
                ),
                prompt=prompt.text,
                context=prompt.context,
                overall_a=base_judgement.overall if base_judgement else None,
                overall_b=final_judgement.overall if final_judgement else None,
            )
            pairs.append(decision)
            records.append(
                pair_decision_record(run_id=self.ctx.run_id, decision=decision, phase="final_eval")
            )
        self._append_new_judgements(self.paths.judgments_path, records)
        return pairs

    def _save_report(self, report: EvaluationReport, path: Path, name: str) -> None:
        write_json(path, report.model_dump(mode="json"))
        self.store.save_artifact(self.ctx.run_id, name, path, kind="json")
        for warning in report.warnings:
            self.ctx.emit("warning", warning, phase="evaluate", variant=report.variant)
        self.ctx.emit(
            "artifact",
            f"{name}: overall {report.overall_mean if report.overall_mean is not None else 'n/a'}",
            phase="evaluate",
            path=str(path),
            scored=report.scored_responses,
            failed=report.failed_judgements,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _refresh_manifest(self, *, extra: dict[str, Any] | None = None) -> None:
        try:
            refresh_manifest(
                self.store,
                self.ctx.run_id,
                self.paths,
                self.config,
                extra=extra,
            )
        except Exception as exc:  # noqa: BLE001 - a manifest refresh must not kill a run
            logger.warning("manifest refresh failed: %s", exc)
            self.ctx.emit("warning", f"manifest refresh failed: {exc}", phase=self.ctx.phase)

    def cancel_all(self) -> None:
        """Signal every stage to stop (used by the TUI's stop action)."""

        self.ctx.cancel_event.set()


__all__ = ["MIN_DPO_PAIRS", "TRAINING_SPLITS", "PipelineRunner"]
