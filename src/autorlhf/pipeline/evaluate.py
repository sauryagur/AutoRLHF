"""Baseline vs fine-tuned evaluation.

Both variants see the *same* held-out prompts, the *same* rubric, and the *same*
judge model; that is what makes the comparison meaningful.  Reports aggregate
over **scored** responses only — a failed judgement is counted as a failure, never
as a zero — and the comparison refuses to claim an improvement it cannot support:

* coverage below ``min_responses_for_claims`` marks the report inconclusive,
* any failed judgement is listed in ``warnings``,
* an overall delta inside the rubric's noise band is reported as noise.

``torch``/``transformers``/``peft`` are imported lazily inside
:func:`load_eval_model`, so importing this module (and therefore the pipeline)
never pulls in the training stack.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from autorlhf.config.rubric import DEFAULT_TIE_EPSILON, Rubric
from autorlhf.config.run_config import RunConfig
from autorlhf.core.errors import InsufficientDataError, ModelLoadError
from autorlhf.core.events import EventBus
from autorlhf.core.logging import get_logger
from autorlhf.llm.jev import JevEvaluator
from autorlhf.llm.schemas import (
    CandidateResponse,
    ComparisonReport,
    CriterionComparison,
    EvalPrompt,
    EvalResponse,
    EvaluationReport,
    Judgement,
    PairDecision,
)
from autorlhf.pipeline.util import wait_if_paused

logger = get_logger("autorlhf.pipeline.evaluate")


def eval_candidate_id(variant: str, prompt_id: str) -> str:
    """Stable id used for the judgements of an evaluation response."""

    return f"{variant}:{prompt_id}"


class EvalModelHandle:
    """A loaded base model, optionally with a PEFT adapter attached."""

    def __init__(
        self,
        *,
        variant: str,
        model: Any,
        tokenizer: Any,
        model_id: str,
        adapter_path: str | None = None,
        disabled_adapter: bool = False,
        notes: list[str] | None = None,
    ) -> None:
        self.variant = variant
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.adapter_path = adapter_path
        #: True for the ``base`` variant when it runs the adapter with
        #: ``disable_adapter()`` — same weights, base behaviour.
        self.disabled_adapter = disabled_adapter
        self.notes: list[str] = list(notes or [])

    @contextlib.contextmanager
    def adapter_context(self) -> Any:
        if self.disabled_adapter:
            with self.model.disable_adapter():
                yield
            return
        yield


def load_eval_model(
    *,
    config: RunConfig,
    variant: str,
    adapter_path: Path | None = None,
    hf_token: str | None = None,
    device: str | None = None,
) -> EvalModelHandle:
    """Load the model for one evaluation variant (imports torch/transformers here)."""

    from autorlhf.training import models as training_models

    loaded = training_models.load_model(
        config.model, config.training, hf_token=hf_token, for_training=False, device=device
    )
    model = loaded.model
    tokenizer = getattr(loaded, "tokenizer", None)
    if tokenizer is None:
        tokenizer, _warnings = training_models.load_tokenizer(config.model, hf_token=hf_token)
    notes: list[str] = list(getattr(loaded, "warnings", []) or [])
    disabled_adapter = False

    if variant == "finetuned":
        if adapter_path is None:
            raise InsufficientDataError(
                "the fine-tuned variant needs an adapter directory",
                hint="Run training first, then evaluate the adapter it produced.",
            )
        from peft import PeftModel

        try:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        except Exception as exc:
            raise ModelLoadError(
                f"could not load the adapter at {adapter_path}: {exc}",
                hint="Check that training completed and the adapter directory is intact.",
            ) from exc
    elif adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        disabled_adapter = True
        notes.append(
            "base variant evaluated with the adapter attached and disabled, so both "
            "variants share identical weights"
        )

    return EvalModelHandle(
        variant=variant,
        model=model,
        tokenizer=tokenizer,
        model_id=config.model.model_id,
        adapter_path=str(adapter_path) if adapter_path else None,
        disabled_adapter=disabled_adapter,
        notes=notes,
    )


def generate_eval_response(
    handle: EvalModelHandle,
    prompt: EvalPrompt,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str | None = None,
) -> EvalResponse:
    """Generate one held-out response, recording failures instead of raising."""

    from autorlhf.core.types import ChatMessage
    from autorlhf.training.models import generate_text

    messages = [ChatMessage(role="user", content=prompt.text)]
    response = EvalResponse(
        response_id=f"{handle.variant}:{prompt.prompt_id}",
        prompt_id=prompt.prompt_id,
        variant=handle.variant,
        model_id=handle.model_id,
        adapter_path=handle.adapter_path,
    )
    try:
        started = time.monotonic()
        with handle.adapter_context():
            text, meta = generate_text(
                handle.model,
                handle.tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                device=device,
            )
        meta = meta or {}
        response.text = text
        response.tokens = meta.get("tokens") or meta.get("completion_tokens")
        response.truncated = bool(meta.get("truncated", False))
        response.latency_s = meta.get("latency_s") or (time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 - recorded on the response, never raised
        logger.warning("generation failed for %s: %s", prompt.prompt_id, exc)
        response.error = str(exc) or type(exc).__name__
    return response


def _judge_once(
    jev: JevEvaluator,
    prompt: EvalPrompt,
    variant: str,
    text: str,
    model_id: str,
    *,
    pass_index: int = 0,
) -> Judgement:
    candidate = CandidateResponse(
        candidate_id=eval_candidate_id(variant, prompt.prompt_id),
        prompt_id=prompt.prompt_id,
        source="local",
        model_id=model_id,
        text=text,
    )
    return jev.score_response(
        prompt_id=prompt.prompt_id,
        candidate=candidate,
        prompt=prompt.text,
        context=prompt.context,
        pass_index=pass_index,
    )


def _merge_judgements(judgements: list[Judgement], rubric: Rubric) -> Judgement:
    """Average repeated judge passes; never invent a value a pass did not return."""

    base = judgements[0]
    if len(judgements) == 1:
        return base
    merged = base.model_copy(deep=True)
    merged.usage = {}
    merged.cached = all(j.cached for j in judgements)
    for key in ("input_tokens", "output_tokens", "cost"):
        total = sum(float(j.usage.get(key, 0) or 0) for j in judgements)
        if total:
            merged.usage[key] = total

    for criterion in rubric.criteria:
        passes = [
            j.criterion_scores[criterion.id]
            for j in judgements
            if criterion.id in j.criterion_scores
        ]
        scores = [p.score for p in passes if p.score is not None]
        target = merged.criterion_scores.get(criterion.id)
        if target is None:
            continue
        if not scores:
            continue
        target.score = sum(scores) / len(scores)
        target.error = None
        confidences = [p.confidence for p in passes if p.confidence is not None]
        target.confidence = sum(confidences) / len(confidences) if confidences else None
        probability_passes = [p.probabilities for p in passes if p.probabilities]
        if probability_passes:
            keys = set.intersection(*(set(p) for p in probability_passes))
            if keys:
                target.probabilities = {
                    key: sum(p[key] for p in probability_passes) / len(probability_passes)
                    for key in sorted(keys)
                }

    merged.overall = rubric.overall(
        {cid: cs.score for cid, cs in merged.criterion_scores.items() if cs.score is not None}
    )
    merged.error = None if merged.overall is not None else "no usable scores after averaging"
    return merged


def evaluate_model(
    *,
    handle: EvalModelHandle,
    prompts: list[EvalPrompt],
    jev: JevEvaluator,
    rubric: Rubric,
    run_id: str,
    eval_prompts_hash: str,
    variant: str,
    judge_model: str,
    judge_repeats: int = 1,
    min_responses_for_claims: int = 6,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    bus: EventBus | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> EvaluationReport:
    """Generate and judge every held-out prompt for one variant."""

    report = EvaluationReport(
        run_id=run_id,
        variant=variant,
        judge_model=judge_model,
        rubric_fingerprint=rubric.fingerprint(),
        eval_prompts_hash=eval_prompts_hash,
        requested_judgements=len(prompts),
    )
    report.warnings.extend(handle.notes)
    if bus is not None:
        bus.emit(
            "log",
            f"loading/generating {variant} responses for {len(prompts)} held-out prompts",
            run_id=run_id,
            phase="evaluate",
            verbose_only=True,
        )

    criterion_sums: dict[str, float] = defaultdict(float)
    criterion_counts: dict[str, int] = defaultdict(int)
    overall_values: list[float] = []
    total = len(prompts)

    for index, prompt in enumerate(prompts):
        if cancel_event is not None and cancel_event.is_set():
            break
        if not wait_if_paused(pause_event, cancel_event):
            break
        if bus is not None:
            bus.emit(
                "log",
                f"local inference start {variant} {index + 1}/{total} prompt={prompt.prompt_id} "
                f"model={handle.model_id} max_new_tokens={max_new_tokens} "
                f"temperature={temperature} top_p={top_p} seed={seed + index}\n"
                f"input prompt: {prompt.text}",
                run_id=run_id,
                phase="evaluate",
                verbose_only=True,
            )
        response = generate_eval_response(
            handle,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed + index,
        )
        report.responses.append(response)
        if bus is not None:
            bus.emit(
                "log",
                (
                    f"generated {variant} response for prompt {prompt.prompt_id} "
                    f"({response.tokens or 0} tokens, {response.latency_s or 0:.2f}s)"
                    if not response.error
                    else f"generation failed for {variant} prompt {prompt.prompt_id}: {response.error}"
                ),
                run_id=run_id,
                phase="evaluate",
                level="warning" if response.error else "debug",
                verbose_only=True,
            )
        judgement: Judgement | None = None
        if response.error or not response.text.strip():
            report.failed_judgements += 1
            report.missing_prompt_ids.append(prompt.prompt_id)
        else:
            passes: list[Judgement] = []
            repeats = max(1, int(judge_repeats))
            for pass_index in range(repeats):
                if bus is not None:
                    bus.emit(
                        "log",
                        f"starting Jev evaluation model={judge_model} variant={variant} "
                        f"prompt={prompt.prompt_id} pass={pass_index + 1}/{repeats}",
                        run_id=run_id,
                        phase="evaluate",
                        verbose_only=True,
                    )
                started = time.monotonic()
                passes.append(
                    _judge_once(
                        jev,
                        prompt,
                        variant,
                        response.text,
                        handle.model_id,
                        pass_index=pass_index,
                    )
                )
                result = passes[-1]
                if bus is not None:
                    bus.emit(
                        "log",
                        f"Jev evaluation complete model={judge_model} prompt={prompt.prompt_id} "
                        f"in {time.monotonic() - started:.2f}s cached={result.cached} usage={result.usage}",
                        run_id=run_id,
                        phase="evaluate",
                        verbose_only=True,
                    )
            judgement = _merge_judgements(passes, rubric)
            report.judgements.append(judgement)
            if judgement.error is not None and all(
                c.failed for c in judgement.criterion_scores.values()
            ):
                report.failed_judgements += 1
                report.missing_prompt_ids.append(prompt.prompt_id)
            else:
                report.scored_responses += 1
                for criterion_id, score in judgement.criterion_scores.items():
                    if score.score is None:
                        continue
                    criterion_sums[criterion_id] += score.score
                    criterion_counts[criterion_id] += 1
                if judgement.overall is not None:
                    overall_values.append(judgement.overall)
        if bus is not None:
            bus.emit(
                "sample",
                f"{variant} evaluation prompt {prompt.prompt_id}",
                run_id=run_id,
                phase="evaluate",
                prompt=prompt.text,
                response=response.text,
                judgement=judgement.model_dump(mode="json") if judgement is not None else None,
                error=response.error,
                verbose_only=True,
            )
            bus.emit(
                "progress",
                f"evaluating {variant}: {index + 1}/{total}",
                run_id=run_id,
                phase="evaluate",
                fraction=(index + 1) / max(1, total),
                verbose_only=True,
                variant=variant,
            )

    report.criterion_means = {
        criterion_id: round(criterion_sums[criterion_id] / criterion_counts[criterion_id], 4)
        for criterion_id in sorted(criterion_counts)
    }
    report.overall_mean = (
        round(sum(overall_values) / len(overall_values), 4) if overall_values else None
    )
    if report.scored_responses < min_responses_for_claims:
        report.warnings.append(
            f"only {report.scored_responses} of {report.requested_judgements} responses "
            f"could be scored (min_responses_for_claims={min_responses_for_claims}); "
            "no claim about improvement can be made from this sample"
        )
    if report.failed_judgements:
        report.warnings.append(f"{report.failed_judgements} response(s) had no usable judgement")
    return report


def compare_reports(
    base: EvaluationReport,
    final: EvaluationReport,
    *,
    pairwise_pairs: list[PairDecision] | None = None,
    prompts_by_id: dict[str, EvalPrompt] | None = None,
    rubric: Rubric | None = None,
    max_examples: int = 5,
    min_responses_for_claims: int = 6,
    noise_epsilon: float = DEFAULT_TIE_EPSILON,
) -> ComparisonReport:
    """Base vs fine-tuned comparison, honest about what the sample can support."""

    pairwise_pairs = pairwise_pairs or []
    comparison = ComparisonReport(
        run_id=base.run_id or final.run_id,
        judge_model=final.judge_model or base.judge_model,
        rubric_fingerprint=final.rubric_fingerprint or base.rubric_fingerprint,
        eval_prompts_hash=final.eval_prompts_hash or base.eval_prompts_hash,
        n_prompts=len(prompts_by_id or {}) or len(base.responses),
        n_scored_base=base.scored_responses,
        n_scored_finetuned=final.scored_responses,
        base_overall=base.overall_mean,
        finetuned_overall=final.overall_mean,
    )
    if base.overall_mean is not None and final.overall_mean is not None:
        comparison.overall_delta = round(final.overall_mean - base.overall_mean, 4)

    criterion_ids = sorted(set(base.criterion_means) | set(final.criterion_means))
    base_counts = _criterion_counts(base)
    final_counts = _criterion_counts(final)
    for criterion_id in criterion_ids:
        base_mean = base.criterion_means.get(criterion_id)
        final_mean = final.criterion_means.get(criterion_id)
        delta = (
            round(final_mean - base_mean, 4)
            if base_mean is not None and final_mean is not None
            else None
        )
        name = criterion_id
        scale_max = 4
        if rubric is not None:
            criterion = rubric.criterion(criterion_id)
            if criterion is not None:
                name = criterion.name
                scale_max = criterion.scale_max
        comparison.criteria.append(
            CriterionComparison(
                criterion_id=criterion_id,
                criterion_name=name,
                base_mean=base_mean,
                finetuned_mean=final_mean,
                base_n=base_counts.get(criterion_id, 0),
                finetuned_n=final_counts.get(criterion_id, 0),
                delta=delta,
                scale_max=scale_max,
            )
        )

    for decision in pairwise_pairs:
        if decision.error is not None or decision.winner is None:
            comparison.pairwise_inconclusive += 1
        elif decision.winner == "a":
            comparison.pairwise_base_wins += 1
        elif decision.winner == "b":
            comparison.pairwise_finetuned_wins += 1
        else:
            comparison.pairwise_ties += 1

    comparison.examples = _examples(
        base, final, prompts_by_id or {}, pairwise_pairs, max_examples=max_examples
    )

    min_scored = min(base.scored_responses, final.scored_responses)
    if min_scored < min_responses_for_claims:
        comparison.warnings.append(
            f"coverage is too low for a claim: base scored {base.scored_responses}, "
            f"fine-tuned scored {final.scored_responses} "
            f"(min_responses_for_claims={min_responses_for_claims})"
        )
        comparison.inconclusive = True
    if base.failed_judgements or final.failed_judgements:
        comparison.warnings.append(
            f"judge failures: base {base.failed_judgements}, fine-tuned {final.failed_judgements}"
        )
    if comparison.overall_delta is not None and abs(comparison.overall_delta) <= noise_epsilon:
        comparison.notes.append(
            f"overall delta {comparison.overall_delta:+.3f} is within the rubric noise band "
            f"(tie_epsilon {noise_epsilon:g})"
        )
        comparison.inconclusive = True
    if not comparison.warnings and not comparison.notes:
        comparison.notes.append("no warnings: coverage and effect size look adequate")
    return comparison


def _criterion_counts(report: EvaluationReport) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for judgement in report.judgements:
        for criterion_id, score in judgement.criterion_scores.items():
            if score.score is not None:
                counts[criterion_id] += 1
    return dict(counts)


def _examples(
    base: EvaluationReport,
    final: EvaluationReport,
    prompts_by_id: dict[str, EvalPrompt],
    pairwise_pairs: list[PairDecision],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    base_responses = {r.prompt_id: r for r in base.responses}
    final_responses = {r.prompt_id: r for r in final.responses}
    base_judgements = {j.candidate_id: j for j in base.judgements}
    final_judgements = {j.candidate_id: j for j in final.judgements}
    wins = {d.prompt_id: d.winner for d in pairwise_pairs}

    examples: list[dict[str, Any]] = []
    for prompt_id in sorted(set(base_responses) & set(final_responses)):
        if len(examples) >= max_examples:
            break
        base_response = base_responses[prompt_id]
        final_response = final_responses[prompt_id]
        if not base_response.text.strip() or not final_response.text.strip():
            continue
        base_judgement = base_judgements.get(eval_candidate_id("base", prompt_id))
        final_judgement = final_judgements.get(eval_candidate_id("finetuned", prompt_id))
        prompt = prompts_by_id.get(prompt_id)
        examples.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt.text if prompt is not None else "",
                "category": prompt.category if prompt is not None else "",
                "base_text": base_response.text,
                "finetuned_text": final_response.text,
                "base_overall": base_judgement.overall if base_judgement else None,
                "finetuned_overall": final_judgement.overall if final_judgement else None,
                "base_judgement": base_judgement.model_dump(mode="json")
                if base_judgement
                else None,
                "finetuned_judgement": final_judgement.model_dump(mode="json")
                if final_judgement
                else None,
                "pairwise_winner": wins.get(prompt_id),
            }
        )
    return examples


__all__ = [
    "EvalModelHandle",
    "compare_reports",
    "eval_candidate_id",
    "evaluate_model",
    "generate_eval_response",
    "load_eval_model",
]
