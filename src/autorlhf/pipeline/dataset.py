"""Dataset records, validation, counters, and reloading.

Every on-disk shape here is frozen by ``docs/DATA_FORMATS.md``: one JSON object
per line, append-only keys, written through
:func:`autorlhf.storage.runs.append_jsonl`.  The module also owns the dataset
audit: :func:`validate_dataset` reports every way a record set can be wrong
(empty responses, duplicates, broken role alternation, invalid preferences,
inconsistent prompt ids, truncated-but-accepted answers) and
:func:`compute_counters` derives the counters the dashboard shows.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from autorlhf.config.character import CharacterSpec
from autorlhf.core.enums import SampleStatus, SplitName
from autorlhf.core.logging import get_logger
from autorlhf.llm.schemas import (
    CandidateResponse,
    EvalPrompt,
    Judgement,
    PairDecision,
    PreferencePair,
    SyntheticPrompt,
)
from autorlhf.pipeline.models import DatasetBundle, ReviewDecision
from autorlhf.pipeline.util import now_iso
from autorlhf.storage.runs import (
    RunPaths,
    append_jsonl,
    read_jsonl,
    write_text_atomic,
)

logger = get_logger("autorlhf.pipeline.dataset")

DATASET_VERSION = "1"

#: Problems a preference rejection can carry, used as counter names.
REJECTION_KINDS: tuple[str, ...] = (
    "tie",
    "low_confidence",
    "invalid",
    "insufficient_candidates",
)

_ROLES = {"system", "user", "assistant"}


# --------------------------------------------------------------------------- #
# record builders
# --------------------------------------------------------------------------- #


def candidate_record(
    *,
    run_id: str,
    prompt: SyntheticPrompt,
    candidate: CandidateResponse,
    judgement: Judgement | None,
    split: str | None = None,
    version: str = DATASET_VERSION,
) -> dict[str, Any]:
    """One ``dataset.jsonl`` record (one candidate response)."""

    criterion_scores: dict[str, float] = {}
    overall: float | None = None
    if judgement is not None:
        criterion_scores = judgement.scored_criteria()
        overall = judgement.overall
    return {
        "record_type": "candidate",
        "dataset_version": version,
        "run_id": run_id,
        "prompt_id": prompt.prompt_id,
        "candidate_id": candidate.candidate_id,
        "split": split or prompt.split,
        "category": prompt.category,
        "prompt": prompt.text,
        "prompt_context": [dict(turn) for turn in prompt.context],
        "response": candidate.text,
        "source": candidate.source,
        "model_id": candidate.model_id,
        "status": candidate.status.value
        if hasattr(candidate.status, "value")
        else str(candidate.status),
        "error": candidate.error,
        "truncated": bool(candidate.truncated),
        "tokens": candidate.tokens,
        "latency_s": candidate.latency_s,
        "cached": bool(candidate.cached),
        "seed": candidate.seed,
        "temperature": candidate.temperature,
        "edits": list(candidate.edits),
        "edited": bool(candidate.edits),
        "overall": overall,
        "criterion_scores": criterion_scores,
        "judgement": judgement.model_dump(mode="json") if judgement is not None else None,
        "reviewer": "",
        "created_at": now_iso(),
    }


def sft_record(
    *,
    run_id: str,
    example_id: str,
    prompt_id: str,
    split: str,
    messages: list[dict[str, str]],
    source: str,
    status: str = SampleStatus.ACCEPTED.value,
    version: str = DATASET_VERSION,
) -> dict[str, Any]:
    return {
        "record_type": "sft",
        "dataset_version": version,
        "run_id": run_id,
        "example_id": example_id,
        "prompt_id": prompt_id,
        "split": split,
        "messages": messages,
        "source": source,
        "status": status,
        "created_at": now_iso(),
    }


def preference_record(
    *,
    run_id: str,
    prompt: SyntheticPrompt,
    pair: PreferencePair,
    chosen: CandidateResponse,
    rejected: CandidateResponse,
    decision: PairDecision | None,
    criterion_deltas: dict[str, float],
    audit: dict[str, Any],
    version: str = DATASET_VERSION,
) -> dict[str, Any]:
    return {
        "record_type": "preference",
        "dataset_version": version,
        "run_id": run_id,
        "pair_id": pair.pair_id,
        "prompt_id": prompt.prompt_id,
        "split": prompt.split,
        "prompt": prompt.text,
        "prompt_context": [dict(turn) for turn in prompt.context],
        "chosen_id": pair.chosen_id,
        "rejected_id": pair.rejected_id,
        "chosen_text": chosen.text,
        "rejected_text": rejected.text,
        "margin": pair.margin,
        "judge_confidence": pair.judge_confidence,
        "source": pair.source,
        "decision": pair.decision,
        "probabilities": decision.probabilities if decision is not None else None,
        "criterion_deltas": dict(criterion_deltas),
        "audit": audit,
        "pair_decision": decision.model_dump(mode="json") if decision is not None else None,
        "reviewer": pair.reviewer,
        "created_at": pair.created_at,
    }


def eval_prompt_record(
    *, run_id: str, prompt: EvalPrompt, version: str = DATASET_VERSION
) -> dict[str, Any]:
    return {
        "record_type": "eval_prompt",
        "dataset_version": version,
        "run_id": run_id,
        "prompt_id": prompt.prompt_id,
        "text": prompt.text,
        "context": [dict(turn) for turn in prompt.context],
        "category": prompt.category,
        "origin": prompt.origin,
    }


def judgement_record(
    *,
    run_id: str,
    judgement: Judgement,
    phase: str,
    kind: str = "score",
    retries: int = 0,
    version: str = DATASET_VERSION,
) -> dict[str, Any]:
    return {
        "record_type": "judgement",
        "dataset_version": version,
        "run_id": run_id,
        "judgement_id": judgement.judgement_id,
        "prompt_id": judgement.prompt_id,
        "candidate_id": judgement.candidate_id,
        "phase": phase,
        "kind": kind,
        "judge_model": judgement.judge_model,
        "judge_build": judgement.judge_build,
        "cached": bool(judgement.cached),
        "error": judgement.error,
        "retries": retries,
        "usage": dict(judgement.usage),
        "human_override": judgement.human_override,
        "created_at": judgement.created_at,
        "judgement": judgement.model_dump(mode="json"),
    }


def pair_decision_record(
    *,
    run_id: str,
    decision: PairDecision,
    phase: str,
    version: str = DATASET_VERSION,
) -> dict[str, Any]:
    return {
        "record_type": "judgement",
        "dataset_version": version,
        "run_id": run_id,
        "judgement_id": decision.pair_id,
        "prompt_id": decision.prompt_id,
        "candidate_id": "",
        "phase": phase,
        "kind": "choice",
        "judge_model": decision.judge_model,
        "judge_build": decision.judge_build,
        "cached": bool(decision.cached),
        "error": decision.error,
        "retries": 0,
        "usage": dict(decision.usage),
        "human_override": decision.human_override,
        "created_at": decision.created_at,
        "pair_decision": decision.model_dump(mode="json"),
    }


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def append_records(path: Path, records: Iterable[dict[str, Any]]) -> int:
    return append_jsonl(path, list(records))


def rewrite_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically replace the whole file (used after a human edit)."""

    text = "".join(json.dumps(record, default=str, sort_keys=True) + "\n" for record in records)
    write_text_atomic(path, text)


def update_record(
    path: Path,
    *,
    match: dict[str, Any],
    changes: dict[str, Any],
) -> bool:
    """Rewrite ``path`` with ``changes`` applied to the first record matching ``match``."""

    records = read_jsonl(path)
    if not records:
        return False
    for record in records:
        if all(record.get(key) == value for key, value in match.items()):
            record.update(changes)
            rewrite_records(path, records)
            return True
    return False


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def _conversation_problems(messages: list[Any], *, must_end_assistant: bool) -> list[str]:
    problems: list[str] = []
    turns = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    if not turns:
        return problems
    if turns[0].get("role") != "user":
        problems.append("conversation does not start with a user turn")
    previous: str | None = None
    for index, turn in enumerate(turns):
        role = turn.get("role")
        if role not in _ROLES:
            problems.append(f"turn {index} has unknown role {role!r}")
            continue
        if previous == role:
            problems.append(f"turns {index - 1} and {index} are both '{role}'")
        previous = role
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            problems.append(f"turn {index} ({role}) is empty")
    if must_end_assistant and turns[-1].get("role") != "assistant":
        problems.append("conversation does not end with an assistant turn")
    return problems


def validate_dataset(records: list[dict[str, Any]]) -> list[str]:
    """Return a human-readable problem for every defect found in ``records``."""

    problems: list[str] = []
    prompt_splits: dict[str, str] = {}
    candidates: dict[str, dict[str, Any]] = {}
    seen_candidate_text: set[tuple[str, str, str]] = set()
    sft_ids: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            problems.append(f"record #{index} is not a JSON object")
            continue
        kind = record.get("record_type")
        if kind == "candidate":
            prompt_id = str(record.get("prompt_id", ""))
            candidate_id = str(record.get("candidate_id", ""))
            text = record.get("response")
            split = str(record.get("split", ""))
            if not prompt_id or not candidate_id:
                problems.append(f"record #{index} has no prompt_id/candidate_id")
            if prompt_id in prompt_splits and prompt_splits[prompt_id] != split:
                problems.append(
                    f"prompt '{prompt_id}' appears with split '{split}' and "
                    f"'{prompt_splits[prompt_id]}'"
                )
            prompt_splits.setdefault(prompt_id, split)
            if split not in {name.value for name in SplitName}:
                problems.append(f"candidate '{candidate_id}' has unknown split '{split}'")
            status = str(record.get("status", ""))
            if status not in {s.value for s in SampleStatus}:
                problems.append(f"candidate '{candidate_id}' has unknown status '{status}'")
            if not isinstance(text, str) or not text.strip():
                if status != SampleStatus.FAILED.value:
                    problems.append(f"candidate '{candidate_id}' ({status}) has an empty response")
                elif not record.get("error"):
                    problems.append(f"failed candidate '{candidate_id}' carries no error message")
            else:
                key = (prompt_id, candidate_id, text)
                if key in seen_candidate_text:
                    problems.append(f"duplicate record for candidate '{candidate_id}'")
                seen_candidate_text.add(key)
            if record.get("truncated") and status == SampleStatus.ACCEPTED.value:
                problems.append(f"candidate '{candidate_id}' is accepted but was truncated")
            if record.get("judgement") is None and record.get("criterion_scores"):
                problems.append(f"candidate '{candidate_id}' has criterion scores but no judgement")
            candidates[candidate_id] = record
        elif kind == "sft":
            example_id = str(record.get("example_id", ""))
            if example_id in sft_ids:
                problems.append(f"duplicate sft example id '{example_id}'")
            sft_ids.add(example_id)
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                problems.append(f"sft example '{example_id}' has no messages")
                continue
            first = messages[0]
            if not isinstance(first, dict) or first.get("role") != "system":
                problems.append(
                    f"sft example '{example_id}' does not start with the character system turn"
                )
            problems.extend(
                f"sft example '{example_id}': {p}"
                for p in _conversation_problems(messages, must_end_assistant=True)
            )
        elif kind == "preference":
            pair_id = str(record.get("pair_id", ""))
            chosen = str(record.get("chosen_id", ""))
            rejected = str(record.get("rejected_id", ""))
            if chosen == rejected:
                problems.append(f"preference '{pair_id}' chooses and rejects the same response")
            for side, candidate_id in (("chosen", chosen), ("rejected", rejected)):
                candidate = candidates.get(candidate_id)
                if candidate is None:
                    problems.append(
                        f"preference '{pair_id}' {side} side '{candidate_id}' is missing"
                    )
                    continue
                if candidate.get("status") == SampleStatus.FAILED.value:
                    problems.append(
                        f"preference '{pair_id}' {side} side '{candidate_id}' failed to generate"
                    )
                if not str(record.get(f"{side}_text", "")).strip():
                    problems.append(f"preference '{pair_id}' {side} text is empty")
            source = record.get("source")
            if source not in {"jev", "human"}:
                problems.append(f"preference '{pair_id}' has unknown source '{source}'")
        elif kind == "eval_prompt":
            continue
        elif kind == "judgement":
            continue
        else:
            problems.append(f"record #{index} has unknown record_type {kind!r}")

    # Preferences must reference candidates of the same prompt.
    for record in records:
        if isinstance(record, dict) and record.get("record_type") == "preference":
            chosen_candidate = candidates.get(str(record.get("chosen_id", "")))
            if (
                chosen_candidate is not None
                and chosen_candidate.get("prompt_id") != record.get("prompt_id")
            ):
                problems.append(
                    f"preference '{record.get('pair_id')}' spans prompts "
                    f"'{record.get('prompt_id')}' and '{chosen_candidate.get('prompt_id')}'"
                )
    return problems


def validation_summary(problems: list[str]) -> dict[str, int]:
    """A coarse, counter-friendly summary of the validation problems."""

    summary: dict[str, int] = {}
    for problem in problems:
        key = problem.split(" ", 1)[0].strip("'\"") or "problem"
        summary[key] = summary.get(key, 0) + 1
    return summary


# --------------------------------------------------------------------------- #
# counters
# --------------------------------------------------------------------------- #


def compute_counters(
    *,
    prompts: list[SyntheticPrompt],
    candidates: list[CandidateResponse],
    judgements: dict[str, Judgement],
    pairs: list[PreferencePair],
    sft_records: list[dict[str, Any]],
    rejections: dict[str, int] | None = None,
) -> dict[str, int]:
    """Every counter the dashboard and the manifest read, derived from the records."""

    rejections = rejections or {}
    counters: dict[str, int] = {
        "prompts_generated": len(prompts),
        "candidates_generated": 0,
        "generation_failed": 0,
        "judgements_ok": 0,
        "judgements_failed": 0,
        "pairs_accepted": len(pairs),
        "pairs_rejected_tie": int(rejections.get("tie", 0)),
        "pairs_rejected_low_confidence": int(rejections.get("low_confidence", 0)),
        "pairs_rejected_invalid": int(rejections.get("invalid", 0))
        + int(rejections.get("insufficient_candidates", 0)),
        "accepted": 0,
        "edited": 0,
        "rejected": 0,
        "sft_examples": len(sft_records),
    }
    for candidate in candidates:
        if candidate.status is SampleStatus.FAILED:
            counters["generation_failed"] += 1
        else:
            counters["candidates_generated"] += 1
        status = (
            candidate.status.value if hasattr(candidate.status, "value") else str(candidate.status)
        )
        if status in counters:
            counters[status] += 1
    for judgement in judgements.values():
        if judgement.error is None and not all(
            c.failed for c in judgement.criterion_scores.values()
        ):
            counters["judgements_ok"] += 1
        else:
            counters["judgements_failed"] += 1
    for split in SplitName:
        counters[f"split_{split.value}"] = sum(1 for p in prompts if p.split == split.value)
        counters[f"candidates_{split.value}"] = sum(
            1
            for c in candidates
            if c.status is not SampleStatus.FAILED
            and next((p.split for p in prompts if p.prompt_id == c.prompt_id), "") == split.value
        )
    return counters


def persist_counters(store: Any, run_id: str, counters: dict[str, int]) -> None:
    """Write every counter through ``RunStore.set_counter`` (idempotent on resume)."""

    for name, value in counters.items():
        store.set_counter(run_id, name, int(value))


# --------------------------------------------------------------------------- #
# evaluation prompts
# --------------------------------------------------------------------------- #


def build_eval_prompts(
    prompts: list[SyntheticPrompt],
    *,
    character: CharacterSpec | None = None,
    include_seed_prompts: bool = False,
    assignment: Any | None = None,
) -> list[EvalPrompt]:
    """The held-out prompt set: synthetic eval-split prompts plus optional seeds."""

    from autorlhf.pipeline.prompts import seed_eval_prompts

    eval_ids = (
        set(assignment.prompt_ids(SplitName.EVAL))
        if assignment is not None
        else {p.prompt_id for p in prompts if p.split == SplitName.EVAL.value}
    )
    out = [
        EvalPrompt(
            prompt_id=prompt.prompt_id,
            text=prompt.text,
            context=[dict(turn) for turn in prompt.context],
            category=prompt.category,
            origin="synthetic",
        )
        for prompt in prompts
        if prompt.prompt_id in eval_ids
    ]
    if include_seed_prompts and character is not None:
        existing = {p.text for p in out}
        for prompt in seed_eval_prompts(character):
            if prompt.text in existing:
                continue
            out.append(prompt)
    return out


def write_eval_prompts(paths: RunPaths, run_id: str, prompts: list[EvalPrompt]) -> int:
    records = [eval_prompt_record(run_id=run_id, prompt=prompt) for prompt in prompts]
    return append_records(paths.eval_prompts_path, records)


def load_eval_prompts(paths: RunPaths) -> list[EvalPrompt]:
    out: list[EvalPrompt] = []
    for record in read_jsonl(paths.eval_prompts_path):
        if record.get("record_type") != "eval_prompt":
            continue
        try:
            out.append(
                EvalPrompt(
                    prompt_id=str(record.get("prompt_id", "")),
                    text=str(record.get("text", "")),
                    context=list(record.get("context") or []),
                    category=str(record.get("category", "ordinary")),
                    origin=str(record.get("origin", "synthetic")),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("skipping an unreadable eval prompt: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# sft records
# --------------------------------------------------------------------------- #


def build_sft_records(
    *,
    run_id: str,
    prompts: list[SyntheticPrompt],
    candidates: list[CandidateResponse],
    character: CharacterSpec,
    decisions: dict[str, ReviewDecision] | None = None,
    splits: Iterable[str] = (SplitName.TRAIN.value,),
    version: str = DATASET_VERSION,
) -> list[dict[str, Any]]:
    """Supervised examples: seed conversations plus accepted/edited candidates.

    The first message is always the character system prompt, so the trained model
    sees exactly the framing it will see at inference.
    """

    allowed = set(splits)
    decisions = decisions or {}
    records: list[dict[str, Any]] = []
    counter = 0

    for index, seed in enumerate(character.valid_seeds()):
        counter += 1
        messages = [{"role": "system", "content": character.system_prompt}]
        messages.extend({"role": m.role, "content": m.content} for m in seed.to_training_messages())
        records.append(
            sft_record(
                run_id=run_id,
                example_id=f"s{counter:04d}",
                prompt_id=f"seed-{index + 1}",
                split=SplitName.TRAIN.value,
                messages=messages,
                source="seed",
                version=version,
            )
        )

    for prompt in prompts:
        if prompt.split not in allowed:
            continue
        for candidate in candidates:
            if candidate.prompt_id != prompt.prompt_id:
                continue
            if candidate.status not in {SampleStatus.ACCEPTED, SampleStatus.EDITED}:
                continue
            if not candidate.text.strip():
                continue
            decision = decisions.get(prompt.prompt_id)
            chosen_override = decision.chosen_id if decision is not None else None
            if decision is not None and decision.action == "reject_both":
                continue
            if chosen_override and candidate.candidate_id != chosen_override:
                continue
            messages = [{"role": "system", "content": character.system_prompt}]
            messages.extend(dict(turn) for turn in prompt.context)
            messages.append({"role": "user", "content": prompt.text})
            messages.append({"role": "assistant", "content": candidate.text})
            counter += 1
            if decision is not None and decision.action == "edit":
                source = "human"
            elif candidate.status is SampleStatus.EDITED:
                source = "edited"
            else:
                source = "accepted"
            records.append(
                sft_record(
                    run_id=run_id,
                    example_id=f"s{counter:04d}",
                    prompt_id=prompt.prompt_id,
                    split=prompt.split,
                    messages=messages,
                    source=source,
                    status=SampleStatus.ACCEPTED.value,
                    version=version,
                )
            )
    return records


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_dataset(paths: RunPaths) -> DatasetBundle:
    """Rebuild a :class:`DatasetBundle` from the run's JSONL artifacts."""

    prompts: dict[str, SyntheticPrompt] = {}
    candidates: list[CandidateResponse] = []
    judgements: dict[str, Judgement] = {}

    for record in read_jsonl(paths.dataset_path):
        if record.get("record_type") != "candidate":
            continue
        prompt_id = str(record.get("prompt_id", ""))
        if prompt_id and prompt_id not in prompts:
            prompts[prompt_id] = SyntheticPrompt(
                prompt_id=prompt_id,
                run_id=str(record.get("run_id", "")),
                split=str(record.get("split", SplitName.TRAIN.value)),
                category=str(record.get("category", "ordinary")),
                text=str(record.get("prompt", "")),
                context=list(record.get("prompt_context") or []),
            )
        candidate = CandidateResponse(
            candidate_id=str(record.get("candidate_id", "")),
            prompt_id=prompt_id,
            source=str(record.get("source", "teacher")),
            model_id=str(record.get("model_id", "")),
            text=str(record.get("response", "")),
            status=SampleStatus(str(record.get("status", SampleStatus.PENDING.value))),
            error=record.get("error"),
            truncated=bool(record.get("truncated", False)),
            tokens=record.get("tokens"),
            latency_s=record.get("latency_s"),
            seed=record.get("seed"),
            temperature=float(record.get("temperature", 0.8) or 0.8),
            cached=bool(record.get("cached", False)),
            edits=list(record.get("edits") or []),
        )
        candidates.append(candidate)
        raw_judgement = record.get("judgement")
        if isinstance(raw_judgement, dict):
            try:
                judgements[candidate.candidate_id] = Judgement.model_validate(raw_judgement)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("skipping an unreadable judgement: %s", exc)

    pair_decisions: list[PairDecision] = []
    pairs: list[PreferencePair] = []
    for record in read_jsonl(paths.preferences_path):
        if record.get("record_type") != "preference":
            continue
        raw_decision = record.get("pair_decision")
        if isinstance(raw_decision, dict):
            try:
                pair_decisions.append(PairDecision.model_validate(raw_decision))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("skipping an unreadable pair decision: %s", exc)
        try:
            pairs.append(
                PreferencePair(
                    pair_id=str(record.get("pair_id", "")),
                    run_id=str(record.get("run_id", "")),
                    prompt_id=str(record.get("prompt_id", "")),
                    chosen_id=str(record.get("chosen_id", "")),
                    rejected_id=str(record.get("rejected_id", "")),
                    margin=record.get("margin"),
                    judge_confidence=record.get("judge_confidence"),
                    source=str(record.get("source", "jev")),
                    decision=str(record.get("decision", "")),
                    reviewer=str(record.get("reviewer", "")),
                    created_at=str(record.get("created_at", "")) or now_iso(),
                    payload={
                        "criterion_deltas": record.get("criterion_deltas") or {},
                        "audit": record.get("audit") or {},
                        "probabilities": record.get("probabilities"),
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("skipping an unreadable preference: %s", exc)

    sft_records = [
        record for record in read_jsonl(paths.sft_path) if record.get("record_type") == "sft"
    ]

    from autorlhf.pipeline.splits import SplitAssignment

    assignment = SplitAssignment.load(paths.split_manifest_path)
    if assignment is not None:
        train_ids = assignment.prompt_ids(SplitName.TRAIN)
        validation_ids = assignment.prompt_ids(SplitName.VALIDATION)
        eval_ids = assignment.prompt_ids(SplitName.EVAL)
    else:
        train_ids = [pid for pid, p in prompts.items() if p.split == SplitName.TRAIN.value]
        validation_ids = [
            pid for pid, p in prompts.items() if p.split == SplitName.VALIDATION.value
        ]
        eval_ids = [pid for pid, p in prompts.items() if p.split == SplitName.EVAL.value]

    eval_file_ids = [prompt.prompt_id for prompt in load_eval_prompts(paths)]
    if eval_file_ids:
        eval_ids = sorted(set(eval_ids) | set(eval_file_ids))

    rejections = _rejections_from_preferences(paths)
    bundle = DatasetBundle(
        run_id=next((p.run_id for p in prompts.values() if p.run_id), ""),
        dataset_hash=assignment.dataset_hash if assignment is not None else "",
        split_hash=assignment.split_hash if assignment is not None else "",
        prompts=sorted(prompts.values(), key=lambda p: p.prompt_id),
        candidates=candidates,
        judgements=judgements,
        pair_decisions=pair_decisions,
        preference_pairs=pairs,
        train_prompt_ids=sorted(train_ids),
        validation_prompt_ids=sorted(validation_ids),
        eval_prompt_ids=sorted(eval_ids),
    )
    bundle.counters = compute_counters(
        prompts=bundle.prompts,
        candidates=candidates,
        judgements=judgements,
        pairs=pairs,
        sft_records=sft_records,
        rejections=rejections,
    )
    return bundle


def _rejections_from_preferences(paths: RunPaths) -> dict[str, int]:
    """Count rejection reasons recorded in the preference audit trail."""

    counts: dict[str, int] = {}
    for record in read_jsonl(paths.preferences_path):
        audit = record.get("audit")
        if not isinstance(audit, dict):
            continue
        for entry in audit.get("rejected_pairs") or []:
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("kind", "invalid"))
            counts[kind] = counts.get(kind, 0) + 1
    return counts


__all__ = [
    "DATASET_VERSION",
    "REJECTION_KINDS",
    "append_records",
    "build_eval_prompts",
    "build_sft_records",
    "candidate_record",
    "compute_counters",
    "eval_prompt_record",
    "judgement_record",
    "load_dataset",
    "load_eval_prompts",
    "pair_decision_record",
    "persist_counters",
    "preference_record",
    "rewrite_records",
    "sft_record",
    "update_record",
    "validate_dataset",
    "validation_summary",
    "write_eval_prompts",
]
