"""Prompt-level dataset splitting.

The split is decided from **prompt ids**, before any response exists, so a
near-duplicate answer can never leak across the train/held-out boundary.  The
assignment is persisted to ``splits.json`` (see ``docs/DATA_FORMATS.md``) and
reused on resume whenever the dataset fingerprint still matches, which makes it
impossible for a resumed run to re-split its own data and leak held-out prompts.

Stratification keeps every category (ordinary, edge case, boundary, ...) present
in each split whenever the category has enough prompts to go round.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autorlhf.config.run_config import GenerationConfig
from autorlhf.core.enums import SplitName
from autorlhf.core.ids import content_hash
from autorlhf.llm.schemas import SyntheticPrompt
from autorlhf.storage.runs import RunPaths, read_json, write_json

SPLIT_VERSION = "1"

#: Fill order used to break ties when several splits have equal room left.
_FILL_ORDER: tuple[str, ...] = ("eval", "validation", "train")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SplitAssignment(BaseModel):
    """The persisted ``splits.json`` payload."""

    run_id: str = ""
    version: str = SPLIT_VERSION
    dataset_hash: str = ""
    split_hash: str = ""
    seed: int = 0
    fractions: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    assignments: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)

    def lists(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {name.value: [] for name in SplitName}
        for prompt_id, split in self.assignments.items():
            out.setdefault(split, []).append(prompt_id)
        for ids in out.values():
            ids.sort()
        return out

    def prompt_ids(self, split: SplitName | str) -> list[str]:
        value = split.value if isinstance(split, SplitName) else str(split)
        return sorted(pid for pid, s in self.assignments.items() if s == value)

    def split_of(self, prompt_id: str) -> str | None:
        return self.assignments.get(prompt_id)

    # -- io ---------------------------------------------------------------- #

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def save(self, path: Path) -> None:
        write_json(path, self.to_payload())

    @classmethod
    def load(cls, path: Path) -> SplitAssignment | None:
        payload = read_json(path, default=None)
        if not isinstance(payload, dict):
            return None
        try:
            return cls.model_validate(payload)
        except Exception:
            return None



def validate_split(assignment: SplitAssignment, prompts: Iterable[SyntheticPrompt]) -> list[str]:
    """Return every problem with ``assignment``; empty means it is a valid partition.

    The critical invariant: every prompt id belongs to exactly one split, and the
    three splits are pairwise disjoint and together cover every prompt.
    """

    problems: list[str] = []
    ids = [p.prompt_id for p in prompts]
    seen: dict[str, str] = {}
    for prompt_id in ids:
        if prompt_id in seen:
            problems.append(f"prompt id '{prompt_id}' appears more than once in the prompt set")
        seen[prompt_id] = ""

    valid = {name.value for name in SplitName}
    for prompt_id, split in assignment.assignments.items():
        if split not in valid:
            problems.append(f"prompt '{prompt_id}' is assigned to unknown split '{split}'")

    missing = [pid for pid in ids if pid not in assignment.assignments]
    if missing:
        problems.append(f"{len(missing)} prompt(s) have no split assignment, e.g. {missing[:3]}")
    unknown = [pid for pid in assignment.assignments if pid not in set(ids)]
    if unknown:
        problems.append(f"{len(unknown)} assigned prompt id(s) are not in the prompt set")

    lists = assignment.lists()
    buckets: dict[str, str] = {}
    for split, prompt_ids in lists.items():
        for prompt_id in prompt_ids:
            if prompt_id in buckets:
                problems.append(
                    f"prompt '{prompt_id}' leaks across splits: "
                    f"'{buckets[prompt_id]}' and '{split}'"
                )
            buckets[prompt_id] = split

    for split, prompt_ids in lists.items():
        expected = assignment.counts.get(split)
        if expected is not None and expected != len(prompt_ids):
            problems.append(
                f"split '{split}' counts {expected} but holds {len(prompt_ids)} prompts"
            )

    if assignment.split_hash and assignment.split_hash != content_hash(assignment.assignments):
        problems.append("split_hash does not match the assignment it describes")
    return problems


def _targets(
    total: int,
    *,
    eval_fraction: float,
    validation_fraction: float,
    min_eval_prompts: int,
    min_train_prompts: int,
) -> dict[str, int]:
    """Per-split prompt counts, honouring the configured minimums."""

    if total <= 0:
        return {"train": 0, "validation": 0, "eval": 0}
    n_eval = max(int(min_eval_prompts), int(round(total * eval_fraction)))
    n_val = int(round(total * validation_fraction))
    n_eval = min(n_eval, total)
    n_val = min(n_val, total - n_eval)

    while n_eval + n_val > total - int(min_train_prompts):
        if n_val > 0:
            n_val -= 1
        elif n_eval > 1:
            n_eval -= 1
        else:
            break

    # Give validation a seat when the configuration asked for one and there is
    # room without starving training.
    if (
        n_val == 0
        and validation_fraction > 0
        and total - n_eval >= 2
        and total - n_eval - 1 >= int(min_train_prompts)
    ):
        n_val = 1

    return {"train": total - n_eval - n_val, "validation": n_val, "eval": n_eval}


def plan_splits(
    prompts: list[SyntheticPrompt],
    *,
    eval_fraction: float,
    validation_fraction: float,
    min_eval_prompts: int,
    min_train_prompts: int,
    seed: int,
    dataset_hash: str = "",
    run_id: str = "",
) -> SplitAssignment:
    """Deterministically assign every prompt to exactly one split, stratified by category."""

    by_category: dict[str, list[str]] = {}
    for prompt in prompts:
        by_category.setdefault(prompt.category or "ordinary", []).append(prompt.prompt_id)

    order: list[str] = []
    for category in sorted(by_category):
        ids = sorted(by_category[category])
        random.Random(f"{seed}:{category}").shuffle(ids)
        order.extend(ids)

    targets = _targets(
        len(prompts),
        eval_fraction=eval_fraction,
        validation_fraction=validation_fraction,
        min_eval_prompts=min_eval_prompts,
        min_train_prompts=min_train_prompts,
    )
    remaining = dict(targets)
    assignments: dict[str, str] = {}
    for prompt_id in order:
        # Fill proportionally to each split's target so every category is spread
        # across the splits instead of being consumed by whichever split happens
        # to be largest first.
        candidates = [split for split in _FILL_ORDER if targets.get(split, 0) > 0]
        open_splits = [split for split in candidates if remaining.get(split, 0) > 0]
        pool = open_splits or candidates
        if not pool:
            assignments[prompt_id] = "train"
            continue
        pick = max(
            pool,
            key=lambda split: (
                remaining.get(split, 0) / targets[split],
                -_FILL_ORDER.index(split),
            ),
        )
        remaining[pick] -= 1
        assignments[prompt_id] = pick

    counts = {split: sum(1 for s in assignments.values() if s == split) for split in _FILL_ORDER}
    return SplitAssignment(
        run_id=run_id,
        version=SPLIT_VERSION,
        dataset_hash=dataset_hash,
        split_hash=content_hash(assignments),
        seed=int(seed),
        fractions={"eval": float(eval_fraction), "validation": float(validation_fraction)},
        counts=counts,
        assignments=assignments,
    )


def load_or_plan(
    paths: RunPaths,
    prompts: list[SyntheticPrompt],
    *,
    generation: GenerationConfig,
    dataset_hash: str,
    run_id: str = "",
) -> SplitAssignment:
    """Reuse ``splits.json`` when the fingerprint matches, otherwise re-split."""

    existing = SplitAssignment.load(paths.split_manifest_path)
    if existing is not None and existing.dataset_hash == dataset_hash:
        if not validate_split(existing, prompts):
            return existing
    assignment = plan_splits(
        prompts,
        eval_fraction=generation.eval_fraction,
        validation_fraction=generation.validation_fraction,
        min_eval_prompts=generation.min_eval_prompts,
        min_train_prompts=generation.min_train_prompts,
        seed=generation.seed,
        dataset_hash=dataset_hash,
        run_id=run_id,
    )
    assignment.save(paths.split_manifest_path)
    return assignment


def apply_splits(prompts: list[SyntheticPrompt], assignment: SplitAssignment) -> None:
    """Stamp each prompt with the split it was assigned."""

    for prompt in prompts:
        split = assignment.split_of(prompt.prompt_id)
        if split:
            prompt.split = split


__all__ = [
    "SPLIT_VERSION",
    "SplitAssignment",
    "apply_splits",
    "load_or_plan",
    "plan_splits",
    "validate_split",
]
