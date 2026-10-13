"""The complete, persisted configuration of an experiment.

A :class:`RunConfig` is written to ``<run>/config.json`` when the run is created
and is **immutable thereafter** — resuming a run always loads the saved config,
never the current UI state, so an experiment can never silently change its base
model, revision, prompt, dataset, or hyperparameters.

:meth:`RunConfig.fingerprint` covers everything that would invalidate cached
generations or judgements.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from autorlhf.config.character import CharacterSpec
from autorlhf.config.rubric import Rubric
from autorlhf.core.enums import (
    FeedbackMode,
    TrainingMethod,
    TrainingObjective,
)
from autorlhf.core.errors import ConfigError
from autorlhf.core.ids import content_hash

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


class HFModelConfig(BaseModel):
    """Which Hugging Face model is trained, pinned to a revision."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    #: Pinned commit/branch.  ``None`` means "whatever ``main`` is at start
    #: time", recorded into the manifest as a resolved sha where possible.
    revision: str | None = None
    #: ``auto`` | ``float32`` | ``bfloat16`` | ``float16``
    torch_dtype: str = "auto"
    #: ``sdpa`` | ``eager`` | ``flash_attention_2``
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    #: Opt in to the documented generic chat template when a tokenizer ships
    #: none.  Off by default: it is a fallback, not an equivalent.
    allow_generic_chat_template: bool = False
    #: Extra kwargs merged into ``from_pretrained``.
    load_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_id must not be empty")
        return value.strip()

    def resolved_revision_label(self) -> str:
        return self.revision or "main (unpinned)"


# --------------------------------------------------------------------------- #
# generation / judging
# --------------------------------------------------------------------------- #


class GenerationConfig(BaseModel):
    """Dataset preparation and OpenRouter request settings."""

    #: OpenRouter model used to write synthetic prompts and teacher responses.
    teacher_model: str = "openai/gpt-4o-mini"
    #: Jev model id used for every decision.
    judge_model: str = "typesafe/jev-1.13"

    #: How many synthetic prompts to generate in total (before split).
    num_prompts: int = 24
    #: Candidate responses generated per prompt.
    candidates_per_prompt: int = 2
    #: ``teacher`` (OpenRouter), ``local`` (the HF model), or ``mixed``.
    candidate_source: str = "teacher"

    #: Held-out fraction of prompts.  Splitting happens at *prompt* level.
    eval_fraction: float = 0.2
    validation_fraction: float = 0.15
    #: Hard minimums so a small run still has usable splits.
    min_eval_prompts: int = 3
    min_train_prompts: int = 4

    #: Fraction of generated prompts deliberately probing edge cases /
    #: boundaries, as opposed to ordinary interactions.
    edge_case_fraction: float = 0.35

    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 0
    #: Concurrent OpenRouter requests.
    max_concurrency: int = 4
    request_timeout_s: float = 120.0
    max_retries: int = 4
    #: Keep the records the UI hides in non-verbose mode.
    persist_verbose: bool = True

    @field_validator("candidate_source")
    @classmethod
    def _source(cls, value: str) -> str:
        if value not in {"teacher", "local", "mixed"}:
            raise ValueError("candidate_source must be one of: teacher, local, mixed")
        return value

    @field_validator("candidates_per_prompt", "num_prompts")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("eval_fraction", "validation_fraction", "edge_case_fraction")
    @classmethod
    def _fraction(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("fraction must be within [0, 1)")
        return value

    def validate_generation(self) -> list[str]:
        problems: list[str] = []
        if self.eval_fraction + self.validation_fraction >= 1.0:
            problems.append("eval_fraction + validation_fraction must be < 1.0")
        if self.num_prompts * self.eval_fraction < self.min_eval_prompts:
            problems.append(
                f"num_prompts={self.num_prompts} with eval_fraction={self.eval_fraction} "
                f"yields fewer than min_eval_prompts={self.min_eval_prompts}; "
                "increase num_prompts or eval_fraction"
            )
        remaining = self.num_prompts - max(
            self.min_eval_prompts, int(round(self.num_prompts * self.eval_fraction))
        )
        if remaining < self.min_train_prompts:
            problems.append(
                f"only ~{remaining} prompts would remain for training "
                f"(min_train_prompts={self.min_train_prompts}); increase num_prompts"
            )
        if self.max_concurrency < 1:
            problems.append("max_concurrency must be >= 1")
        if self.max_retries < 0:
            problems.append("max_retries must be >= 0")
        return problems


class EvaluationConfig(BaseModel):
    """Held-out evaluation settings."""

    #: Cap on held-out prompts evaluated (all are used when smaller).
    max_eval_prompts: int = 12
    #: Pairwise base-vs-tuned judgements in addition to per-response scores.
    pairwise: bool = True
    #: Number of independent judge passes per response.  >1 averages out noise
    #: but costs proportionally more.
    judge_repeats: int = 1
    #: Include the seed conversations' user turns as extra held-out prompts.
    include_seed_prompts: bool = True
    #: Below this many scored responses the report is marked inconclusive.
    min_responses_for_claims: int = 6


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


class Hyperparameters(BaseModel):
    """Every tunable the trainers consume.

    Defaults are conservative but *valid*; feasibility against real hardware is
    decided by :mod:`autorlhf.config.presets`, never guessed here.
    """

    method: TrainingMethod = TrainingMethod.LORA
    objective: TrainingObjective = TrainingObjective.DPO

    # -- LoRA / QLoRA ------------------------------------------------------ #
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=list)
    #: ``none`` | ``all`` | ``lora_only``
    lora_bias: str = "none"
    use_dora: bool = False

    # -- quantization ------------------------------------------------------ #
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # -- optimisation ------------------------------------------------------ #
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    seed: int = 0
    bf16: bool = False
    fp16: bool = False

    # -- checkpointing / evaluation --------------------------------------- #
    save_steps: int = 25
    eval_steps: int = 25
    logging_steps: int = 5
    save_total_limit: int = 3

    # -- objective specific ------------------------------------------------ #
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    #: Run an SFT warm-up before DPO when supervised data exists.
    sft_warmup: bool = True
    sft_epochs: float = 1.0
    sft_learning_rate: float | None = None
    #: SFT loss masking (`trl` `completion_only_loss`).
    sft_completion_only_loss: bool = True

    # -- bookkeeping ------------------------------------------------------- #
    preset_id: str | None = None
    run_name: str = ""

    # -- validation -------------------------------------------------------- #

    def validate_hyperparameters(self) -> list[str]:
        problems: list[str] = []
        if self.learning_rate <= 0:
            problems.append("learning_rate must be > 0")
        if self.per_device_train_batch_size < 1:
            problems.append("per_device_train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            problems.append("gradient_accumulation_steps must be >= 1")
        if self.max_steps == 0 or self.max_steps < -1:
            problems.append("max_steps must be -1 (use epochs) or >= 1")
        if self.num_train_epochs <= 0 and self.max_steps < 1:
            problems.append("need num_train_epochs > 0 or max_steps >= 1")
        if self.max_seq_length < 64:
            problems.append("max_seq_length must be >= 64")
        if self.save_steps < 1 or self.eval_steps < 1 or self.logging_steps < 1:
            problems.append("save_steps, eval_steps and logging_steps must be >= 1")
        if self.save_total_limit < 1:
            problems.append("save_total_limit must be >= 1")
        if not 0.0 <= self.lora_dropout < 1.0:
            problems.append("lora_dropout must be within [0, 1)")
        if self.method in (TrainingMethod.LORA, TrainingMethod.QLORA):
            if self.lora_r < 1:
                problems.append("lora_r must be >= 1")
            if self.lora_alpha < 1:
                problems.append("lora_alpha must be >= 1")
            if self.lora_bias not in {"none", "all", "lora_only"}:
                problems.append("lora_bias must be one of: none, all, lora_only")
            # An empty `lora_target_modules` is allowed here on purpose: the
            # module names can only be discovered from the model itself, and the
            # initial configuration screen must not download weights.  The
            # trainer discovers them from the real module names and raises
            # ``TargetModulesError`` when nothing matches, so training can never
            # silently proceed with zero adapted modules.
        if self.method is TrainingMethod.QLORA:
            if not self.load_in_4bit:
                problems.append("QLoRA requires load_in_4bit=True")
            if self.bnb_4bit_quant_type not in {"nf4", "fp4"}:
                problems.append("bnb_4bit_quant_type must be 'nf4' or 'fp4'")
            if self.bnb_4bit_compute_dtype not in {"bfloat16", "float16", "float32"}:
                problems.append("bnb_4bit_compute_dtype must be a torch dtype name")
        if self.method is TrainingMethod.FFT and self.load_in_4bit:
            problems.append("full fine-tuning cannot be combined with 4-bit loading")
        if self.bf16 and self.fp16:
            problems.append("bf16 and fp16 are mutually exclusive")
        if (
            self.method in (TrainingMethod.QLORA,)
            and self.fp16
            and self.bnb_4bit_compute_dtype == "bfloat16"
        ):
            problems.append("fp16 training with a bfloat16 compute dtype is inconsistent")
        if self.objective is TrainingObjective.DPO:
            if not 0.0 < self.dpo_beta <= 1.0:
                problems.append("dpo_beta must be within (0, 1]")
            if self.dpo_loss_type not in {"sigmoid", "hinge", "ipo", "exo_pair", "bco_pair"}:
                problems.append(f"unsupported dpo_loss_type '{self.dpo_loss_type}'")
            # DPO on a LoRA adapter needs a reference path; running the adapter
            # disabled as the reference is the supported TRL route.
            if self.gradient_accumulation_steps < 2 and self.per_device_train_batch_size < 2:
                problems.append(
                    "DPO needs an effective batch size >= 2 so chosen/rejected pairs "
                    "land in the same step; raise gradient_accumulation_steps"
                )
        if self.lr_scheduler_type not in {
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        }:
            problems.append(f"unknown lr_scheduler_type '{self.lr_scheduler_type}'")
        if self.optim not in {
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_8bit",
            "paged_adamw_8bit",
            "paged_adamw_32bit",
            "adafactor",
            "sgd",
        }:
            problems.append(f"unknown optim '{self.optim}'")
        return problems

    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def assert_valid(self) -> None:
        problems = self.validate_hyperparameters()
        if problems:
            raise ConfigError(
                "hyperparameters are invalid: " + "; ".join(problems),
                hint="Fix them on the Training screen; presets give a valid starting point.",
            )

    def resolved_sft_lr(self) -> float:
        return self.sft_learning_rate if self.sft_learning_rate else self.learning_rate * 10

    def fingerprint(self) -> str:
        return content_hash(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #


class RunConfig(BaseModel):
    """Everything needed to reproduce and resume a run."""

    schema_version: int = SCHEMA_VERSION
    feedback_mode: FeedbackMode = FeedbackMode.AUTOMATED
    slug: str = "run"

    model: HFModelConfig = Field(default_factory=HFModelConfig)
    character: CharacterSpec = Field(default_factory=CharacterSpec)
    rubric: Rubric = Field(default_factory=Rubric)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    training: Hyperparameters = Field(default_factory=Hyperparameters)

    #: Log training metrics to W&B (credentials live in the secret file).
    wandb_enabled: bool = False
    wandb_project: str = ""
    #: Trace OpenRouter/Jev calls to Weave.  Requires explicit consent because
    #: prompts, responses and judgements leave the machine.
    weave_enabled: bool = False
    weave_project: str = ""
    #: Recorded when the user acknowledged the Weave data-egress notice.
    weave_consent: bool = False

    verbose: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    # -- validation -------------------------------------------------------- #

    def validate_config(self) -> list[str]:
        problems: list[str] = []
        if not self.model.model_id.strip():
            problems.append("no Hugging Face model id configured")
        if self.feedback_mode is FeedbackMode.INTERACTIVE and self.generation.num_prompts < 1:
            problems.append("interactive review needs at least one prompt")
        problems.extend(f"character: {p}" for p in self.character.validate_spec())
        problems.extend(f"rubric: {p}" for p in self.rubric.validate_rubric())
        problems.extend(f"generation: {p}" for p in self.generation.validate_generation())
        problems.extend(f"training: {p}" for p in self.training.validate_hyperparameters())
        if self.weave_enabled and not self.weave_consent:
            problems.append(
                "weave_enabled requires weave_consent=True (prompts and judgements "
                "would be sent to an external service)"
            )
        return problems

    def assert_valid(self) -> None:
        problems = self.validate_config()
        if problems:
            raise ConfigError(
                "run configuration is invalid:\n  - " + "\n  - ".join(problems),
                hint="Fix the listed fields; the Diagnostics action re-runs validation.",
            )

    # -- derived ----------------------------------------------------------- #

    @property
    def objective(self) -> TrainingObjective:
        return self.training.objective

    @property
    def method(self) -> TrainingMethod:
        return self.training.method

    def fingerprint(self) -> str:
        """Hash of everything that would change generated/judged content.

        Runtime toggles (verbose, tracking) are excluded so flipping them does
        not invalidate a cache.
        """

        payload = self.model_dump(mode="json", exclude={"verbose", "created_at", "slug"})
        return content_hash(payload)

    def dataset_fingerprint(self) -> str:
        """Hash of the inputs that determine the dataset, for split reuse."""

        return content_hash(
            {
                "character": self.character.fingerprint(),
                "generation": {
                    k: v
                    for k, v in self.generation.model_dump(mode="json").items()
                    # These only affect retrieval/concurrency, not content.
                    if k not in {"max_concurrency", "request_timeout_s", "persist_verbose"}
                },
                "teacher_model": self.generation.teacher_model,
            }
        )

    def hyperparameter_fingerprint(self) -> str:
        return self.training.fingerprint()

    def summary_rows(self) -> list[tuple[str, str]]:
        """Compact ``(label, value)`` pairs for dashboard/confirmation views."""

        return [
            ("Run slug", self.slug),
            ("Feedback mode", self.feedback_mode.value),
            ("HF model", f"{self.model.model_id} @ {self.model.resolved_revision_label()}"),
            ("Objective", self.training.objective.value.upper()),
            ("Method", self.training.method.value),
            ("Teacher model", self.generation.teacher_model),
            ("Judge (Jev) model", self.generation.judge_model),
            (
                "Prompts / candidates",
                f"{self.generation.num_prompts} × {self.generation.candidates_per_prompt}",
            ),
            (
                "Splits (eval/val)",
                f"{self.generation.eval_fraction:.0%} / {self.generation.validation_fraction:.0%}",
            ),
            ("LoRA r / alpha", f"{self.training.lora_r} / {self.training.lora_alpha}"),
            ("Learning rate", f"{self.training.learning_rate:g}"),
            (
                "Epochs / max steps",
                f"{self.training.num_train_epochs:g} / {self.training.max_steps}",
            ),
            (
                "Batch × accum",
                f"{self.training.per_device_train_batch_size} × {self.training.gradient_accumulation_steps}",
            ),
            ("Max seq len", str(self.training.max_seq_length)),
            ("DPO beta", f"{self.training.dpo_beta:g}"),
            ("SFT warm-up", "yes" if self.training.sft_warmup else "no"),
            ("W&B / Weave", f"{self.wandb_enabled} / {self.weave_enabled}"),
        ]

    # -- io ---------------------------------------------------------------- #

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RunConfig:
        return cls.model_validate(json.loads(text))

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json() + "\n", encoding="utf-8")
        tmp.replace(path)


__all__ = [
    "SCHEMA_VERSION",
    "EvaluationConfig",
    "GenerationConfig",
    "HFModelConfig",
    "Hyperparameters",
    "RunConfig",
]
