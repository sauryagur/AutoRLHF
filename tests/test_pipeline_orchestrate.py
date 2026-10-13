"""PipelineRunner: preflight, dataset preparation, review, planning, evaluation.

Everything network-shaped is faked; the training stack is reached through the
real modules but with model loading and generation monkeypatched, so nothing is
downloaded and no GPU is needed.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from autorlhf.config.character import CharacterSpec
from autorlhf.config.paths import AppPaths
from autorlhf.config.run_config import Hyperparameters, RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.enums import (
    FeedbackMode,
    RunStatus,
    SampleStatus,
    TrainingMethod,
    TrainingObjective,
)
from autorlhf.core.errors import ConfigError, CredentialError, InsufficientDataError
from autorlhf.core.events import EventBus
from autorlhf.core.ids import run_folder_name
from autorlhf.core.types import HardwareInfo, ResourceEstimate
from autorlhf.llm.schemas import (
    CandidateResponse,
    CriterionScore,
    EvalResponse,
    Judgement,
    PairDecision,
    SyntheticPrompt,
)
from autorlhf.pipeline import evaluate as evaluate_module
from autorlhf.pipeline import orchestrate as orchestrate_module
from autorlhf.pipeline.context import PipelineContext
from autorlhf.pipeline.dataset import rewrite_records, sft_record
from autorlhf.pipeline.evaluate import EvalModelHandle
from autorlhf.pipeline.models import DatasetBundle, ReviewDecision
from autorlhf.pipeline.orchestrate import PipelineRunner
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import RunPaths, read_json, read_jsonl

RUN_ID = "20260101-120000-ab12"

CPU_HW = HardwareInfo(
    device_type="cpu",
    device_name="test cpu",
    device_count=0,
    total_ram_mib=15000,
    available_ram_mib=9000,
    cpu_count=4,
    platform="linux",
)

GPU_HW = HardwareInfo(
    device_type="cuda",
    device_name="Test GPU",
    device_count=1,
    total_vram_mib=8192,
    free_vram_mib=7000,
    total_ram_mib=32000,
    cpu_count=8,
    bf16_supported=True,
    bitsandbytes_available=True,
    platform="linux",
)


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #


class StubTrainingPlan(BaseModel):
    """A stand-in used only if ``autorlhf.training.plan`` is unavailable."""

    run_id: str
    objective: TrainingObjective
    method: TrainingMethod
    hyperparameters: Hyperparameters
    preset_id: str | None = None
    rationale: list[str] = Field(default_factory=list)
    jev_ranking: dict | None = None
    resource_estimate: ResourceEstimate | None = None
    hardware: HardwareInfo
    sft_examples: list[dict] = Field(default_factory=list)
    preference_pairs: list[dict] = Field(default_factory=list)
    train_count: int = 0
    validation_count: int = 0
    warnings: list[str] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def plan_type():
    """The real ``TrainingPlan`` when it exists, otherwise the local stand-in."""

    try:
        from autorlhf.training.plan import TrainingPlan
    except ImportError:  # pragma: no cover - only before agent B lands
        return StubTrainingPlan
    return TrainingPlan


def make_judgement(prompt_id: str, candidate_id: str, overall: float) -> Judgement:
    return Judgement(
        judgement_id=f"j-{candidate_id}",
        run_id=RUN_ID,
        prompt_id=prompt_id,
        candidate_id=candidate_id,
        criterion_scores={
            "character_consistency": CriterionScore(
                criterion_id="character_consistency",
                criterion_name="Character consistency",
                score=overall,
            )
        },
        overall=overall,
        judge_model="typesafe/jev-1.13",
        judge_build="typesafe/jev-1.13-20260917",
    )


class StubJev:
    """A JevEvaluator stand-in with deterministic, controllable answers."""

    def __init__(
        self,
        *,
        decision_winner: str = "a",
        ranking: tuple[str | None, dict] | None = None,
        base_overall: float = 2.0,
        finetuned_overall: float = 3.5,
    ) -> None:
        self.run_id = RUN_ID
        self.decision_winner = decision_winner
        self.ranking = ranking
        self.base_overall = base_overall
        self.finetuned_overall = finetuned_overall
        self.score_calls = 0
        self.compare_calls = 0
        self.rank_calls = 0

    def score_response(self, *, prompt_id, candidate, prompt, context=()):
        self.score_calls += 1
        if candidate.candidate_id.startswith("base:"):
            overall = self.base_overall
        elif candidate.candidate_id.startswith("finetuned:"):
            overall = self.finetuned_overall
        else:
            overall = 4.0 if candidate.candidate_id.endswith("c0") else 2.0
        return make_judgement(prompt_id, candidate.candidate_id, overall)

    def compare_responses(self, **kwargs):
        self.compare_calls += 1
        a = kwargs["option_a"]
        b = kwargs["option_b"]
        overall_a = kwargs.get("overall_a")
        overall_b = kwargs.get("overall_b")
        margin = overall_a - overall_b if overall_a is not None and overall_b is not None else 2.0
        return PairDecision(
            pair_id=f"{kwargs['prompt_id']}-{a.candidate_id}-vs-{b.candidate_id}",
            run_id=RUN_ID,
            prompt_id=kwargs["prompt_id"],
            option_a_id=a.candidate_id,
            option_b_id=b.candidate_id,
            winner=self.decision_winner,  # type: ignore[arg-type]
            choice_raw=self.decision_winner,
            probabilities={"a": 0.8, "b": 0.2},
            confidence=0.8,
            margin=margin,
            judge_model="typesafe/jev-1.13",
            judge_build="typesafe/jev-1.13-20260917",
        )

    def rank_presets(self, *, state, options):
        self.rank_calls += 1
        if self.ranking is not None:
            return self.ranking
        return None, {"options": options, "chosen": None, "error": "no ranking configured"}

    def check_condition(self, *, instructions, state, key="condition"):
        return 0.9, {"probability": 0.9}


class FakeGenerator:
    """Replaces CandidateGenerator so no model is ever loaded."""

    instances: list[FakeGenerator] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = kwargs["config"]
        self.existing_seen: list[CandidateResponse] | None = None
        self.closed = False
        FakeGenerator.instances.append(self)

    def generate(self, prompts, *, existing=None):
        self.existing_seen = existing
        if existing:
            return list(existing)
        out: list[CandidateResponse] = []
        for prompt in prompts:
            for index in range(self.config.generation.candidates_per_prompt):
                out.append(
                    CandidateResponse(
                        candidate_id=f"{prompt.prompt_id}-c{index}",
                        prompt_id=prompt.prompt_id,
                        source="teacher",
                        model_id=self.config.generation.teacher_model,
                        text=f"answer {index} for {prompt.prompt_id}",
                        status=SampleStatus.PENDING,
                    )
                )
        return out

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #


def build_config(character: CharacterSpec, rubric) -> RunConfig:
    config = RunConfig(
        slug="test-run",
        feedback_mode=FeedbackMode.AUTOMATED,
        character=character,
        rubric=rubric,
    )
    config.generation.num_prompts = 10
    config.generation.candidates_per_prompt = 2
    config.generation.eval_fraction = 0.3
    config.generation.validation_fraction = 0.15
    config.training.method = TrainingMethod.LORA
    config.training.objective = TrainingObjective.DPO
    config.training.lora_target_modules = ["c_attn"]
    config.training.max_steps = 2
    return config


def make_ctx(
    tmp_path: Path,
    *,
    character: CharacterSpec,
    rubric,
    key: str | None = "sk-or-v1-test",
    config: RunConfig | None = None,
    interactive: bool = False,
) -> PipelineContext:
    paths = AppPaths.for_tests(tmp_path).ensure()
    env = Environment(paths)
    if key:
        env.secrets.update({"OPENROUTER_API_KEY": key})
        env.refresh()
    store = RunStore(paths.db_path)
    config = config or build_config(character, rubric)
    folder = run_folder_name(RUN_ID, config.slug)
    store.create_run(config, folder=folder, run_id=RUN_ID)
    run_paths = RunPaths(paths.run_dir(folder)).ensure()
    return PipelineContext(
        env=env,
        store=store,
        paths=run_paths,
        config=config,
        bus=EventBus(),
        interactive=interactive,
    )


@pytest.fixture(autouse=True)
def fast_manifest(monkeypatch):
    """Keep manifest refreshes from importing torch."""

    monkeypatch.setattr("autorlhf.storage.runs.runtime_versions", lambda: {"python": "3.12"})


@pytest.fixture()
def fake_plan_type(monkeypatch):
    monkeypatch.setattr(orchestrate_module, "_training_plan_type", plan_type)
    return plan_type()


def fake_prompts(count: int = 10) -> list[SyntheticPrompt]:
    categories = (
        "ordinary",
        "varied_phrasing",
        "context_following",
        "multi_turn",
        "edge_case",
        "boundary",
    )
    return [
        SyntheticPrompt(
            prompt_id=f"p{index + 1:04d}",
            category=categories[index % len(categories)],
            text=f"synthetic prompt {index}",
        )
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #


def test_jev_property_builds_the_client_without_deadlocking(tmp_path, minimal_character, rubric):
    """``ctx.jev`` builds the OpenRouter client while holding the context lock."""

    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric)
    built: dict[str, object] = {}

    def _build() -> None:
        built["jev"] = ctx.jev

    worker = threading.Thread(target=_build, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), "ctx.jev deadlocked"
    assert built["jev"].client is ctx.openrouter
    assert ctx.jev is built["jev"]  # cached, not rebuilt


def test_preflight_requires_credentials(tmp_path, minimal_character, rubric):
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric, key=None)
    runner = PipelineRunner(ctx)
    with pytest.raises(CredentialError):
        runner.preflight()


def test_preflight_raises_config_error_for_an_invalid_config(tmp_path, minimal_character, rubric):
    config = build_config(minimal_character, rubric)
    config.character = CharacterSpec(name="broken")
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric, config=config)
    with pytest.raises(ConfigError) as excinfo:
        PipelineRunner(ctx).preflight()
    assert "system prompt" in str(excinfo.value)


def test_preflight_reports_hardware_and_model_metadata(
    tmp_path, minimal_character, rubric, monkeypatch
):
    monkeypatch.setattr(orchestrate_module, "_detect_hardware", lambda: CPU_HW)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: 494_000_000)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric)
    report = PipelineRunner(ctx).preflight()
    assert report["ok"] is True
    assert report["hardware"]["device_type"] == "cpu"
    assert report["param_count"] == 494_000_000
    assert report["judge_model"] == ctx.config.generation.judge_model
    assert report["config_fingerprint"]


def test_preflight_degrades_when_hardware_detection_fails(
    tmp_path, minimal_character, rubric, monkeypatch
):
    def _boom():
        raise RuntimeError("no torch")

    monkeypatch.setattr(orchestrate_module, "_detect_hardware", _boom)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: None)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric)
    report = PipelineRunner(ctx).preflight()
    assert report["ok"] is True
    assert report["hardware"] is None
    assert any("hardware detection failed" in w for w in report["warnings"])


# --------------------------------------------------------------------------- #
# training plan
# --------------------------------------------------------------------------- #


def _write_sft_example(paths: RunPaths) -> None:
    rewrite_records(
        paths.sft_path,
        [
            sft_record(
                run_id=RUN_ID,
                example_id="s0001",
                prompt_id="seed-1",
                split="train",
                messages=[
                    {"role": "system", "content": "You are a terse test assistant."},
                    {"role": "user", "content": "Say hi."},
                    {"role": "assistant", "content": "Hi."},
                ],
                source="seed",
            )
        ],
    )


def test_build_training_plan_falls_back_to_sft_without_pairs(
    tmp_path, minimal_character, rubric, monkeypatch, fake_plan_type
):
    monkeypatch.setattr(orchestrate_module, "_detect_hardware", lambda: CPU_HW)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: None)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric)
    _write_sft_example(ctx.paths)
    bundle = DatasetBundle(run_id=RUN_ID)

    plan = PipelineRunner(ctx).build_training_plan(bundle)
    assert isinstance(plan, fake_plan_type)
    assert plan.objective is TrainingObjective.SFT
    assert plan.preset_id == "cpu_tiny_lora"
    assert plan.train_count == 1
    assert any("too few preference pairs" in w for w in plan.warnings)
    assert any("switching to SFT" in line for line in plan.rationale)
    assert plan.hardware.device_type == "cpu"
    assert plan.resource_estimate is not None
    assert plan.sft_examples[0]["example_id"] == "s0001"


def test_build_training_plan_raises_when_there_is_no_data(
    tmp_path, minimal_character, rubric, monkeypatch, fake_plan_type
):
    monkeypatch.setattr(orchestrate_module, "_detect_hardware", lambda: CPU_HW)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: None)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric)
    with pytest.raises(InsufficientDataError) as excinfo:
        PipelineRunner(ctx).build_training_plan(DatasetBundle(run_id=RUN_ID))
    message = str(excinfo.value)
    assert "0 valid preference pair" in message
    assert "0 supervised examples" in message


def test_build_training_plan_uses_jev_ranking_when_several_presets_fit(
    tmp_path, minimal_character, rubric, monkeypatch, fake_plan_type
):
    monkeypatch.setattr(orchestrate_module, "_detect_hardware", lambda: GPU_HW)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: 500_000_000)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric)
    _write_sft_example(ctx.paths)
    stub = StubJev(
        ranking=(
            "cpu_tiny_lora",
            {
                "options": {"cpu_tiny_lora": "CPU / tiny LoRA"},
                "chosen": "cpu_tiny_lora",
                "probabilities": {"cpu_tiny_lora": 0.6, "single_gpu_safe_lora": 0.4},
                "confidence": 0.55,
                "model_build": "typesafe/jev-1.13-20260917",
                "error": None,
            },
        )
    )
    monkeypatch.setattr(PipelineContext, "jev", property(lambda self: stub))

    plan = PipelineRunner(ctx).build_training_plan(DatasetBundle(run_id=RUN_ID))
    assert stub.rank_calls == 1
    assert plan.preset_id == "cpu_tiny_lora"
    assert plan.jev_ranking is not None
    assert plan.jev_ranking["chosen"] == "cpu_tiny_lora"
    assert any("Jev ranked" in line for line in plan.rationale)
    assert any("not a probability that training will succeed" in line for line in plan.rationale)


# --------------------------------------------------------------------------- #
# dataset preparation
# --------------------------------------------------------------------------- #


@pytest.fixture()
def prepared(tmp_path, minimal_character, rubric, monkeypatch):
    monkeypatch.setattr(orchestrate_module, "_detect_hardware", lambda: CPU_HW)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: None)
    calls = {"prompts": 0}

    def _generate_prompts(**kwargs):
        calls["prompts"] += 1
        return fake_prompts(10)

    monkeypatch.setattr(orchestrate_module, "generate_prompts", _generate_prompts)
    FakeGenerator.instances.clear()
    monkeypatch.setattr(orchestrate_module, "CandidateGenerator", FakeGenerator)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric)
    stub = StubJev()
    monkeypatch.setattr(PipelineContext, "jev", property(lambda self: stub))
    runner = PipelineRunner(ctx)
    bundle = runner.prepare_dataset()
    return SimpleNamespace(
        ctx=ctx,
        runner=runner,
        bundle=bundle,
        stub=stub,
        calls=calls,
        paths=ctx.paths,
        events=ctx.bus.history,
    )


def test_prepare_dataset_emits_verbose_generation_stages(prepared):
    logs = [event for event in prepared.events() if event.kind == "log"]
    assert logs
    assert all(event.verbose_only for event in logs)
    messages = [event.message for event in logs]
    assert any("dataset generation started" in message for message in messages)
    assert any("prompt generation complete: 10 prompts" in message for message in messages)
    assert any("prompt splits assigned" in message for message in messages)
    assert any("candidate generation complete: 20 candidates" in message for message in messages)
    assert any("candidate judging complete: 20 judgements" in message for message in messages)
    assert any("dataset records written: 20 candidates" in message for message in messages)
    assert any("supervised fine-tuning records written" in message for message in messages)


def test_prepare_dataset_writes_every_artifact(prepared):
    paths = prepared.paths
    assert paths.dataset_path.is_file()
    assert paths.eval_prompts_path.is_file()
    assert paths.sft_path.is_file()
    assert paths.split_manifest_path.is_file()

    records = read_jsonl(paths.dataset_path)
    assert len(records) == 20
    assert {record["record_type"] for record in records} == {"candidate"}
    assert all(record["judgement"] is not None for record in records)
    assert all(record["status"] == "accepted" for record in records)
    assert {record["split"] for record in records} == {"train", "validation", "eval"}

    eval_prompts = read_jsonl(paths.eval_prompts_path)
    assert len(eval_prompts) >= 3
    assert all(record["record_type"] == "eval_prompt" for record in eval_prompts)

    splits = read_json(paths.split_manifest_path)
    assert sum(splits["counts"].values()) == 10
    assert splits["dataset_hash"] == prepared.ctx.config.dataset_fingerprint()

    judgements = read_jsonl(paths.judgments_path)
    assert len(judgements) == 20
    assert all(record["kind"] == "score" for record in judgements)

    from autorlhf.pipeline.dataset import validate_dataset

    assert validate_dataset(records) == []
    assert prepared.ctx.store.counters(RUN_ID)["dataset_problems"] == 0


def test_prepare_dataset_updates_counters_and_status(prepared):
    counters = prepared.ctx.store.counters(RUN_ID)
    assert counters["prompts_generated"] == 10
    assert counters["candidates_generated"] == 20
    assert counters["generation_failed"] == 0
    assert counters["judgements_ok"] == 20
    assert counters["accepted"] == 20
    assert counters["sft_examples"] == 11  # 10 accepted candidates + 1 seed
    assert counters["split_train"] == 5
    assert counters["split_eval"] == 3
    assert prepared.ctx.store.get_run(RUN_ID).status is RunStatus.GENERATING
    assert prepared.bundle.train_prompt_ids
    assert prepared.bundle.eval_prompt_ids
    assert prepared.bundle.counters["judgements_ok"] == 20


def test_prepare_dataset_is_idempotent_on_resume(prepared):
    first_records = read_jsonl(prepared.paths.dataset_path)
    stub_calls = prepared.stub.score_calls
    prepared.runner.prepare_dataset(resume=True)
    assert prepared.calls["prompts"] == 1  # prompts were reused, not regenerated
    assert prepared.stub.score_calls == stub_calls  # judgements were reused
    assert len(read_jsonl(prepared.paths.dataset_path)) == len(first_records)
    assert len(read_jsonl(prepared.paths.judgments_path)) == 20  # no duplicates


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #


def test_run_review_builds_pairs_and_supervised_examples(prepared):
    bundle = prepared.runner.run_review(prepared.bundle)
    records = read_jsonl(prepared.paths.preferences_path)
    # train (5) + validation (2) prompts each contribute one pair
    assert len(records) == 7
    assert all(record["record_type"] == "preference" for record in records)
    assert all(record["source"] == "jev" for record in records)
    assert all(record["chosen_id"] != record["rejected_id"] for record in records)
    assert all(
        record["audit"]["thresholds"]["tie_epsilon"] == prepared.ctx.config.rubric.tie_epsilon
        for record in records
    )

    assert len(bundle.preference_pairs) == 7
    assert len(bundle.pair_decisions) == 7
    counters = prepared.ctx.store.counters(RUN_ID)
    assert counters["pairs_accepted"] == 7
    assert counters["pairs_rejected_tie"] == 0

    decisions = read_jsonl(prepared.paths.judgments_path)
    assert any(record["kind"] == "choice" for record in decisions)
    assert prepared.ctx.store.get_run(RUN_ID).status is RunStatus.REVIEWING

    sft = read_jsonl(prepared.paths.sft_path)
    assert any(record["source"] == "seed" for record in sft)
    assert any(record["source"] == "accepted" for record in sft)
    assert all(record["messages"][0]["role"] == "system" for record in sft)


def test_run_review_interactive_applies_human_decisions(
    tmp_path, minimal_character, rubric, monkeypatch
):
    """A TUI-style reviewer thread drives the queue while run_review blocks."""

    monkeypatch.setattr(orchestrate_module, "_detect_hardware", lambda: CPU_HW)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: None)
    monkeypatch.setattr(orchestrate_module, "generate_prompts", lambda **kw: fake_prompts(10))
    monkeypatch.setattr(orchestrate_module, "CandidateGenerator", FakeGenerator)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric, interactive=True)
    stub = StubJev()
    monkeypatch.setattr(PipelineContext, "jev", property(lambda self: stub))
    runner = PipelineRunner(ctx)
    bundle = runner.prepare_dataset()

    stop = threading.Event()
    reviewed: list[str] = []

    def _reviewer() -> None:
        while not stop.is_set():
            for item in ctx.review_queue.pending():
                chosen = item.candidates[1].candidate_id
                ctx.review_queue.decide(
                    ReviewDecision(item_id=item.item_id, action="choose_b", chosen_id=chosen)
                )
                reviewed.append(item.item_id)
            time.sleep(0.02)

    worker = threading.Thread(target=_reviewer, daemon=True)
    worker.start()
    try:
        bundle = runner.run_review(bundle)
    finally:
        stop.set()
        worker.join(timeout=5)

    assert reviewed
    assert len(bundle.preference_pairs) == 7
    assert all(pair.source == "human" for pair in bundle.preference_pairs)
    records = read_jsonl(ctx.paths.preferences_path)
    assert all(record["source"] == "human" for record in records)
    assert all(record["reviewer"] == "human" for record in records)
    assert all(record["pair_decision"] is not None for record in records)
    assert ctx.store.counters(RUN_ID)["pairs_accepted"] == 7
    assert any("human review decisions" in warning for warning in bundle.warnings)


def test_run_review_trains_on_the_humans_edited_text(
    tmp_path, minimal_character, rubric, monkeypatch
):
    """A human edit must reach sft.jsonl and preferences.jsonl, not just the dataset."""

    monkeypatch.setattr(orchestrate_module, "_detect_hardware", lambda: CPU_HW)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: None)
    monkeypatch.setattr(orchestrate_module, "generate_prompts", lambda **kw: fake_prompts(10))
    monkeypatch.setattr(orchestrate_module, "CandidateGenerator", FakeGenerator)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric, interactive=True)
    monkeypatch.setattr(PipelineContext, "jev", property(lambda self: StubJev()))
    runner = PipelineRunner(ctx)
    bundle = runner.prepare_dataset()

    stop = threading.Event()
    edited = ["HUMAN EDITED text"]

    def _reviewer() -> None:
        while not stop.is_set():
            for item in ctx.review_queue.pending():
                ctx.review_queue.decide(
                    ReviewDecision(
                        item_id=item.item_id,
                        action="edit",
                        edited_texts={item.candidates[0].candidate_id: edited[0]},
                    )
                )
            time.sleep(0.02)

    worker = threading.Thread(target=_reviewer, daemon=True)
    worker.start()
    try:
        runner.run_review(bundle)
    finally:
        stop.set()
        worker.join(timeout=5)

    sft = [r for r in read_jsonl(ctx.paths.sft_path) if r["source"] == "human"]
    assert edited[0] in {record["messages"][-1]["content"] for record in sft}

    pairs = read_jsonl(ctx.paths.preferences_path)
    assert all(record["chosen_text"] == edited[0] for record in pairs)


def test_plan_training_persists_and_announces_the_plan(prepared, fake_plan_type):
    bundle = prepared.runner.run_review(prepared.bundle)
    plan = prepared.runner.plan_training(bundle)
    assert plan.objective is TrainingObjective.DPO
    assert plan.preset_id == "cpu_tiny_lora"
    plan_path = prepared.paths.root / "training_plan.json"
    assert plan_path.is_file()
    stored = read_json(plan_path)
    assert stored["preset_id"] == "cpu_tiny_lora"
    assert stored["objective"] == "dpo"
    assert prepared.ctx.store.counters(RUN_ID)["train_count"] == plan.train_count

    sft_train = [
        record for record in read_jsonl(prepared.paths.sft_path) if record["split"] == "train"
    ]
    train_pairs = [
        record
        for record in read_jsonl(prepared.paths.preferences_path)
        if record["split"] == "train"
    ]
    validation_pairs = [
        record
        for record in read_jsonl(prepared.paths.preferences_path)
        if record["split"] == "validation"
    ]
    assert plan.train_count == len(sft_train) + len(train_pairs)
    assert plan.validation_count == len(validation_pairs)
    assert len(plan.sft_examples) == len(sft_train)
    assert len(plan.preference_pairs) == len(train_pairs) + len(validation_pairs)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


@pytest.fixture()
def eval_ready(prepared, monkeypatch):
    prepared.runner.run_review(prepared.bundle)

    def _load_eval_model(*, config, variant, adapter_path=None, hf_token=None, device=None):
        return EvalModelHandle(
            variant=variant,
            model=None,
            tokenizer=None,
            model_id="tiny",
            adapter_path=str(adapter_path) if adapter_path else None,
        )

    def _generate(handle, prompt, **kwargs):
        return EvalResponse(
            response_id=f"{handle.variant}:{prompt.prompt_id}",
            prompt_id=prompt.prompt_id,
            variant=handle.variant,
            text=f"{handle.variant} answer for {prompt.prompt_id}",
        )

    monkeypatch.setattr(orchestrate_module, "load_eval_model", _load_eval_model)
    monkeypatch.setattr(evaluate_module, "generate_eval_response", _generate)
    return prepared


def test_run_baseline_eval_writes_a_report(eval_ready):
    report = eval_ready.runner.run_baseline_eval()
    assert report.variant == "base"
    assert (
        report.scored_responses == len(report.responses) == len(eval_ready.bundle.eval_prompt_ids)
    )
    assert report.overall_mean == pytest.approx(2.0)
    assert eval_ready.paths.baseline_eval_path.is_file()
    stored = read_json(eval_ready.paths.baseline_eval_path)
    assert stored["variant"] == "base"
    assert stored["eval_prompts_hash"]
    assert eval_ready.ctx.store.get_run(RUN_ID).status is RunStatus.EVALUATING_BASE


def test_run_final_eval_compares_against_the_baseline(eval_ready, tmp_path):
    eval_ready.runner.run_baseline_eval()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    report, comparison = eval_ready.runner.run_final_eval(adapter_path=adapter)
    assert report.variant == "finetuned"
    assert eval_ready.paths.final_eval_path.is_file()
    assert eval_ready.paths.comparison_path.is_file()

    assert comparison.base_overall == pytest.approx(2.0)
    assert comparison.finetuned_overall == pytest.approx(3.5)
    assert comparison.overall_delta == pytest.approx(1.5)
    assert comparison.criteria[0].delta == pytest.approx(1.5)
    assert comparison.n_scored_base == report.scored_responses
    assert (
        comparison.pairwise_base_wins
        + comparison.pairwise_finetuned_wins
        + comparison.pairwise_ties
        + comparison.pairwise_inconclusive
        == len(report.responses)
    )
    # Four held-out prompts is below min_responses_for_claims, so no claim is made.
    assert comparison.inconclusive is True
    assert any("coverage" in w for w in comparison.warnings)
    stored = read_json(eval_ready.paths.comparison_path)
    assert stored["inconclusive"] is True
    assert eval_ready.ctx.store.get_run(RUN_ID).status is RunStatus.EVALUATING_FINAL


def test_run_final_eval_requires_a_baseline(eval_ready, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(InsufficientDataError):
        eval_ready.runner.run_final_eval(adapter_path=adapter)


def test_run_baseline_eval_requires_eval_prompts(tmp_path, minimal_character, rubric, monkeypatch):
    monkeypatch.setattr(orchestrate_module, "_detect_hardware", lambda: CPU_HW)
    monkeypatch.setattr(orchestrate_module, "_param_count", lambda cfg, token: None)
    ctx = make_ctx(tmp_path, character=minimal_character, rubric=rubric)
    with pytest.raises(InsufficientDataError):
        PipelineRunner(ctx).run_baseline_eval()
