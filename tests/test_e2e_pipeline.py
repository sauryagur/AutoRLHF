"""End-to-end integration test: generate -> judge -> curate -> train -> evaluate.

This is the lead-owned acceptance harness.  It drives the *real* pipeline,
storage, and trainers against:

* a fake OpenRouter/Jev backend (no network, no paid calls), and
* a tiny real causal LM on CPU (built offline by ``tests/conftest.py``).

The fake judge is deliberately opinionated: every second candidate generated for
a prompt is marked ``EXCELLENT`` and the others ``MEDIOCRE``, and the fake scores
them 4.0 and 2.0 with high confidence.  That yields a decisive preference whose
margin is far above the rubric's tie epsilon, so the run must construct real
preference pairs and invoke the real DPO trainer.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from autorlhf.config.character import CharacterSpec, SeedConversation
from autorlhf.config.paths import AppPaths
from autorlhf.config.rubric import default_rubric
from autorlhf.config.run_config import RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.enums import (
    FeedbackMode,
    RunStatus,
    SampleStatus,
    TrainingMethod,
    TrainingObjective,
)
from autorlhf.core.events import EventBus
from autorlhf.core.ids import new_run_id, run_folder_name
from autorlhf.core.types import ChatMessage, GenerationResult
from autorlhf.llm.schemas import (
    ChoiceAnswer,
    DecisionsResponse,
    DecisionsUsage,
    NoulAnswer,
    ScoreAnswer,
)
from autorlhf.pipeline.context import PipelineContext
from autorlhf.pipeline.models import ReviewDecision
from autorlhf.pipeline.orchestrate import PipelineRunner
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import (
    RunPaths,
    export_run,
    file_hash,
    load_config,
    read_json,
    read_jsonl,
    save_config,
)

# --------------------------------------------------------------------------- #
# fake remote backend
# --------------------------------------------------------------------------- #

EXCELLENT = "EXCELLENT-REPLY"
MEDIOCRE = "MEDIOCRE-REPLY"
EXCELLENT_SCORE = 4.0
MEDIOCRE_SCORE = 2.0

FAKE_KEY = "sk-or-v1-fake-key-for-tests-only"


def _requested_prompts(messages: Any) -> tuple[str, int]:
    """Parse the requested category and count out of the prompt-generation request."""

    text = _collect_text(messages)
    category = "ordinary"
    match = re.search(r"Category:\s*([A-Za-z_]+)", text)
    if match:
        category = match.group(1)
    count = 2
    match = re.search(r"Write\s+(\d+)\s+distinct", text)
    if match:
        count = max(1, min(8, int(match.group(1))))
    return category, count


def _collect_text(value: Any) -> str:
    """Flatten every string inside a nested state object.

    Handles pydantic models (``ChatMessage`` and friends) as well as plain
    dicts/strings, because the pipeline passes both.
    """

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_collect_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_collect_text(item) for item in value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _collect_text(dump())
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    return ""


class FakeOpenRouterClient:
    """Stand-in for :class:`autorlhf.llm.client.OpenRouterClient`."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 4,
        max_concurrency: int = 4,
        cache: Any = None,
        run_id: str = "",
        app_title: str = "autorlhf",
        trace: Any = None,
    ) -> None:
        assert api_key, "the pipeline must hand the client a key"
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.cache = cache
        self.run_id = run_id
        self.trace = trace
        self.chat_calls = 0
        self.decide_calls = 0
        self._per_prompt: dict[str, int] = {}
        self.closed = False

    # -- text generation --------------------------------------------------- #

    def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage] | list[dict],
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
        seed: int | None = None,
        response_format: dict | None = None,
    ) -> GenerationResult:
        self.chat_calls += 1
        key = _collect_text(messages)[:200]
        index = self._per_prompt.get(key, 0)
        self._per_prompt[key] = index + 1
        marker = EXCELLENT if index % 2 else MEDIOCRE
        text = f"{marker} Reply {index}: a plausible in-character answer."
        return GenerationResult(
            request_id=f"fake-chat-{self.chat_calls}",
            model_id=model,
            text=text,
            source=model,
            prompt_tokens=64,
            completion_tokens=len(text.split()),
            latency_s=0.01,
        )

    def chat_json(
        self,
        *,
        model: str,
        messages: list[ChatMessage] | list[dict],
        schema_hint: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        seed: int | None = None,
        response_format: dict | None = None,
    ) -> tuple[dict[str, Any], GenerationResult]:
        self.chat_calls += 1
        category, count = _requested_prompts(messages)
        # Honour the requested category and count: the generator fills a per
        # category quota and stops as soon as the quota is met, so over-delivering
        # (or mislabelling categories) would inflate the prompt set.
        items = [
            {
                "text": f"Synthetic {category} prompt #{self.chat_calls}.{index}",
                "category": category,
                "context": [{"role": "user", "content": "hello"}]
                if category in {"context_following", "multi_turn"}
                else [],
            }
            for index in range(count)
        ]
        payload = {"prompts": items}
        return payload, GenerationResult(
            request_id=f"fake-chatjson-{self.chat_calls}",
            model_id=model,
            text=json.dumps(payload),
            source=model,
            prompt_tokens=128,
            completion_tokens=64,
            latency_s=0.01,
        )

    # -- Jev decisions ----------------------------------------------------- #

    def decide(self, request: Any) -> DecisionsResponse:
        self.decide_calls += 1
        state_text = _collect_text(request.state)
        answers: dict[str, Any] = {}
        for key, question in request.questions.items():
            kind = getattr(question, "type", None)
            if kind == "score":
                levels = list(getattr(question, "criteria", []) or [])
                top = str(max(0, len(levels) - 1))
                if EXCELLENT in state_text:
                    score = EXCELLENT_SCORE
                    probabilities = {top: 0.9, "0": 0.0}
                else:
                    score = MEDIOCRE_SCORE
                    probabilities = {"2": 0.8, top: 0.1}
                answers[key] = ScoreAnswer(
                    type="score",
                    score=score,
                    legend={str(i): level for i, level in enumerate(levels)},
                    probabilities=probabilities,
                    confidence=0.92,
                )
            elif kind == "choice":
                options = list((getattr(question, "criteria", {}) or {}).keys())
                if {"a", "b", "tie"} <= set(options):
                    a_text = _collect_text(request.state.get("response_a", ""))
                    b_text = _collect_text(request.state.get("response_b", ""))
                    if EXCELLENT in a_text and MEDIOCRE in b_text:
                        winner, probs = "a", {"a": 0.85, "b": 0.14, "tie": 0.01}
                    elif EXCELLENT in b_text and MEDIOCRE in a_text:
                        winner, probs = "b", {"a": 0.14, "b": 0.85, "tie": 0.01}
                    else:
                        winner, probs = "tie", {"a": 0.4, "b": 0.4, "tie": 0.2}
                    answers[key] = ChoiceAnswer(
                        type="choice", choice=winner, probabilities=probs, confidence=0.8
                    )
                else:
                    chosen = options[0] if options else "none"
                    rest = max(1, len(options) - 1)
                    answers[key] = ChoiceAnswer(
                        type="choice",
                        choice=chosen,
                        probabilities={
                            option: (0.7 if option == chosen else 0.3 / rest) for option in options
                        },
                        confidence=0.7,
                    )
            else:
                answers[key] = NoulAnswer(type="noul", noul=0.9)
        return DecisionsResponse(
            id=f"fake-decision-{self.decide_calls}",
            model="typesafe/jev-1.13-fake-20260101",
            provider="FakeProvider",
            answers=answers,
            usage=DecisionsUsage(input_tokens=256, output_tokens=16, cost=0.00003),
        )

    def decide_many(
        self, requests: Any, *, on_progress: Any = None, cancel_event: Any = None
    ) -> list[Any]:
        out = []
        for request in requests:
            out.append(self.decide(request))
            if on_progress is not None:
                on_progress(len(out))
        return out

    def verify_credentials(self) -> tuple[bool, str]:
        return True, "fake credentials accepted"

    def usage_summary(self) -> dict[str, float]:
        return {
            "calls": float(self.chat_calls + self.decide_calls),
            "input_tokens": 0.0,
            "cost_usd": 0.0,
        }

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def e2e_env(tmp_path: Path) -> Environment:
    env = Environment(AppPaths.for_tests(tmp_path))
    env.secrets.update({"OPENROUTER_API_KEY": FAKE_KEY})
    env.refresh()
    return env


@pytest.fixture()
def e2e_config(tiny_chat_model_dir: Path) -> RunConfig:
    character = CharacterSpec(
        name="Aria the Lighthouse Keeper",
        system_prompt=(
            "You are Aria, a kindly lighthouse keeper. Answer briefly and in character. "
            "Never mention being an AI, and never discuss programming."
        ),
        behavioral_constraints=[
            "Never mention being an AI.",
            "Refuse programming questions.",
            "Answer in at most three sentences.",
        ],
        seed_conversations=[
            SeedConversation(
                messages=[
                    ChatMessage(role="user", content="Will the fog lift today?"),
                    ChatMessage(role="assistant", content="Not before the tide turns, I reckon."),
                ]
            )
        ],
    )
    config = RunConfig(
        slug="e2e",
        feedback_mode=FeedbackMode.AUTOMATED,
        character=character,
        rubric=default_rubric(),
    )
    config.model.model_id = str(tiny_chat_model_dir)
    config.model.revision = None
    config.model.torch_dtype = "float32"
    config.model.attn_implementation = "eager"

    config.generation.teacher_model = "fake/teacher"
    config.generation.judge_model = "typesafe/jev-1.13"
    config.generation.num_prompts = 8
    config.generation.candidates_per_prompt = 2
    config.generation.candidate_source = "teacher"
    config.generation.eval_fraction = 0.25
    config.generation.validation_fraction = 0.25
    config.generation.min_eval_prompts = 2
    config.generation.min_train_prompts = 2
    config.generation.max_concurrency = 2
    # Greedy local generation with a modest budget: the 512-token context of the
    # tiny fixture would be overrun by the default 512 new tokens, and its random
    # weights can otherwise emit an empty completion.  Greedy + a small budget
    # keeps the acceptance test deterministic and judged.
    config.generation.temperature = 0.0
    config.generation.max_new_tokens = 24
    config.evaluation.max_eval_prompts = 8
    config.evaluation.pairwise = True

    config.training.method = TrainingMethod.LORA
    config.training.objective = TrainingObjective.DPO
    config.training.lora_target_modules = ["c_attn", "c_proj"]
    config.training.lora_r = 4
    config.training.lora_alpha = 8
    config.training.lora_dropout = 0.0
    config.training.learning_rate = 1e-3
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 2
    config.training.max_steps = 2
    config.training.num_train_epochs = 1.0
    config.training.max_seq_length = 96
    config.training.save_steps = 2
    config.training.eval_steps = 2
    config.training.logging_steps = 1
    config.training.gradient_checkpointing = False
    config.training.bf16 = False
    config.training.fp16 = False
    config.training.sft_warmup = False
    config.verbose = True
    assert config.validate_config() == [], config.validate_config()
    return config


@pytest.fixture()
def e2e_run(e2e_env: Environment, e2e_config: RunConfig) -> tuple[str, RunPaths]:
    """Create the run exactly the way the CLI and the TUI do."""

    run_id = new_run_id()
    folder = run_folder_name(run_id, e2e_config.slug)
    paths = RunPaths(e2e_env.paths.run_dir(folder)).ensure()
    store = RunStore(e2e_env.paths.db_path)
    record = store.create_run(
        e2e_config, run_id=run_id, folder=folder, config_path=paths.config_path
    )
    save_config(paths, e2e_config)
    store.close()
    return record.id, paths


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeOpenRouterClient]:
    """Patch the client the pipeline constructs, and hand back the class."""

    monkeypatch.setattr("autorlhf.pipeline.context.OpenRouterClient", FakeOpenRouterClient)
    return FakeOpenRouterClient


def _make_context(
    env: Environment, run_id: str, paths: RunPaths, *, interactive: bool = False
) -> tuple[PipelineContext, EventBus]:
    store = RunStore(env.paths.db_path)
    bus = EventBus()
    ctx = PipelineContext(
        env=env,
        store=store,
        paths=paths,
        config=load_config(paths),
        bus=bus,
        interactive=interactive,
        verbose=True,
    )
    assert ctx.run_id == run_id, "the context must resolve the run's SQLite row"
    return ctx, bus


# --------------------------------------------------------------------------- #
# generate -> judge -> curate
# --------------------------------------------------------------------------- #


def test_prepare_dataset_review_and_baseline(
    e2e_env: Environment, e2e_run: tuple[str, RunPaths], fake_client: Any
) -> None:
    run_id, paths = e2e_run
    ctx, _bus = _make_context(e2e_env, run_id, paths)
    runner = PipelineRunner(ctx)
    try:
        bundle = runner.prepare_dataset()

        # -- prompts, candidates and splits --------------------------------- #
        records = read_jsonl(paths.dataset_path)
        assert records, "dataset.jsonl must contain candidate records"
        candidates = [r for r in records if r["record_type"] == "candidate"]
        n_prompts = len(bundle.prompts)
        assert n_prompts >= 4
        assert len(candidates) == n_prompts * ctx.config.generation.candidates_per_prompt

        split_file = read_json(paths.split_manifest_path)
        assert split_file, "splits.json must be persisted"
        assignments = split_file["assignments"]
        assert set(assignments) == {p.prompt_id for p in bundle.prompts}
        assert set(assignments.values()) <= {"train", "validation", "eval"}

        eval_ids = {pid for pid, split in assignments.items() if split == "eval"}
        assert eval_ids, "the split must produce held-out prompts"
        for record in candidates:
            # No prompt may appear in two splits, so each candidate's split must
            # equal its prompt's split.
            assert record["split"] == assignments[record["prompt_id"]]

        # -- judgements ----------------------------------------------------- #
        judged = [r for r in candidates if r.get("judgement")]
        assert len(judged) == len(candidates), "every candidate must carry a judgement"
        for record in judged:
            assert record["overall"] is not None
            assert record["criterion_scores"]
            assert not record["judgement"]["error"]

        # -- curation (automated) ------------------------------------------- #
        reviewed = runner.run_review(bundle)
        pairs = read_jsonl(paths.preferences_path)
        assert pairs, "automated review must build preference pairs"
        for pair in pairs:
            assert pair["chosen_id"] != pair["rejected_id"]
            assert pair["chosen_text"] and pair["rejected_text"]
            assert pair["source"] == "jev"
            # Pairs may come from train or validation prompts (validation pairs
            # evaluate DPO during training); held-out eval prompts are never
            # used to build preference data.
            assert pair["split"] in {"train", "validation"}, pair["split"]
            assert pair["split"] != "eval"
            assert pair["margin"] is not None and pair["margin"] > 0
            assert pair["judge_confidence"] is not None
            assert pair["audit"]["thresholds"]
        assert reviewed.counters["pairs_accepted"] >= 1
        sft_records = read_jsonl(paths.sft_path)
        assert sft_records, "supervised examples must be written for the SFT path"

        # -- baseline evaluation on the held-out prompts --------------------- #
        baseline = runner.run_baseline_eval()
        assert baseline.variant == "base"
        assert baseline.scored_responses > 0
        assert (
            baseline.failed_judgements + baseline.scored_responses == baseline.requested_judgements
        )
        assert baseline.criterion_means
        assert baseline.overall_mean is not None
        assert paths.baseline_eval_path.is_file()

        eval_prompts = [
            r for r in read_jsonl(paths.eval_prompts_path) if r.get("record_type") == "eval_prompt"
        ]
        # Held-out = the split's eval prompts plus any seed-derived prompts; both
        # are never used for training.
        assert len(eval_prompts) >= len(eval_ids)
        assert eval_ids <= {r["prompt_id"] for r in eval_prompts}, (
            "the split's eval prompts must all appear in eval_prompts.jsonl"
        )
        assert baseline.eval_prompts_hash == file_hash(paths.eval_prompts_path)

        # -- training plan --------------------------------------------------- #
        plan = runner.build_training_plan(bundle)
        assert plan.objective is TrainingObjective.DPO
        assert plan.method is TrainingMethod.LORA
        assert plan.preference_pairs, "the DPO plan must carry the real preference pairs"
        assert plan.hyperparameters.lora_target_modules
        assert plan.rationale
        assert plan.resource_estimate is not None

        # -- audit trail and status transitions ------------------------------ #
        assert read_jsonl(paths.judgments_path)
        assert ctx.store.counters(run_id)["judgements_ok"] >= 1
        statuses = [t["to_status"] for t in ctx.store.transitions(run_id)]
        assert RunStatus.PREPARING_DATA.value in statuses
        assert RunStatus.REVIEWING.value in statuses

        # -- resume is idempotent: no paid work is repeated ------------------ #
        calls_before = ctx.openrouter.chat_calls + ctx.openrouter.decide_calls
        runner.prepare_dataset()
        runner.run_review(bundle)
        assert ctx.openrouter.chat_calls + ctx.openrouter.decide_calls == calls_before, (
            "re-running the dataset/review stages must reuse cached results"
        )
    finally:
        ctx.close()


# --------------------------------------------------------------------------- #
# interactive review
# --------------------------------------------------------------------------- #


def test_interactive_review_blocks_and_records_human_override(
    e2e_env: Environment, e2e_run: tuple[str, RunPaths], fake_client: Any
) -> None:
    """Interactive mode must block on the queue, and human decisions must win."""

    run_id, paths = e2e_run
    ctx, _bus = _make_context(e2e_env, run_id, paths, interactive=True)
    runner = PipelineRunner(ctx)
    result: dict[str, Any] = {}
    try:
        bundle = runner.prepare_dataset()

        def _work() -> None:
            try:
                result["bundle"] = runner.run_review(bundle)
            except BaseException as exc:  # surface worker failures in the test
                result["error"] = exc

        worker = threading.Thread(target=_work, daemon=True)
        worker.start()

        overridden: dict[str, Any] = {}
        decided: set[str] = set()
        deadline = time.monotonic() + 120
        while worker.is_alive() and time.monotonic() < deadline:
            pending = [i for i in ctx.review_queue.pending() if i.item_id not in decided]
            if pending:
                item = pending[0]
                decided.add(item.item_id)
                if not overridden:
                    # Disagree with Jev: pick the candidate it ranked worse.
                    chosen = item.candidates[-1]
                    rejected = item.candidates[0]
                    overridden = {
                        "chosen": chosen.candidate_id,
                        "rejected": rejected.candidate_id,
                        "item": item.item_id,
                        "recommended": item.recommended_candidate_id,
                    }
                    ctx.review_queue.decide(
                        ReviewDecision(
                            item_id=item.item_id,
                            reviewer="test-human",
                            # ``accept`` with explicit ids selects by candidate id
                            # (``choose_a``/``choose_b`` mean the pair's A/B roles).
                            action="accept",
                            chosen_id=chosen.candidate_id,
                            rejected_id=rejected.candidate_id,
                            scores={
                                c.candidate_id: {"character_consistency": 5.0}
                                for c in item.candidates
                            },
                            comments="I prefer this one.",
                            created_at="2026-01-01T00:00:00",
                        )
                    )
                else:
                    chosen_id = item.recommended_candidate_id or item.candidates[0].candidate_id
                    rejected_id = next(
                        c.candidate_id for c in item.candidates if c.candidate_id != chosen_id
                    )
                    ctx.review_queue.decide(
                        ReviewDecision(
                            item_id=item.item_id,
                            reviewer="test-human",
                            action="accept",
                            chosen_id=chosen_id,
                            rejected_id=rejected_id,
                            scores={
                                c.candidate_id: {"character_consistency": 4.0}
                                for c in item.candidates
                            },
                            created_at="2026-01-01T00:00:00",
                        )
                    )
                continue
            time.sleep(0.02)

        worker.join(timeout=60)
        assert "error" not in result, result.get("error")
        assert not worker.is_alive(), "the review stage must not hang once all items are decided"
        assert overridden, "the review queue must have offered items to review"
        assert ctx.review_queue.stats().get("completed", 0) >= 1

        # The human override is persisted and used for the preference data.
        records = read_jsonl(paths.dataset_path)
        by_id = {r["candidate_id"]: r for r in records if r["record_type"] == "candidate"}
        assert by_id[overridden["chosen"]]["status"] == SampleStatus.ACCEPTED.value
        assert by_id[overridden["rejected"]]["status"] == SampleStatus.REJECTED.value
        assert by_id[overridden["chosen"]]["reviewer"] == "test-human"

        pairs = read_jsonl(paths.preferences_path)
        human_pairs = [p for p in pairs if p["source"] == "human"]
        assert human_pairs, "human decisions must produce human-sourced preference pairs"
        assert any(p["reviewer"] == "test-human" for p in human_pairs)
        assert any(
            p["prompt_id"] == overridden["item"] and p["chosen_id"] == overridden["chosen"]
            for p in human_pairs
        ), "the human's choice must be the chosen side of the pair"
        for pair in human_pairs:
            # Jev's recommendation must be kept for the audit trail, not discarded.
            assert pair["pair_decision"]["choice_raw"] in {"a", "b", "tie"}
            assert pair["pair_decision"]["judge_model"] == "typesafe/jev-1.13"
            assert pair["audit"]["thresholds"]
    finally:
        ctx.cancel_event.set()
        ctx.close()


# --------------------------------------------------------------------------- #
# self-describing run + credential-free export
# --------------------------------------------------------------------------- #


def test_run_is_self_describing_and_exports_without_secrets(
    e2e_env: Environment, e2e_run: tuple[str, RunPaths], fake_client: Any, tmp_path: Path
) -> None:
    run_id, paths = e2e_run
    ctx, _bus = _make_context(e2e_env, run_id, paths)
    runner = PipelineRunner(ctx)
    try:
        runner.prepare_dataset()
        ctx.store.update_status(run_id, RunStatus.PAUSED, note="test pause")

        record = ctx.store.require_run(run_id)
        assert record.status is RunStatus.PAUSED
        assert record.model_id == str(ctx.config.model.model_id)
        assert run_id in {r.id for r in ctx.store.incomplete_runs()}
        assert RunStatus.PAUSED.value in [t["to_status"] for t in ctx.store.transitions(run_id)]

        manifest = read_json(paths.manifest_path)
        assert manifest["base_model"]["model_id"] == str(ctx.config.model.model_id)
        assert manifest["judge_model"] == "typesafe/jev-1.13"
        assert manifest["teacher_model"] == "fake/teacher"
        assert manifest["training"]["objective"] == "dpo"
        assert manifest["fingerprints"]["config"]
        assert manifest["runtime_versions"]["python"]
        assert FAKE_KEY not in json.dumps(manifest)

        destination = export_run(
            run_id=run_id,
            paths=paths,
            config=ctx.config,
            destination=tmp_path / "export",
        )
        blob = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in destination.rglob("*")
            if path.is_file()
        )
        assert FAKE_KEY not in blob, "the export must never contain a credential"
        assert "sk-or-v1" not in blob
        assert (destination / "config.json").is_file()
        assert (destination / "manifest.json").is_file()
        assert (destination / "dataset.jsonl").is_file()
    finally:
        ctx.close()


# --------------------------------------------------------------------------- #
# train (real DPO through TRL) then compare base with fine-tuned
# --------------------------------------------------------------------------- #


def test_train_dpo_then_compare_base_and_finetuned(
    e2e_env: Environment, e2e_run: tuple[str, RunPaths], fake_client: Any
) -> None:
    """The acceptance path: real preference data -> real DPO trainer -> comparison."""

    from autorlhf.storage.runs import (
        is_adapter_complete,
        is_checkpoint_complete,
        list_checkpoint_dirs,
    )
    from autorlhf.training.jobs import TrainingJob
    from autorlhf.training.trainers import find_resume_checkpoint

    run_id, paths = e2e_run
    ctx, bus = _make_context(e2e_env, run_id, paths)
    runner = PipelineRunner(ctx)
    try:
        bundle = runner.prepare_dataset()
        bundle = runner.run_review(bundle)
        baseline = runner.run_baseline_eval()
        plan = runner.build_training_plan(bundle)
        assert plan.objective is TrainingObjective.DPO

        job = TrainingJob(
            run_id=run_id,
            paths=paths,
            config=ctx.config,
            plan=plan,
            bus=bus,
            cancel_event=ctx.cancel_event,
            pause_event=ctx.pause_event,
            env=e2e_env,
            store=ctx.store,
        )
        job.request_resume()  # clears the pause flag; a no-op for a fresh run
        result = job.run()

        assert result.error is None, result.error
        assert result.status in {"completed", "interrupted"}, result.status
        assert result.steps_completed >= 1
        assert result.adapter_path, "training must produce an adapter"
        adapter = Path(result.adapter_path)
        assert is_adapter_complete(adapter), f"incomplete adapter directory: {adapter}"

        # -- checkpoints are real and resumable ------------------------------ #
        # Trainers write into checkpoints/<objective>/ so the two objectives
        # never collide (checkpoint_root in trainers.py).
        ckpt_root = paths.checkpoints_dir / TrainingObjective.DPO.value
        assert ckpt_root.is_dir(), f"no DPO checkpoint root: {ckpt_root}"
        checkpoint_dirs = list_checkpoint_dirs(ckpt_root)
        assert checkpoint_dirs, "training must write at least one checkpoint"
        complete = [p for p in checkpoint_dirs if is_checkpoint_complete(p)]
        assert complete, "at least one checkpoint must be marked complete"
        resume_from = find_resume_checkpoint(paths, objective=TrainingObjective.DPO)
        assert resume_from is not None and resume_from in complete, (
            f"resume={resume_from} complete={[str(p) for p in complete]} "
            f"result_ckpts={result.checkpoints} ckpt_root={ckpt_root}"
        )

        # -- metrics reached SQLite and the bus ------------------------------ #
        metric_names = ctx.store.metric_names(run_id)
        assert any("loss" in name for name in metric_names), metric_names
        losses = ctx.store.metrics(run_id, next(n for n in metric_names if "loss" in n))
        assert losses and all(isinstance(step, int) for step, _ in losses)

        # -- final evaluation reuses the identical held-out prompts ---------- #
        final, comparison = runner.run_final_eval(adapter_path=adapter)
        assert final.variant == "finetuned"
        assert final.scored_responses > 0
        assert final.eval_prompts_hash == baseline.eval_prompts_hash, (
            "base and fine-tuned evaluations must use the same held-out prompts"
        )
        assert comparison.n_prompts == len(baseline.responses)
        assert comparison.criteria, "the comparison must report per-criterion means"
        for criterion in comparison.criteria:
            assert criterion.base_mean is not None
            assert criterion.finetuned_mean is not None
            assert criterion.delta is not None
        assert comparison.overall_delta is not None
        assert comparison.rubric_fingerprint == ctx.config.rubric.fingerprint()
        assert paths.comparison_path.is_file()
        assert paths.final_eval_path.is_file()

        # -- the run is now self-describing and complete --------------------- #
        record = ctx.store.require_run(run_id)
        assert record.status in {RunStatus.COMPLETED, RunStatus.EVALUATING_FINAL}
        manifest = read_json(paths.manifest_path)
        assert manifest["datasets"]["dataset"]["hash"]
        assert manifest["datasets"]["preferences"]["hash"]
        assert manifest["last_complete_checkpoint"]
        assert FAKE_KEY not in json.dumps(manifest)
    finally:
        ctx.close()
