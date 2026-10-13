"""JevEvaluator tests: request shape and response parsing, with the SDK faked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autorlhf.config.character import DEFAULT_CHARACTER
from autorlhf.core.errors import JudgeError, RemoteAPIError
from autorlhf.llm.client import DECISIONS_SERVER_URL, OpenRouterClient
from autorlhf.llm.jev import MAX_STATE_CHARS, RAW_TRUNCATION_KEY, TRUNCATION_MARKER, JevEvaluator
from autorlhf.llm.schemas import CandidateResponse
from autorlhf.storage.cache import ResponseCache

JUDGE = "typesafe/jev-1.13"


class FakeDecisions:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, BaseException):
            raise result
        return result


def make_client(decisions) -> OpenRouterClient:
    client = OpenRouterClient("sk-or-v1-test", max_retries=0)
    client._sdk = SimpleNamespace(alpha=SimpleNamespace(decisions=decisions))
    return client


def make_evaluator(decisions, *, rubric, character=None, cache=None, max_retries=0) -> JevEvaluator:
    client = make_client(decisions)
    return JevEvaluator(
        client,
        rubric,
        character or DEFAULT_CHARACTER,
        judge_model=JUDGE,
        cache=cache,
        run_id="run-1",
        max_retries=max_retries,
    )


def candidate(text: str = "Not before the afternoon tide.") -> CandidateResponse:
    return CandidateResponse(
        candidate_id="p0001-c0", prompt_id="p0001", text=text, source="teacher"
    )


def score_response_for(rubric, *, omit_optional: bool = False) -> dict:
    answers = {}
    for criterion in rubric.criteria:
        answer = {"type": "score", "score": float(criterion.scale_max)}
        if not omit_optional:
            answer["legend"] = {str(i): level for i, level in enumerate(criterion.levels)}
            answer["probabilities"] = {
                str(i): 1.0 / len(criterion.levels) for i in range(len(criterion.levels))
            }
            answer["confidence"] = 0.8
        answers[criterion.id] = answer
    return {
        "id": "gen-dec-1",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": answers,
        "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002},
    }


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def test_score_request_shape_matches_the_wire_contract(rubric):
    decisions = FakeDecisions([score_response_for(rubric)])
    jev = make_evaluator(decisions, rubric=rubric)
    judgement = jev.score_response(
        prompt_id="p0001",
        candidate=candidate(),
        prompt="Will the fog lift today?",
        context=[{"role": "user", "content": "hello"}],
    )
    call = decisions.calls[0]
    assert call["server_url"] == DECISIONS_SERVER_URL
    assert call["model"] == JUDGE
    assert call["session_id"] == "run-1"
    assert set(call["questions"]) == {c.id for c in rubric.criteria}
    for criterion in rubric.criteria:
        question = call["questions"][criterion.id]
        assert question["type"] == "score"
        assert question["criteria"] == criterion.levels  # ordered lowest-first
        assert question["instructions"] == criterion.instructions_for_judge()
    state = call["state"]
    assert state["character"]["name"] == DEFAULT_CHARACTER.name
    assert state["prompt"] == "Will the fog lift today?"
    assert state["response"] == "Not before the afternoon tide."
    assert state["context"] == [{"role": "user", "content": "hello"}]
    assert state["prompt_id"] == "p0001"
    assert state["candidate_id"] == "p0001-c0"

    assert judgement.error is None
    assert judgement.judge_build == "typesafe/jev-1.13-20260917"
    assert judgement.provider == "TypeSafe"
    assert judgement.usage["input_tokens"] == 476
    assert set(judgement.criterion_scores) == {c.id for c in rubric.criteria}
    for score in judgement.criterion_scores.values():
        assert score.score == pytest.approx(score.scale_max)
        assert score.probabilities is not None
        assert score.confidence == pytest.approx(0.8)
        assert score.legend
    assert judgement.overall == pytest.approx(
        rubric.overall({c.id: c.scale_max for c in rubric.criteria})
    )


def test_score_omitted_probabilities_and_confidence_are_not_invented(rubric):
    decisions = FakeDecisions([score_response_for(rubric, omit_optional=True)])
    jev = make_evaluator(decisions, rubric=rubric)
    judgement = jev.score_response(prompt_id="p0001", candidate=candidate(), prompt="p")
    assert judgement.error is None
    for score in judgement.criterion_scores.values():
        assert score.score is not None
        assert score.probabilities is None
        assert score.confidence is None
        assert score.legend is None


def test_failed_judgement_is_none_not_zero(rubric):
    decisions = FakeDecisions([RemoteAPIError("boom")])
    jev = make_evaluator(decisions, rubric=rubric)
    judgement = jev.score_response(prompt_id="p0001", candidate=candidate(), prompt="p")
    assert judgement.error is not None
    assert judgement.overall is None
    for score in judgement.criterion_scores.values():
        assert score.score is None
        assert score.error


def test_score_missing_answer_key_raises_inside_the_client(rubric):
    payload = score_response_for(rubric)
    payload["answers"].pop(next(iter(payload["answers"])))
    decisions = FakeDecisions([payload])
    jev = make_evaluator(decisions, rubric=rubric)
    judgement = jev.score_response(prompt_id="p0001", candidate=candidate(), prompt="p")
    assert judgement.error is not None
    assert "no answer" in judgement.error


def test_score_truncates_only_the_response_and_records_it(rubric):
    decisions = FakeDecisions([score_response_for(rubric)])
    jev = make_evaluator(decisions, rubric=rubric)
    huge = "x" * (MAX_STATE_CHARS + 5000)
    judgement = jev.score_response(prompt_id="p0001", candidate=candidate(huge), prompt="p")
    state = decisions.calls[0]["state"]
    assert len(state["response"]) < len(huge)
    assert state["response"].endswith(TRUNCATION_MARKER)
    assert judgement.raw[RAW_TRUNCATION_KEY] is True


def test_score_caches_second_identical_call(store, rubric):
    decisions = FakeDecisions([score_response_for(rubric)])
    cache = ResponseCache(store)
    jev = make_evaluator(decisions, rubric=rubric, cache=cache)
    first = jev.score_response(prompt_id="p0001", candidate=candidate(), prompt="p")
    second = jev.score_response(prompt_id="p0001", candidate=candidate(), prompt="p")
    assert len(decisions.calls) == 1
    assert first.cached is False
    assert second.cached is True


def test_score_repeat_passes_are_not_cache_hits_of_each_other(store, rubric):
    """judge_repeats only averages noise if each pass really reaches the judge."""

    decisions = FakeDecisions([score_response_for(rubric)])
    cache = ResponseCache(store)
    jev = make_evaluator(decisions, rubric=rubric, cache=cache)
    first = jev.score_response(prompt_id="p0001", candidate=candidate(), prompt="p", pass_index=0)
    second = jev.score_response(prompt_id="p0001", candidate=candidate(), prompt="p", pass_index=1)
    assert len(decisions.calls) == 2
    assert decisions.calls[1]["state"]["pass"] == 1
    assert first.judgement_id != second.judgement_id


# --------------------------------------------------------------------------- #
# choice: preference
# --------------------------------------------------------------------------- #


def test_compare_responses_choice_shape_and_winner(rubric, jev_choice_response):
    decisions = FakeDecisions([jev_choice_response])
    jev = make_evaluator(decisions, rubric=rubric)
    a = CandidateResponse(candidate_id="p0001-c0", prompt_id="p0001", text="A text")
    b = CandidateResponse(candidate_id="p0001-c1", prompt_id="p0001", text="B text")
    decision = jev.compare_responses(
        prompt_id="p0001",
        option_a=a,
        option_b=b,
        prompt="Will the fog lift?",
        context=[{"role": "user", "content": "hi"}],
        overall_a=3.0,
        overall_b=4.0,
    )
    call = decisions.calls[0]
    assert call["server_url"] == DECISIONS_SERVER_URL
    question = call["questions"]["preference"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"a", "b", "tie"}
    assert "no meaningful preference" in question["criteria"]["tie"]
    state = call["state"]
    assert state["response_a"] == "A text"
    assert state["response_b"] == "B text"
    assert state["overall_a"] == 3.0
    assert state["overall_b"] == 4.0
    assert state["prompt"] == "Will the fog lift?"

    assert decision.error is None
    assert decision.winner == "b"
    assert decision.choice_raw == "b"
    assert decision.probabilities == {"a": 0.19, "b": 0.81, "tie": 0.0}
    assert decision.confidence == pytest.approx(0.67)
    assert decision.margin == pytest.approx(-1.0)
    assert decision.judge_build == "typesafe/jev-1.13-20260917"
    assert decision.usable is True
    assert decision.winner_id() == "p0001-c1"


def test_compare_responses_tie_is_usable_as_a_tie(rubric, jev_choice_response):
    payload = dict(jev_choice_response)
    payload["answers"] = {
        "preference": {
            "type": "choice",
            "choice": "tie",
            "probabilities": {"a": 0.3, "b": 0.3, "tie": 0.4},
            "confidence": 0.5,
        }
    }
    decisions = FakeDecisions([payload])
    jev = make_evaluator(decisions, rubric=rubric)
    decision = jev.compare_responses(
        prompt_id="p1",
        option_a=CandidateResponse(candidate_id="a", prompt_id="p1", text="A"),
        option_b=CandidateResponse(candidate_id="b", prompt_id="p1", text="B"),
        prompt="p",
    )
    assert decision.winner == "tie"
    assert decision.usable is False
    assert decision.margin is None


def test_compare_responses_unknown_option_is_an_error(rubric, jev_choice_response):
    payload = dict(jev_choice_response)
    payload["answers"] = {"preference": {"type": "choice", "choice": "z"}}
    decisions = FakeDecisions([payload])
    jev = make_evaluator(decisions, rubric=rubric)
    decision = jev.compare_responses(
        prompt_id="p1",
        option_a=CandidateResponse(candidate_id="a", prompt_id="p1", text="A"),
        option_b=CandidateResponse(candidate_id="b", prompt_id="p1", text="B"),
        prompt="p",
    )
    assert decision.error is not None
    assert decision.winner is None


def test_compare_responses_failure_records_error(rubric):
    decisions = FakeDecisions([RemoteAPIError("nope")])
    jev = make_evaluator(decisions, rubric=rubric)
    decision = jev.compare_responses(
        prompt_id="p1",
        option_a=CandidateResponse(candidate_id="a", prompt_id="p1", text="A"),
        option_b=CandidateResponse(candidate_id="b", prompt_id="p1", text="B"),
        prompt="p",
    )
    assert decision.error is not None
    assert decision.winner is None


# --------------------------------------------------------------------------- #
# choice: preset ranking
# --------------------------------------------------------------------------- #


def test_rank_presets_returns_real_numbers(rubric, jev_choice_response):
    options = {
        "cpu_tiny_lora": "CPU / tiny LoRA — smallest viable run",
        "single_gpu_safe_lora": "Single GPU / safe LoRA — default",
    }
    payload = dict(jev_choice_response)
    payload["answers"] = {
        "preset": {
            "type": "choice",
            "choice": "cpu_tiny_lora",
            "probabilities": {"cpu_tiny_lora": 0.7, "single_gpu_safe_lora": 0.3},
            "confidence": 0.64,
        }
    }
    decisions = FakeDecisions([payload])
    jev = make_evaluator(decisions, rubric=rubric)
    chosen, info = jev.rank_presets(state={"task": "rank_training_presets"}, options=options)
    call = decisions.calls[0]
    assert call["questions"]["preset"]["criteria"] == options
    assert chosen == "cpu_tiny_lora"
    assert info["probabilities"] == {"cpu_tiny_lora": 0.7, "single_gpu_safe_lora": 0.3}
    assert info["confidence"] == pytest.approx(0.64)
    assert info["model_build"] == "typesafe/jev-1.13-20260917"
    assert info["error"] is None


def test_rank_presets_rejects_choice_outside_the_options(rubric, jev_choice_response):
    payload = dict(jev_choice_response)
    payload["answers"] = {"preset": {"type": "choice", "choice": "multi_gpu_fft"}}
    decisions = FakeDecisions([payload])
    jev = make_evaluator(decisions, rubric=rubric)
    chosen, info = jev.rank_presets(state={}, options={"cpu_tiny_lora": "x"})
    assert chosen is None
    assert info["error"]


def test_rank_presets_without_options_makes_no_call(rubric):
    decisions = FakeDecisions([])
    jev = make_evaluator(decisions, rubric=rubric)
    chosen, info = jev.rank_presets(state={}, options={})
    assert chosen is None
    assert decisions.calls == []
    assert info["error"]


# --------------------------------------------------------------------------- #
# noul
# --------------------------------------------------------------------------- #


def test_check_condition_noul_shape_and_probability(rubric, jev_noul_response):
    decisions = FakeDecisions([jev_noul_response])
    jev = make_evaluator(decisions, rubric=rubric)
    probability, info = jev.check_condition(
        instructions="does the response break a constraint?", state={"task": "x"}
    )
    call = decisions.calls[0]
    assert call["server_url"] == DECISIONS_SERVER_URL
    question = call["questions"]["condition"]
    assert question["type"] == "noul"
    assert set(question["criteria"]) == {"true", "false"}
    assert call["state"] == {"task": "x"}
    assert probability == pytest.approx(0.96)
    assert info["probability"] == pytest.approx(0.96)
    assert info["model_build"] == "typesafe/jev-1.13-20260917"


def test_check_condition_custom_key(rubric, jev_noul_response):
    payload = dict(jev_noul_response)
    payload["answers"] = {"hallucination": {"type": "noul", "noul": 0.2}}
    decisions = FakeDecisions([payload])
    jev = make_evaluator(decisions, rubric=rubric)
    probability, _info = jev.check_condition(instructions="x", state={}, key="hallucination")
    assert "hallucination" in decisions.calls[0]["questions"]
    assert probability == pytest.approx(0.2)


def test_check_condition_failure_raises_judge_error(rubric):
    decisions = FakeDecisions([RemoteAPIError("down")])
    jev = make_evaluator(decisions, rubric=rubric)
    with pytest.raises(JudgeError):
        jev.check_condition(instructions="x", state={})


def test_all_three_primitives_are_exercised(
    rubric, jev_score_response, jev_choice_response, jev_noul_response
):
    """A single evaluator can run score, choice, and noul requests."""

    decisions = FakeDecisions([jev_score_response, jev_choice_response, jev_noul_response])
    jev = make_evaluator(decisions, rubric=rubric)
    judgement = jev.score_response(prompt_id="p1", candidate=candidate(), prompt="p")
    decision = jev.compare_responses(
        prompt_id="p1",
        option_a=CandidateResponse(candidate_id="a", prompt_id="p1", text="A"),
        option_b=CandidateResponse(candidate_id="b", prompt_id="p1", text="B"),
        prompt="p",
    )
    probability, _info = jev.check_condition(instructions="x", state={})
    assert {
        call["questions"][next(iter(call["questions"]))]["type"] for call in decisions.calls
    } == {
        "score",
        "choice",
        "noul",
    }
    assert judgement.prompt_id == "p1"
    assert decision.prompt_id == "p1"
    assert 0.0 <= probability <= 1.0
