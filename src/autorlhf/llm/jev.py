"""Jev: the typed judge behind every automatic judgement in autorlhf.

Jev is a *decision* model, not a chat model.  It answers one of three typed
questions and returns a structured answer with the judge's own probabilities and
confidence:

* ``score`` — an ordered scale (used per rubric criterion),
* ``choice`` — one of several named options (used for A/B preference and preset
  ranking),
* ``noul`` — yes/no, returned as ``P(true)`` (used for explicit condition checks).

Two rules are load-bearing and must not be relaxed:

* autorlhf never invents a probability or a confidence.  When the judge omits
  ``probabilities``/``confidence`` (the schema marks them optional) the fields
  stay ``None``; when a judgement fails, the score is ``None`` plus an error
  string — never ``0``.
* a chat model is never prompted to *emulate* a decision.  If Jev cannot answer,
  the call fails and the failure is recorded.
"""

from __future__ import annotations

import random
import time
from typing import Any

from autorlhf.config.character import CharacterSpec
from autorlhf.config.rubric import Rubric
from autorlhf.core.enums import DecisionKind
from autorlhf.core.errors import JudgeError, RemoteAPIError
from autorlhf.core.ids import content_hash
from autorlhf.core.logging import get_logger
from autorlhf.llm.client import OpenRouterClient
from autorlhf.llm.schemas import (
    CandidateResponse,
    ChoiceAnswer,
    CriterionScore,
    DecisionsRequest,
    DecisionsResponse,
    Judgement,
    NoulAnswer,
    PairDecision,
    ScoreAnswer,
)
from autorlhf.storage.cache import NS_JEV_CHOICE, NS_JEV_NOUL, NS_JEV_SCORE, ResponseCache

logger = get_logger("autorlhf.llm.jev")

#: Jev's context window is 32k tokens.  autorlhf budgets in characters with a
#: conservative 4 chars/token, so this is a little under the hard limit.
MAX_STATE_CHARS = 120_000

#: Sent to the judge so the response is truncated visibly rather than silently.
TRUNCATION_MARKER = "\n[... response truncated to fit the judge's context ...]"

#: Marker recorded in ``Judgement.raw`` when the response had to be shortened.
RAW_TRUNCATION_KEY = "response_truncated"

_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 8.0


def _backoff(attempt: int) -> float:
    return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt)) * (0.5 + random.random())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class JevEvaluator:
    """Scores candidate responses, decides preferences, ranks presets, and checks
    conditions with Jev's typed primitives."""

    def __init__(
        self,
        client: OpenRouterClient,
        rubric: Rubric,
        character: CharacterSpec,
        *,
        judge_model: str,
        cache: ResponseCache | None = None,
        run_id: str = "",
        # The client already retries with backoff and classifies retryable vs
        # credential errors; a second loop here would multiply the two.
        max_retries: int = 0,
    ) -> None:
        self.client = client
        self.rubric = rubric
        self.character = character
        self.judge_model = judge_model
        self.cache = cache if cache is not None else client.cache
        self.run_id = run_id
        self.max_retries = max(0, int(max_retries))

    # ------------------------------------------------------------------ #
    # request building
    # ------------------------------------------------------------------ #

    def _character_state(self) -> dict[str, Any]:
        return self.character.as_state()

    def _fit_response(self, text: str, *, budget: int) -> tuple[str, bool]:
        """Truncate only the *response*; the rubric and character spec are never cut."""

        if len(text) <= budget:
            return text, False
        keep = max(0, budget - len(TRUNCATION_MARKER))
        return text[:keep] + TRUNCATION_MARKER, True

    def _response_budget(self, state: dict[str, Any]) -> int:
        """Characters left for the candidate response inside the context budget."""

        import json

        fixed = json.dumps(
            {k: v for k, v in state.items() if k not in {"response", "response_a", "response_b"}},
            ensure_ascii=False,
            default=str,
        )
        return max(512, MAX_STATE_CHARS - len(fixed))

    def _decide(
        self,
        *,
        namespace: str,
        cache_key: str,
        request: DecisionsRequest,
    ) -> tuple[DecisionsResponse | None, str | None, bool]:
        """Call ``client.decide`` with jev-level caching and bounded retries."""

        if self.cache is not None:
            cached = self.cache.get(namespace, cache_key)
            if isinstance(cached, dict) and "response" in cached:
                try:
                    return DecisionsResponse.model_validate(cached["response"]), None, True
                except Exception as exc:  # pragma: no cover - corrupt cache entry
                    logger.warning("ignoring an unreadable cached judgement: %s", exc)

        attempt = 0
        last_error: str | None = None
        while attempt <= self.max_retries:
            try:
                response = self.client.decide(request)
            except RemoteAPIError as exc:
                last_error = str(exc) or type(exc).__name__
                if attempt >= self.max_retries:
                    logger.warning("Jev decision failed after retries: %s", last_error)
                    return None, last_error, False
                time.sleep(_backoff(attempt))
                attempt += 1
                continue
            if self.cache is not None:
                self.cache.put(
                    namespace,
                    cache_key,
                    {"response": response.model_dump(mode="json")},
                    model=self.judge_model,
                    input_tokens=response.usage.input_tokens,
                    cost_usd=response.usage.cost,
                )
            return response, None, False
        return None, last_error or "the judge did not answer", False

    # ------------------------------------------------------------------ #
    # score
    # ------------------------------------------------------------------ #

    def score_response(
        self,
        *,
        prompt_id: str,
        candidate: CandidateResponse,
        prompt: str,
        context: list[dict[str, str]] = (),
        pass_index: int = 0,
    ) -> Judgement:
        """Score one candidate response against every rubric criterion.

        ``pass_index`` separates repeated evaluation passes: without it every pass
        is a cache hit on the first one and the average is a value with itself.
        """

        questions = self.rubric.questions_for_judge()
        state: dict[str, Any] = {
            **self._character_state(),
            "task": "score_response",
            "prompt_id": prompt_id,
            "candidate_id": candidate.candidate_id,
            "prompt": prompt,
            "context": list(context),
            "response": candidate.text,
        }
        # Only a repeat pass carries it, so single-pass callers keep their exact
        # cache key and judgement id.
        if pass_index:
            state["pass"] = pass_index
        budget = self._response_budget(state)
        fitted, truncated = self._fit_response(candidate.text, budget=budget)
        state["response"] = fitted

        id_parts: dict[str, Any] = {
            "kind": "score",
            "run_id": self.run_id,
            "prompt_id": prompt_id,
            "candidate_id": candidate.candidate_id,
            "judge_model": self.judge_model,
        }
        if pass_index:
            id_parts["pass"] = pass_index
        judgement_id = "j-" + content_hash(id_parts)
        request = DecisionsRequest(
            model=self.judge_model,
            state=state,
            questions=questions,
            session_id=self.run_id or None,
        )
        cache_key = content_hash(
            {
                "kind": "jev.score",
                "model": self.judge_model,
                "rubric": self.rubric.fingerprint(),
                "state": state,
                "questions": {
                    k: v.model_dump(mode="json", exclude_none=True)
                    for k, v in request.questions.items()
                },
            }
        )
        response, error, cached = self._decide(
            namespace=NS_JEV_SCORE, cache_key=cache_key, request=request
        )
        if response is None or error is not None:
            return self._failed_judgement(
                judgement_id=judgement_id,
                prompt_id=prompt_id,
                candidate_id=candidate.candidate_id,
                error=error or "the judge returned no response",
                state_truncated=truncated,
                raw_state_keys=sorted(state),
            )

        judgement = Judgement(
            judgement_id=judgement_id,
            run_id=self.run_id,
            prompt_id=prompt_id,
            candidate_id=candidate.candidate_id,
            kind=DecisionKind.SCORE,
            judge_model=self.judge_model,
            judge_build=response.model,
            provider=response.provider,
            usage=response.usage.model_dump(mode="json"),
            created_at=_now(),
            cached=cached,
            raw=self._raw(response, truncated),
        )
        scores: dict[str, float] = {}
        for criterion in self.rubric.criteria:
            answer = response.answers.get(criterion.id)
            score = self._criterion_score(criterion, answer)
            judgement.criterion_scores[criterion.id] = score
            if score.score is not None:
                scores[criterion.id] = score.score
        judgement.overall = self.rubric.overall(scores)
        if judgement.overall is None:
            judgement.error = "the judge returned no usable scores"
        return judgement

    def _criterion_score(self, criterion: Any, answer: Any) -> CriterionScore:
        base = CriterionScore(
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            scale_min=criterion.scale_min,
            scale_max=criterion.scale_max,
            answer_type=getattr(answer, "type", "score") or "score",
        )
        if answer is None:
            base.error = "the judge returned no answer for this criterion"
            return base
        if not isinstance(answer, ScoreAnswer):
            base.error = f"the judge answered this criterion as '{answer.type}', not 'score'"
            return base
        base.score = float(answer.score)
        base.probabilities = dict(answer.probabilities) if answer.probabilities else None
        base.legend = dict(answer.legend) if answer.legend else None
        base.confidence = answer.confidence
        return base

    def _failed_judgement(
        self,
        *,
        judgement_id: str,
        prompt_id: str,
        candidate_id: str,
        error: str,
        state_truncated: bool,
        raw_state_keys: list[str],
    ) -> Judgement:
        judgement = Judgement(
            judgement_id=judgement_id,
            run_id=self.run_id,
            prompt_id=prompt_id,
            candidate_id=candidate_id,
            kind=DecisionKind.SCORE,
            judge_model=self.judge_model,
            created_at=_now(),
            error=error,
            raw={
                "error": error,
                RAW_TRUNCATION_KEY: state_truncated,
                "state_keys": raw_state_keys,
            },
        )
        for criterion in self.rubric.criteria:
            judgement.criterion_scores[criterion.id] = CriterionScore(
                criterion_id=criterion.id,
                criterion_name=criterion.name,
                scale_min=criterion.scale_min,
                scale_max=criterion.scale_max,
                error=error,
            )
        return judgement

    @staticmethod
    def _raw(response: DecisionsResponse, truncated: bool) -> dict[str, Any]:
        raw = response.model_dump(mode="json")
        raw[RAW_TRUNCATION_KEY] = truncated
        return raw

    # ------------------------------------------------------------------ #
    # choice: preference between two responses
    # ------------------------------------------------------------------ #

    def compare_responses(
        self,
        *,
        prompt_id: str,
        option_a: CandidateResponse,
        option_b: CandidateResponse,
        prompt: str,
        context: list[dict[str, str]] = (),
        overall_a: float | None = None,
        overall_b: float | None = None,
    ) -> PairDecision:
        """Ask Jev which of two responses is better, with an explicit tie option."""

        pair_id = f"{prompt_id}-{option_a.candidate_id}-vs-{option_b.candidate_id}"
        questions = {
            "preference": {
                "type": "choice",
                "instructions": (
                    "Two responses to the same user prompt are shown. Pick the one the "
                    "specified character should produce. Choose 'tie' when neither is "
                    "meaningfully better; do not force a preference."
                ),
                "criteria": {
                    "a": f"Response A ({option_a.candidate_id})",
                    "b": f"Response B ({option_b.candidate_id})",
                    "tie": "no meaningful preference between A and B",
                },
            }
        }
        state: dict[str, Any] = {
            **self._character_state(),
            "task": "compare_responses",
            "prompt_id": prompt_id,
            "prompt": prompt,
            "context": list(context),
            "response_a": option_a.text,
            "response_b": option_b.text,
            "overall_a": overall_a,
            "overall_b": overall_b,
        }
        budget = self._response_budget(state) // 2
        state["response_a"], truncated_a = self._fit_response(option_a.text, budget=budget)
        state["response_b"], truncated_b = self._fit_response(option_b.text, budget=budget)

        decision = PairDecision(
            pair_id=pair_id,
            run_id=self.run_id,
            prompt_id=prompt_id,
            option_a_id=option_a.candidate_id,
            option_b_id=option_b.candidate_id,
            judge_model=self.judge_model,
            created_at=_now(),
            margin=(
                overall_a - overall_b if overall_a is not None and overall_b is not None else None
            ),
        )
        request = DecisionsRequest(
            model=self.judge_model,
            state=state,
            questions=questions,
            session_id=self.run_id or None,
        )
        cache_key = content_hash(
            {
                "kind": "jev.choice",
                "model": self.judge_model,
                "state": state,
                "questions": questions,
            }
        )
        response, error, cached = self._decide(
            namespace=NS_JEV_CHOICE, cache_key=cache_key, request=request
        )
        decision.cached = cached
        if response is None or error is not None:
            decision.error = error or "the judge returned no response"
            decision.raw = {
                "error": decision.error,
                RAW_TRUNCATION_KEY: truncated_a or truncated_b,
            }
            return decision

        decision.judge_build = response.model
        decision.provider = response.provider
        decision.usage = response.usage.model_dump(mode="json")
        decision.raw = self._raw(response, truncated_a or truncated_b)
        answer = response.answers.get("preference")
        if not isinstance(answer, ChoiceAnswer):
            decision.error = "the judge did not return a choice answer"
            return decision
        decision.choice_raw = answer.choice
        decision.probabilities = dict(answer.probabilities) if answer.probabilities else None
        decision.confidence = answer.confidence
        if answer.choice in {"a", "b", "tie"}:
            decision.winner = answer.choice  # type: ignore[assignment]
        else:
            decision.error = f"the judge chose an unknown option '{answer.choice}'"
        return decision

    # ------------------------------------------------------------------ #
    # choice: preset ranking
    # ------------------------------------------------------------------ #

    def rank_presets(self, *, state: dict, options: dict[str, str]) -> tuple[str | None, dict]:
        """Ask Jev to rank the *feasible* presets, returning the chosen key and info.

        The returned distribution is the judge's **preference among the listed
        options only**.  It is not, and must never be presented as, a probability
        that training will succeed; the UI states this next to the numbers.
        """

        info: dict[str, Any] = {
            "options": dict(options),
            "chosen": None,
            "probabilities": None,
            "confidence": None,
            "model_build": "",
            "provider": "",
            "usage": {},
            "error": None,
        }
        if not options:
            info["error"] = "no feasible presets to rank"
            return None, info

        questions = {
            "preset": {
                "type": "choice",
                "instructions": (
                    "Choose the single training preset that best fits the hardware and the "
                    "stated goal. This is a preference among the listed options only."
                ),
                "criteria": dict(options),
            }
        }
        request = DecisionsRequest(
            model=self.judge_model,
            state=state,
            questions=questions,
            session_id=self.run_id or None,
        )
        cache_key = content_hash(
            {
                "kind": "jev.choice.preset",
                "model": self.judge_model,
                "state": state,
                "questions": questions,
            }
        )
        response, error, cached = self._decide(
            namespace=NS_JEV_CHOICE, cache_key=cache_key, request=request
        )
        info["cached"] = cached
        if response is None or error is not None:
            info["error"] = error or "the judge returned no response"
            return None, info

        info["model_build"] = response.model
        info["provider"] = response.provider
        info["usage"] = response.usage.model_dump(mode="json")
        answer = response.answers.get("preset")
        if not isinstance(answer, ChoiceAnswer):
            info["error"] = "the judge did not return a choice answer"
            return None, info
        info["probabilities"] = dict(answer.probabilities) if answer.probabilities else None
        info["confidence"] = answer.confidence
        chosen = answer.choice if answer.choice in options else None
        if chosen is None:
            info["error"] = f"the judge chose an option outside the list: '{answer.choice}'"
        info["chosen"] = chosen
        return chosen, info

    # ------------------------------------------------------------------ #
    # noul: condition check
    # ------------------------------------------------------------------ #

    def check_condition(
        self, *, instructions: str, state: dict, key: str = "condition"
    ) -> tuple[float, dict]:
        """Ask a yes/no question, returning ``(P(true), info)``.

        Raises :class:`JudgeError` when the judge cannot answer — a probability is
        never fabricated.
        """

        questions = {
            key: {
                "type": "noul",
                "instructions": instructions,
                "criteria": {
                    "true": "the condition holds for the supplied state",
                    "false": "the condition does not hold for the supplied state",
                },
            }
        }
        request = DecisionsRequest(
            model=self.judge_model,
            state=state,
            questions=questions,
            session_id=self.run_id or None,
        )
        cache_key = content_hash(
            {
                "kind": "jev.noul",
                "model": self.judge_model,
                "key": key,
                "instructions": instructions,
                "state": state,
            }
        )
        response, error, cached = self._decide(
            namespace=NS_JEV_NOUL, cache_key=cache_key, request=request
        )
        info: dict[str, Any] = {
            "key": key,
            "cached": cached,
            "model_build": "",
            "provider": "",
            "usage": {},
            "error": error,
        }
        if response is None or error is not None:
            raise JudgeError(
                f"the judge could not evaluate '{key}': {error or 'no response'}",
                hint="Retry the condition check, or pin a different judge build.",
            )
        info["model_build"] = response.model
        info["provider"] = response.provider
        info["usage"] = response.usage.model_dump(mode="json")
        answer = response.answers.get(key)
        if not isinstance(answer, NoulAnswer):
            raise JudgeError(
                f"the judge did not return a yes/no answer for '{key}'",
                hint="Check the judge model and the question type.",
            )
        probability = float(answer.noul)
        info["probability"] = probability
        info["raw"] = response.model_dump(mode="json")
        return probability, info


__all__ = ["MAX_STATE_CHARS", "RAW_TRUNCATION_KEY", "TRUNCATION_MARKER", "JevEvaluator"]
