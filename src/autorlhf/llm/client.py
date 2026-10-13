"""The single module in autorlhf that performs network IO.

Two upstream surfaces are wrapped:

* **OpenRouter chat completions** (``POST /api/v1/chat/completions``) — the
  teacher model used to write synthetic prompts and candidate responses;
* **Jev decisions** (``POST /api/alpha/decisions``) — the typed judge.

Everything the rest of the app sees is a frozen domain model
(:class:`~autorlhf.core.types.GenerationResult`,
:class:`~autorlhf.llm.schemas.DecisionsResponse`).  Every failure is translated
into a typed :mod:`autorlhf.core.errors` exception; a network problem never
becomes an empty string or an empty answer set.

Design notes that matter:

* the SDK client is built lazily on first use and cached, so constructing an
  :class:`OpenRouterClient` performs no IO and opens no sockets;
* ``alpha.decisions.create`` is called with ``server_url="https://openrouter.ai"``
  because the Decisions path is ``/api/alpha/decisions`` while the SDK's default
  server already ends in ``/api/v1``;
* paid calls are content-addressed through :class:`~autorlhf.storage.cache.ResponseCache`,
  so resuming a run never repeats a charge;
* Weave tracing is opt-in and explicit: it only happens when Weave is already
  active in the process, and the integration import is guarded.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
from pydantic import ValidationError

from autorlhf.core.errors import (
    CredentialError,
    GenerationError,
    JudgeError,
    RateLimitedError,
    RemoteAPIError,
)
from autorlhf.core.ids import content_hash
from autorlhf.core.logging import get_logger
from autorlhf.core.types import ChatMessage, GenerationResult
from autorlhf.llm.schemas import ChatCompletionResponse, DecisionsRequest, DecisionsResponse
from autorlhf.storage.cache import NS_CHAT, ResponseCache

try:  # pragma: no cover - the integrations package lands independently
    from autorlhf.integrations import tracking as _tracking
except ImportError:  # pragma: no cover
    _tracking = None

logger = get_logger("autorlhf.llm.client")

#: Jev's Decisions endpoint lives outside the SDK's default ``/api/v1`` server.
DECISIONS_SERVER_URL = "https://openrouter.ai"

#: Cache namespace for the raw decisions response (Jev-level caching happens in
#: :mod:`autorlhf.llm.jev` and stores the parsed judgement).
NS_DECISIONS = "jev.decisions"

#: HTTP statuses that mean "your credential is the problem", not "try again".
_CREDENTIAL_STATUSES = frozenset({401, 402, 403})

#: Name fragments used to recover an HTTP status when the SDK error carries none.
_STATUS_BY_NAME: tuple[tuple[str, int], ...] = (
    ("Unauthorized", 401),
    ("PaymentRequired", 402),
    ("Forbidden", 403),
    ("NotFound", 404),
    ("TooManyRequests", 429),
    ("Conflict", 409),
    ("Gone", 410),
    ("BadRequest", 400),
    ("GatewayTimeout", 504),
    ("BadGateway", 502),
    ("InternalServer", 500),
    ("EdgeNetworkTimeout", 504),
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL)

_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 20.0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _to_dict(value: Any) -> dict[str, Any]:
    """Best-effort conversion of an SDK response object into a plain dict."""

    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise RemoteAPIError(
        f"cannot interpret the SDK response of type {type(value).__name__}",
        hint="This is an autorlhf bug: report the OpenRouter SDK version in use.",
    )


def _normalise_messages(messages: Iterable[Any]) -> list[dict[str, str]]:
    """Coerce ``ChatMessage`` / pydantic models / dicts into wire messages."""

    out: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            out.append({"role": message.role, "content": message.content})
            continue
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if not isinstance(role, str) or not isinstance(content, str):
            raise GenerationError(
                f"invalid chat message {message!r}; role and content must be strings",
                hint="Build messages with autorlhf.core.types.ChatMessage.",
            )
        out.append({"role": role, "content": content})
    return out


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter, capped so a run cannot stall forever."""

    window = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt))
    return window * (0.5 + random.random())


def _status_from_name(name: str) -> int | None:
    for fragment, status in _STATUS_BY_NAME:
        if fragment in name:
            return status
    return None


def _retry_after(exc: BaseException) -> float | None:
    raw = getattr(exc, "raw_response", None)
    headers = getattr(raw, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except Exception:  # pragma: no cover - exotic header objects
        return None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _safe_message(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text[:400]


def classify_error(exc: BaseException, *, what: str) -> RemoteAPIError:
    """Translate an SDK/httpx exception into a typed autorlhf error.

    Never returns; always raises through the caller.  The returned exception
    carries a ``retryable`` attribute so :meth:`OpenRouterClient._request` knows
    whether another attempt is worth it.
    """

    if isinstance(exc, RemoteAPIError):
        return exc

    if isinstance(exc, httpx.TimeoutException):
        err = RemoteAPIError(
            f"{what} timed out: {_safe_message(exc)}",
            hint="Raise request_timeout_s on the run config, or retry later.",
        )
        err.retryable = True
        return err
    if isinstance(exc, httpx.TransportError):
        err = RemoteAPIError(
            f"{what} could not reach OpenRouter: {_safe_message(exc)}",
            hint="Check network access and the OpenRouter status page, then retry.",
        )
        err.retryable = True
        return err

    status = None
    raw = getattr(exc, "raw_response", None)
    if raw is not None:
        status = getattr(raw, "status_code", None)
    if status is None:
        status = _status_from_name(type(exc).__name__)

    if status in _CREDENTIAL_STATUSES:
        return CredentialError(
            f"OpenRouter rejected the credential (HTTP {status}): {_safe_message(exc)}",
            hint="Run `autorlhf --setup` and paste a fresh OPENROUTER_API_KEY, then retry.",
        )
    if status == 429:
        err = RateLimitedError(
            f"OpenRouter rate limit reached (HTTP 429): {_safe_message(exc)}",
            retry_after=_retry_after(exc),
            hint="autorlhf backs off automatically; lower max_concurrency if it persists.",
        )
        err.retryable = True
        return err
    if status is not None and status >= 500:
        err = RemoteAPIError(
            f"OpenRouter server error (HTTP {status}) during {what}: {_safe_message(exc)}",
            hint="This is usually transient; retry the step.",
        )
        err.retryable = True
        return err

    detail = f" (HTTP {status})" if status is not None else ""
    return RemoteAPIError(
        f"{what} failed{detail}: {_safe_message(exc)}",
        hint="Inspect the message; fix the request or the credential, then retry.",
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object in ``text``, or raise :class:`GenerationError`.

    Handles a bare object, an object wrapped in a ```` ```json ```` fence, and
    an object embedded in prose.
    """

    candidates: list[str] = [text.strip()]
    for match in _JSON_FENCE_RE.finditer(text):
        candidates.append(match.group("body").strip())
    for candidate in candidates:
        if not candidate:
            continue
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    embedded = _first_balanced_object(text)
    if embedded is not None:
        parsed = _loads_object(embedded)
        if parsed is not None:
            return parsed
    raise GenerationError(
        "the model did not return a JSON object",
        hint="Retry, or lower the requested schema complexity for this model.",
    )


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _first_balanced_object(text: str) -> str | None:
    """Extract the first balanced ``{...}`` block, ignoring braces in strings."""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #


class OpenRouterClient:
    """Thread-safe, retrying, caching wrapper over the OpenRouter SDK."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 4,
        max_concurrency: int = 4,
        cache: ResponseCache | None = None,
        run_id: str = "",
        app_title: str = "autorlhf",
        trace: Callable[[str], None] | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise CredentialError(
                "no OpenRouter API key is configured",
                hint="Run autorlhf --setup to store OPENROUTER_API_KEY, then retry.",
            )
        self._api_key = api_key.strip()
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.max_concurrency = max(1, int(max_concurrency))
        self.cache = cache
        self.run_id = run_id
        self.app_title = app_title
        self.trace = trace

        self._sdk: Any | None = None
        self._lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._usage: dict[str, float] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "cache_hits": 0,
            "errors": 0,
        }

    # -- sdk plumbing ------------------------------------------------------ #

    def _get_sdk(self) -> Any:
        with self._lock:
            if self._sdk is None:
                from openrouter import OpenRouter

                self._sdk = OpenRouter(
                    api_key=self._api_key,
                    x_open_router_title=self.app_title,
                    timeout_ms=int(self.timeout_s * 1000),
                )
            return self._sdk

    def _trace_api(self, message: str) -> None:
        if self.trace is None:
            return
        try:
            self.trace(message.replace(self._api_key, "[redacted]"))
        except Exception:  # noqa: BLE001 - observability must not break an API call
            logger.exception("verbose API trace callback failed")

    def _trace_payload(self, label: str, payload: Any) -> None:
        if self.trace is not None:
            self._trace_api(
                f"{label}\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
            )

    def _call_sdk(self, fn: Callable[[], Any], *, what: str) -> Any:
        """Invoke an SDK callable, translating every failure into a typed error."""

        try:
            return fn()
        except RemoteAPIError:
            raise
        except Exception as exc:  # noqa: BLE001 - classification is the point
            raise classify_error(exc, what=what) from exc

    def _request(self, fn: Callable[[], Any], *, what: str) -> Any:
        """Run ``fn`` with bounded, jittered exponential backoff."""

        attempt = 0
        while True:
            try:
                return self._call_sdk(fn, what=what)
            except RateLimitedError as exc:
                if attempt >= self.max_retries:
                    self._bump("errors")
                    raise
                delay = exc.retry_after if exc.retry_after is not None else _backoff(attempt)
                self._trace_api(
                    f"API retry {what}: rate limited; waiting {delay:.1f}s "
                    f"(retry {attempt + 1}/{self.max_retries})"
                )
                logger.warning(
                    "%s was rate limited; retrying in %.1fs (attempt %d/%d)",
                    what,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(max(0.0, delay))
                attempt += 1
            except CredentialError:
                self._bump("errors")
                raise
            except RemoteAPIError as exc:
                if attempt >= self.max_retries or not getattr(exc, "retryable", False):
                    self._bump("errors")
                    raise
                delay = _backoff(attempt)
                self._trace_api(
                    f"API retry {what}: {exc}; waiting {delay:.1f}s "
                    f"(retry {attempt + 1}/{self.max_retries})"
                )
                logger.warning(
                    "%s failed (%s); retrying in %.1fs (attempt %d/%d)",
                    what,
                    exc,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)
                attempt += 1

    # -- usage bookkeeping ------------------------------------------------- #

    def _bump(self, name: str, delta: float = 1.0) -> None:
        with self._usage_lock:
            self._usage[name] = self._usage.get(name, 0.0) + delta

    def _record_chat_usage(self, completion: ChatCompletionResponse) -> None:
        usage = completion.usage
        with self._usage_lock:
            self._usage["calls"] += 1
            self._usage["prompt_tokens"] += int(usage.prompt_tokens or 0)
            self._usage["completion_tokens"] += int(usage.completion_tokens or 0)
            self._usage["cost_usd"] += float(usage.cost or 0.0)

    def usage_summary(self) -> dict[str, float]:
        """Totals for the dashboard: calls, tokens, dollars, cache hits, errors."""

        with self._usage_lock:
            summary = dict(self._usage)
        summary["total_tokens"] = summary["prompt_tokens"] + summary["completion_tokens"]
        if self.cache is not None:
            summary["cache_entries"] = float(
                sum(int(ns["entries"]) for ns in self.cache.stats()["namespaces"].values())
            )
        return summary

    # -- caching ----------------------------------------------------------- #

    def _cache_get(self, namespace: str, key: str) -> Any | None:
        if self.cache is None:
            return None
        return self.cache.get(namespace, key)

    def _cache_put(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        model: str = "",
        input_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        if self.cache is None:
            return
        self.cache.put(
            namespace,
            key,
            value,
            model=model,
            input_tokens=input_tokens,
            cost_usd=cost_usd,
        )

    # -- weave ------------------------------------------------------------- #

    @staticmethod
    def _weave_log(name: str, payload: dict[str, Any]) -> None:
        """Attach ``payload`` to the active Weave call, if Weave is live.

        Deliberately a no-op unless ``weave`` has already been imported: autorlhf
        never pays the import cost, and never traces, without explicit consent.
        """

        if _tracking is None:
            return
        import sys

        if "weave" not in sys.modules:
            return
        try:
            _tracking.weave_log(name, payload)
        except Exception as exc:  # pragma: no cover - tracing must never break a call
            logger.debug("weave_log failed: %s", exc)

    # -- chat -------------------------------------------------------------- #

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
        """One chat completion, cached by the content hash of the whole request."""

        wire_messages = _normalise_messages(messages)
        request_payload: dict[str, Any] = {
            "model": model,
            "messages": wire_messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "seed": seed,
            "response_format": response_format,
        }
        key = content_hash({"kind": "chat", **request_payload})
        cached = self._cache_get(NS_CHAT, key)
        if isinstance(cached, dict) and "result" in cached:
            result = GenerationResult.model_validate(cached["result"])
            result.cached = True
            self._bump("cache_hits")
            self._trace_api(f"API cache HIT POST /api/v1/chat/completions model={model}; no request sent")
            return result

        self._trace_payload("API request POST https://openrouter.ai/api/v1/chat/completions", request_payload)
        started = time.monotonic()
        try:
            raw = self._request(
                lambda: self._get_sdk().chat.send(
                    model=model,
                    messages=wire_messages,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens),
                    seed=seed,
                    response_format=response_format,
                ),
                what=f"chat completion ({model})",
            )
        except Exception as exc:
            self._trace_api(f"API error POST /api/v1/chat/completions model={model}: {exc}")
            raise
        latency_s = time.monotonic() - started
        self._trace_payload(
            f"API response POST /api/v1/chat/completions model={model} latency={latency_s:.2f}s",
            _to_dict(raw),
        )

        try:
            completion = ChatCompletionResponse.model_validate(_to_dict(raw))
        except ValidationError as exc:
            raise GenerationError(
                f"could not parse the chat completion from {model}: {exc}",
                hint="The provider returned an unexpected shape; retry or pick another model.",
            ) from exc

        if not completion.choices:
            raise GenerationError(
                f"{model} returned no choices",
                hint="Retry the request; if it persists, choose another teacher model.",
            )
        text = completion.first_text()
        finish_reason = completion.finish_reason
        if not text.strip():
            raise GenerationError(
                f"{model} returned an empty completion (finish_reason={finish_reason})",
                hint="Lower the prompt size or raise max_tokens, then retry.",
            )

        result = GenerationResult(
            model_id=model,
            text=text,
            source=model,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            latency_s=latency_s,
            cached=False,
            error=None,
            truncated=finish_reason == "length",
            meta={
                "finish_reason": finish_reason,
                "response_model": completion.model,
                "provider": completion.provider,
                "response_id": completion.id,
                "cost_usd": completion.usage.cost,
                "response_format": response_format,
            },
        )
        self._record_chat_usage(completion)
        self._cache_put(
            NS_CHAT,
            key,
            {"result": result.model_dump(mode="json")},
            model=model,
            input_tokens=completion.usage.prompt_tokens,
            cost_usd=completion.usage.cost,
        )
        self._weave_log(
            "autorlhf.chat",
            {
                "model": model,
                "tokens": completion.usage.total_tokens,
                "cost_usd": completion.usage.cost,
                "truncated": result.truncated,
            },
        )
        return result

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
    ) -> tuple[dict, GenerationResult]:
        """Chat completion in JSON mode, returning ``(parsed_object, result)``.

        ``schema_hint`` is appended as a final system message so the model knows
        the exact shape to emit; it is part of the cache key because it is part
        of the prompt.
        """

        wire = _normalise_messages(messages)
        if schema_hint.strip():
            wire = [*wire, {"role": "system", "content": schema_hint.strip()}]
        result = self.chat(
            model=model,
            messages=wire,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            response_format=response_format or {"type": "json_object"},
        )
        return extract_json_object(result.text), result

    # -- decisions --------------------------------------------------------- #

    def decide(self, request: DecisionsRequest) -> DecisionsResponse:
        """One Jev decision, validated against the frozen wire models."""

        questions = {
            key: question.model_dump(mode="json", exclude_none=True)
            for key, question in request.questions.items()
        }
        payload: dict[str, Any] = {
            "model": request.model,
            "state": request.state,
            "questions": questions,
            "session_id": request.session_id,
            "user": request.user,
        }
        key = content_hash({"kind": "decisions", **payload})
        cached = self._cache_get(NS_DECISIONS, key)
        if isinstance(cached, dict) and "response" in cached:
            self._bump("cache_hits")
            response = DecisionsResponse.model_validate(cached["response"])
            self._check_answers(request, response)
            self._trace_api(f"API cache HIT POST /api/alpha/decisions model={request.model}; no request sent")
            return response

        self._trace_payload("API request POST https://openrouter.ai/api/alpha/decisions", payload)
        started = time.monotonic()
        try:
            raw = self._request(
                lambda: self._decisions_call(request, questions),
                what=f"Jev decision ({request.model})",
            )
        except Exception as exc:
            self._trace_api(f"API error POST /api/alpha/decisions model={request.model}: {exc}")
            raise
        self._trace_payload(
            f"API response POST /api/alpha/decisions model={request.model} latency={time.monotonic() - started:.2f}s",
            _to_dict(raw),
        )
        try:
            response = DecisionsResponse.model_validate(_to_dict(raw))
        except ValidationError as exc:
            raise JudgeError(
                f"could not parse the Jev response from {request.model}: {exc}",
                hint="The judge returned an unexpected shape; retry or pin another judge build.",
            ) from exc

        self._check_answers(request, response)
        with self._usage_lock:
            self._usage["calls"] += 1
            self._usage["prompt_tokens"] += int(response.usage.input_tokens or 0)
            self._usage["completion_tokens"] += int(response.usage.output_tokens or 0)
            self._usage["cost_usd"] += float(response.usage.cost or 0.0)
        self._cache_put(
            NS_DECISIONS,
            key,
            {"response": response.model_dump(mode="json")},
            model=request.model,
            input_tokens=response.usage.input_tokens,
            cost_usd=response.usage.cost,
        )
        self._weave_log(
            "autorlhf.decisions",
            {
                "model": request.model,
                "judge_build": response.model,
                "questions": sorted(questions),
                "input_tokens": response.usage.input_tokens,
                "cost_usd": response.usage.cost,
            },
        )
        return response

    def _decisions_call(self, request: DecisionsRequest, questions: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "state": request.state,
            "questions": questions,
            "server_url": DECISIONS_SERVER_URL,
        }
        if request.session_id:
            kwargs["session_id"] = request.session_id
        if request.user:
            kwargs["user"] = request.user
        return self._get_sdk().alpha.decisions.create(**kwargs)

    @staticmethod
    def _check_answers(request: DecisionsRequest, response: DecisionsResponse) -> None:
        """A missing key or a mismatched answer type is an error, never a default."""

        for key, question in request.questions.items():
            answer = response.answers.get(key)
            if answer is None:
                raise JudgeError(
                    f"the judge returned no answer for question '{key}'",
                    hint="Retry the decision; if it persists, pin a different judge build.",
                )
            if answer.type != question.type:
                raise JudgeError(
                    f"the judge answered '{key}' as '{answer.type}' but '{question.type}' was asked",
                    hint="This is a wire-format mismatch: check the judge model and schema.",
                )

    def decide_many(
        self,
        requests: Sequence[DecisionsRequest],
        *,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_event: Any | None = None,
    ) -> list[DecisionsResponse]:
        """Run several decisions with bounded concurrency, preserving order."""

        total = len(requests)
        if total == 0:
            return []
        results: list[DecisionsResponse | None] = [None] * total
        workers = max(1, min(self.max_concurrency, total))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jev") as pool:
            futures = {}
            for index, request in enumerate(requests):
                if cancel_event is not None and cancel_event.is_set():
                    break
                futures[pool.submit(self.decide, request)] = index
            done = 0
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if on_progress is not None:
                    on_progress(done, total)
        return [item for item in results if item is not None]

    # -- diagnostics ------------------------------------------------------- #

    def verify_credentials(self) -> tuple[bool, str]:
        """Cheap live credential check.  Never raises, never logs the key."""

        try:
            meta = self._request(
                lambda: self._get_sdk().api_keys.get_current_key_metadata(),
                what="credential check",
            )
        except CredentialError as exc:
            return False, f"{exc} {exc.hint}".strip()
        except RemoteAPIError as exc:
            return False, f"Could not verify the OpenRouter credential: {exc}"

        data = getattr(meta, "data", None)
        label = getattr(data, "label", None) if data is not None else None
        suffix = f" (key label: {label})" if label else ""
        return True, f"OpenRouter credential accepted{suffix}."

    def close(self) -> None:
        """Release the SDK client's sockets.  Safe to call more than once."""

        with self._lock:
            sdk, self._sdk = self._sdk, None
        if sdk is None:
            return
        for name in ("close", "close_clients"):
            closer = getattr(sdk, name, None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:  # pragma: no cover - best effort
                    logger.debug("closing the OpenRouter SDK failed: %s", exc)
                return
        try:  # pragma: no cover - depends on the installed SDK build
            import openrouter

            close_all = getattr(openrouter, "close_clients", None)
            if callable(close_all):
                close_all()
        except Exception as exc:  # pragma: no cover
            logger.debug("closing OpenRouter clients failed: %s", exc)


__all__ = [
    "DECISIONS_SERVER_URL",
    "NS_DECISIONS",
    "OpenRouterClient",
    "classify_error",
    "extract_json_object",
]
