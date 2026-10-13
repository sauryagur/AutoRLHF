"""Thread-safe event bus connecting background work to the Textual UI.

The pipeline and trainers never import Textual.  They publish :class:`Event`
objects to an :class:`EventBus`; the TUI subscribes and forwards them onto the
main event loop.  This keeps the engines importable in headless tests.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from autorlhf.core.ids import content_hash

EventKind = Literal[
    "status",
    "phase",
    "progress",
    "log",
    "sample",
    "judgment",
    "metric",
    "warning",
    "error",
    "artifact",
    "counters",
]


class Event(BaseModel):
    """A single observable thing that happened.

    ``seq`` is monotonically increasing per bus, which lets the UI drop stale
    events and lets the in-memory history be replayed in order.
    """

    kind: EventKind
    message: str = ""
    #: Free-form structured payload.  Keep it JSON-serializable.
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: str = ""
    phase: str = ""
    step: int | None = None
    #: 0.0–1.0 when the event describes bounded progress.
    fraction: float | None = None
    #: ``info`` | ``debug`` | ``warning`` | ``error``
    level: str = "info"
    #: True for records the UI only shows in verbose mode (still persisted).
    verbose_only: bool = False
    ts: float = Field(default_factory=time.time)
    seq: int = 0

    def to_jsonl(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def dedupe_key(self) -> str:
        return content_hash(self.model_dump(mode="json", exclude={"ts", "seq"}))


Listener = Callable[[Event], None]


class EventBus:
    """Fan-out bus with a bounded in-memory ring buffer.

    ``emit`` is safe to call from any thread.  Listener exceptions are swallowed
    (and recorded) so one broken consumer cannot kill a training run.
    """

    def __init__(self, *, history_size: int = 2000) -> None:
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._history: list[Event] = []
        self._history_size = history_size
        self._seq = 0
        self._errors: list[str] = []

    # -- subscription ------------------------------------------------------ #

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register ``listener``; returns an unsubscribe callable."""

        with self._lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return _unsubscribe

    # -- publishing -------------------------------------------------------- #

    def emit(
        self,
        kind: EventKind,
        message: str = "",
        *,
        run_id: str = "",
        phase: str = "",
        step: int | None = None,
        fraction: float | None = None,
        level: str = "info",
        verbose_only: bool = False,
        **data: Any,
    ) -> Event:
        with self._lock:
            self._seq += 1
            event = Event(
                kind=kind,
                message=message,
                data=data,
                run_id=run_id,
                phase=phase,
                step=step,
                fraction=fraction,
                level=level,
                verbose_only=verbose_only,
                seq=self._seq,
            )
            listeners = list(self._listeners)
            self._history.append(event)
            if len(self._history) > self._history_size:
                del self._history[: len(self._history) - self._history_size]

        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:  # pragma: no cover - defensive
                with self._lock:
                    self._errors.append(f"listener failed: {exc}")
        return event

    # -- introspection ----------------------------------------------------- #

    def history(self, *, since_seq: int = 0) -> list[Event]:
        with self._lock:
            return [e for e in self._history if e.seq > since_seq]

    def errors(self) -> list[str]:
        with self._lock:
            return list(self._errors)

    def print_listener(self, *, verbose: bool = False) -> Listener:
        """A listener that prints events to stdout — used by ``--headless``."""

        def _print(event: Event) -> None:
            if event.verbose_only and not verbose:
                return
            prefix = f"[{event.phase or event.kind}]"
            if event.kind == "progress" and event.fraction is not None:
                print(f"{prefix} {event.message} ({event.fraction:.0%})", flush=True)
            elif event.kind == "metric":
                print(f"{prefix} {event.message} {event.data.get('value')}", flush=True)
            elif event.kind == "log" and not verbose:
                pass
            else:
                print(f"{prefix} {event.message}", flush=True)

        return _print

    def __iter__(self) -> Iterator[Event]:  # pragma: no cover - convenience
        return iter(self.history())


__all__ = ["Event", "EventBus", "EventKind", "Listener"]
