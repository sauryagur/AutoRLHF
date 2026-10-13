"""Foundation tests: status machine, id helpers, and the event bus.

Covers spec items 5 and 9.
"""

from __future__ import annotations

import re
import time

import pytest

from autorlhf.core.enums import (
    RESUMABLE_STATUSES,
    TERMINAL_STATUSES,
    RunStatus,
    assert_transition,
    can_transition,
)
from autorlhf.core.events import EventBus
from autorlhf.core.ids import canonical_json, content_hash, new_run_id, run_folder_name, slugify

_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4,}$")

# --------------------------------------------------------------------------- #
# item 5 - status machine
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        ("created", "preparing_data"),
        (RunStatus.PAUSED, RunStatus.TRAINING_DPO),
        (RunStatus.INTERRUPTED, RunStatus.TRAINING_SFT),
        (RunStatus.PREPARING_DATA, RunStatus.COMPLETED),
        (RunStatus.FAILED, RunStatus.PREPARING_DATA),
        (RunStatus.GENERATING, RunStatus.REVIEWING),
        (RunStatus.EVALUATING_FINAL, RunStatus.COMPLETED),
    ],
)
def test_legal_transitions(src: RunStatus | str, dst: RunStatus | str) -> None:
    assert can_transition(src, dst) is True
    assert_transition(src, dst)


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        ("created", "completed"),
        ("created", "training_sft"),
        (RunStatus.TRAINING_SFT, RunStatus.REVIEWING),
        (RunStatus.COMPLETED, RunStatus.FAILED),
    ],
)
def test_illegal_transitions(src: RunStatus | str, dst: RunStatus | str) -> None:
    assert can_transition(src, dst) is False
    with pytest.raises(ValueError):
        assert_transition(src, dst)


def test_completed_reaches_nothing_else() -> None:
    for status in RunStatus:
        if status is RunStatus.COMPLETED:
            continue
        assert can_transition(RunStatus.COMPLETED, status) is False


def test_same_status_is_always_allowed() -> None:
    for status in RunStatus:
        assert can_transition(status, status) is True


def test_resumable_and_terminal_status_sets() -> None:
    assert RunStatus.PAUSED in RESUMABLE_STATUSES
    assert RunStatus.INTERRUPTED in RESUMABLE_STATUSES
    assert RunStatus.FAILED in RESUMABLE_STATUSES
    assert RunStatus.COMPLETED not in RESUMABLE_STATUSES
    assert RunStatus.COMPLETED in TERMINAL_STATUSES
    assert RunStatus.CANCELLED in TERMINAL_STATUSES
    assert not (RESUMABLE_STATUSES & TERMINAL_STATUSES)


# --------------------------------------------------------------------------- #
# id helpers
# --------------------------------------------------------------------------- #


def test_slugify_and_run_folder_name() -> None:
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("!!!") == "run"
    assert slugify("a" * 50, max_len=5) == "aaaaa"
    assert run_folder_name("20260101-120000-ab12", "My Run") == "20260101-120000-ab12-my-run"


def test_new_run_id_has_a_sortable_timestamped_format() -> None:
    fixed = 1_700_000_000.0
    prefix = time.strftime("%Y%m%d-%H%M%S", time.localtime(fixed))
    ids = [new_run_id(now=fixed) for _ in range(20)]

    assert all(_RUN_ID_RE.match(run_id) for run_id in ids)
    assert all(run_id.startswith(f"{prefix}-") for run_id in ids)

    # The id is the `runs` primary key, so same-second ids must not collide.
    # 200 draws at 4 bytes of entropy collide with ~5e-6 probability.
    assert len({new_run_id(now=fixed) for _ in range(200)}) == 200

    later_prefix = time.strftime("%Y%m%d-%H%M%S", time.localtime(fixed + 1))
    assert later_prefix > prefix
    assert new_run_id(now=fixed)[:15] < new_run_id(now=fixed + 1)[:15]


def test_content_hash_is_deterministic_and_key_order_independent() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})
    assert len(content_hash("payload")) == 32
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


# --------------------------------------------------------------------------- #
# item 9 - event bus
# --------------------------------------------------------------------------- #


def test_sequence_numbers_increase_monotonically(bus: EventBus) -> None:
    seqs = [bus.emit("log", f"message-{index}").seq for index in range(5)]
    assert len(set(seqs)) == 5
    assert seqs == sorted(seqs)
    assert [event.seq for event in bus.history()] == seqs


def test_broken_listener_does_not_break_other_listeners(bus: EventBus) -> None:
    received = []

    def bad_listener(event) -> None:
        raise RuntimeError("listener boom")

    bus.subscribe(bad_listener)
    bus.subscribe(received.append)
    bus.emit("log", "hello")

    assert len(received) == 1
    assert any("boom" in error for error in bus.errors())


def test_history_records_every_event(bus: EventBus) -> None:
    bus.emit("status", "a")
    bus.emit("metric", "b", value=1.0)

    assert [event.kind for event in bus.history()] == ["status", "metric"]


def test_history_is_bounded_by_history_size() -> None:
    bus = EventBus(history_size=3)
    for index in range(6):
        bus.emit("log", str(index))

    history = bus.history()
    assert [event.message for event in history] == ["3", "4", "5"]
    assert [event.message for event in bus.history(since_seq=history[0].seq)] == ["4", "5"]


def test_unsubscribe_stops_delivery(bus: EventBus) -> None:
    received = []
    unsubscribe = bus.subscribe(received.append)
    bus.emit("log", "first")
    unsubscribe()
    bus.emit("log", "second")
    assert [event.message for event in received] == ["first"]


def test_emit_carries_structured_payload(bus: EventBus) -> None:
    event = bus.emit(
        "progress",
        "halfway",
        run_id="run-1",
        phase="train",
        step=3,
        fraction=0.5,
        value=7,
    )
    assert event.kind == "progress"
    assert event.data == {"value": 7}
    assert event.fraction == 0.5
    assert event.step == 3
    assert event.run_id == "run-1"
    assert event.phase == "train"
    assert event.to_jsonl()["kind"] == "progress"
