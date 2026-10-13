"""Foundation tests: SQLite run store, artifact IO, checkpoints, exports, cache.

Covers spec items 6, 7 and 8.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from autorlhf.config.run_config import RunConfig
from autorlhf.core.enums import RunStatus
from autorlhf.storage.cache import NS_CHAT, ResponseCache
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import (
    CHECKPOINT_SENTINEL,
    RunPaths,
    append_jsonl,
    build_manifest,
    checkpoint_step,
    count_jsonl,
    export_run,
    file_hash,
    is_adapter_complete,
    is_checkpoint_complete,
    iter_jsonl,
    list_checkpoint_dirs,
    load_config,
    mark_checkpoint_complete,
    read_json,
    read_jsonl,
    save_config,
    select_resume_checkpoint,
    write_json,
)

# --------------------------------------------------------------------------- #
# item 6 - SQLite run store
# --------------------------------------------------------------------------- #


def test_create_get_and_list_runs(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)

    assert record.status is RunStatus.CREATED
    assert record.is_resumable is True
    assert store.get_run(record.id) == record
    assert store.require_run(record.id).id == record.id
    assert record.id in {r.id for r in store.list_runs()}
    assert store.latest_run() is not None


def test_empty_status_filter_means_no_filter(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)

    assert record.id in {r.id for r in store.list_runs(statuses=[])}


def test_illegal_transition_is_rejected_and_legal_one_recorded(
    store: RunStore, run_config: RunConfig
) -> None:
    record = store.create_run(run_config)

    with pytest.raises(ValueError):
        store.update_status(record.id, RunStatus.COMPLETED)
    assert store.get_run(record.id).status is RunStatus.CREATED

    store.update_status(record.id, RunStatus.PREPARING_DATA, phase="data")
    transitions = store.transitions(record.id)
    assert [(t["from_status"], t["to_status"]) for t in transitions] == [
        ("", "created"),
        ("created", "preparing_data"),
    ]
    assert store.get_run(record.id).phase == "data"


def test_counters_accumulate(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)
    assert store.bump_counter(record.id, "generated") == 1
    assert store.bump_counter(record.id, "generated", 4) == 5
    store.set_counter(record.id, "generated", 2)
    assert store.counters(record.id) == {"generated": 2}


def test_metrics_append_and_latest_metric(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)
    store.add_metric(record.id, "loss", 1.0, step=1)
    store.add_metric(record.id, "loss", 0.5, step=2)

    assert store.metrics(record.id, "loss") == [(1, 1.0), (2, 0.5)]
    assert store.latest_metric(record.id, "loss") == 0.5
    assert store.metric_names(record.id) == ["loss"]
    assert store.latest_metric(record.id, "missing") is None


def test_datasets_are_recorded(store: RunStore, run_config: RunConfig, tmp_path: Path) -> None:
    record = store.create_run(run_config)
    store.record_dataset(
        record.id, "train", tmp_path / "dataset.jsonl", count=3, hash_="abc", meta={"k": 1}
    )

    rows = store.datasets(record.id)
    assert len(rows) == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["count"] == 3
    assert rows[0]["meta"] == {"k": 1}


def test_reviews_upsert_and_review_stats(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)
    store.save_review(record.id, "p0001", prompt_id="p0001", status="pending", payload={"a": 1})
    store.save_review(record.id, "p0001", prompt_id="p0001", status="completed", payload={"a": 2})
    store.save_review(record.id, "p0002", prompt_id="p0002", status="pending")

    assert len(store.list_reviews(record.id)) == 2
    assert store.get_review(record.id, "p0001")["payload"] == {"a": 2}
    assert store.review_stats(record.id) == {"completed": 1, "pending": 1}
    assert len(store.list_reviews(record.id, status="pending")) == 1
    assert store.get_review(record.id, "missing") is None


def test_preferences_upsert_on_pair_id(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)
    store.save_preference(
        record.id, "p0001-a-vs-b", prompt_id="p0001", chosen_id="c1", rejected_id="c0", margin=0.9
    )
    store.save_preference(
        record.id, "p0001-a-vs-b", prompt_id="p0001", chosen_id="c0", rejected_id="c1", margin=0.2
    )

    rows = store.list_preferences(record.id)
    assert len(rows) == 1
    assert rows[0]["chosen_id"] == "c0"
    assert rows[0]["margin"] == 0.2


def test_checkpoints_complete_only_and_last_complete(
    store: RunStore, run_config: RunConfig
) -> None:
    record = store.create_run(run_config)
    store.save_checkpoint(record.id, "/ckpt/checkpoint-10", step=10, complete=True, size_bytes=100)
    store.save_checkpoint(record.id, "/ckpt/checkpoint-20", step=20, complete=False, size_bytes=50)

    assert [c["step"] for c in store.list_checkpoints(record.id)] == [20, 10]
    assert [c["step"] for c in store.list_checkpoints(record.id, complete_only=True)] == [10]
    assert store.last_complete_checkpoint(record.id)["step"] == 10


def test_artifacts_upsert_and_list(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)
    store.save_artifact(record.id, "manifest", "/run/manifest.json", kind="json")
    store.save_artifact(record.id, "manifest", "/run/manifest.json", kind="json", meta={"v": 2})

    rows = store.list_artifacts(record.id)
    assert len(rows) == 1
    assert rows[0]["meta"] == {"v": 2}


def test_run_summary_reports_state(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)
    store.bump_counter(record.id, "generated", 2)
    summary = store.run_summary(record.id)
    assert summary["run"]["id"] == record.id
    assert summary["counters"] == {"generated": 2}


def test_delete_run_removes_every_row(store: RunStore, run_config: RunConfig) -> None:
    record = store.create_run(run_config)
    store.update_status(record.id, RunStatus.PREPARING_DATA)
    store.bump_counter(record.id, "c")
    store.add_metric(record.id, "loss", 1.0)
    store.record_dataset(record.id, "train", "/d.jsonl", count=1)
    store.save_review(record.id, "i1")
    store.save_preference(record.id, "pair", prompt_id="p", chosen_id="a", rejected_id="b")
    store.save_checkpoint(record.id, "/ckpt", step=1, complete=True)
    store.save_artifact(record.id, "manifest", "/m.json")
    store.log_event(record.id, seq=1, kind="log")

    store.delete_run(record.id)

    assert store.get_run(record.id) is None
    assert store.transitions(record.id) == []
    assert store.counters(record.id) == {}
    assert store.metrics(record.id) == []
    assert store.datasets(record.id) == []
    assert store.list_reviews(record.id) == []
    assert store.list_preferences(record.id) == []
    assert store.list_checkpoints(record.id) == []
    assert store.list_artifacts(record.id) == []
    assert store.events(record.id) == []


def test_incomplete_runs_excludes_terminal_statuses(store: RunStore, run_config: RunConfig) -> None:
    completed = store.create_run(run_config)
    store.update_status(completed.id, RunStatus.PREPARING_DATA)
    store.update_status(completed.id, RunStatus.COMPLETED)

    cancelled = store.create_run(run_config)
    store.update_status(cancelled.id, RunStatus.CANCELLED)

    interrupted = store.create_run(run_config)
    store.update_status(interrupted.id, RunStatus.PREPARING_DATA)
    store.update_status(interrupted.id, RunStatus.INTERRUPTED)

    failed = store.create_run(run_config)
    store.update_status(failed.id, RunStatus.FAILED)

    ids = {record.id for record in store.incomplete_runs()}
    assert interrupted.id in ids
    assert failed.id in ids
    assert completed.id not in ids
    assert cancelled.id not in ids


# --------------------------------------------------------------------------- #
# item 7 - artifact IO
# --------------------------------------------------------------------------- #


def test_write_and_read_json_round_trip_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"
    payload = {"b": [1, 2, 3], "a": {"nested": True}}

    write_json(path, payload)

    assert read_json(path) == payload
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []
    assert read_json(tmp_path / "missing.json", default={"x": 1}) == {"x": 1}


def test_append_read_and_count_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    assert append_jsonl(path, {"id": 1}) == 1
    assert append_jsonl(path, [{"id": 2}, {"id": 3}]) == 2

    assert read_jsonl(path) == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert count_jsonl(path) == 3
    assert list(iter_jsonl(path)) == read_jsonl(path)

    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    assert count_jsonl(path) == 4
    assert len(read_jsonl(path)) == 3

    assert append_jsonl(path, []) == 0
    assert read_jsonl(tmp_path / "missing.jsonl") == []
    assert count_jsonl(tmp_path / "missing.jsonl") == 0


def test_file_hash_is_stable_and_missing_is_empty(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("hello", encoding="utf-8")
    second.write_text("hello", encoding="utf-8")

    assert file_hash(first) == file_hash(second)
    assert len(file_hash(first)) == 32

    second.write_text("hello!", encoding="utf-8")
    assert file_hash(first) != file_hash(second)
    assert file_hash(tmp_path / "nope.txt") == ""


def test_save_and_load_config_round_trip(tmp_path: Path, run_config: RunConfig) -> None:
    paths = RunPaths(tmp_path / "run").ensure()
    save_config(paths, run_config)

    loaded = load_config(paths)
    assert loaded == run_config
    assert loaded.fingerprint() == run_config.fingerprint()


# --------------------------------------------------------------------------- #
# item 7 - checkpoints
# --------------------------------------------------------------------------- #


def _make_checkpoint(
    directory: Path,
    *,
    sentinel: bool = True,
    state: bytes = b'{"global_step": 5}',
    weights: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if sentinel:
        (directory / CHECKPOINT_SENTINEL).write_text('{"completed_at": "now"}\n', encoding="utf-8")
    if state is not None:
        (directory / "trainer_state.json").write_bytes(state)
    if weights:
        (directory / "adapter_model.safetensors").write_bytes(b"weights")
    return directory


def test_is_checkpoint_complete_requires_every_part(tmp_path: Path) -> None:
    assert is_checkpoint_complete(_make_checkpoint(tmp_path / "ok")) is True
    assert (
        is_checkpoint_complete(_make_checkpoint(tmp_path / "no-sentinel", sentinel=False)) is False
    )
    assert is_checkpoint_complete(_make_checkpoint(tmp_path / "empty-state", state=b"")) is False
    assert is_checkpoint_complete(_make_checkpoint(tmp_path / "no-weights", weights=False)) is False
    assert is_checkpoint_complete(tmp_path / "missing") is False


def test_mark_checkpoint_complete_writes_the_sentinel(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint-1"
    directory.mkdir()
    (directory / "trainer_state.json").write_text('{"global_step": 1}', encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"w")
    assert is_checkpoint_complete(directory) is False

    mark_checkpoint_complete(directory)

    assert (directory / CHECKPOINT_SENTINEL).is_file()
    assert is_checkpoint_complete(directory) is True
    assert checkpoint_step(directory) == 1


def test_select_resume_checkpoint_ignores_incomplete_newer(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    highest_complete = _make_checkpoint(checkpoints / "checkpoint-10")
    _make_checkpoint(checkpoints / "checkpoint-20", sentinel=False)
    _make_checkpoint(checkpoints / "checkpoint-5")

    assert [p.name for p in list_checkpoint_dirs(checkpoints)] == [
        "checkpoint-20",
        "checkpoint-10",
        "checkpoint-5",
    ]
    assert select_resume_checkpoint(checkpoints) == highest_complete
    assert select_resume_checkpoint(tmp_path / "empty") is None


def test_is_adapter_complete(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert is_adapter_complete(adapter) is False

    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    assert is_adapter_complete(adapter) is True


# --------------------------------------------------------------------------- #
# item 7 - manifest and export
# --------------------------------------------------------------------------- #


def test_build_manifest_contains_no_key_material(tmp_path: Path, run_config: RunConfig) -> None:
    paths = RunPaths(tmp_path / "run").ensure()
    manifest = build_manifest(run_id="run-1", config=run_config, paths=paths)

    text = json.dumps(manifest)
    assert "sk-or-v1-" not in text
    assert "OPENROUTER_API_KEY" not in text
    assert manifest["run_id"] == "run-1"
    assert manifest["fingerprints"]["config"] == run_config.fingerprint()


def test_export_run_is_self_contained_and_secret_free(
    tmp_path: Path, run_config: RunConfig
) -> None:
    root = tmp_path / "run"
    paths = RunPaths(root).ensure()
    save_config(paths, run_config)
    append_jsonl(paths.dataset_path, {"record_type": "candidate", "run_id": "run-1"})
    append_jsonl(paths.preferences_path, {"record_type": "preference", "run_id": "run-1"})

    planted_key = "sk-or-v1-planted1234567890"
    planted_token = "hf_plantedsecret1234567890"
    (root / "config.env").write_text(f"OPENROUTER_API_KEY={planted_key}\n", encoding="utf-8")
    (root / ".env").write_text(f"HF_TOKEN={planted_token}\n", encoding="utf-8")

    destination = export_run(
        run_id="run-1",
        paths=paths,
        config=run_config,
        destination=tmp_path / "export",
    )

    assert (destination / "config.json").is_file()
    assert (destination / "dataset.jsonl").is_file()
    assert (destination / "preferences.jsonl").is_file()
    assert not (destination / "config.env").exists()
    assert not (destination / ".env").exists()

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-1"
    assert manifest["fingerprints"]["config"] == run_config.fingerprint()

    for exported in destination.rglob("*"):
        if not exported.is_file():
            continue
        content = exported.read_text(encoding="utf-8", errors="ignore")
        assert planted_key not in content
        assert planted_token not in content
        assert "OPENROUTER_API_KEY" not in content


# --------------------------------------------------------------------------- #
# item 8 - response cache
# --------------------------------------------------------------------------- #


def test_cache_same_payload_hits_and_changed_payload_misses(store: RunStore) -> None:
    cache = ResponseCache(store, enabled=True)
    key = ResponseCache.make_key({"model": "m", "prompt": "hi"})

    assert cache.get(NS_CHAT, key) is None
    cache.put(NS_CHAT, key, {"text": "hello"}, model="m", input_tokens=5, cost_usd=0.01)
    assert cache.get(NS_CHAT, key) == {"text": "hello"}
    assert cache.hits == 1
    assert cache.misses == 1

    other_key = ResponseCache.make_key({"model": "m", "prompt": "bye"})
    assert other_key != key
    assert cache.get(NS_CHAT, other_key) is None


def test_cache_disabled_performs_no_writes(store: RunStore) -> None:
    cache = ResponseCache(store, enabled=False)
    key = ResponseCache.make_key("payload")

    cache.put(NS_CHAT, key, {"v": 1})

    assert store.cache_stats() == {}
    assert cache.get(NS_CHAT, key) is None


def test_lookup_or_compute_stores_usage_metadata(store: RunStore) -> None:
    cache = ResponseCache(store, enabled=True)
    calls: list[int] = []

    def compute() -> dict[str, str]:
        calls.append(1)
        return {"text": "fresh"}

    def on_store(value: dict[str, str]) -> dict[str, object]:
        return {"model": "openai/gpt-4o-mini", "input_tokens": 42, "cost_usd": 0.25}

    value, from_cache = cache.lookup_or_compute(NS_CHAT, "k1", compute, on_store=on_store)
    assert value == {"text": "fresh"}
    assert from_cache is False
    assert calls == [1]

    stats = store.cache_stats()
    assert stats[NS_CHAT]["entries"] == 1
    assert stats[NS_CHAT]["cost_usd"] == 0.25

    with sqlite3.connect(store.db_path) as connection:
        row = connection.execute(
            "SELECT model, input_tokens, cost_usd FROM cache WHERE namespace=? AND key=?",
            (NS_CHAT, "k1"),
        ).fetchone()
    assert row == ("openai/gpt-4o-mini", 42, 0.25)

    again, from_cache_again = cache.lookup_or_compute(NS_CHAT, "k1", compute, on_store=on_store)
    assert again == {"text": "fresh"}
    assert from_cache_again is True
    assert calls == [1]
