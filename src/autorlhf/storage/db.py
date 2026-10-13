"""SQLite persistence: runs, status transitions, metrics, reviews, cache.

Everything hot (status, counters, decisions, cache) lives in SQLite so an
interrupted process can be resumed.  Bulky payloads (datasets, judgements,
checkpoints) live in files; SQLite stores their paths and hashes.

Design notes:

* WAL mode + a process-wide lock.  Textual runs workers on other threads, and
  the pipeline may run in yet another, so connections are per-call and
  ``check_same_thread=False``.
* Status changes go through :func:`autorlhf.core.enums.assert_transition`, so an
  illegal transition raises instead of being silently persisted.
* Nothing in here ever stores a credential.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autorlhf.config.run_config import RunConfig
from autorlhf.core.enums import (
    RESUMABLE_STATUSES,
    FeedbackMode,
    RunStatus,
    TrainingMethod,
    TrainingObjective,
    assert_transition,
)
from autorlhf.core.ids import content_hash, new_run_id, run_folder_name, slugify

SCHEMA_VERSION = 3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                  TEXT PRIMARY KEY,
    folder              TEXT NOT NULL,
    slug                TEXT NOT NULL,
    status              TEXT NOT NULL,
    phase               TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    model_id            TEXT NOT NULL DEFAULT '',
    revision            TEXT,
    feedback_mode       TEXT NOT NULL DEFAULT 'automated',
    objective           TEXT NOT NULL DEFAULT 'dpo',
    method              TEXT NOT NULL DEFAULT 'lora',
    config_path         TEXT NOT NULL DEFAULT '',
    config_fingerprint  TEXT NOT NULL DEFAULT '',
    dataset_hash        TEXT NOT NULL DEFAULT '',
    step                INTEGER NOT NULL DEFAULT 0,
    last_checkpoint     TEXT,
    parent_run_id       TEXT,
    error               TEXT,
    notes               TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status   TEXT NOT NULL,
    ts          TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_transitions_run ON transitions(run_id, id);

CREATE TABLE IF NOT EXISTS metrics (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT NOT NULL,
    name    TEXT NOT NULL,
    step    INTEGER NOT NULL DEFAULT 0,
    value   REAL NOT NULL,
    ts      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_name ON metrics(run_id, name, step);

CREATE TABLE IF NOT EXISTS counters (
    run_id TEXT NOT NULL,
    name   TEXT NOT NULL,
    value  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, name)
);

CREATE TABLE IF NOT EXISTS datasets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    split      TEXT NOT NULL,
    path       TEXT NOT NULL,
    count      INTEGER NOT NULL DEFAULT 0,
    hash       TEXT NOT NULL DEFAULT '',
    version    TEXT NOT NULL DEFAULT '1',
    created_at TEXT NOT NULL,
    meta_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_datasets_run ON datasets(run_id);

CREATE TABLE IF NOT EXISTS reviews (
    run_id     TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    prompt_id  TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'pending',
    payload    TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(run_id, status);

CREATE TABLE IF NOT EXISTS preferences (
    run_id            TEXT NOT NULL,
    pair_id           TEXT NOT NULL,
    prompt_id         TEXT NOT NULL DEFAULT '',
    chosen_id         TEXT NOT NULL,
    rejected_id       TEXT NOT NULL,
    margin            REAL,
    judge_confidence  REAL,
    source            TEXT NOT NULL DEFAULT 'jev',
    decision          TEXT NOT NULL DEFAULT '',
    reviewer          TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    payload           TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, pair_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    run_id     TEXT NOT NULL,
    path       TEXT NOT NULL,
    step       INTEGER NOT NULL DEFAULT 0,
    complete   INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    metrics    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, path)
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id, step DESC);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id     TEXT NOT NULL,
    name       TEXT NOT NULL,
    path       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'file',
    meta       TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, name)
);

CREATE TABLE IF NOT EXISTS cache (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd   REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (namespace, key)
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT NOT NULL,
    seq     INTEGER NOT NULL DEFAULT 0,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    phase   TEXT NOT NULL DEFAULT '',
    level   TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL DEFAULT '',
    data    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _dumps(payload: Any) -> str:
    return json.dumps(payload, default=str, sort_keys=True)


def _loads(text: str | None) -> Any:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


class RunRecord(BaseModel):
    """Row of the ``runs`` table."""

    id: str
    folder: str
    slug: str
    status: RunStatus
    phase: str = ""
    created_at: str
    updated_at: str
    model_id: str = ""
    revision: str | None = None
    feedback_mode: FeedbackMode = FeedbackMode.AUTOMATED
    objective: TrainingObjective = TrainingObjective.DPO
    method: TrainingMethod = TrainingMethod.LORA
    config_path: str = ""
    config_fingerprint: str = ""
    dataset_hash: str = ""
    step: int = 0
    last_checkpoint: str | None = None
    parent_run_id: str | None = None
    error: str | None = None
    notes: str = ""

    @property
    def is_resumable(self) -> bool:
        return self.status in RESUMABLE_STATUSES

    @property
    def label(self) -> str:
        return f"{self.id} [{self.status.value}] {self.model_id}"


class RunStore:
    """All SQLite access.  Safe to share across threads."""

    def __init__(self, db_path: Path | str, *, event_bus: Any | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._bus = event_bus
        self._local = threading.local()
        self.init_schema()

    # -- plumbing ---------------------------------------------------------- #

    def _connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()

    # -- runs -------------------------------------------------------------- #

    def create_run(
        self,
        config: RunConfig,
        *,
        folder: str | None = None,
        run_id: str | None = None,
        config_path: Path | str = "",
        notes: str = "",
    ) -> RunRecord:
        run_id = run_id or new_run_id()
        folder = folder or run_folder_name(run_id, config.slug or config.character.name)
        now = _now()
        record = RunRecord(
            id=run_id,
            folder=folder,
            slug=slugify(config.slug or config.character.name),
            status=RunStatus.CREATED,
            phase="",
            created_at=now,
            updated_at=now,
            model_id=config.model.model_id,
            revision=config.model.revision,
            feedback_mode=config.feedback_mode,
            objective=config.training.objective,
            method=config.training.method,
            config_path=str(config_path),
            config_fingerprint=config.fingerprint(),
            dataset_hash=config.dataset_fingerprint(),
            step=0,
            notes=notes,
        )
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO runs (id, folder, slug, status, phase, created_at, updated_at,
                                  model_id, revision, feedback_mode, objective, method,
                                  config_path, config_fingerprint, dataset_hash, step, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.id,
                    record.folder,
                    record.slug,
                    record.status.value,
                    record.phase,
                    record.created_at,
                    record.updated_at,
                    record.model_id,
                    record.revision,
                    record.feedback_mode.value,
                    record.objective.value,
                    record.method.value,
                    record.config_path,
                    record.config_fingerprint,
                    record.dataset_hash,
                    record.step,
                    record.notes,
                ),
            )
            conn.execute(
                "INSERT INTO transitions (run_id, from_status, to_status, ts, note) VALUES (?,?,?,?,?)",
                (run_id, "", RunStatus.CREATED.value, now, "created"),
            )
            conn.commit()
        self._emit("status", f"run {run_id} created", run_id=run_id, status=record.status.value)
        return record

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            folder=row["folder"],
            slug=row["slug"],
            status=RunStatus(row["status"]),
            phase=row["phase"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            model_id=row["model_id"],
            revision=row["revision"],
            feedback_mode=FeedbackMode(row["feedback_mode"]),
            objective=TrainingObjective(row["objective"]),
            method=TrainingMethod(row["method"]),
            config_path=row["config_path"],
            config_fingerprint=row["config_fingerprint"],
            dataset_hash=row["dataset_hash"],
            step=row["step"],
            last_checkpoint=row["last_checkpoint"],
            parent_run_id=row["parent_run_id"],
            error=row["error"],
            notes=row["notes"],
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._connect().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def require_run(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(f"unknown run '{run_id}'")
        return record

    def list_runs(
        self,
        *,
        limit: int = 100,
        statuses: Iterable[RunStatus | str] | None = None,
    ) -> list[RunRecord]:
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        values = [RunStatus(s).value for s in statuses] if statuses else []
        if values:
            sql += f" WHERE status IN ({','.join('?' * len(values))})"
            params.extend(values)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def incomplete_runs(self) -> list[RunRecord]:
        """Runs a restarting app should offer to resume, inspect, or abandon."""

        return self.list_runs(
            limit=200,
            statuses=[s for s in RESUMABLE_STATUSES if s is not RunStatus.CREATED],
        )

    def latest_run(self) -> RunRecord | None:
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    # -- status ------------------------------------------------------------ #

    def update_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        phase: str | None = None,
        error: str | None = None,
        note: str = "",
        step: int | None = None,
        validate: bool = True,
    ) -> RunRecord:
        new_status = RunStatus(status)
        current = self.require_run(run_id)
        if validate:
            assert_transition(current.status, new_status)
        now = _now()
        fields = ["status=?", "updated_at=?"]
        params: list[Any] = [new_status.value, now]
        if phase is not None:
            fields.append("phase=?")
            params.append(phase)
        if error is not None:
            fields.append("error=?")
            params.append(error)
        if step is not None:
            fields.append("step=?")
            params.append(step)
        if note:
            fields.append("notes=?")
            params.append(note)
        params.append(run_id)
        with self._lock:
            conn = self._connect()
            conn.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id=?", params)
            conn.execute(
                "INSERT INTO transitions (run_id, from_status, to_status, ts, note) VALUES (?,?,?,?,?)",
                (run_id, current.status.value, new_status.value, now, note or ""),
            )
            conn.commit()
        self._emit(
            "status",
            f"{current.status.value} -> {new_status.value}",
            run_id=run_id,
            status=new_status.value,
            previous=current.status.value,
            phase=phase or current.phase,
            error=error,
        )
        return self.require_run(run_id)

    def set_step(self, run_id: str, step: int) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute("UPDATE runs SET step=?, updated_at=? WHERE id=?", (step, _now(), run_id))
            conn.commit()

    def set_last_checkpoint(self, run_id: str, path: str | None, step: int | None = None) -> None:
        with self._lock:
            conn = self._connect()
            if step is None:
                conn.execute(
                    "UPDATE runs SET last_checkpoint=?, updated_at=? WHERE id=?",
                    (path, _now(), run_id),
                )
            else:
                conn.execute(
                    "UPDATE runs SET last_checkpoint=?, step=?, updated_at=? WHERE id=?",
                    (path, step, _now(), run_id),
                )
            conn.commit()

    def transitions(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = (
                self._connect()
                .execute("SELECT * FROM transitions WHERE run_id=? ORDER BY id", (run_id,))
                .fetchall()
            )
        return [dict(r) for r in rows]

    # -- counters ---------------------------------------------------------- #

    def bump_counter(self, run_id: str, name: str, delta: int = 1) -> int:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO counters(run_id, name, value) VALUES (?,?,?)
                ON CONFLICT(run_id, name) DO UPDATE SET value = value + excluded.value
                """,
                (run_id, name, delta),
            )
            conn.commit()
            row = conn.execute(
                "SELECT value FROM counters WHERE run_id=? AND name=?", (run_id, name)
            ).fetchone()
        value = int(row["value"]) if row else 0
        self._emit("counters", name, run_id=run_id, name_key=name, value=value)
        return value

    def set_counter(self, run_id: str, name: str, value: int) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO counters(run_id, name, value) VALUES (?,?,?)
                ON CONFLICT(run_id, name) DO UPDATE SET value = excluded.value
                """,
                (run_id, name, value),
            )
            conn.commit()
        self._emit("counters", name, run_id=run_id, name_key=name, value=value)

    def counters(self, run_id: str) -> dict[str, int]:
        with self._lock:
            rows = (
                self._connect()
                .execute("SELECT name, value FROM counters WHERE run_id=?", (run_id,))
                .fetchall()
            )
        return {r["name"]: int(r["value"]) for r in rows}

    # -- metrics ----------------------------------------------------------- #

    def add_metric(self, run_id: str, name: str, value: float, step: int = 0) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO metrics(run_id, name, step, value, ts) VALUES (?,?,?,?,?)",
                (run_id, name, step, float(value), _now()),
            )
            conn.commit()
        self._emit("metric", name, run_id=run_id, name_key=name, value=float(value), step=step)

    def metrics(self, run_id: str, name: str | None = None) -> list[tuple[int, float]]:
        sql = "SELECT step, value FROM metrics WHERE run_id=?"
        params: list[Any] = [run_id]
        if name:
            sql += " AND name=?"
            params.append(name)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
        return [(int(r["step"]), float(r["value"])) for r in rows]

    def metric_names(self, run_id: str) -> list[str]:
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    "SELECT DISTINCT name FROM metrics WHERE run_id=? ORDER BY name", (run_id,)
                )
                .fetchall()
            )
        return [r["name"] for r in rows]

    def latest_metric(self, run_id: str, name: str) -> float | None:
        with self._lock:
            row = (
                self._connect()
                .execute(
                    "SELECT value FROM metrics WHERE run_id=? AND name=? ORDER BY id DESC LIMIT 1",
                    (run_id, name),
                )
                .fetchone()
            )
        return float(row["value"]) if row else None

    # -- datasets ---------------------------------------------------------- #

    def record_dataset(
        self,
        run_id: str,
        split: str,
        path: Path | str,
        *,
        count: int,
        hash_: str = "",
        version: str = "1",
        meta: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO datasets(run_id, split, path, count, hash, version, created_at, meta_json)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (run_id, split, str(path), count, hash_, version, _now(), _dumps(meta or {})),
            )
            conn.commit()

    def datasets(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = (
                self._connect()
                .execute("SELECT * FROM datasets WHERE run_id=? ORDER BY id", (run_id,))
                .fetchall()
            )
        return [{**dict(r), "meta": _loads(r["meta_json"])} for r in rows]

    # -- reviews ----------------------------------------------------------- #

    def save_review(
        self,
        run_id: str,
        item_id: str,
        *,
        prompt_id: str = "",
        status: str = "pending",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a review item.  ``status`` is ``pending``/``completed``/``skipped``."""

        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO reviews(run_id, item_id, prompt_id, status, payload, updated_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(run_id, item_id) DO UPDATE SET
                    status=excluded.status,
                    payload=excluded.payload,
                    prompt_id=excluded.prompt_id,
                    updated_at=excluded.updated_at
                """,
                (run_id, item_id, prompt_id, status, _dumps(payload or {}), _now()),
            )
            conn.commit()

    def get_review(self, run_id: str, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = (
                self._connect()
                .execute("SELECT * FROM reviews WHERE run_id=? AND item_id=?", (run_id, item_id))
                .fetchone()
            )
        if row is None:
            return None
        data = dict(row)
        data["payload"] = _loads(row["payload"])
        return data

    def list_reviews(
        self, run_id: str, *, status: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM reviews WHERE run_id=?"
        params: list[Any] = [run_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY updated_at, item_id"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["payload"] = _loads(row["payload"])
            out.append(data)
        return out

    def review_stats(self, run_id: str) -> dict[str, int]:
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    "SELECT status, COUNT(*) AS n FROM reviews WHERE run_id=? GROUP BY status",
                    (run_id,),
                )
                .fetchall()
            )
        return {r["status"]: int(r["n"]) for r in rows}

    # -- preferences ------------------------------------------------------- #

    def save_preference(
        self,
        run_id: str,
        pair_id: str,
        *,
        prompt_id: str,
        chosen_id: str,
        rejected_id: str,
        margin: float | None = None,
        judge_confidence: float | None = None,
        source: str = "jev",
        decision: str = "",
        reviewer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO preferences(run_id, pair_id, prompt_id, chosen_id, rejected_id,
                                        margin, judge_confidence, source, decision, reviewer,
                                        created_at, payload)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id, pair_id) DO UPDATE SET
                    chosen_id=excluded.chosen_id,
                    rejected_id=excluded.rejected_id,
                    margin=excluded.margin,
                    judge_confidence=excluded.judge_confidence,
                    source=excluded.source,
                    decision=excluded.decision,
                    reviewer=excluded.reviewer,
                    payload=excluded.payload
                """,
                (
                    run_id,
                    pair_id,
                    prompt_id,
                    chosen_id,
                    rejected_id,
                    margin,
                    judge_confidence,
                    source,
                    decision,
                    reviewer,
                    _now(),
                    _dumps(payload or {}),
                ),
            )
            conn.commit()

    def list_preferences(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    "SELECT * FROM preferences WHERE run_id=? ORDER BY created_at, pair_id",
                    (run_id,),
                )
                .fetchall()
            )
        return [{**dict(r), "payload": _loads(r["payload"])} for r in rows]

    # -- checkpoints ------------------------------------------------------- #

    def save_checkpoint(
        self,
        run_id: str,
        path: Path | str,
        *,
        step: int,
        complete: bool,
        size_bytes: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO checkpoints(run_id, path, step, complete, size_bytes, metrics, created_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(run_id, path) DO UPDATE SET
                    step=excluded.step,
                    complete=excluded.complete,
                    size_bytes=excluded.size_bytes,
                    metrics=excluded.metrics,
                    created_at=excluded.created_at
                """,
                (run_id, str(path), step, int(complete), size_bytes, _dumps(metrics or {}), _now()),
            )
            conn.commit()

    def list_checkpoints(self, run_id: str, *, complete_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM checkpoints WHERE run_id=?"
        if complete_only:
            sql += " AND complete=1"
        sql += " ORDER BY step DESC, created_at DESC"
        with self._lock:
            rows = self._connect().execute(sql, (run_id,)).fetchall()
        return [
            {**dict(r), "complete": bool(r["complete"]), "metrics": _loads(r["metrics"])}
            for r in rows
        ]

    def last_complete_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        items = self.list_checkpoints(run_id, complete_only=True)
        return items[0] if items else None

    # -- artifacts --------------------------------------------------------- #

    def save_artifact(
        self,
        run_id: str,
        name: str,
        path: Path | str,
        *,
        kind: str = "file",
        meta: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO artifacts(run_id, name, path, kind, meta, created_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(run_id, name) DO UPDATE SET
                    path=excluded.path, kind=excluded.kind, meta=excluded.meta,
                    created_at=excluded.created_at
                """,
                (run_id, name, str(path), kind, _dumps(meta or {}), _now()),
            )
            conn.commit()

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = (
                self._connect()
                .execute("SELECT * FROM artifacts WHERE run_id=? ORDER BY name", (run_id,))
                .fetchall()
            )
        return [{**dict(r), "meta": _loads(r["meta"])} for r in rows]

    # -- cache ------------------------------------------------------------- #

    @staticmethod
    def cache_key(namespace: str, payload: Any) -> str:
        return content_hash({"namespace": namespace, "payload": payload})

    def cache_get(self, namespace: str, key: str) -> Any | None:
        with self._lock:
            row = (
                self._connect()
                .execute("SELECT value FROM cache WHERE namespace=? AND key=?", (namespace, key))
                .fetchone()
            )
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return None

    def cache_put(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        model: str = "",
        input_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO cache(namespace, key, value, created_at, model, input_tokens, cost_usd)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value=excluded.value, model=excluded.model,
                    input_tokens=excluded.input_tokens, cost_usd=excluded.cost_usd
                """,
                (
                    namespace,
                    key,
                    json.dumps(value, default=str),
                    _now(),
                    model,
                    int(input_tokens),
                    float(cost_usd),
                ),
            )
            conn.commit()

    def cache_stats(self) -> dict[str, Any]:
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    "SELECT namespace, COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost "
                    "FROM cache GROUP BY namespace"
                )
                .fetchall()
            )
        return {
            r["namespace"]: {"entries": int(r["n"]), "cost_usd": round(float(r["cost"]), 6)}
            for r in rows
        }

    # -- events ------------------------------------------------------------ #

    def log_event(
        self,
        run_id: str,
        *,
        seq: int,
        kind: str,
        phase: str = "",
        level: str = "info",
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO events(run_id, seq, ts, kind, phase, level, message, data) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, seq, _now(), kind, phase, level, message, _dumps(data or {})),
            )
            conn.commit()

    def events(self, run_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)
                )
                .fetchall()
            )
        return [{**dict(r), "data": _loads(r["data"])} for r in reversed(rows)]

    # -- maintenance ------------------------------------------------------- #

    def delete_run(self, run_id: str) -> None:
        """Remove a run's rows.  On-disk artifacts are the caller's business."""

        with self._lock:
            conn = self._connect()
            for table in (
                "runs",
                "transitions",
                "metrics",
                "counters",
                "datasets",
                "reviews",
                "preferences",
                "checkpoints",
                "artifacts",
                "events",
            ):
                column = "id" if table == "runs" else "run_id"
                conn.execute(f"DELETE FROM {table} WHERE {column}=?", (run_id,))
            conn.commit()

    def run_summary(self, run_id: str) -> dict[str, Any]:
        """Everything the dashboard shows for one run."""

        record = self.require_run(run_id)
        return {
            "run": record.model_dump(mode="json"),
            "counters": self.counters(run_id),
            "datasets": self.datasets(run_id),
            "reviews": self.review_stats(run_id),
            "preferences": len(self.list_preferences(run_id)),
            "checkpoints": len(self.list_checkpoints(run_id, complete_only=True)),
            "last_checkpoint": self.last_complete_checkpoint(run_id),
            "metrics": {
                name: (series[-1] if (series := self.metrics(run_id, name)) else None)
                for name in self.metric_names(run_id)
            },
            "artifacts": self.list_artifacts(run_id),
            "transitions": self.transitions(run_id)[-20:],
        }

    # -- helpers ----------------------------------------------------------- #

    def _emit(self, kind: str, message: str, *, run_id: str, **data: Any) -> None:
        if self._bus is not None:
            try:
                self._bus.emit(kind, message, run_id=run_id, **data)
            except Exception:  # pragma: no cover - never fail a DB write for the UI
                pass


__all__ = ["SCHEMA_VERSION", "RunRecord", "RunStore", "run_folder_name"]
