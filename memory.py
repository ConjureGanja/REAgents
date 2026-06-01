"""
SQLite-backed memory and knowledge persistence.

Uses a write-queue pattern so the training loop never blocks on I/O:
  - training thread  → puts events onto _queue
  - writer thread    → drains queue and commits to SQLite

Read queries (dashboard, LLM context) are synchronous and safe because SQLite
supports concurrent reads alongside one writer.
"""

import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time   REAL    NOT NULL,
    end_time     REAL,
    total_reward REAL    DEFAULT 0,
    steps        INTEGER DEFAULT 0,
    death_count  INTEGER DEFAULT 0,
    chapter      TEXT    DEFAULT '',
    curriculum   TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS step_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   INTEGER NOT NULL,
    step         INTEGER NOT NULL,
    timestamp    REAL    NOT NULL,
    action       TEXT    NOT NULL,   -- JSON [mv,cam,inter,comb,ev,inv]
    reward       REAL    NOT NULL,
    health_pct   REAL,
    ammo_clip    INTEGER,
    ammo_res     INTEGER,
    enemy_count  INTEGER,
    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE TABLE IF NOT EXISTS llm_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id     INTEGER NOT NULL,
    step           INTEGER NOT NULL,
    timestamp      REAL    NOT NULL,
    model          TEXT    NOT NULL,
    prompt_summary TEXT,
    response       TEXT    NOT NULL,
    decision       TEXT,
    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE TABLE IF NOT EXISTS knowledge (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_step_ep   ON step_log(episode_id);
CREATE INDEX IF NOT EXISTS idx_llm_ep    ON llm_log(episode_id);
CREATE INDEX IF NOT EXISTS idx_step_ts   ON step_log(timestamp);
"""


@dataclass
class EpisodeStart:
    episode_id: int = 0          # explicit ID — avoids AUTOINCREMENT ambiguity
    chapter: str = ""
    curriculum: str = ""
    start_time: float = field(default_factory=time.time)


@dataclass
class EpisodeEnd:
    episode_id: int
    total_reward: float
    steps: int
    death_count: int
    end_time: float = field(default_factory=time.time)


@dataclass
class StepEvent:
    episode_id: int
    step: int
    action: List[int]
    reward: float
    health_pct: float = 1.0
    ammo_clip: int = 0
    ammo_res: int = 0
    enemy_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMEvent:
    episode_id: int
    step: int
    model: str
    response: str
    decision: str = ""
    prompt_summary: str = ""
    timestamp: float = field(default_factory=time.time)


_SENTINEL = object()  # signals writer thread to exit


class MemorySystem:
    """Async-safe SQLite memory with a background writer thread."""

    def __init__(self, db_path: str = "data/re_agent_memory.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue()
        self._episode_id: int = 0   # current episode (set by start_episode)
        self._next_id: int = 1      # monotonic counter, initialised from DB in start()
        self._id_lock = threading.Lock()

        self._writer = threading.Thread(target=self._writer_loop, daemon=True, name="MemWriter")

    def start(self) -> None:
        self._init_schema()
        # Seed the counter from whatever is already in the DB
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM episodes").fetchone()
        conn.close()
        self._next_id = (row[0] or 0) + 1
        self._writer.start()
        logger.info("Memory system started — db: %s  next_episode_id=%d",
                    self._db_path, self._next_id)

    def stop(self) -> None:
        self._queue.put(_SENTINEL)
        self._writer.join(timeout=5)

    # ── Public write API (non-blocking) ───────────────────────────────────────

    def start_episode(self, chapter: str = "", curriculum: str = "") -> int:
        """Assign a new episode_id and queue the DB INSERT — non-blocking."""
        with self._id_lock:
            self._episode_id = self._next_id
            self._next_id += 1

        self._queue.put(EpisodeStart(
            episode_id=self._episode_id,
            chapter=chapter,
            curriculum=curriculum,
        ))
        return self._episode_id

    def end_episode(self, total_reward: float, steps: int, death_count: int) -> None:
        self._queue.put(EpisodeEnd(
            episode_id=self._episode_id,
            total_reward=total_reward,
            steps=steps,
            death_count=death_count,
        ))

    def log_step(self, step: int, action: List[int], reward: float, hud: Dict) -> None:
        self._queue.put(StepEvent(
            episode_id=self._episode_id,
            step=step,
            action=action,
            reward=reward,
            health_pct=float(hud.get("health_pct", 1.0)),
            ammo_clip=int(hud.get("ammo_clip", 0) or 0),
            ammo_res=int(hud.get("ammo_res", 0) or 0),
            enemy_count=hud.get("enemy_count", 0),
        ))

    def log_llm(self, step: int, model: str, response: str,
                decision: str = "", prompt_summary: str = "") -> None:
        self._queue.put(LLMEvent(
            episode_id=self._episode_id,
            step=step,
            model=model,
            response=response,
            decision=decision,
            prompt_summary=prompt_summary,
        ))

    def set_knowledge(self, key: str, value: Any) -> None:
        self._queue.put(("knowledge", key, json.dumps(value), time.time()))

    # ── Public read API (synchronous, safe with WAL) ──────────────────────────

    def get_knowledge(self, key: str) -> Optional[Any]:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT value FROM knowledge WHERE key=?", (key,)).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None

    def get_reward_history(self, last_n: int = 100) -> List[float]:
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute(
            "SELECT total_reward FROM episodes ORDER BY id DESC LIMIT ?", (last_n,)
        ).fetchall()
        conn.close()
        return [r[0] or 0.0 for r in reversed(rows)]

    def get_recent_llm_decisions(self, limit: int = 20) -> List[Dict]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM llm_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_training_summary(self) -> Dict:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("""
            SELECT
                COUNT(id)            AS episode_count,
                SUM(steps)           AS total_steps,
                AVG(total_reward)    AS avg_reward,
                MAX(total_reward)    AS best_reward,
                SUM(death_count)     AS total_deaths
            FROM episodes
        """).fetchone()
        conn.close()
        return {
            "episode_count": row[0] or 0,
            "total_steps": row[1] or 0,
            "avg_reward": round(row[2] or 0.0, 2),
            "best_reward": round(row[3] or 0.0, 2),
            "total_deaths": row[4] or 0,
        }

    # ── Internal writer loop ──────────────────────────────────────────────────

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        batch: List = []

        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if batch:
                    self._flush(conn, batch)
                    batch = []
                continue

            if item is _SENTINEL:
                if batch:
                    self._flush(conn, batch)
                conn.close()
                return

            batch.append(item)
            if len(batch) >= 50:
                self._flush(conn, batch)
                batch = []

    def _flush(self, conn: sqlite3.Connection, batch: List) -> None:
        try:
            with conn:
                for item in batch:
                    if isinstance(item, EpisodeStart):
                        conn.execute(
                            "INSERT INTO episodes (id, start_time, chapter, curriculum) VALUES (?,?,?,?)",
                            (item.episode_id, item.start_time, item.chapter, item.curriculum),
                        )
                    elif isinstance(item, EpisodeEnd):
                        conn.execute(
                            "UPDATE episodes SET end_time=?, total_reward=?, steps=?, death_count=? WHERE id=?",
                            (item.end_time, item.total_reward, item.steps, item.death_count, item.episode_id),
                        )
                    elif isinstance(item, StepEvent):
                        # Convert each action element to a plain Python int before
                        # serialising — SB3 returns numpy int64 from its training loop,
                        # and json.dumps raises TypeError on any numpy scalar.
                        # Analogy: like converting a foreign currency before depositing it —
                        # the bank (json.dumps) only accepts local currency (Python int).
                        safe_action = [int(x) for x in item.action]
                        conn.execute(
                            """INSERT INTO step_log
                               (episode_id,step,timestamp,action,reward,health_pct,ammo_clip,ammo_res,enemy_count)
                               VALUES (?,?,?,?,?,?,?,?,?)""",
                            (item.episode_id, item.step, item.timestamp,
                             json.dumps(safe_action), item.reward,
                             item.health_pct, item.ammo_clip, item.ammo_res, item.enemy_count),
                        )
                    elif isinstance(item, LLMEvent):
                        conn.execute(
                            """INSERT INTO llm_log
                               (episode_id,step,timestamp,model,prompt_summary,response,decision)
                               VALUES (?,?,?,?,?,?,?)""",
                            (item.episode_id, item.step, item.timestamp,
                             item.model, item.prompt_summary, item.response, item.decision),
                        )
                    elif isinstance(item, tuple) and item[0] == "knowledge":
                        _, key, value, ts = item
                        conn.execute(
                            "INSERT OR REPLACE INTO knowledge (key, value, updated_at) VALUES (?,?,?)",
                            (key, value, ts),
                        )
        except sqlite3.Error as exc:
            logger.error("Memory flush error: %s", exc)
