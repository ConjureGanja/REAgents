"""
Thread-safe shared state container.
All modules read/write through this object so no direct inter-thread references are needed.

SIMULATION EXTENSIONS (sim_mode fields)
─────────────────────────────────────────
When running in simulation mode (sim_main.py), several extra fields are populated:

  worker_snapshots      — lightweight dict per worker (positions, health, enemy states)
                          used by the dashboard to render the 2D village grid without
                          pickling full numpy frames across process boundaries.
  worker_rewards        — most-recent episode reward per worker (list of N floats).
  worker_episodes       — episode counter per worker.
  sim_steps_per_sec     — real-time throughput; shown in the dashboard speed panel.
  objectives_log        — rolling list of objective events (e.g. "W3 reached barn").
                          Analogy: a radio dispatch log — each worker calls in when
                          something notable happens.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SharedState:
    # ── Current game frame (BGR numpy, updated every step) ────────────────────
    frame: Optional[np.ndarray] = None

    # ── Parsed HUD data ───────────────────────────────────────────────────────
    hud: Dict[str, Any] = field(default_factory=lambda: {
        "health_pct": 1.0,
        "ammo_clip": 0,
        "ammo_res": 0,
    })

    # ── YOLO detections from perception system ────────────────────────────────
    detections: List[Dict] = field(default_factory=list)

    # ── LLM outputs ───────────────────────────────────────────────────────────
    gpt_analysis: str = ""
    claude_plan: str = ""
    grok_tactical: str = ""
    llm_objective: str = ""          # High-level goal set by LLM for reward shaping
    llm_action_override: Optional[List[int]] = None  # If LLM wants to hard-override RL

    # ── RL training metrics ───────────────────────────────────────────────────
    episode_count: int = 0
    total_steps: int = 0
    episode_reward: float = 0.0
    best_reward: float = float("-inf")
    reward_history: List[float] = field(default_factory=list)  # per-episode totals
    step_reward_history: List[float] = field(default_factory=list)  # recent step rewards
    episode_length_history: List[int] = field(default_factory=list)
    death_count: int = 0

    # ── Last RL action and reward ─────────────────────────────────────────────
    current_action: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    last_reward: float = 0.0

    # ── Control flags ─────────────────────────────────────────────────────────
    is_training: bool = False
    stop_requested: bool = False
    force_llm_call: bool = False
    curriculum_stage: str = "exploration"
    game_focused: bool = True    # False while the game window is not in the foreground

    # paused is DISTINCT from stop_requested:
    #   paused       — temporarily suspends action execution; training loop keeps
    #                  running (steps accumulate, rewards compute) but NO keyboard
    #                  or mouse inputs are sent to the game.  Resumable at any time.
    #   stop_requested — terminates the entire training session; not resumable.
    #
    # Analogy: paused is like a TV remote's ⏸ button (you can press ▶ to continue);
    # stop_requested is like unplugging the TV (you have to restart from scratch).
    #
    # auto_pause_on_focus_loss — when True (default), the focus-monitor thread
    # automatically sets paused=True whenever the RE4 window loses foreground focus,
    # and auto-resumes (paused=False) when the game regains focus.
    # Set to False if you prefer to manage pausing manually.
    paused: bool = False
    auto_pause_on_focus_loss: bool = True

    # ── Simulation-mode fields (populated by sim_trainer.py) ──────────────────
    #
    # worker_snapshots: one dict per worker containing:
    #   {"player_pos": (x,y), "player_health": float, "enemy_positions": [...],
    #    "items_remaining": int, "step": int, "episode_reward": float}
    # These are small (< 1 KB each) and safe to ship across process pipes.
    #
    # sim_steps_per_sec: measured throughput across all workers combined.
    #   Analogy: the speedometer on the training run — shows how much faster
    #   the simulation is compared to waiting for the real game.
    #
    # objectives_log: circular buffer of notable events from any worker.
    #   Example entries: "W2 killed 3 enemies", "W5 collected shotgun"
    worker_snapshots: List[Dict] = field(default_factory=list)
    worker_rewards: List[float] = field(default_factory=list)
    worker_episodes: List[int] = field(default_factory=list)
    worker_positions: List[Tuple[float, float]] = field(default_factory=list)
    sim_steps_per_sec: float = 0.0
    objectives_log: List[str] = field(default_factory=list)   # rolling 200-entry log

    # ── Internal lock — not part of the public interface ─────────────────────
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a deep-enough copy for the dashboard to render safely."""
        with self._lock:
            return {
                "frame": self.frame.copy() if self.frame is not None else None,
                "hud": dict(self.hud),
                "detections": list(self.detections),
                "gpt_analysis": self.gpt_analysis,
                "claude_plan": self.claude_plan,
                "grok_tactical": self.grok_tactical,
                "llm_objective": self.llm_objective,
                "episode_count": self.episode_count,
                "total_steps": self.total_steps,
                "episode_reward": self.episode_reward,
                "best_reward": self.best_reward,
                "reward_history": list(self.reward_history),
                "step_reward_history": list(self.step_reward_history[-200:]),
                "episode_length_history": list(self.episode_length_history),
                "death_count": self.death_count,
                "current_action": list(self.current_action),
                "last_reward": self.last_reward,
                "is_training": self.is_training,
                "paused": self.paused,
                "auto_pause_on_focus_loss": self.auto_pause_on_focus_loss,
                "curriculum_stage": self.curriculum_stage,
                "game_focused": self.game_focused,
                # Simulation fields
                "worker_snapshots": list(self.worker_snapshots),
                "worker_rewards": list(self.worker_rewards),
                "worker_episodes": list(self.worker_episodes),
                "sim_steps_per_sec": self.sim_steps_per_sec,
                "objectives_log": list(self.objectives_log[-50:]),
            }

    def log_objective_event(self, message: str) -> None:
        """Append a notable simulation event to the objectives log (capped at 200)."""
        with self._lock:
            self.objectives_log.append(message)
            if len(self.objectives_log) > 200:
                self.objectives_log.pop(0)

    def update_worker(self, worker_id: int, snapshot: Dict, reward: float) -> None:
        """
        Update per-worker data from the DashboardCallback.

        This is called once per VecEnv step for each worker — must be fast.
        Initialises the lists on first call so we don't need to know the worker
        count at SharedState construction time.

        Analogy: like a sports scoreboard operator updating one player's stats
        after each play — must be quick so the board stays current.
        """
        with self._lock:
            # Grow lists lazily on first encounter of a new worker_id
            while len(self.worker_snapshots) <= worker_id:
                self.worker_snapshots.append({})
                self.worker_rewards.append(0.0)
                self.worker_episodes.append(0)
                self.worker_positions.append((25.0, 42.0))

            self.worker_snapshots[worker_id] = snapshot
            self.worker_rewards[worker_id] = reward
            pos = snapshot.get("player_pos", (25.0, 42.0))
            self.worker_positions[worker_id] = (float(pos[0]), float(pos[1]))

    def append_reward(self, step_reward: float) -> None:
        with self._lock:
            self.step_reward_history.append(step_reward)
            if len(self.step_reward_history) > 500:
                self.step_reward_history.pop(0)

    def finalize_episode(self, total_reward: float, length: int) -> None:
        with self._lock:
            self.episode_count += 1
            self.episode_reward = total_reward
            self.reward_history.append(total_reward)
            self.episode_length_history.append(length)
            if total_reward > self.best_reward:
                self.best_reward = total_reward
            # Keep last 500 episodes
            if len(self.reward_history) > 500:
                self.reward_history.pop(0)
            if len(self.episode_length_history) > 500:
                self.episode_length_history.pop(0)
