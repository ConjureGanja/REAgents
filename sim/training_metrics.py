"""
Growth Tracking & Evaluation Metrics
====================================

PURPOSE
-------
"See actual growth in the capabilities of the agent" — this module is the
microscope that lets you watch the policy improve in real time.

The standard SB3 / TensorBoard metrics (mean reward, episode length) tell
you *that* learning is happening.  This module tells you *what* is being
learned:

    - Combat skill        — kills per episode, hit accuracy, shots fired
    - Survival skill      — episodes that reach the bell, average HP at end
    - Exploration skill   — % of episodes that reach the barn / find shotgun
    - Item discipline     — herbs picked up, ammo wasted
    - Decision quality    — inventory mid-combat events (a strong anti-pattern)

Why these specifically?  Each maps to a *behaviour we want the agent to learn*
and a *failure mode we want to detect*.  If "kills/episode" is climbing but
"shots fired" is climbing twice as fast, the agent is becoming a spray-and-
pray shooter — not the careful, ammo-conscious Leon Kennedy we want.

Analogy
-------
A football coach doesn't just look at the score.  They track passes
completed, tackles won, time of possession, etc.  Each metric tells a
different part of the story.  We do the same here, with metrics that map
to RE4-specific skills.

OUTPUTS
-------
    runs/<RUN_NAME>/
        eval_log.csv          — one row per eval pass
        learning_curve.png    — multi-panel PNG of all metrics over time
        best_checkpoint.txt   — name of the best policy seen so far

Each row of the CSV looks like:

    timestep, mean_reward, mean_kills, mean_survive_steps, shotgun_rate,
    bell_rate, mean_hp_end, mean_shots, inv_in_combat_per_ep, eval_episodes
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Per-eval summary
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalSummary:
    """
    Aggregated metrics from one evaluation pass (N episodes).

    This is the "skill snapshot" at a single point in training.  Compare
    snapshots over time to see growth.

    Fields
    ------
    timestep         : training step at which this eval ran
    n_episodes       : how many eval episodes were averaged
    mean_reward      : mean total episode reward
    std_reward       : standard deviation of total reward
    mean_kills       : mean enemy kill count per episode
    max_kills        : max kills in a single episode
    mean_survive     : mean survival steps before death/timeout
    bell_rate        : fraction of eps that reached `episode_success`
    shotgun_rate     : fraction of eps that picked up the shotgun
    mean_hp_end      : mean HP at episode end (0..1)
    mean_shots_fired : mean clip-bullets spent per episode
    deaths           : count of episodes that ended in player_dead
    inv_in_combat    : count of inventory opens while enemies were near
    """
    timestep:         int   = 0
    n_episodes:       int   = 0
    mean_reward:      float = 0.0
    std_reward:       float = 0.0
    mean_kills:       float = 0.0
    max_kills:        int   = 0
    mean_survive:     float = 0.0
    bell_rate:        float = 0.0
    shotgun_rate:     float = 0.0
    mean_hp_end:      float = 0.0
    mean_shots_fired: float = 0.0
    deaths:           int   = 0
    inv_in_combat:    int   = 0

    def pretty(self) -> str:
        """Compact human-readable line for log output."""
        return (
            f"step={self.timestep:>9,}  "
            f"R={self.mean_reward:6.2f}±{self.std_reward:4.2f}  "
            f"kills={self.mean_kills:4.1f} (max {self.max_kills})  "
            f"surv={self.mean_survive:5.0f}  "
            f"bell={self.bell_rate:.0%}  "
            f"shotgun={self.shotgun_rate:.0%}  "
            f"HP_end={self.mean_hp_end:.2f}  "
            f"deaths={self.deaths}/{self.n_episodes}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Growth tracker
# ──────────────────────────────────────────────────────────────────────────────

class GrowthTracker:
    """
    Maintains a running history of EvalSummary records and writes them to disk.

    Lives in the main training process, NOT inside SubprocVecEnv workers.
    The trainer calls `record(summary)` after each evaluation pass.

    OUTPUT LIFECYCLE
    ----------------
    Initialise with a run directory.  On each `record()`:
        1. Append a row to eval_log.csv
        2. Re-render learning_curve.png with the full history
        3. Update best_checkpoint.txt if this run beat the best mean_reward

    The PNG is overwritten every time so you always see the latest plot.
    Open it in your file explorer or a previewer that auto-refreshes.

    Why CSV + PNG?  CSV is the source of truth (any plotting tool can read it,
    you can re-plot later, you can grep for milestones).  PNG is the at-a-
    glance dashboard.  Best of both worlds.
    """

    def __init__(self, run_dir: str, run_name: str = "sim_run"):
        self.run_dir   = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name  = run_name
        self.csv_path  = self.run_dir / "eval_log.csv"
        self.png_path  = self.run_dir / "learning_curve.png"
        self.best_path = self.run_dir / "best_checkpoint.txt"

        self.history: List[EvalSummary] = []
        self.best_reward: float = -float("inf")
        self.best_step:   int   = 0

        # Write CSV header on first init (overwrites any previous run's CSV)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(asdict(EvalSummary()).keys()))

        logger.info("GrowthTracker initialised → %s", self.run_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, summary: EvalSummary) -> bool:
        """
        Append a summary to history.  Returns True if this is the best so far.
        """
        self.history.append(summary)
        self._append_csv(summary)
        self._render_png()

        is_best = summary.mean_reward > self.best_reward
        if is_best:
            self.best_reward = summary.mean_reward
            self.best_step   = summary.timestep
            self.best_path.write_text(
                f"step={summary.timestep}\nreward={summary.mean_reward:.4f}\n",
                encoding="utf-8",
            )
            logger.info("★ New best policy at step %d  reward=%.2f",
                        summary.timestep, summary.mean_reward)
        return is_best

    def latest(self) -> Optional[EvalSummary]:
        return self.history[-1] if self.history else None

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _append_csv(self, summary: EvalSummary) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(asdict(summary).values()))

    def _render_png(self) -> None:
        """
        Plot all metrics in a single multi-panel PNG.

        Lazy-imports matplotlib so the module is usable without it (e.g. in a
        worker process where you don't want the heavy import overhead).
        """
        if not self.history:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")           # No interactive window — write file only
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping learning_curve.png")
            return

        steps = [h.timestep for h in self.history]
        if len(steps) < 1:
            return

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), tight_layout=True)
        fig.suptitle(f"Training growth — {self.run_name}", fontsize=14, weight="bold")

        # Panel 1 — mean reward (with std band)
        ax = axes[0, 0]
        means = np.array([h.mean_reward for h in self.history])
        stds  = np.array([h.std_reward  for h in self.history])
        ax.plot(steps, means, color="C0", linewidth=2, label="Mean")
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color="C0")
        ax.set_title("Episode Reward (eval)")
        ax.set_xlabel("Timestep");  ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3);   ax.legend(loc="lower right")

        # Panel 2 — combat skill: kills per episode
        ax = axes[0, 1]
        ax.plot(steps, [h.mean_kills for h in self.history],
                color="C3", linewidth=2, label="Mean kills")
        ax.plot(steps, [h.max_kills  for h in self.history],
                color="C3", linewidth=1, linestyle="--", alpha=0.5, label="Max kills")
        ax.set_title("Combat Skill (kills/ep)")
        ax.set_xlabel("Timestep"); ax.set_ylabel("Kills")
        ax.grid(True, alpha=0.3);  ax.legend(loc="lower right")

        # Panel 3 — survival skill (mean steps before episode end)
        ax = axes[0, 2]
        ax.plot(steps, [h.mean_survive for h in self.history],
                color="C2", linewidth=2)
        ax.set_title("Survival (steps/ep)")
        ax.set_xlabel("Timestep"); ax.set_ylabel("Steps survived")
        ax.grid(True, alpha=0.3)

        # Panel 4 — exploration skill: shotgun + bell rates
        ax = axes[1, 0]
        ax.plot(steps, [h.bell_rate    for h in self.history],
                color="C4", linewidth=2, label="Reached bell")
        ax.plot(steps, [h.shotgun_rate for h in self.history],
                color="C5", linewidth=2, label="Got shotgun")
        ax.set_title("Objective Completion Rate")
        ax.set_xlabel("Timestep"); ax.set_ylabel("Rate (0..1)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3);  ax.legend(loc="lower right")

        # Panel 5 — health discipline: mean HP at episode end
        ax = axes[1, 1]
        ax.plot(steps, [h.mean_hp_end for h in self.history],
                color="C6", linewidth=2)
        ax.set_title("Mean HP at episode end")
        ax.set_xlabel("Timestep"); ax.set_ylabel("HP (0..1)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Panel 6 — anti-pattern detector: inventory mid-combat
        ax = axes[1, 2]
        ax.plot(steps, [h.inv_in_combat for h in self.history],
                color="C9", linewidth=2)
        ax.set_title("Inventory mid-combat (lower = better)")
        ax.set_xlabel("Timestep"); ax.set_ylabel("Count over eval")
        ax.grid(True, alpha=0.3)

        # Annotate latest point on the reward panel for at-a-glance context
        latest = self.history[-1]
        axes[0, 0].annotate(
            f"now: {latest.mean_reward:.1f}",
            xy=(latest.timestep, latest.mean_reward),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, color="C0", weight="bold",
        )

        plt.savefig(self.png_path, dpi=110, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Eval episode runner
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    eval_env,
    model,
    n_episodes: int = 5,
    timestep: int = 0,
    deterministic: bool = True,
) -> EvalSummary:
    """
    Run N evaluation episodes and return an EvalSummary.

    The eval env should be **single, non-vectorised, NOT randomised** — we want
    a clean read on the policy's actual capability.  Use:

        from sim.domain_randomization import DomainRandomizationWrapper, RandomizationConfig
        eval_env = DomainRandomizationWrapper(
            SimResidentEvilEnv(...),
            RandomizationConfig.eval_preset(),
        )

    For RecurrentPPO, we manage the LSTM state manually between steps.

    Analogy
    -------
    Like a chess engine playing a measured rating game — same rules every time,
    no random variation, so any improvement we see is real skill, not luck.
    """
    # Detect RecurrentPPO by class name — more reliable than introspecting
    # predict()'s signature, which can vary across SB3 versions.
    is_recurrent = type(model).__name__ == "RecurrentPPO"

    rewards:    List[float] = []
    kill_counts:List[int]   = []
    survives:   List[int]   = []
    bell_hits = 0
    shotgun   = 0
    hp_ends:    List[float] = []
    shots_fired:List[int]   = []
    deaths    = 0
    inv_combat = 0

    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        ep_reward = 0.0
        ep_kills  = 0
        ep_steps  = 0
        ep_shots  = 0
        ep_inv_in_combat = 0
        terminated = False
        truncated  = False
        info: Dict[str, Any] = {}     # initialise so terminal-step info is in scope

        while not (terminated or truncated):
            # SB3 expects a batched obs for predict().  Wrap each value in
            # a singleton batch dimension.
            obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}

            if is_recurrent:
                action, lstm_states = model.predict(
                    obs_batch,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
            else:
                action, _ = model.predict(obs_batch, deterministic=deterministic)

            action = action[0]
            episode_starts = np.zeros((1,), dtype=bool)

            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += float(reward)
            ep_steps  += 1

            # Shots fired — counted whenever combat action is shoot (2) or aim+shoot (3)
            if int(action[3]) in (2, 3):
                ep_shots += 1
            # Inventory mid-combat anti-pattern — counted only when enemies are present
            if int(action[5]) == 1 and info.get("enemy_count", 0) > 0:
                ep_inv_in_combat += 1

        # End-of-episode bookkeeping.  The episode block in `info` is set by
        # SimResidentEvilEnv on the terminal step and contains the authoritative
        # totals for kills/items/success/shotgun.
        ep_block = info.get("episode", {})
        ep_kills = int(ep_block.get("kills", ep_kills))   # prefer episode total

        rewards.append(ep_reward)
        survives.append(ep_steps)
        kill_counts.append(ep_kills)
        shots_fired.append(ep_shots)
        inv_combat += ep_inv_in_combat
        hp_ends.append(float(info.get("health_pct", 0.0)))

        if ep_block.get("success", False):
            bell_hits += 1
        if ep_block.get("shotgun", False):
            shotgun += 1
        if terminated:           # terminated = player_dead (episode_success uses truncated)
            deaths += 1

    return EvalSummary(
        timestep         = timestep,
        n_episodes       = n_episodes,
        mean_reward      = float(np.mean(rewards)),
        std_reward       = float(np.std(rewards)),
        mean_kills       = float(np.mean(kill_counts)),
        max_kills        = int(np.max(kill_counts)) if kill_counts else 0,
        mean_survive     = float(np.mean(survives)),
        bell_rate        = bell_hits / max(1, n_episodes),
        shotgun_rate     = shotgun  / max(1, n_episodes),
        mean_hp_end      = float(np.mean(hp_ends)),
        mean_shots_fired = float(np.mean(shots_fired)),
        deaths           = deaths,
        inv_in_combat    = inv_combat,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate fake history to verify the plot works
    import random, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    tracker = GrowthTracker("./_metrics_test", "smoke")
    for step in range(0, 200_000, 10_000):
        # Simulated S-curve learning trajectory
        x = step / 200_000
        progress = 1 / (1 + math.exp(-10 * (x - 0.5)))
        s = EvalSummary(
            timestep         = step,
            n_episodes       = 5,
            mean_reward      = -10 + 30 * progress + random.uniform(-3, 3),
            std_reward       = 5.0 - 3.0 * progress,
            mean_kills       = 8.0 * progress + random.uniform(-1, 1),
            max_kills        = int(12 * progress) + random.randint(0, 3),
            mean_survive     = 200 + 400 * progress,
            bell_rate        = progress,
            shotgun_rate     = 0.7 * progress,
            mean_hp_end      = 0.2 + 0.5 * progress,
            mean_shots_fired = 30 - 15 * progress,
            deaths           = max(0, int(5 - 5 * progress)),
            inv_in_combat    = max(0, int(8 - 8 * progress)),
        )
        tracker.record(s)
    print(f"OK — wrote {tracker.png_path}")
    print(f"OK — wrote {tracker.csv_path}")
