"""
Combat-specific metric tracking for the Combat curriculum stage.

Spec'd in NEXT_STEPS.md §1.3: during combat-stage training the dashboard needs
accuracy (kills per shot), efficiency (damage dealt vs taken), and rolling
kill counts — metrics the generic reward curve can't show.

Data is derived from HUD deltas by MemoryCallback (trainer.py):
  shots  = ammo clip decrease between steps
  kills  = enemy-count decrease within 1.5 s of a shot
  damage = validated health drop
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CombatMetrics:
    shots_fired: int = 0
    kills: int = 0
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    headshots: int = 0        # if detectable
    melee_kills: int = 0

    @property
    def accuracy(self) -> float:
        """Kill/shot ratio (proxy for accuracy)."""
        return self.kills / max(1, self.shots_fired)

    @property
    def efficiency(self) -> float:
        """Damage dealt per damage taken ratio."""
        return self.damage_dealt / max(0.1, self.damage_taken)


class CombatLogger:
    """Accumulates per-episode combat stats and rolling-window summaries."""

    # A kill only counts if a shot happened this recently (YOLO dropouts alone
    # must not count — same rule as the reward function).
    KILL_WINDOW_S = 1.5

    def __init__(self) -> None:
        self.episode_metrics: List[CombatMetrics] = []
        self.current = CombatMetrics()
        self._last_shot_t: float = 0.0

    def log_shot(self, rounds: int = 1) -> None:
        self.current.shots_fired += max(0, int(rounds))
        self._last_shot_t = time.time()

    def log_kill(self, count: int = 1, is_melee: bool = False) -> None:
        self.current.kills += max(0, int(count))
        if is_melee:
            self.current.melee_kills += max(0, int(count))

    def log_damage(self, damage: float, is_player: bool) -> None:
        if is_player:
            self.current.damage_taken += damage
        else:
            self.current.damage_dealt += damage

    def shot_recently(self) -> bool:
        return (time.time() - self._last_shot_t) <= self.KILL_WINDOW_S

    def finalize_episode(self) -> CombatMetrics:
        self.episode_metrics.append(self.current)
        self.current = CombatMetrics()
        return self.episode_metrics[-1]

    def get_rolling_stats(self, window: int = 10) -> Optional[Dict[str, float]]:
        recent = self.episode_metrics[-window:]
        if not recent:
            return None
        n = len(recent)
        return {
            "window":         n,
            "mean_accuracy":  sum(m.accuracy for m in recent) / n,
            "mean_efficiency": sum(m.efficiency for m in recent) / n,
            "mean_kills":     sum(m.kills for m in recent) / n,
            "mean_shots":     sum(m.shots_fired for m in recent) / n,
            "total_kills":    sum(m.kills for m in self.episode_metrics),
        }
