"""
Gymnasium environment backed by the abstract RE4 simulation.

This is a DROP-IN REPLACEMENT for environment.py when running in simulation
mode.  The observation space, action space, and reward structure are identical.
The only difference is the source of observations:

  Real game (environment.py)     Simulation (this file)
  ──────────────────────────     ──────────────────────
  ScreenCapture.get_frame()  →   SimGameState.render_frame()
  GameControls._take_action()→   SimGameState.step(action)
  PerceptionSystem.read_hud()→   SimGameState.get_hud()
  time.sleep(0.08)           →   (nothing — pure computation)

Because there are no I/O calls or sleep() waits, a single simulation step
takes ~50–200 µs instead of ~80 ms.  With 8 SubprocVecEnv workers, total
throughput reaches ~80,000 steps/second.

POLICY TRANSFER
───────────────
Because the observation space is identical (same shapes, same dtypes, same
normalisation), a policy trained here can be directly loaded into the real
environment.py for fine-tuning.  Command:

    python main.py --resume models/checkpoints/sim_re_agent_final.zip

The CNN has learned spatial reasoning from the top-down rendering.  When it
sees the first real game frames, the HUD vector and action feedback are
identical, so learning continues rather than starting from scratch.
"""

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple  # Optional kept for render()

import numpy as np
import yaml
from gymnasium import spaces
import gymnasium as gym

from sim_game_state import SimGameState, EnemyState, MAP_W, MAP_H

logger = logging.getLogger(__name__)

# ── Curriculum reward weights ─────────────────────────────────────────────────
# Gentle multipliers — survival and objective always count fully; we only shift
# emphasis between exploration (navigation) and combat across stages.  The old
# weights (exploration 2.5) let a per-step movement bonus dominate every other
# signal, so the policy learned to wiggle in place instead of pursuing the
# shotgun/bell.  Keeping the spread small lets the actual mission rewards lead.
_CURRICULUM_WEIGHTS: Dict[str, Dict[str, float]] = {
    "exploration": {"survival": 1.0, "exploration": 1.0, "combat": 0.7, "objective": 1.0},
    "combat":      {"survival": 1.0, "exploration": 0.6, "combat": 1.3, "objective": 1.0},
    "completion":  {"survival": 1.0, "exploration": 0.4, "combat": 1.0, "objective": 1.5},
}

# ── HUD normalisation caps (must match environment.py exactly for transfer) ───
_AMMO_CLIP_MAX  = 30
_AMMO_RES_MAX   = 60
_ENEMY_MAX      = 8
_DIVERSITY_WINDOW = 20

# ── Mission objective points (for potential-based navigation shaping) ─────────
# Phase 0: head to the shotgun inside the barn.  Phase 1 (after pickup): head to
# the bell zone (the well) and survive until the timer expires ("bell rings").
_SHOTGUN_POS = (41.0, 10.0)   # Matches the shotgun ITEM_SPAWN in sim_game_state
_BELL_POS    = (25.0, 25.0)   # Centre of BELL_ZONE / the well

# Coarse visitation grid for genuine map-coverage exploration (replaces the old
# "movement diversity" bonus, which rewarded oscillating in place).
_COVERAGE_CELLS = 10          # 10×10 grid over the 50×50 map → 5×5-unit cells
_PROGRESS_SCALE = 0.25        # Reward per world-unit of progress toward the goal
_NEW_CELL_BONUS = 0.15        # One-off reward the first time each cell is entered


class SimResidentEvilEnv(gym.Env):
    """
    Gym environment wrapping SimGameState for simulation-mode training.

    Every public method (reset, step, render) has the same signature and
    return types as ResidentEvilEnv in environment.py.

    WORKER ID
    ─────────
    Each parallel SubprocVecEnv worker gets a unique worker_id so:
      1. Snapshots in the info dict can be routed to the right slot in
         SharedState (for the dashboard's live grid view).
      2. The random seed is offset per worker, ensuring different workers
         explore different trajectories.
         Analogy: eight students each given a different chapter of a book
         to read — collectively they cover more ground than eight students
         all reading the same chapter.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config_path: str = "config.yaml",
        worker_id: int = 0,
    ):
        """
        NOTE: SharedState is intentionally NOT accepted here.

        SubprocVecEnv spawns real OS processes and must pickle everything
        passed to the env factory.  SharedState contains a threading.RLock
        which is not picklable — passing it would crash on Windows spawn.

        Worker envs are fully self-contained.  They communicate back to the
        main process only through the return values of step()/reset() — in
        particular the `info` dict, which carries sim_snapshot and episode
        stats.  The SimDashboardCallback in the main process reads those and
        writes to SharedState.

        Analogy: submarine crew (worker) can't hold a radio to the surface
        the whole time, but they surface periodically (via info dict) to
        report what they found.
        """
        super().__init__()
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        self._worker_id = worker_id

        sim_cfg = self._cfg.get("simulation", {})
        rl_cfg  = self._cfg["rl_hyperparameters"]

        # Observation dimensions
        h, w = rl_cfg["obs_frame_size"]
        self._obs_h, self._obs_w = h, w
        self._grayscale: bool   = rl_cfg.get("grayscale", False)
        channels_per_frame      = 1 if self._grayscale else 3
        self._stack_size: int   = rl_cfg.get("frame_stack", 4)
        total_channels          = channels_per_frame * self._stack_size
        self._frame_buffer: deque = deque(maxlen=self._stack_size)

        # Observation / action spaces — MUST match environment.py exactly
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(
                low=0, high=255,
                shape=(h, w, total_channels),
                dtype=np.uint8,
            ),
            "hud": spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32),
        })
        self.action_space = spaces.MultiDiscrete([9, 5, 2, 4, 3, 2])

        # Curriculum
        self._curriculum_stage: str = self._cfg["curriculum"].get(
            "initial_stage", "exploration"
        )

        # How often to include a snapshot in the info dict (to avoid pipe overhead)
        self._snapshot_every = int(sim_cfg.get("snapshot_every_n", 50))

        # Simulation core — seeded differently per worker
        self._sim = SimGameState(sim_cfg, rng_seed=worker_id * 1000)

        # Episode tracking
        self._prev_health:    float  = 1.0
        self._prev_ammo_clip: int    = 0
        self._prev_ammo_res:  int    = 0
        self._episode_step:   int    = 0
        self._episode_reward: float  = 0.0
        self._death_count:    int    = 0
        self._in_combat:      bool   = False
        self._last_comb:      int    = 0
        self._was_aiming:     bool   = False
        self._last_inv_time:  float  = 0.0
        self._last_mv_action: int    = 0
        self._inv_opened:     bool   = False
        self._recent_mv:      deque  = deque(maxlen=_DIVERSITY_WINDOW)

        # ── Navigation / coverage shaping state ───────────────────────────────
        self._goal_phase:     int    = 0      # 0=to shotgun, 1=to bell
        self._prev_goal_dist: float  = 0.0    # distance to current goal last step
        self._visited_cells:  set    = set()  # coarse grid cells entered this episode
        self._barn_reward_given: bool = False

        logger.info(
            "SimResidentEvilEnv[W%d] init: obs %dx%d %s ×%d-stack",
            worker_id, h, w,
            "gray" if self._grayscale else "rgb",
            self._stack_size,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_curriculum_stage(self, stage: str) -> None:
        if stage in _CURRICULUM_WEIGHTS:
            self._curriculum_stage = stage
            logger.info("[W%d] Curriculum → %s", self._worker_id, stage)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)

        self._sim.reset()
        self._frame_buffer.clear()

        hud        = self._sim.get_hud()
        detections = self._sim.get_detections()
        frame      = self._sim.render_frame(self._obs_h, self._obs_w)

        # Reset episode tracking
        self._prev_health    = float(hud.get("health_pct", 1.0))
        self._prev_ammo_clip = int(hud.get("ammo_clip", 0))
        self._prev_ammo_res  = int(hud.get("ammo_res", 0))
        self._episode_step   = 0
        self._episode_reward = 0.0
        self._in_combat      = False
        self._last_comb      = 0
        self._was_aiming     = False
        self._inv_opened     = False
        self._recent_mv.clear()

        # Navigation / coverage shaping — reset for the new episode.
        self._goal_phase        = 0
        self._prev_goal_dist    = self._dist_to_goal()
        self._visited_cells     = {self._cell_of(self._sim.player_x, self._sim.player_y)}
        self._barn_reward_given = False

        obs = self._build_obs(frame, hud, detections)
        return obs, hud

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action_list = [int(a) for a in action]
        mv, cam, inter, comb, ev, inv = action_list

        self._last_mv_action = mv
        self._last_comb      = comb
        self._recent_mv.append(mv)

        # LLM action override is not supported in worker processes
        # (workers are isolated — overrides are applied in the main process only)
        mv, cam, inter, comb, ev, inv = action_list

        # Advance simulation
        self._sim.step(action_list)
        self._episode_step += 1

        hud        = self._sim.get_hud()
        detections = self._sim.get_detections()
        frame      = self._sim.render_frame(self._obs_h, self._obs_w)

        # Build obs BEFORE updating prev values (same order as environment.py)
        obs = self._build_obs(frame, hud, detections)

        # Combat tracking for reward function — MUST mirror environment.py
        # exactly (the in_combat flag is part of the HUD obs vector, so any
        # divergence here is a sim-to-real distribution shift).
        if comb in (1, 3):
            self._in_combat = True
            self._was_aiming = True
        elif comb == 2:
            self._in_combat = True
        else:
            self._in_combat = False
            self._was_aiming = False

        self._inv_opened = (inv == 1)

        # Reward and termination
        reward, terminated = self._calculate_reward(hud, detections)
        self._episode_reward += reward

        # Episode truncation: max sim steps reached (bell rings)
        truncated = self._sim.episode_success

        # Build info dict
        info: Dict[str, Any] = {
            **hud,
            "detections": detections,
            "episode_step": self._episode_step,
            "worker_id": self._worker_id,
        }

        # Include a lightweight snapshot every N steps for the dashboard
        # (avoids pickling large arrays through SubprocVecEnv pipes every step)
        if self._episode_step % self._snapshot_every == 0:
            snap = self._sim.get_snapshot()
            snap["episode_reward"] = round(self._episode_reward, 2)
            info["sim_snapshot"] = snap

        if terminated or truncated:
            info["episode"] = {
                "r":      self._episode_reward,
                "l":      self._episode_step,
                "deaths": self._death_count,
                "kills":  self._sim.total_kills,
                "items":  self._sim.items_collected,
                "shotgun": self._sim.shotgun_collected,
                "success": truncated,   # truncated = bell rang = success
            }
            if terminated:
                self._death_count += 1

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return the current simulation frame as an RGB array."""
        return self._sim.render_frame(self._obs_h, self._obs_w)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_obs(
        self,
        frame: np.ndarray,
        hud: Dict,
        detections: List[Dict],
    ) -> Dict[str, np.ndarray]:
        """
        Build the observation dict from simulation state.

        Frame stacking works identically to environment.py:
          - grayscale: (H,W,1) × N_stack → (H,W,N_stack)
          - RGB:       (H,W,3) × N_stack → (H,W,3×N_stack)

        The CNN sees the last N frames simultaneously, giving it information
        about motion (enemies moving between frames = different pixels).
        This is the same principle as a flip-book animation — each page alone
        shows a still; the sequence shows movement.
        """
        # Preprocess frame
        if self._grayscale:
            import cv2
            small = cv2.resize(frame, (self._obs_w, self._obs_h))
            small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        else:
            import cv2
            small = cv2.resize(frame, (self._obs_w, self._obs_h))
            # frame is already RGB from render_frame() — no colour conversion needed

        self._frame_buffer.append(small)

        frames_to_stack = list(self._frame_buffer)
        while len(frames_to_stack) < self._stack_size:
            frames_to_stack.insert(0, frames_to_stack[0])

        stacked = np.concatenate(frames_to_stack, axis=2)

        # HUD vector (7 elements — matches environment.py exactly)
        health    = float(hud.get("health_pct", 1.0))
        clip_raw  = int(hud.get("ammo_clip", 0) or 0)
        res_raw   = int(hud.get("ammo_res",  0) or 0)
        clip_norm = min(clip_raw / _AMMO_CLIP_MAX, 1.0)
        res_norm  = min(res_raw  / _AMMO_RES_MAX,  1.0)

        n_enemies  = sum(1 for d in detections if d["label"] == "enemy")
        enemy_norm = min(n_enemies / _ENEMY_MAX, 1.0)
        in_combat  = 1.0 if self._in_combat else 0.0

        # Time pressure: fraction of max episode steps elapsed
        time_pres = min(self._sim.sim_step / max(1, self._sim.max_steps), 1.0)

        ammo_delta      = clip_raw - self._prev_ammo_clip
        ammo_delta_norm = max(min(ammo_delta / _AMMO_CLIP_MAX, 1.0), -1.0) * 0.5 + 0.5

        hud_vec = np.array(
            [health, clip_norm, res_norm, enemy_norm, in_combat, time_pres, ammo_delta_norm],
            dtype=np.float32,
        )
        return {"frame": stacked, "hud": hud_vec}

    def _goal_pos(self) -> Tuple[float, float]:
        """Current navigation target: the shotgun until collected, then the bell."""
        return _BELL_POS if self._sim.shotgun_collected else _SHOTGUN_POS

    def _dist_to_goal(self) -> float:
        gx, gy = self._goal_pos()
        dx, dy = self._sim.player_x - gx, self._sim.player_y - gy
        return float((dx * dx + dy * dy) ** 0.5)

    @staticmethod
    def _cell_of(x: float, y: float) -> Tuple[int, int]:
        """Map a world position to a coarse visitation-grid cell."""
        cx = int(min(_COVERAGE_CELLS - 1, max(0, x / MAP_W * _COVERAGE_CELLS)))
        cy = int(min(_COVERAGE_CELLS - 1, max(0, y / MAP_H * _COVERAGE_CELLS)))
        return cx, cy

    def _calculate_reward(
        self,
        hud: Dict,
        detections: List[Dict],
    ) -> Tuple[float, bool]:
        """
        Mission-shaped reward for the village siege.

        The agent is guided through the actual objective arc — navigate to the
        barn → grab the shotgun → reach the bell → survive until it rings —
        rather than the old design where a per-step "movement diversity" bonus
        dominated everything and the shotgun was physically uncollectable.

        Design principles:
          • Potential-based progress shaping gives a dense *gradient* toward the
            current goal (shotgun, then bell).  Because it telescopes, the total
            navigation reward depends only on how much closer the agent gets, not
            on the path length — so it can't be farmed by wiggling.
          • Genuine map coverage (first visit to each grid cell) replaces the old
            in-place "diversity" wiggle bonus.
          • Combat and damage are scaled so fighting and avoiding hits actually
            matter relative to navigation.
          • Big sparse payoffs anchor the milestones: shotgun pickup, barn entry,
            and — crucially — surviving to the bell (which previously paid zero).
        """
        weights    = _CURRICULUM_WEIGHTS.get(self._curriculum_stage,
                                             _CURRICULUM_WEIGHTS["exploration"])
        terminated = self._sim.player_dead

        # Snapshot BEFORE any updates — the herb-recovery check below needs the
        # health value from the previous step, not the one we're about to write.
        health_at_prev_step = self._prev_health

        # ── 1. Survival ───────────────────────────────────────────────────────
        curr_health = self._sim.health_pct
        survival_r  = 0.02   # small heartbeat for being alive

        if self._sim.damage_this_step > 0:
            # Getting hit must hurt more than a step of movement is worth, or the
            # agent learns to facetank the crowd.
            survival_r -= 0.15 * self._sim.damage_this_step   # 12-dmg hit → -1.8

        if self._sim.player_dead:
            survival_r -= 25.0

        if curr_health < 0.2:
            survival_r -= 0.1   # "Danger zone" — find a herb / disengage

        self._prev_health = curr_health

        # ── 2. Combat ─────────────────────────────────────────────────────────
        kills   = self._sim.kills_this_step
        n_now   = len([d for d in detections if d["label"] == "enemy"])
        combat_r = 5.0 * kills   # clean, dominant kill signal

        # Did a bullet actually leave the gun this step?  (Clip decreased, or a
        # kill registered.)  The old reward paid +0.5/step for merely HOLDING
        # aim+shoot at a visible enemy — with an empty clip that was an
        # infinite free-reward farm, and it's what taught the policy to hold
        # comb=3 permanently (the "rooted Leon" pathology in the real game).
        curr_clip  = self._sim.ammo_clip
        ammo_spent = max(0, self._prev_ammo_clip - curr_clip)
        shot_fired = ammo_spent > 0 or kills > 0
        self._prev_ammo_clip = curr_clip

        comb = self._last_comb
        if n_now > 0:
            if comb == 3 and shot_fired:
                combat_r += 0.5   # genuine aimed shot at a visible enemy
            elif comb == 1:
                combat_r += 0.05  # holding aim (tiny — don't make it farmable)
            elif comb == 2 and not self._was_aiming:
                combat_r -= 0.3   # hip-fire (inaccurate in RE4 — discourage)
        else:
            # No enemy in sight: aiming roots you and firing wastes ammo — both
            # are strictly bad in RE4.  The slowdown physics alone proved too
            # indirect to unlearn the constant-aim habit; this makes the cost
            # explicit per step.
            if comb in (1, 3):
                combat_r -= 0.1
            if comb in (2, 3) and ammo_spent > 0:
                combat_r -= 0.1

        if self._inv_opened and n_now > 0:
            combat_r -= 3.0   # inventory mid-combat = frozen = bad

        # Ammo discipline (mild)
        if ammo_spent > 0:
            combat_r -= 0.05 * ammo_spent

        # ── 3. Navigation + coverage (the new "exploration") ──────────────────
        # Potential-based progress toward the current goal.  Switching goals
        # (shotgun → bell) re-baselines the distance so the discontinuity does
        # NOT register as a huge one-step reward.
        phase_now = 1 if self._sim.shotgun_collected else 0
        curr_goal_dist = self._dist_to_goal()
        if phase_now != self._goal_phase:
            self._goal_phase = phase_now
            progress_r = 0.0                       # re-baseline on goal switch
        else:
            progress_r = _PROGRESS_SCALE * (self._prev_goal_dist - curr_goal_dist)
        self._prev_goal_dist = curr_goal_dist

        # Genuine coverage: reward the first entry into each coarse grid cell.
        exploration_r = progress_r
        cell = self._cell_of(self._sim.player_x, self._sim.player_y)
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            exploration_r += _NEW_CELL_BONUS

        # Mild anti-idle nudge (keeps it moving without the wiggle exploit).
        if self._last_mv_action == 0:
            exploration_r -= 0.02

        # ── 4. Item collection ────────────────────────────────────────────────
        item_r   = 0.0
        curr_res = self._sim.ammo_reserve

        if curr_res > self._prev_ammo_res:
            delta_res = curr_res - self._prev_ammo_res
            item_r += 2.0 * min(delta_res / 10.0, 3.0)
        self._prev_ammo_res = curr_res

        # Health recovery (using a herb).  Compares against the PRE-update
        # snapshot — self._prev_health was already overwritten above, so the
        # old comparison could never fire (dead code until now).
        if curr_health > health_at_prev_step + 0.05:
            item_r += 2.0

        # ── 5. Mission objectives ─────────────────────────────────────────────
        objective_r = item_r

        # First time entering the barn — a milestone on the way to the shotgun.
        if self._sim.barn_visited and not self._barn_reward_given:
            objective_r += 8.0
            self._barn_reward_given = True

        # The main objective: collect the barn shotgun.
        if self._sim.shotgun_this_step:
            objective_r += 25.0

        # After the shotgun, reward holding the bell zone (the well) — the place
        # to be when the bell rings.
        if self._sim.shotgun_collected and self._sim.bell_area_reached:
            objective_r += 0.1

        # THE WIN: survive until the timer expires (the bell rings).  The payoff
        # is GATED on having collected the shotgun first, so "just camp the bell
        # zone and dodge everyone" is no longer a near-optimal strategy — the
        # full reward requires completing the actual objective chain
        # (barn → shotgun → survive).  Surviving without the shotgun still earns
        # a small credit (don't punish staying alive), but it's ~7× less than
        # doing the whole mission, which makes the barn detour clearly worth it.
        if self._sim.episode_success:
            objective_r += 35.0 if self._sim.shotgun_collected else 5.0

        # ── Weighted sum ──────────────────────────────────────────────────────
        reward = (
            weights["survival"]      * survival_r
            + weights["combat"]      * combat_r
            + weights["exploration"] * exploration_r
            + weights["objective"]   * objective_r
        )

        return float(reward), terminated


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from stable_baselines3.common.env_checker import check_env

    env = SimResidentEvilEnv()
    print("Running SB3 environment check…")
    check_env(env, warn=True)
    print("check_env PASSED")

    obs, info = env.reset()
    print("Obs shapes:", {k: v.shape for k, v in obs.items()})

    # Throughput benchmark
    start = time.perf_counter()
    N = 5_000
    done_count = 0
    for _ in range(N):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, info = env.reset()
            done_count += 1

    elapsed = time.perf_counter() - start
    print(f"{N:,} steps in {elapsed:.3f}s  ({N/elapsed:,.0f} steps/sec)  {done_count} episodes")
    print("SimResidentEvilEnv test PASSED.")
