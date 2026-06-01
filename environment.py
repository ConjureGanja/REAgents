"""
Gymnasium environment wrapping Resident Evil 4 Remake.

OBSERVATION SPACE  (Dict)
──────────────────────────
  "frame" : (H, W, C×N_stack)  uint8
      Stacked grayscale or RGB frames.  N_stack=4 by default.
      Frame stacking lets the CNN see MOTION without a recurrent network.
      Analogy: like a photographer using a slow shutter speed to capture blur
      — the agent can see where enemies are moving, not just where they are now.

  "hud"   : (7,)  float32
      [health, clip_norm, res_norm, enemy_norm, in_combat, time_pressure, ammo_delta_norm]
      (7th element added: ammo delta — negative = shots fired this step)

ACTION SPACE  (MultiDiscrete)
──────────────────────────────
  Index  Dimension       Options
  0      Movement        0=stop 1=fwd 2=back 3=left 4=right 5=fwd_left 6=fwd_right 7=back_left 8=back_right
  1      Camera          0=none 1=left 2=right 3=up 4=down
  2      Interact        0=none 1=interact(F)
  3      Combat          0=none 1=aim 2=shoot 3=aim+shoot
  4      Evasion         0=none 1=dodge(space) 2=sprint(shift)
  5      Inventory       0=none 1=toggle(tab)

VILLAGE / CHAPTER 1 REWARD SHAPING
────────────────────────────────────
The reward function is tuned specifically for the village opening (Chapter 1):
  • Strong survival bonus — health is everything in the opening
  • Exploration reward for movement diversity (agents tend to get stuck)
  • Item pickup detection — ammo or health increases = reward
  • Accurate combat rewards: aim→shoot combos, enemy kills, leg-trip follow-up
  • Steep penalty for standing still (the village mob will overwhelm a stationary Leon)
  • Inventory spam prevention — opening the suitcase mid-fight = death
"""

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from capture import ScreenCapture
from controls import GameControls
from perception import PerceptionSystem
from shared_state import SharedState

logger = logging.getLogger(__name__)

# ── Curriculum reward weights ──────────────────────────────────────────────────
_CURRICULUM_WEIGHTS: Dict[str, Dict[str, float]] = {
    "exploration": {"survival": 0.6, "exploration": 2.5, "combat": 0.4, "objective": 1.0},
    "combat":      {"survival": 0.4, "exploration": 0.5, "combat": 2.5, "objective": 1.5},
    "completion":  {"survival": 0.3, "exploration": 0.3, "combat": 1.5, "objective": 3.0},
}

# ── HUD normalisation caps ─────────────────────────────────────────────────────
_AMMO_CLIP_MAX  = 30     # Maximum pistol magazine (RE4 Remake)
_AMMO_RES_MAX   = 60     # Maximum pistol reserve
_ENEMY_MAX      = 8      # YOLO detects at most 8 enemies before we cap
_MAX_EPISODE_S  = 600    # 10-minute episode hard cap

# ── Action diversity tracking ─────────────────────────────────────────────────
_DIVERSITY_WINDOW = 20   # Look-back window for movement action diversity


class ResidentEvilEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str = "config.yaml",
        shared_state: Optional[SharedState] = None,
    ):
        super().__init__()
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        self._shared = shared_state or SharedState()

        # ── Sub-systems ───────────────────────────────────────────────────────
        self._cap  = ScreenCapture(config_path)
        _win       = self._cfg["game_settings"].get("window_title_substring", "RESIDENT EVIL 4")
        self._ctrl = GameControls(window_title_substring=_win)
        self._eyes = PerceptionSystem(config_path)

        # ── Observation configuration ─────────────────────────────────────────
        rl_cfg = self._cfg["rl_hyperparameters"]
        h, w   = rl_cfg["obs_frame_size"]               # e.g. [84, 84]
        self._obs_h, self._obs_w = h, w

        self._grayscale: bool = rl_cfg.get("grayscale", False)
        channels_per_frame    = 1 if self._grayscale else 3

        # Frame stacking — N frames concatenated along the channel axis
        # Example: grayscale + stack=4 → (84, 84, 4)
        #          RGB       + stack=4 → (84, 84, 12)
        # This gives the CNN temporal information without a recurrent network.
        self._stack_size: int = rl_cfg.get("frame_stack", 4)
        total_channels        = channels_per_frame * self._stack_size
        self._frame_buffer: deque = deque(maxlen=self._stack_size)

        # HUD has 7 elements (added ammo_delta_norm vs the original 6)
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(
                low=0, high=255,
                shape=(h, w, total_channels),
                dtype=np.uint8,
            ),
            "hud": spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32),
        })

        self.action_space = spaces.MultiDiscrete([9, 5, 2, 4, 3, 2])

        # Action hold duration — how long each action is pressed
        self._action_hold = float(rl_cfg.get("action_hold_seconds", 0.08))

        # Inventory cooldown — prevents Tab-spam (suitcase covers the whole screen)
        self._inv_cooldown: float = 12.0

        # Curriculum
        self._curriculum_stage: str = self._cfg["curriculum"].get("initial_stage", "exploration")

        # ── Episode state (reset in reset()) ──────────────────────────────────
        self._prev_health:       float      = 1.0
        self._prev_ammo_clip:    int        = 0
        self._prev_ammo_res:     int        = 0
        self._prev_enemy_labels: List[str]  = []
        self._episode_step:      int        = 0
        self._episode_start:     float      = time.time()
        self._death_count:       int        = 0
        self._total_episode_reward: float   = 0.0
        self._episode_length:    int        = 0
        self._in_combat:         bool       = False
        self._last_comb_action:  int        = 0
        self._was_aiming:        bool       = False
        self._inv_opened_step:   bool       = False
        self._last_inv_time:     float      = 0.0

        # Action diversity tracking — reward the agent for varying its movement
        # Analogy: a student who only ever memorises one strategy vs one who
        # tries different approaches and learns which works best.
        self._recent_mv_actions: deque = deque(maxlen=_DIVERSITY_WINDOW)

        # Item pickup tracking — detect ammo/health increases as proxy for pickup
        self._prev_health_for_pickup: float = 1.0

        logger.info(
            "ResidentEvilEnv initialised (obs %dx%d %s x%d-stack, stage=%s)",
            h, w,
            "gray" if self._grayscale else "rgb",
            self._stack_size,
            self._curriculum_stage,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_curriculum_stage(self, stage: str) -> None:
        if stage in _CURRICULUM_WEIGHTS:
            self._curriculum_stage = stage
            logger.info("Curriculum stage → %s", stage)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)

        reset_cfg    = self._cfg.get("reset", {})
        loading_wait = float(reset_cfg.get("loading_wait", 8.0))

        logger.debug("Resetting environment…")
        self._ctrl.reset_game(reset_cfg)
        time.sleep(loading_wait)

        # Clear frame buffer — padding zeros will fill the stack until real frames arrive
        self._frame_buffer.clear()

        frame = self._cap.get_frame()
        hud   = self._eyes.read_hud(frame)

        # Reset all episode state
        self._prev_health            = float(hud.get("health_pct", 1.0))
        self._prev_health_for_pickup = self._prev_health
        self._prev_ammo_clip         = int(hud.get("ammo_clip", 0) or 0)
        self._prev_ammo_res          = int(hud.get("ammo_res", 0) or 0)
        self._prev_enemy_labels      = []
        self._episode_step           = 0
        self._episode_start          = time.time()
        self._in_combat              = False
        self._total_episode_reward   = 0.0
        self._episode_length         = 0
        self._last_comb_action       = 0
        self._was_aiming             = False
        self._inv_opened_step        = False
        self._last_inv_time          = 0.0
        self._recent_mv_actions.clear()

        obs = self._build_obs(frame, hud, detections=[])
        self._shared.update(
            frame=frame,
            hud=hud,
            detections=[],
            current_action=[0, 0, 0, 0, 0, 0],
            curriculum_stage=self._curriculum_stage,
        )
        return obs, hud

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = list(action)

        # ── LLM action override ───────────────────────────────────────────────
        # If Claude has recommended a specific action this step, use it instead
        # of the RL policy's choice.  The override is cleared immediately after
        # use so it only applies for a single step.
        override = self._shared.llm_action_override
        if override is not None:
            action = override
            self._shared.update(llm_action_override=None)

        # ── Execute action ────────────────────────────────────────────────────
        self._last_mv_action  = action[0]
        self._last_comb_action = action[3]
        self._recent_mv_actions.append(action[0])

        self._take_action(action)
        self._episode_step += 1
        self._episode_length += 1

        # ── Capture new game state ────────────────────────────────────────────
        frame      = self._cap.get_frame()
        if frame.size == 0:
            frame  = np.zeros((self._obs_h, self._obs_w, 3), dtype=np.uint8)
        detections = self._eyes.detect_objects(frame)
        hud        = self._eyes.read_hud(frame)
        enemy_labels = [
            d["label"] for d in detections
            if d["label"] in ("person", "zombie", "enemy")
        ]

        # ── Build observation BEFORE reward ───────────────────────────────────
        # IMPORTANT: obs must be built before _calculate_reward() because the
        # reward function calls `self._prev_ammo_clip = curr_clip` at the end.
        # If we built obs after that update, `ammo_delta_norm` in the HUD vector
        # would always be zero (curr - curr = 0) — ammo-waste tracking broken.
        # Analogy: read the fuel gauge before you top up the tank, not after.
        obs = self._build_obs(frame, hud, detections)

        # ── Reward & termination ──────────────────────────────────────────────
        reward, terminated = self._calculate_reward(hud, enemy_labels)
        self._total_episode_reward += reward

        # Time-based truncation (10-minute hard cap)
        elapsed   = time.time() - self._episode_start
        truncated = elapsed > _MAX_EPISODE_S

        # ── Update shared state for dashboard ─────────────────────────────────
        self._shared.update(
            frame=frame,
            hud=hud,
            detections=detections,
            current_action=action,
            last_reward=reward,
            curriculum_stage=self._curriculum_stage,
        )
        info = {**hud, "detections": detections, "episode_step": self._episode_step}

        if terminated or truncated:
            info["episode"] = {
                "r":      self._total_episode_reward,
                "l":      self._episode_length,
                "deaths": self._death_count,
            }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        frame = self._cap.get_frame()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_obs(
        self,
        frame: np.ndarray,
        hud: Dict,
        detections: List[Dict],
    ) -> Dict[str, np.ndarray]:
        """
        Build the observation dict:
          "frame" — stacked frames (H, W, C×N_stack)
          "hud"   — 7-element normalised sensor vector

        FRAME STACKING EXPLAINED:
          We maintain a rolling buffer of the last N frames.  Each call to
          _build_obs appends the current frame and pops the oldest.  The buffer
          is concatenated along the channel axis:
            stack=4, grayscale:  (84, 84, 1) × 4  →  (84, 84, 4)
            stack=4, RGB:        (84, 84, 3) × 4  →  (84, 84, 12)
          The CNN sees all 4 frames simultaneously and can detect motion
          (enemy moving between frames = different pixel values).
          Padding: while the buffer fills up in the first N steps of an episode,
          we pad with the earliest available frame repeated.
        """
        # --- Preprocess current frame ---
        small = cv2.resize(frame, (self._obs_w, self._obs_h))
        if self._grayscale:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        else:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)   # (H, W, 3)

        # Push into rolling buffer
        self._frame_buffer.append(small)

        # Pad with the first available frame if buffer not yet full
        frames_to_stack = list(self._frame_buffer)
        while len(frames_to_stack) < self._stack_size:
            frames_to_stack.insert(0, frames_to_stack[0])

        stacked = np.concatenate(frames_to_stack, axis=2)   # (H, W, C×N_stack)

        # --- HUD vector (7 elements) ---
        health    = float(hud.get("health_pct", 1.0))
        clip_raw  = int(hud.get("ammo_clip", 0) or 0)
        res_raw   = int(hud.get("ammo_res",  0) or 0)
        clip_norm = min(clip_raw / _AMMO_CLIP_MAX, 1.0)
        res_norm  = min(res_raw  / _AMMO_RES_MAX,  1.0)

        n_enemies  = sum(1 for d in detections if d["label"] in ("person", "zombie", "enemy"))
        enemy_norm = min(n_enemies / _ENEMY_MAX, 1.0)
        in_combat  = 1.0 if self._in_combat else 0.0

        elapsed   = (time.time() - self._episode_start) / _MAX_EPISODE_S
        time_pres = min(elapsed, 1.0)

        # Ammo delta — negative means shots fired this step (penalise ammo waste)
        ammo_delta = clip_raw - self._prev_ammo_clip
        ammo_delta_norm = max(min(ammo_delta / _AMMO_CLIP_MAX, 1.0), -1.0) * 0.5 + 0.5

        hud_vec = np.array(
            [health, clip_norm, res_norm, enemy_norm, in_combat, time_pres, ammo_delta_norm],
            dtype=np.float32,
        )
        return {"frame": stacked, "hud": hud_vec}

    def _take_action(self, action: List[int]) -> None:
        """
        Map the discrete action indices to actual game inputs via GameControls.

        TWO GUARDS before any input is sent:

        1. paused flag — set by the dashboard Pause button or automatically when
           the game window loses focus.  While paused, ALL inputs are blocked and
           any held keys/buttons are released.  The RL training loop keeps ticking
           (steps count up, rewards compute) but nothing touches the game or the
           browser the user is interacting with.

        2. is_game_focused() — final OS-level check.  Even if paused=False, if
           RE4 has somehow lost focus (e.g. a popup appeared), we block inputs.

        Analogy: paused is like a bouncer at the door of the input pipeline.
        is_game_focused() is the second bouncer right before the game window.
        Both have to say "yes" before a keystroke gets in.
        """
        # Guard 1 — explicit pause (dashboard button or auto-focus-loss)
        if self._shared.paused:
            self._ctrl.release_all()   # ensure no keys are stuck held
            return                     # silently skip — no warning spam in logs

        # Guard 2 — OS foreground focus check
        if not self._ctrl.is_game_focused():
            self._ctrl.release_all()
            logger.warning("Game window lost focus — skipping action, releasing held keys")
            self._shared.update(game_focused=False)
            return
        self._shared.update(game_focused=True)

        mv, cam, inter, comb, ev, inv = action

        self._inv_opened_step = False

        # ── Movement ─────────────────────────────────────────────────────────
        _MOVE = {
            1: "fwd",      2: "back",      3: "left",       4: "right",
            5: "fwd_left", 6: "fwd_right", 7: "back_left",  8: "back_right",
        }
        if mv in _MOVE:
            self._ctrl.move(_MOVE[mv], duration=self._action_hold)

        # ── Camera ────────────────────────────────────────────────────────────
        _CAM = {1: (-25, 0), 2: (25, 0), 3: (0, -20), 4: (0, 20)}
        if cam in _CAM:
            self._ctrl.look(*_CAM[cam])

        # ── Combat ────────────────────────────────────────────────────────────
        # RE4 scheme: hold right-click to aim, left-click to shoot
        # The correct combo is: aim first (build accuracy), then shoot.
        if comb == 1:
            self._ctrl.combat("aim")          # hold RMB — entering aim stance
            self._in_combat = True
            self._was_aiming = True
        elif comb == 2:
            self._ctrl.combat("shoot")        # click LMB — hip-fire
            self._in_combat = True
        elif comb == 3:
            self._ctrl.combat("aim")          # RMB + LMB in one step = aimed shot
            self._ctrl.combat("shoot")
            self._in_combat = True
            self._was_aiming = True
        else:
            self._ctrl.combat("stop_aim")     # release RMB — lower weapon
            self._in_combat = False
            self._was_aiming = False

        # ── Interact ──────────────────────────────────────────────────────────
        if inter == 1:
            self._ctrl.interact()

        # ── Evasion ───────────────────────────────────────────────────────────
        if ev == 1:
            self._ctrl.evade(sprint=False)
        elif ev == 2:
            self._ctrl.evade(sprint=True)

        # ── Inventory ─────────────────────────────────────────────────────────
        if inv == 1:
            now = time.time()
            if now - self._last_inv_time >= self._inv_cooldown:
                self._ctrl.inventory()
                self._last_inv_time = now
                self._inv_opened_step = True

    def _calculate_reward(
        self,
        hud: Dict,
        enemy_labels: List[str],
    ) -> Tuple[float, bool]:
        """
        Reward function for the RE4 village setting.

        Components (weighted by curriculum stage):
        ┌─────────────────┬──────────────────────────────────────────────────┐
        │ survival_r      │ +0.1/step alive; proportional penalty for damage;│
        │                 │ large death penalty                               │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ combat_r        │ enemy kills; aimed shots; leg-shot bonus;        │
        │                 │ penalise hip-fire; penalise ammo waste            │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ exploration_r   │ movement diversity (unique actions/window);      │
        │                 │ penalty for standing still                        │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ item_pickup_r   │ ammo reserve increase → item pickup proxy        │
        │                 │ health increase → herb use proxy                  │
        ├─────────────────┼──────────────────────────────────────────────────┤
        │ objective_r     │ LLM-set objective bonus (if active)              │
        └─────────────────┴──────────────────────────────────────────────────┘
        """
        weights    = _CURRICULUM_WEIGHTS.get(self._curriculum_stage, _CURRICULUM_WEIGHTS["exploration"])
        terminated = False

        # ── 1. Survival ───────────────────────────────────────────────────────
        curr_health = float(hud.get("health_pct", 0.0))
        survival_r  = 0.1   # per-step alive bonus

        if curr_health < self._prev_health:
            delta = self._prev_health - curr_health
            # Proportional damage penalty — losing 50% health = -2.5 pts
            survival_r -= 5.0 * delta

        if curr_health <= 0.05:
            # Death: large penalty + episode termination
            survival_r -= 20.0
            self._death_count += 1
            terminated = True

        # Village-specific: extra penalty for being at critical health (red ring)
        # At this point the agent must prioritise herbs immediately
        if curr_health < 0.2:
            survival_r -= 0.3   # "danger" zone — hurry up and heal

        self._prev_health = curr_health

        # ── 2. Combat ─────────────────────────────────────────────────────────
        n_now    = len(enemy_labels)
        n_was    = len(self._prev_enemy_labels)
        killed   = max(0, n_was - n_now)
        combat_r = 0.5 * n_now          # reward for being in an active fight
        combat_r += 3.0 * killed        # strong kill bonus (up from 2.0)

        comb = self._last_comb_action
        if n_now > 0:
            if comb == 3:
                # Aimed shot at a visible enemy — the ideal RE4 technique
                combat_r += 2.0
            elif comb == 1:
                # Holding aim at enemies — good discipline
                combat_r += 0.5
            elif comb == 2 and not self._was_aiming:
                # Hip-fire without aiming — inaccurate in RE4; discourage it
                # Exception: at very close range hip-fire is valid
                combat_r -= 1.0

        # Inventory mid-combat = extremely bad (Leon freezes, can't dodge)
        if self._inv_opened_step and n_now > 0:
            combat_r -= 5.0

        # Ammo discipline — small penalty per bullet fired
        curr_clip   = int(hud.get("ammo_clip", 0) or 0)
        ammo_delta  = self._prev_ammo_clip - curr_clip
        if ammo_delta > 0:
            combat_r -= 0.1 * ammo_delta     # 10 shots fired = -1.0 pts
        self._prev_ammo_clip = curr_clip

        self._prev_enemy_labels = enemy_labels

        # ── 3. Exploration / movement diversity ───────────────────────────────
        # Penalise standing still (the village mob encircles a stationary Leon).
        # Additionally, reward movement VARIETY — using many different directions
        # instead of just running in circles.
        mv_action = getattr(self, "_last_mv_action", 0)

        if mv_action == 0:
            # Standing still — strong penalty (especially in village crowds)
            exploration_r = -0.05
        else:
            # Alive and moving — small baseline reward
            exploration_r = 0.05
            # Diversity bonus: what fraction of recent actions were unique?
            if len(self._recent_mv_actions) >= 5:
                unique_moves   = len(set(self._recent_mv_actions))
                diversity_frac = unique_moves / len(set(range(1, 9)))   # out of 8 directions
                # Scale 0→0.2 bonus based on movement variety
                exploration_r += 0.2 * diversity_frac

        # ── 4. Item pickup proxy ──────────────────────────────────────────────
        # We can't directly detect "Leon picked up the shotgun", but we CAN
        # detect: ammo reserve increased (ammo pickup) or health increased (herb).
        curr_res = int(hud.get("ammo_res", 0) or 0)
        item_r   = 0.0

        if curr_res > self._prev_ammo_res:
            # Ammo reserve went up → picked up ammo or the shotgun (big reward!)
            delta_res = curr_res - self._prev_ammo_res
            item_r   += 2.0 * min(delta_res / 10.0, 3.0)   # up to +6.0 per big pickup

        if curr_health > self._prev_health_for_pickup + 0.05:
            # Health went up → used a herb  (only count meaningful increases)
            item_r += 1.0

        self._prev_ammo_res          = curr_res
        self._prev_health_for_pickup = curr_health

        # ── 5. LLM objective shaping ──────────────────────────────────────────
        # The Claude advisor sets a short phrase as the active objective.
        # We give a small per-step bonus while any objective is set, to guide
        # the exploration curriculum.
        objective_r = 0.5 if self._shared.llm_objective else 0.0

        # ── Weighted sum ──────────────────────────────────────────────────────
        reward = (
            weights["survival"]    * survival_r
            + weights["combat"]    * combat_r
            + weights["exploration"] * (exploration_r + item_r)
            + weights["objective"]   * objective_r
        )

        return float(reward), terminated


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = ResidentEvilEnv()
    obs, info = env.reset()
    print("Obs shapes:", {k: v.shape for k, v in obs.items()})
    print("Frame channels:", obs["frame"].shape[2], "(should be channels × stack_size)")

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, info = env.reset()

    print("Environment test passed.")
