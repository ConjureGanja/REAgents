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

# ── Curriculum reward weights (FALLBACK defaults) ─────────────────────────────
# These are only used when config.yaml doesn't provide per-stage reward_weights.
# config.yaml promises "edit this file to tune the agent without touching any
# Python code" — previously these hardcoded values silently OVERRODE the config,
# so tuning curriculum.stages.*.reward_weights did nothing.  Now the config wins
# and this dict is the safety net.
# Analogy: the config is the thermostat on the wall; this dict is the factory
# default the furnace falls back to if the thermostat is unplugged.
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
        self._ctrl = GameControls(
            window_title_substring=_win,
            smoothing=self._cfg.get("controls", {}).get("smoothing"),
        )
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

        # ── Perception throttle ───────────────────────────────────────────────
        # YOLO + EasyOCR together cost ~150 ms, far too slow to run every step.
        # The raw frame is always fresh (threaded ScreenCapture), but the
        # detections/HUD only need to refresh ~10×/sec.  We re-run perception at
        # most every `perception_interval` seconds and reuse the cached result
        # in between, so the decision loop can hit ≥20 fps.
        _perc_fps = float(self._cfg["game_settings"].get("perception_fps", 10))
        self._perception_interval = 1.0 / max(_perc_fps, 1.0)
        self._last_perception_t   = 0.0
        self._cached_detections: List[Dict] = []
        self._cached_hud: Dict[str, Any]    = {"health_pct": 1.0, "ammo_clip": 0, "ammo_res": 0}

        # Inventory cooldown — prevents Tab-spam (suitcase covers the whole screen)
        self._inv_cooldown: float = 12.0

        # ── Death detection robustness ────────────────────────────────────────
        # The health detector frequently misreads 0.0 when the HUD ring isn't
        # visible (cutscenes, menus, dark frames, the binocular scene).  Trusting
        # those zeros caused phantom deaths → 1-step episodes → an endless
        # reset/Load-Game menu loop.  We therefore: (a) treat health 0.0/None as
        # "HUD not visible / unknown" and hold the last good value rather than
        # calling it death; (b) ignore implausibly large single-step health drops
        # (a real hit can't take you from full to zero in one 0.04 s step); and
        # (c) only declare death after health stays genuinely low for several
        # consecutive steps, and never during a post-reset grace window.
        self._death_confirm_steps = int(rl_cfg.get("death_confirm_steps", 10))
        self._reset_grace_steps   = int(rl_cfg.get("reset_grace_steps", 15))
        self._max_plausible_drop  = float(rl_cfg.get("max_plausible_health_drop", 0.5))
        self._low_health_streak   = 0

        # Death-screen detection (PRIMARY death signal).  The health ring
        # vanishes the instant Leon dies, so health-based detection alone is
        # structurally unreliable — instead we look for the "YOU ARE DEAD"
        # screen (dark centre + red lettering) over several consecutive
        # perception reads.  Health-based death remains as a fallback but now
        # also requires a plausible descent (you don't go from full HP to 5%
        # without ever passing through the middle).
        self._death_screen_reads  = int(rl_cfg.get("death_screen_confirm_reads", 8))
        self._low_health_thresh   = float(rl_cfg.get("low_health_threshold", 0.12))
        self._death_screen_streak = 0
        self._health_before_low   = 1.0
        self._cached_death_screen = False
        self._perception_fresh    = False

        # Enemy latch — the aim/shoot gate used to check only the PREVIOUS
        # frame's YOLO detections, so a single missed detection strobed combat
        # off mid-fight.  Now an enemy sighting stays "hot" for a few seconds.
        self._enemy_latch_s: float = float(rl_cfg.get("enemy_latch_seconds", 3.0))
        self._last_enemy_seen_t: float = 0.0
        self._last_shot_t: float = 0.0
        self._llm_steering: bool = False

        # ── LLM action override persistence ───────────────────────────────────
        # Claude consults only every few seconds, but its movement override used
        # to be applied for a SINGLE 0.04 s step and then discarded — so "sprint
        # north to the shotgun house" made Leon twitch for one frame and stop,
        # never actually navigating.  We now latch a fresh override and re-apply
        # it for `override_hold_steps` steps so the advisor can genuinely steer.
        self._override_hold = int(
            self._cfg.get("llm_settings", {}).get("override_hold_steps", 20)
        )
        self._llm_override: Optional[List[int]] = None
        self._llm_override_steps_left: int = 0

        # ── Anti-freeze exploration prior ─────────────────────────────────────
        # The (sim-trained) policy frequently picks "stop", which leaves Leon
        # standing still in the open until the village mob encircles him.  When
        # no enemies are visible and the policy idles, nudge him to keep walking
        # and sweeping the camera so he actually traverses the village toward the
        # shotgun house / bell.  Toggle off via config to train pure RL.
        self._anti_freeze: bool = bool(
            rl_cfg.get("anti_freeze_explore", True)
        )
        self._explore_tick: int = 0

        # Curriculum
        self._curriculum_stage: str = self._cfg["curriculum"].get("initial_stage", "exploration")

        # ── Curriculum weights: config-driven with hardcoded fallback ─────────
        # Merge order per stage: hardcoded defaults ← config reward_weights.
        # This honours config.yaml's contract that reward tuning lives there.
        # Example: setting curriculum.stages.combat.reward_weights.combat: 4.0
        # in config.yaml now actually changes the combat-stage reward emphasis.
        self._curriculum_weights: Dict[str, Dict[str, float]] = {
            name: dict(w) for name, w in _CURRICULUM_WEIGHTS.items()
        }
        for _stage, _scfg in self._cfg["curriculum"].get("stages", {}).items():
            _rw = (_scfg or {}).get("reward_weights")
            if _rw:
                base = self._curriculum_weights.setdefault(
                    _stage, dict(_CURRICULUM_WEIGHTS["exploration"])
                )
                base.update({k: float(v) for k, v in _rw.items()})
        logger.info("Curriculum reward weights in effect: %s", self._curriculum_weights)

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
        if stage in self._curriculum_weights:
            self._curriculum_stage = stage
            logger.info("Curriculum stage → %s", stage)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)

        reset_cfg    = self._cfg.get("reset", {})
        loading_wait = float(reset_cfg.get("loading_wait", 8.0))

        logger.debug("Resetting environment…")
        # Clear any open pause-menu sub-screens before navigating Load Game.
        self._ctrl.escape_to_gameplay()
        self._ctrl.reset_game(reset_cfg)
        time.sleep(loading_wait)

        # Clear frame buffer — padding zeros will fill the stack until real frames arrive
        self._frame_buffer.clear()

        # Clear perception's per-episode buffers (health smoothing, ammo cache)
        # so stale low readings from the last episode can't re-trigger a death.
        if hasattr(self._eyes, "reset_state"):
            self._eyes.reset_state()

        frame = self._cap.get_frame()
        hud   = self._eyes.read_hud(frame)

        # Seed the perception cache with this fresh read so the first steps have
        # valid detections/HUD before the throttle interval elapses.
        self._cached_hud        = hud
        self._cached_detections = []
        self._last_perception_t = time.time()

        # Reset all episode state
        self._low_health_streak      = 0
        self._death_screen_streak    = 0
        self._health_before_low      = 1.0
        self._cached_death_screen    = False
        self._perception_fresh       = False
        self._last_enemy_seen_t      = 0.0
        self._last_shot_t            = 0.0
        self._llm_steering           = False
        _seed_h = float(hud.get("health_pct", 1.0) or 0.0)
        # If the very first read is a bogus 0, assume full health rather than
        # starting the episode "already dying".
        self._prev_health            = _seed_h if _seed_h > 0.0 else 1.0
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

        # ── LLM action override (held for several steps) ──────────────────────
        # If Claude has recommended a specific action, latch it and re-apply it
        # for `override_hold_steps` steps so its navigation actually moves Leon
        # (a one-step override evaporates in 0.04 s and steers nothing).
        override = self._shared.llm_action_override
        if override is not None:
            self._llm_override = list(override)
            self._llm_override_steps_left = self._override_hold
            self._shared.update(llm_action_override=None)

        self._llm_steering = (
            self._llm_override_steps_left > 0 and self._llm_override is not None
        )
        if self._llm_steering:
            # Claude is actively steering — follow it, don't second-guess.
            action = list(self._llm_override)
            self._llm_override_steps_left -= 1
        elif self._anti_freeze:
            # No LLM steering this step — break any idle freeze so Leon explores.
            action = self._maybe_explore(action)

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

        # Throttled perception: only re-run YOLO + OCR every perception_interval
        # seconds; reuse the cached detections/HUD in between.  The frame itself
        # is always fresh, so vision stays ≥20 fps even though annotations lag.
        now = time.time()
        if now - self._last_perception_t >= self._perception_interval:
            self._cached_detections   = self._eyes.detect_objects(frame)
            self._cached_hud          = self._eyes.read_hud(frame)
            self._cached_death_screen = (
                self._eyes.detect_death_screen(frame)
                if hasattr(self._eyes, "detect_death_screen") else False
            )
            self._last_perception_t = now
            self._perception_fresh  = True
        else:
            # Cached data — death/health streaks must not advance on repeats.
            self._perception_fresh = False
        detections = self._cached_detections
        hud        = self._cached_hud
        enemy_labels = [
            d["label"] for d in detections
            if d["label"] in ("person", "zombie", "enemy")
        ]
        if enemy_labels:
            self._last_enemy_seen_t = now

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
            # NOTE: SB3's Monitor wrapper OVERWRITES info["episode"] with its own
            # {"r","l","t"} dict on episode end, so any custom keys placed inside
            # it (like "deaths") never survive to the callbacks.  We therefore
            # also expose deaths as a TOP-LEVEL info key, which Monitor leaves
            # untouched.  Analogy: don't put your note inside an envelope the
            # post office is going to replace — tape it to the outside of the box.
            info["episode"] = {
                "r":      self._total_episode_reward,
                "l":      self._episode_length,
                "deaths": self._death_count,
            }
            info["death_count"] = self._death_count

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
        # health_pct is None when the ring isn't visible (Leon not aiming) —
        # feed the policy the last known-good value instead of a fake 0/1.
        _h_raw    = hud.get("health_pct", None)
        health    = float(_h_raw) if _h_raw is not None else float(self._prev_health)
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

    def _maybe_explore(self, action: List[int]) -> List[int]:
        """
        Anti-freeze exploration prior.

        Problem: the policy often selects mv=0 (stop).  With no enemies nearby
        that just leaves Leon standing in the open until the village crowd closes
        in — the agent looks "stuck" and never reaches the shotgun house or the
        bell.  When it's safe (no enemies on the last frame) AND the policy chose
        to stand still, we substitute an exploratory movement so he keeps making
        ground.  This is a light, *varied* nudge (it changes heading and sweeps
        the camera over time, occasionally sprints) rather than a fixed script,
        and it only kicks in on idle-and-safe steps — whenever the policy actually
        wants to move or fight, we leave its choice untouched.
        """
        # Only intervene when it's safe and the policy is idling.  Uses the
        # same time latch as the aim gate — a YOLO flicker mid-fight shouldn't
        # count as "safe to wander off".
        if (time.time() - self._last_enemy_seen_t) <= self._enemy_latch_s:
            return action
        mv, cam, inter, comb, ev, inv = action
        if mv != 0:
            return action

        self._explore_tick += 1
        t = self._explore_tick
        # Rotate heading every ~30 steps so we don't wall-hug one direction.
        mv = {0: 1, 1: 6, 2: 1, 3: 5}[(t // 30) % 4]   # fwd / fwd-right / fwd / fwd-left
        # Gentle camera sweep to scan for the route and approaching enemies.
        if cam == 0:
            cam = 2 if (t % 12) < 6 else 1             # alternate look right / left
        # Sprint a fraction of the time to cover the village faster.
        if ev == 0 and (t % 6 == 0):
            ev = 2
        return [mv, cam, inter, comb, ev, inv]

    def _take_action(self, action: List[int]) -> None:
        """
        Translate discrete action indices into a single atomic gamepad update.

        Guard: if shared.paused is True (dashboard Pause button, or auto-pause
        when the focus monitor detects RE4 lost foreground), all axes are zeroed
        and the step is skipped.  The RL training loop still ticks so episode
        state is consistent, but nothing reaches the game.

        With the virtual gamepad backend, window focus is NOT required for
        training steps — inputs go directly to the XInput device layer.
        """
        if self._shared.paused:
            self._ctrl.release_all()
            return

        mv, cam, inter, comb, ev, inv = action

        # ── Aim/shoot gate ────────────────────────────────────────────────────
        # The sim-trained policy tends to hold aim+shoot (combat=3) almost every
        # step.  In RE4 holding aim (LT) ROOTS Leon — he raises the gun and can
        # only shuffle, so a constant-aim policy looks frozen and never explores.
        # Heuristic fix: only permit aim/shoot when an enemy has been seen
        # RECENTLY (time latch, not just the previous frame — YOLO flickers,
        # and a single missed detection used to strobe combat off mid-fight).
        # Claude's explicit overrides bypass the gate entirely: if the advisor
        # says aim, it can see something YOLO can't.
        enemy_recent = (time.time() - self._last_enemy_seen_t) <= self._enemy_latch_s
        if comb in (1, 2, 3) and not enemy_recent and not self._llm_steering:
            comb = 0

        self._inv_opened_step = False

        # Inventory cooldown check (prevent Tab-spam mid-fight)
        effective_inv = 0
        if inv == 1:
            now = time.time()
            if now - self._last_inv_time >= self._inv_cooldown:
                effective_inv = 1
                self._last_inv_time = now
                self._inv_opened_step = True

        # Track combat state for reward calculation
        if comb in (1, 3):
            self._in_combat = True
            self._was_aiming = True
        elif comb == 2:
            self._in_combat = True
        else:
            self._in_combat = False
            self._was_aiming = False

        # Single atomic gamepad update — all six dimensions applied at once,
        # held for action_hold_seconds, then returned to neutral.
        self._ctrl.execute_step(mv, cam, inter, comb, ev, effective_inv,
                                hold=self._action_hold)

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
        weights    = self._curriculum_weights.get(
            self._curriculum_stage, self._curriculum_weights["exploration"]
        )
        terminated = False

        # ── 1. Survival ───────────────────────────────────────────────────────
        # Validate the health reading FIRST.  None (or a bogus 0.0) means the
        # HUD ring wasn't visible this frame (not aiming, cutscene, menu, dark
        # frame) — NOT death.  Hold the last known-good value in that case.
        raw_health   = hud.get("health_pct", None)
        health_valid = raw_health is not None and float(raw_health) > 0.0
        curr_health  = float(raw_health) if health_valid else self._prev_health

        survival_r = 0.1   # per-step alive bonus

        if health_valid and curr_health < self._prev_health:
            delta = self._prev_health - curr_health
            # Implausibly large single-step drops are misreads, not damage —
            # distrust the whole reading (don't penalise AND don't let it
            # poison _prev_health / the death streak below).
            if delta <= self._max_plausible_drop:
                survival_r -= 5.0 * delta   # losing 50% health = -2.5 pts
            else:
                health_valid = False
                curr_health  = self._prev_health

        # ── Death detection ───────────────────────────────────────────────────
        # PRIMARY: the "YOU ARE DEAD" screen (dark centre + red lettering),
        # detected in perception and confirmed over several consecutive fresh
        # reads.  The health ring vanishes at the moment of death, so health
        # alone can never catch it reliably.
        # FALLBACK: sustained genuinely-low health, but only if we saw a
        # plausible descent first (health was already below 50% when the low
        # streak began).  Streaks only advance on FRESH perception reads —
        # cached repeats of one bad frame no longer count as N confirmations.
        in_grace = self._episode_step <= self._reset_grace_steps
        if self._perception_fresh and not in_grace:
            if self._cached_death_screen:
                self._death_screen_streak += 1
            else:
                self._death_screen_streak = 0

            if health_valid and curr_health <= self._low_health_thresh:
                if self._low_health_streak == 0:
                    self._health_before_low = self._prev_health
                self._low_health_streak += 1
            elif health_valid:
                self._low_health_streak = 0

        died_by_screen = self._death_screen_streak >= self._death_screen_reads
        died_by_health = (
            self._low_health_streak >= self._death_confirm_steps
            and self._health_before_low <= 0.5
        )
        if died_by_screen or died_by_health:
            survival_r -= 20.0
            self._death_count += 1
            terminated = True
            self._low_health_streak   = 0
            self._death_screen_streak = 0
            # Push the running death total to SharedState — previously this
            # counter lived only inside the env, so the dashboard's death stat
            # stayed frozen at 0 forever no matter how many times Leon died.
            # Analogy: keeping score on a napkin in your pocket instead of the
            # scoreboard everyone is watching.
            self._shared.update(death_count=self._death_count)
            logger.info(
                "Death detected (%s) — ending episode.",
                "death screen" if died_by_screen else "sustained low health",
            )

        # Danger zone — extra urgency, but only on a trusted low reading.
        if health_valid and curr_health < 0.2:
            survival_r -= 0.3

        if health_valid:
            self._prev_health = curr_health

        # ── 2. Combat ─────────────────────────────────────────────────────────
        n_now    = len(enemy_labels)
        n_was    = len(self._prev_enemy_labels)
        killed   = max(0, n_was - n_now)

        # Did a bullet actually leave the gun?  (OCR clip count decreased.)
        # The RE4R ammo counter is only visible while aiming, which is exactly
        # when shots happen, so this reads reliably at the moment it matters.
        curr_clip  = int(hud.get("ammo_clip", 0) or 0)
        ammo_spent = max(0, self._prev_ammo_clip - curr_clip)
        self._prev_ammo_clip = curr_clip
        if ammo_spent > 0:
            self._last_shot_t = time.time()

        # Small presence term (was 0.5×n — big enough to farm by standing in
        # the crowd) and a kill bonus that requires a RECENT genuine shot:
        # a YOLO detection dropout used to count as a "kill" (+3) for free.
        combat_r = 0.1 * n_now
        recently_shot = (time.time() - getattr(self, "_last_shot_t", 0.0)) <= 1.5
        if killed and recently_shot:
            combat_r += 3.0 * killed

        comb = self._last_comb_action
        if n_now > 0:
            if comb == 3 and ammo_spent > 0:
                # Genuine aimed shot at a visible enemy — the ideal RE4 technique.
                # Gated on ammo actually decreasing: merely HOLDING aim+shoot
                # used to pay +2.0/step forever, which trained the rooted
                # constant-aim behaviour.
                combat_r += 2.0
            elif comb == 1:
                # Holding aim at enemies — small, not farmable
                combat_r += 0.2
            elif comb == 2 and not self._was_aiming:
                # Hip-fire without aiming — inaccurate in RE4; discourage it
                combat_r -= 1.0

        # Inventory mid-combat = extremely bad (Leon freezes, can't dodge)
        if self._inv_opened_step and n_now > 0:
            combat_r -= 5.0

        # Ammo discipline — small penalty per bullet fired
        if ammo_spent > 0:
            combat_r -= 0.1 * ammo_spent     # 10 shots fired = -1.0 pts

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
        # Flat per-step bonus while an objective is set.  Kept small so it nudges
        # toward the objective without rewarding mere idle existence (the old 0.5
        # was the biggest per-step term and encouraged passive wandering).
        objective_r = 0.2 if self._shared.llm_objective else 0.0

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
