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
from constants import ACTION_SPACE_SIZES
from env.obs_builder import ObsBuilder
from env.death_detection import DeathDetector
from env.rewards import RewardCalculator, RewardInputs

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
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        from config_loader import ensure_config
        self._cfg = ensure_config(config, config_path)

        self._shared = shared_state or SharedState()

        # ── Sub-systems ───────────────────────────────────────────────────────
        self._cap  = ScreenCapture(config=self._cfg)
        _win       = self._cfg["game_settings"].get("window_title_substring", "RESIDENT EVIL 4")
        self._ctrl = GameControls(
            window_title_substring=_win,
            smoothing=self._cfg.get("controls", {}).get("smoothing"),
        )
        self._eyes = PerceptionSystem(config=self._cfg)

        # ── Observation configuration ─────────────────────────────────────────
        rl_cfg = self._cfg["rl_hyperparameters"]
        h, w   = rl_cfg["obs_frame_size"]               # e.g. [84, 84]
        self._obs_h, self._obs_w = h, w

        self._grayscale: bool = rl_cfg.get("grayscale", False)
        channels_per_frame    = 1 if self._grayscale else 3

        # Frame stacking — N frames concatenated along the channel axis
        # (grayscale+stack=4 → (84,84,4); RGB+stack=4 → (84,84,12)).
        # The rolling buffer lives in env.obs_builder.ObsBuilder.
        self._stack_size: int = rl_cfg.get("frame_stack", 4)
        total_channels        = channels_per_frame * self._stack_size
        self._obs_builder = ObsBuilder(self._obs_h, self._obs_w, self._grayscale, self._stack_size)

        # HUD has 7 elements (added ammo_delta_norm vs the original 6)
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(
                low=0, high=255,
                shape=(h, w, total_channels),
                dtype=np.uint8,
            ),
            "hud": spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32),
        })

        self.action_space = spaces.MultiDiscrete(ACTION_SPACE_SIZES)

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

        # Dedicated perception worker — runs YOLO/OCR/health/death-screen on a
        # background thread and publishes into SharedState, so the step loop
        # never blocks on the ~40-150 ms vision tick.  All PerceptionSystem
        # access is serialized through this worker (YOLO/EasyOCR are not
        # thread-safe); reset() and the startup fallback use read_sync().
        from perception_worker import PerceptionWorker
        self._perception_worker = PerceptionWorker(
            eyes=self._eyes, cap=self._cap, shared=self._shared, fps=_perc_fps,
        )
        self._perception_worker.start()

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
        # Death detection lives in env.death_detection.DeathDetector — it owns
        # the health-validation state machine, the streak counters, and the
        # post-reset grace window.  Config keys unchanged.
        self._death = DeathDetector(
            confirm_steps=int(rl_cfg.get("death_confirm_steps", 10)),
            screen_reads=int(rl_cfg.get("death_screen_confirm_reads", 8)),
            grace_steps=int(rl_cfg.get("reset_grace_steps", 15)),
            max_plausible_drop=float(rl_cfg.get("max_plausible_health_drop", 0.5)),
            low_health_thresh=float(rl_cfg.get("low_health_threshold", 0.12)),
        )
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

        # Reward computation lives in env.rewards.RewardCalculator.
        self._rewards = RewardCalculator(self._curriculum_weights)
        self._rewards.set_stage(self._curriculum_stage)

        # ── Episode state (reset in reset()) ──────────────────────────────────
        # Health/ammo/enemy bookkeeping lives in the DeathDetector and
        # RewardCalculator modules; the env keeps only orchestration state.
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
            self._rewards.set_stage(stage)
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

        # Clear the frame buffer (module) — padding refills until real frames arrive
        self._obs_builder.reset()

        # Clear perception's per-episode buffers (health smoothing, ammo cache)
        # so stale low readings from the last episode can't re-trigger a death.
        if hasattr(self._eyes, "reset_state"):
            self._eyes.reset_state()

        frame = self._cap.get_frame()
        # Serialized synchronous read (shares the worker's lock — YOLO/EasyOCR
        # are not thread-safe, so no direct self._eyes calls from this thread).
        hud   = self._perception_worker.read_sync(frame)[1]

        # Seed the perception cache with this fresh read so the first steps have
        # valid detections/HUD before the throttle interval elapses.
        self._cached_hud        = hud
        self._cached_detections = []
        self._last_perception_t = time.time()

        # Reset all episode state
        self._cached_death_screen    = False
        self._perception_fresh       = False
        self._last_enemy_seen_t      = 0.0
        self._llm_steering           = False
        _seed_h = float(hud.get("health_pct", 1.0) or 0.0)
        # If the very first read is a bogus 0, assume full health rather than
        # starting the episode "already dying".
        seed_health = _seed_h if _seed_h > 0.0 else 1.0
        self._death.reset(prev_health=seed_health)
        self._rewards.reset(
            health=seed_health,
            ammo_clip=int(hud.get("ammo_clip", 0) or 0),
            ammo_res=int(hud.get("ammo_res", 0) or 0),
        )
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

        obs = self._obs_builder.build(
            frame, hud, [],
            prev_health=self._death.prev_health,
            prev_ammo_clip=self._rewards.prev_ammo_clip,
            in_combat=False,
            episode_start=self._episode_start,
        )
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

        # Perception now runs on the dedicated worker thread and lands in
        # SharedState; adopt the newest snapshot instead of running YOLO/OCR
        # inline.  The frame itself is always fresh (threaded ScreenCapture).
        # Synchronous fallback covers the startup steps before the worker's
        # first publish (perception_at == 0.0).
        now = time.time()
        if self._shared.perception_at > self._last_perception_t:
            self._cached_detections   = list(self._shared.detections)
            self._cached_hud          = dict(self._shared.hud)
            self._cached_death_screen = bool(self._shared.death_screen)
            self._last_perception_t   = self._shared.perception_at
            self._perception_fresh    = True
        elif self._shared.perception_at == 0.0 and now - self._last_perception_t >= self._perception_interval:
            # Worker hasn't published yet (startup) — one serialized sync read.
            dets, hud_now, death      = self._perception_worker.read_sync(frame)
            self._cached_detections   = dets
            self._cached_hud          = hud_now
            self._cached_death_screen = death
            self._last_perception_t   = now
            self._perception_fresh    = True
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
        obs = self._obs_builder.build(
            frame, hud, detections,
            prev_health=self._death.prev_health,
            prev_ammo_clip=self._rewards.prev_ammo_clip,
            in_combat=self._in_combat,
            episode_start=self._episode_start,
        )

        # ── Reward & termination ──────────────────────────────────────────────
        verdict = self._death.update(
            raw_health=hud.get("health_pct", None),
            death_screen=self._cached_death_screen,
            perception_fresh=self._perception_fresh,
            episode_step=self._episode_step,
        )
        terminated = verdict.died
        if terminated:
            self._death_count += 1
            # Push the running total to SharedState so the dashboard's death
            # stat actually moves.
            self._shared.update(death_count=self._death_count)
            logger.info("Death detected (%s) — ending episode.", verdict.cause)

        reward = self._rewards.compute(RewardInputs(
            hud=hud,
            enemy_labels=enemy_labels,
            died=verdict.died,
            health=verdict.health,
            health_valid=verdict.health_valid,
            damage_taken=verdict.damage_taken,
            last_comb_action=self._last_comb_action,
            was_aiming=self._was_aiming,
            inv_opened=self._inv_opened_step,
            last_mv_action=self._last_mv_action,
            recent_mv_actions=list(self._recent_mv_actions),
            llm_objective=self._shared.llm_objective or "",
        ))
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
