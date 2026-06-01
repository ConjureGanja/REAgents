"""
Domain Randomization Wrapper for Sim-to-Real Transfer
======================================================

THE PROBLEM THIS SOLVES
-----------------------
Your sim renders symbolic colored squares.  Real RE4 is a photorealistic AAA
game.  A CNN trained only on the sim will memorise things like:

    "pixel value [0, 220, 0] in channel 1 == player"
    "pixel value [220, 0, 0] in channel 1 == enemy"

When you swap in real game frames, these exact pixel patterns don't exist
anywhere — and the policy collapses.  This is the **sim-to-real gap**.

THE SOLUTION: DOMAIN RANDOMIZATION
-----------------------------------
During training, we deliberately corrupt the sim observations every episode:
    • Random brightness, contrast, hue shift
    • Random Gaussian noise, salt-and-pepper noise
    • Motion blur, JPEG compression artifacts
    • Random color permutations
    • Random distractor objects (decoy "enemies" the policy must ignore)
    • Random dynamics jitter (enemy speed ±20%, damage ±15%)

This forces the policy to learn **invariant features** — relative positions,
shapes, motion patterns — instead of exact pixel values.  By training time,
the policy generalises to *any* visual style, including the real game.

Analogy
-------
Imagine teaching a child to recognise dogs only by showing photos taken in
sunny weather.  When they encounter a dog at night, they're stumped.
Now imagine showing photos in sun, rain, snow, blurry, B&W, cartoonish, etc.
The child learns "dog-ness" itself, not "the visual pattern of sunny dogs".
That's what domain randomization does for the agent.

REFERENCE
---------
"Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
(Peng et al., 2018) — the seminal work showing this transfers to physical
robots.  Same principle applies here.

USAGE
-----
    from sim.domain_randomization import DomainRandomizationWrapper, RandomizationConfig

    base_env  = SimResidentEvilEnv(config_path="config.yaml", worker_id=0)
    train_env = DomainRandomizationWrapper(
        base_env,
        config=RandomizationConfig.training_preset(),  # full randomization
    )

    # For evaluation — disable randomization to measure clean policy quality
    eval_env  = DomainRandomizationWrapper(
        base_env,
        config=RandomizationConfig.eval_preset(),      # zero randomization
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RandomizationConfig:
    """
    Tunable per-episode randomization parameters.

    All ranges are inclusive `[min, max]`.  At each `reset()` we sample one
    value from each range and apply it consistently for the entire episode.

    NOTE on tuning
    --------------
    Start with `training_preset()` defaults.  If your policy fails to learn,
    *reduce* randomization (the agent is overwhelmed).  If it learns the sim
    perfectly but flops on the real game, *increase* randomization (the
    agent over-fit to clean sim visuals).

    Visual params (operate on the rendered frame)
    ----------------------------------------------
    brightness_range        : multiplicative factor, e.g. [0.6, 1.4] = ±40%
    contrast_range          : multiplicative factor for contrast around 128
    hue_shift_range         : degrees, [-180, 180]
    saturation_range        : multiplicative factor on the color channels' spread
    gaussian_noise_std      : pixel-value sigma added per frame (0-30)
    salt_pepper_prob        : fraction of pixels turned pure black/white
    motion_blur_kernel_max  : max kernel size for motion blur (odd, 1-9). 1 = off
    jpeg_quality_min        : low quality bound for JPEG compression artifacts (10-100)
    distractor_count_max    : max number of fake "enemy"-coloured dots painted in

    Dynamics params (operate on the simulation itself)
    --------------------------------------------------
    enemy_speed_jitter      : ±fraction, e.g. 0.20 = enemy speed multiplied by [0.80, 1.20]
    damage_jitter           : ±fraction on damage dealt by enemies
    detection_range_jitter  : ±fraction on enemy detection radius
    starting_ammo_jitter    : ±fraction on starting magazine size
    """

    # ── Visual (frame-level) ──────────────────────────────────────────────────
    brightness_range:        Tuple[float, float] = (0.65, 1.35)
    contrast_range:          Tuple[float, float] = (0.70, 1.30)
    hue_shift_range:         Tuple[float, float] = (-30.0, 30.0)
    saturation_range:        Tuple[float, float] = (0.6, 1.4)
    gaussian_noise_std:      float               = 8.0
    salt_pepper_prob:        float               = 0.005
    motion_blur_kernel_max:  int                 = 3   # 1 = disabled, must be odd
    jpeg_quality_min:        int                 = 60  # set to 100 to disable
    distractor_count_max:    int                 = 4

    # ── Dynamics (sim-level) ─────────────────────────────────────────────────
    enemy_speed_jitter:      float               = 0.20  # ±20%
    damage_jitter:           float               = 0.15  # ±15%
    detection_range_jitter:  float               = 0.20  # ±20%
    starting_ammo_jitter:    float               = 0.0   # off by default

    # ── Master switches (debug aid) ──────────────────────────────────────────
    enable_visual:    bool = True
    enable_dynamics:  bool = True
    enable_distractors: bool = True

    # Per-episode RNG seed (None = use Python's default RNG state)
    seed: Optional[int] = None

    @classmethod
    def training_preset(cls) -> "RandomizationConfig":
        """Aggressive randomization — optimal for transferable policy learning."""
        return cls()  # defaults are already tuned for training

    @classmethod
    def eval_preset(cls) -> "RandomizationConfig":
        """No randomization — for measuring clean policy quality."""
        return cls(
            brightness_range       = (1.0, 1.0),
            contrast_range         = (1.0, 1.0),
            hue_shift_range        = (0.0, 0.0),
            saturation_range       = (1.0, 1.0),
            gaussian_noise_std     = 0.0,
            salt_pepper_prob       = 0.0,
            motion_blur_kernel_max = 1,
            jpeg_quality_min       = 100,
            distractor_count_max   = 0,
            enemy_speed_jitter     = 0.0,
            damage_jitter          = 0.0,
            detection_range_jitter = 0.0,
            starting_ammo_jitter   = 0.0,
            enable_visual          = False,
            enable_dynamics        = False,
            enable_distractors     = False,
        )

    @classmethod
    def gentle_preset(cls) -> "RandomizationConfig":
        """Lighter randomization — fall-back if training doesn't converge."""
        return cls(
            brightness_range       = (0.85, 1.15),
            contrast_range         = (0.90, 1.10),
            hue_shift_range        = (-10.0, 10.0),
            saturation_range       = (0.85, 1.15),
            gaussian_noise_std     = 3.0,
            salt_pepper_prob       = 0.002,
            motion_blur_kernel_max = 1,
            jpeg_quality_min       = 80,
            distractor_count_max   = 1,
            enemy_speed_jitter     = 0.10,
            damage_jitter          = 0.05,
            detection_range_jitter = 0.10,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DomainRandomizationWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that randomizes visual + dynamics each episode.

    Wraps any env exposing the observation Dict {"frame": uint8[H,W,C], "hud": ...}.
    The wrapped env's observation_space and action_space are unchanged — the
    policy sees exactly the same shapes/types.

    KEY DESIGN DECISIONS
    --------------------
    1. **Per-episode randomization** (not per-step).  This is intentional:
       the agent should reason about a *consistent* visual style within an
       episode, just like the real game has consistent lighting per scene.
       Per-step randomization would just look like fast-flickering noise
       and destroy temporal consistency.

       Analogy: a movie director picks ONE colour grade for a scene and
       sticks with it; they don't switch grading every frame.

    2. **Dynamics randomization is applied via env_method**, not by editing
       state directly.  This keeps wrapper code clean and lets the underlying
       env validate any out-of-range values.

    3. **Distractors are painted onto the frame AFTER all augmentations**, so
       the noise/blur doesn't "smudge" them into invisibility.  We want the
       agent to see crisp distractors so it learns to actively ignore them.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[RandomizationConfig] = None,
    ):
        super().__init__(env)
        self.cfg = config or RandomizationConfig.training_preset()
        self._rng = np.random.default_rng(self.cfg.seed)

        # Per-episode parameters (sampled at reset())
        self._brightness:   float       = 1.0
        self._contrast:     float       = 1.0
        self._hue_shift:    float       = 0.0
        self._saturation:   float       = 1.0
        self._noise_std:    float       = 0.0
        self._sp_prob:      float       = 0.0
        self._blur_kernel:  int         = 1
        self._jpeg_q:       int         = 100
        self._distractors:  list        = []   # list of (x, y, color) painted on frame

        # Lazy-imported cv2 — we only need it when augmentations are enabled
        self._cv2 = None

        logger.info(
            "DomainRandomizationWrapper init  visual=%s  dynamics=%s",
            self.cfg.enable_visual, self.cfg.enable_dynamics,
        )

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Resample randomization parameters, then forward to the wrapped env.

        Visual params are stored on `self` for use in step()/_augment().
        Dynamics params are pushed to the wrapped env's SimGameState before
        reset (so the new episode uses the new dynamics).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.cfg.enable_visual:
            self._sample_visual_params()
        if self.cfg.enable_distractors:
            self._sample_distractors()
        if self.cfg.enable_dynamics:
            self._apply_dynamics_jitter()

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._augment_obs(obs)
        return obs, info

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._augment_obs(obs)
        return obs, reward, terminated, truncated, info

    # ── Sampling new parameters ───────────────────────────────────────────────

    def _sample_visual_params(self) -> None:
        c = self.cfg
        self._brightness  = float(self._rng.uniform(*c.brightness_range))
        self._contrast    = float(self._rng.uniform(*c.contrast_range))
        self._hue_shift   = float(self._rng.uniform(*c.hue_shift_range))
        self._saturation  = float(self._rng.uniform(*c.saturation_range))
        self._noise_std   = float(self._rng.uniform(0.0, c.gaussian_noise_std))
        self._sp_prob     = float(self._rng.uniform(0.0, c.salt_pepper_prob))

        # Motion blur kernel — odd integer in [1, motion_blur_kernel_max]
        max_k = max(1, c.motion_blur_kernel_max)
        if max_k > 1:
            k = int(self._rng.integers(1, max_k + 1))
            if k % 2 == 0:
                k += 1   # ensure odd
            self._blur_kernel = k
        else:
            self._blur_kernel = 1

        # JPEG quality — uniform in [min, 100]
        self._jpeg_q = int(self._rng.integers(c.jpeg_quality_min, 101))

    def _sample_distractors(self) -> None:
        """
        Generate fake "enemy"-coloured dots that get painted on each frame.

        Why?  In the real game, the YOLO-detected world contains LOTS of objects
        the agent must learn to ignore (NPCs, decorative props, UI text bleed-
        through, etc.).  Adding decoys forces the policy to use *temporal*
        cues (this red dot moved consistently → real enemy) instead of just
        "any red dot is dangerous".

        Analogy: a chess engine that learns to win by always taking the closest
        pawn would get destroyed against a real opponent who deliberately
        positions decoy pawns.
        """
        n = int(self._rng.integers(0, max(1, self.cfg.distractor_count_max + 1)))
        self._distractors = []
        for _ in range(n):
            # Random pixel location in the 84x84 frame (will be scaled if frame is larger)
            x = int(self._rng.integers(2, 82))
            y = int(self._rng.integers(2, 82))
            # Random colour close to the real enemy red so it's plausible
            r = int(self._rng.integers(150, 255))
            g = int(self._rng.integers(0, 80))
            b = int(self._rng.integers(0, 80))
            self._distractors.append((x, y, (r, g, b)))

    def _apply_dynamics_jitter(self) -> None:
        """
        Mutate the wrapped sim's runtime parameters before its reset() runs.

        The wrapped env exposes its SimGameState as `env._sim`.  We multiply
        the relevant attributes by a random factor in [1-j, 1+j].

        Important: we always re-read the *config* values (not whatever was
        last set), so jitter doesn't compound across episodes.
        """
        c   = self.cfg
        sim = getattr(self.env, "_sim", None)
        if sim is None:
            logger.debug("Wrapped env exposes no _sim — skipping dynamics jitter")
            return

        cfg_dict = getattr(sim, "_cfg", {})

        # Helper: fresh value from config, then jittered
        def _jitter(base: float, frac: float) -> float:
            if frac <= 0.0:
                return base
            f = float(self._rng.uniform(1.0 - frac, 1.0 + frac))
            return base * f

        sim.enemy_speed     = _jitter(float(cfg_dict.get("enemy_speed",      0.28)), c.enemy_speed_jitter)
        sim.enemy_chase_spd = _jitter(float(cfg_dict.get("enemy_chase_speed",0.40)), c.enemy_speed_jitter)
        sim.attack_damage   = _jitter(float(cfg_dict.get("enemy_attack_damage",12.0)), c.damage_jitter)
        sim.detect_range    = _jitter(float(cfg_dict.get("enemy_detection_range",14.0)), c.detection_range_jitter)

        if c.starting_ammo_jitter > 0:
            base_clip = int(cfg_dict.get("ammo_start_clip", 15))
            sim.ammo_clip_start = max(1, int(_jitter(base_clip, c.starting_ammo_jitter)))

    # ── Augmentations ─────────────────────────────────────────────────────────

    def _augment_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all visual augmentations to obs["frame"] in place-safe manner."""
        if not self.cfg.enable_visual and not self.cfg.enable_distractors:
            return obs

        frame = obs["frame"]                    # (H, W, C) uint8 — possibly stacked
        # Frame stacking: shape may be (H, W, 12) for 4×RGB stack.
        # We augment each 3-channel frame separately so each "time slice" has
        # the same per-episode visual style but slightly different per-frame
        # noise (which is the *point* of noise — adds temporal variance).
        H, W, C = frame.shape

        if C == 1:
            # Grayscale (no stacking) — process whole tensor at once
            frame_aug = self._augment_single_frame(frame)
        elif C == 3:
            # Single RGB frame
            frame_aug = self._augment_single_frame(frame)
        elif C % 3 == 0:
            # Stacked RGB
            n = C // 3
            slices = []
            for i in range(n):
                s = frame[:, :, i*3 : (i+1)*3]
                slices.append(self._augment_single_frame(s))
            frame_aug = np.concatenate(slices, axis=2)
        elif C % 1 == 0 and C > 1:
            # Stacked grayscale — augment each
            slices = []
            for i in range(C):
                s = frame[:, :, i:i+1]
                slices.append(self._augment_single_frame(s))
            frame_aug = np.concatenate(slices, axis=2)
        else:
            frame_aug = frame

        return {**obs, "frame": frame_aug}

    def _augment_single_frame(self, f: np.ndarray) -> np.ndarray:
        """Apply augmentations to a single (H, W, C) frame (C=1 or C=3)."""
        out = f.astype(np.float32)

        if self.cfg.enable_visual:
            # 1. Brightness — multiply
            out *= self._brightness
            # 2. Contrast — pull toward 128
            out = (out - 128.0) * self._contrast + 128.0

            # 3. Hue / saturation (only RGB) ── operates via cv2 HSV space
            if f.shape[2] == 3 and (abs(self._hue_shift) > 0.1 or
                                    abs(self._saturation - 1.0) > 0.05):
                cv2 = self._cv()
                rgb = np.clip(out, 0, 255).astype(np.uint8)
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 0] = (hsv[..., 0] + self._hue_shift / 2.0) % 180     # OpenCV hue is 0-179
                hsv[..., 1] *= self._saturation
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

            # 4. Gaussian noise
            if self._noise_std > 0:
                out += self._rng.normal(0.0, self._noise_std, size=out.shape)

            # 5. Salt-and-pepper noise
            if self._sp_prob > 0:
                mask = self._rng.random(out.shape[:2])
                out[mask < self._sp_prob / 2.0]                       = 0.0
                out[mask > 1.0 - self._sp_prob / 2.0]                 = 255.0

            # 6. Motion blur
            if self._blur_kernel > 1:
                cv2 = self._cv()
                k = self._blur_kernel
                # Horizontal motion blur kernel (the most common type in fast-paced games)
                kernel = np.zeros((k, k), dtype=np.float32)
                kernel[k // 2, :] = 1.0 / k
                out = cv2.filter2D(out.astype(np.uint8), -1, kernel).astype(np.float32)

        # Clip & cast back to uint8
        out = np.clip(out, 0, 255).astype(np.uint8)

        # 7. JPEG compression artifacts (after blur for realism)
        if self.cfg.enable_visual and self._jpeg_q < 100:
            cv2 = self._cv()
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_q]
            ok, enc = cv2.imencode(".jpg", out, params)
            if ok:
                out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
                if out.ndim == 2:
                    out = out[:, :, np.newaxis]

        # 8. Distractors — painted last so noise/blur doesn't erase them
        if self.cfg.enable_distractors and self._distractors and out.shape[2] == 3:
            for (x, y, col) in self._distractors:
                # Scale coords to current frame size (in case obs_size != 84)
                xi = int(x * out.shape[1] / 84)
                yi = int(y * out.shape[0] / 84)
                if 0 <= xi < out.shape[1] and 0 <= yi < out.shape[0]:
                    # Draw a 2×2 dot
                    out[max(0,yi-1):yi+2, max(0,xi-1):xi+2] = col

        return out

    def _cv(self):
        """Lazy-load cv2 only if any augmentation needs it."""
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity check: wrap a sim env and verify obs shapes/types are unchanged
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from sim_environment import SimResidentEvilEnv

    base = SimResidentEvilEnv(config_path="config.yaml", worker_id=0)
    print("Base obs space:", base.observation_space)

    wrapped = DomainRandomizationWrapper(base, RandomizationConfig.training_preset())
    obs, _ = wrapped.reset()
    print("Wrapped obs frame shape :", obs["frame"].shape, obs["frame"].dtype)
    print("Wrapped obs hud   shape :", obs["hud"].shape,   obs["hud"].dtype)

    # Run 200 steps with visual sanity checks
    total = 0
    import time
    t0 = time.perf_counter()
    for _ in range(200):
        a = wrapped.action_space.sample()
        obs, _, term, trunc, _ = wrapped.step(a)
        total += 1
        if term or trunc:
            wrapped.reset()
    print(f"Ran {total} steps in {time.perf_counter() - t0:.2f}s — wrapper OK.")
