"""
RL training pipeline for the RE4 agent.

ALGORITHM GUIDE  — why RecurrentPPO is the primary choice
──────────────────────────────────────────────────────────
When choosing an RL algorithm for a 3D action game like RE4, the key
constraints are:

  1. PARTIAL OBSERVABILITY — you can't see around corners.  The agent needs
     *memory* of recent events (heard a chainsaw 3 seconds ago = danger nearby).
     → RecurrentPPO adds an LSTM layer that maintains a hidden state across
       steps.  This is like a short-term memory notebook for the agent.

  2. VISUAL RICHNESS — 3D environments need a powerful CNN backbone.
     → We use the Impala ResNet extractor (feature_extractor.py) instead
       of SB3's default NatureCNN.

  3. REACTION SPEED — RE4 requires ~8-12 decisions per second.
     → Our action_hold_seconds = 0.08 s achieves this.  The LSTM keeps track
       of what happened between calls.

  4. SINGLE GPU — we're training on one machine, not a distributed cluster.
     → On-policy methods (PPO, RecurrentPPO) work well on a single GPU.
       Off-policy methods (Rainbow DQN, Ape-X) are more sample-efficient
       but require replay buffers and are harder to stabilise with Dict obs.

ALGORITHM COMPARISON for RE4:
  RecurrentPPO   ← PRIMARY   — handles memory, visual input, stable training
  PPO            ← FALLBACK  — faster per-step, use with frame_stack=4 for memory
  Rainbow DQN    ← FUTURE    — better sample efficiency; requires a replay buffer
                                compatible with Dict observations (not in SB3 yet)
  DreamerV3      ← ADVANCED  — world-model based, SOTA on complex games, but
                                needs 32 GB+ RAM and days to reach competence

Set `algorithm: "PPO"` or `"RecurrentPPO"` in config.yaml to switch.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import ResidentEvilEnv
from memory import MemorySystem
from shared_state import SharedState

logger = logging.getLogger(__name__)


# ── Callbacks ─────────────────────────────────────────────────────────────────

class DashboardCallback(BaseCallback):
    """
    Pushes RL metrics into SharedState every step so the Gradio dashboard
    can display them in real-time.

    Why a callback instead of polling?  SB3's training loop is entirely
    controlled by model.learn().  Callbacks are the official hook points —
    they're called at precise step boundaries without race conditions.
    """

    def __init__(self, shared: SharedState, verbose: int = 0):
        super().__init__(verbose)
        self._shared    = shared
        self._ep_reward = 0.0
        self._ep_length = 0

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])

        reward = float(rewards[0]) if rewards else 0.0
        self._ep_reward += reward
        self._ep_length += 1

        # SB3's num_timesteps is the authoritative, de-duplicated step count
        self._shared.update(total_steps=self.num_timesteps)
        self._shared.append_reward(reward)

        # Episode boundary — finalise stats and reset accumulators
        info = infos[0] if infos else {}
        if "episode" in info:
            ep = info["episode"]
            self._shared.finalize_episode(
                total_reward=ep.get("r", self._ep_reward),
                length=ep.get("l", self._ep_length),
            )
            self._ep_reward = 0.0
            self._ep_length = 0

        # Returning False signals SB3 to stop training (dashboard Stop button)
        return not self._shared.stop_requested


class LLMCallback(BaseCallback):
    """
    Triggers a non-blocking LLM consultation every N training steps.

    Design note: the LLM runs in a separate daemon thread (LLMConsultant).
    If a previous call is still in-flight when we fire again, the new call is
    silently dropped.  This ensures the RL loop never blocks on API latency.
    The consultation result (action override, objective) is written to
    SharedState and read by the environment on the NEXT step.
    """

    def __init__(
        self,
        consultant,           # LLMConsultant — imported lazily to avoid circular dep
        memory: MemorySystem,
        shared: SharedState,
        call_every: int = 60,
        cooldown_seconds: float = 3.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._consultant     = consultant
        self._memory         = memory
        self._shared         = shared
        self._call_every     = call_every
        self._cooldown       = cooldown_seconds
        self._last_call_time = 0.0

    def _on_step(self) -> bool:
        import time
        should_call = (
            self.n_calls % self._call_every == 0
            or self._shared.force_llm_call
        )
        if should_call:
            now = time.monotonic()
            if now - self._last_call_time >= self._cooldown:
                self._last_call_time = now
                self._shared.update(force_llm_call=False)
                self._consultant.trigger(episode_step=self.n_calls)
        return True


class MemoryCallback(BaseCallback):
    """
    Logs step events and LLM decisions into the SQLite memory DB.

    We write only every 10 steps to avoid I/O bottleneck (the DB write queue
    can still fall behind during heavy combat scenes with many YOLO detections).
    """

    def __init__(self, memory: MemorySystem, shared: SharedState, verbose: int = 0):
        super().__init__(verbose)
        self._memory = memory
        self._shared = shared
        self._ep_id  = 0
        self._step   = 0

        # Combat metrics (combat_metrics.py): shots/kills/damage are derived
        # from step-over-step HUD deltas — no game-memory reading required.
        from combat_metrics import CombatLogger
        self._combat        = CombatLogger()
        self._prev_clip     = 0
        self._prev_enemies  = 0
        self._prev_hp: Optional[float] = None

    def _on_training_start(self) -> None:
        self._ep_id = self._memory.start_episode(curriculum=self._shared.curriculum_stage)

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        actions = self.locals.get("actions", [[0, 0, 0, 0, 0, 0]])

        # Convert numpy int64 → Python int so json.dumps in memory.py never fails.
        # SB3 returns actions as numpy arrays; list() preserves the numpy dtype.
        # Using int(x) for x in ... converts each element to a plain Python int.
        action = [int(x) for x in actions[0]] if len(actions) > 0 else [0] * 6
        reward = float(rewards[0]) if rewards else 0.0
        hud    = dict(self._shared.hud)

        hud["enemy_count"] = sum(
            1 for d in self._shared.detections
            if d.get("label") in ("person", "zombie", "enemy")
        )

        # ── Combat deltas ─────────────────────────────────────────────────────
        clip = int(hud.get("ammo_clip", 0) or 0)
        if 0 < clip < self._prev_clip:
            self._combat.log_shot(self._prev_clip - clip)
        self._prev_clip = clip

        enemies_now = hud["enemy_count"]
        if enemies_now < self._prev_enemies and self._combat.shot_recently():
            self._combat.log_kill(self._prev_enemies - enemies_now)
        self._prev_enemies = enemies_now

        hp = hud.get("health_pct")
        if hp is not None and self._prev_hp is not None:
            drop = self._prev_hp - float(hp)
            # Ignore implausible drops — same misread guard as the env's
            # death detection (max 50% per step is real damage).
            if 0.0 < drop <= 0.5:
                self._combat.log_damage(drop, is_player=True)
        if hp is not None and float(hp) > 0.0:
            self._prev_hp = float(hp)

        if self._step % 10 == 0:
            self._memory.log_step(self._step, action, reward, hud)

        info = infos[0] if infos else {}
        if "episode" in info:
            ep = info["episode"]
            final = self._combat.finalize_episode()
            self._memory.end_episode(
                total_reward=ep.get("r", 0.0),
                steps=ep.get("l", 0),
                # Monitor replaces info["episode"] with its own dict, wiping the
                # env's "deaths" key — read the top-level "death_count" the env
                # now also sets (falls back to the old key for compatibility).
                death_count=info.get("death_count", ep.get("deaths", 0)),
                combat={
                    "shots_fired": final.shots_fired,
                    "kills":       final.kills,
                    "accuracy":    final.accuracy,
                },
            )
            # Publish rolling combat stats for the dashboard Combat panel.
            rolling = self._combat.get_rolling_stats()
            if rolling:
                self._shared.update(combat_stats=rolling)
            self._ep_id = self._memory.start_episode(
                curriculum=self._shared.curriculum_stage
            )
            self._step = 0
            self._prev_clip = 0
            self._prev_enemies = 0
            self._prev_hp = None
        else:
            self._step += 1

        return True

    def _on_training_end(self) -> None:
        self._memory.end_episode(0.0, self._step, 0)


class CurriculumCallback(BaseCallback):
    """
    Advances the curriculum stage (exploration → combat → completion) based
    on timestep thresholds set in config.yaml.

    Can also be manually overridden from the Controls tab in the dashboard
    (the dashboard writes to shared.curriculum_stage, which the env reads).

    Analogy: like a videogame's difficulty ramp — the agent starts in a
    "tourist mode" focused on exploring, then gets harder objectives once
    it's competent at staying alive.
    """

    def __init__(
        self,
        env: ResidentEvilEnv,
        shared: SharedState,
        curriculum_cfg: dict,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._env    = env
        self._shared = shared
        self._stages = sorted(
            [
                (cfg.get("until_timestep", 10 ** 9), name)
                for name, cfg in curriculum_cfg["stages"].items()
            ],
            key=lambda x: x[0],
        )
        self._stage_idx = 0

    def _on_step(self) -> bool:
        if self._stage_idx >= len(self._stages):
            return True
        threshold, stage_name = self._stages[self._stage_idx]
        if self.num_timesteps >= threshold:
            self._stage_idx += 1
            if self._stage_idx < len(self._stages):
                next_stage = self._stages[self._stage_idx][1]
                self._env.set_curriculum_stage(next_stage)
                self._shared.update(curriculum_stage=next_stage)
                logger.info(
                    "Curriculum advanced → %s at step %d", next_stage, self.num_timesteps
                )
        return True


# ── Trainer ───────────────────────────────────────────────────────────────────

class RETrainer:
    """
    Builds the VecEnv, model, and callbacks; then runs the training loop.

    Supports RecurrentPPO (default) and plain PPO.  The algorithm is chosen
    by `rl_hyperparameters.algorithm` in config.yaml.

    Model construction happens in build() (called once before train()).
    Checkpoints are saved automatically at `checkpoint_freq` steps.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        shared: Optional[SharedState] = None,
        memory: Optional[MemorySystem] = None,
        consultant=None,   # Optional[LLMConsultant] — kept as Any to avoid import cycles
        config: Optional[dict] = None,
    ):
        # Store the path so build() can pass it to ResidentEvilEnv.
        # Without this, build() was hardcoding "config.yaml" and ignoring any
        # --config flag the user passed on the command line.
        self._config_path = config_path

        from config_loader import ensure_config
        self._cfg = ensure_config(config, config_path)

        self._shared     = shared or SharedState()
        self._memory     = memory or MemorySystem(self._cfg["storage"]["db_path"])
        self._consultant = consultant

        self._rl_cfg    = self._cfg["rl_hyperparameters"]
        self._train_cfg = self._cfg["training"]
        self._cur_cfg   = self._cfg["curriculum"]
        self._storage   = self._cfg["storage"]

        # Create output directories
        Path(self._storage["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self._storage["tensorboard_dir"]).mkdir(parents=True, exist_ok=True)

        self._base_env: Optional[ResidentEvilEnv] = None
        self._env:      Optional[DummyVecEnv]     = None
        self._model:    Optional[object]           = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self) -> None:
        """
        Construct the vectorised environment, optional VecNormalize wrapper,
        and the RL model.

        VecNormalize standardises rewards (mean 0, std 1 running average).
        This massively helps training stability — without it, a reward of +5
        on step 1 and +50 on step 10000 confuse the optimiser.
        Analogy: like converting temperatures to Celsius so the scale is always
        interpretable, regardless of whether you're in Iceland or the Sahara.
        """
        self._base_env = ResidentEvilEnv(config=self._cfg, shared_state=self._shared)

        monitored = Monitor(self._base_env)
        raw_env   = DummyVecEnv([lambda: monitored])
        # Kept so load_checkpoint() can re-wrap it with restored VecNormalize stats.
        self._raw_env = raw_env

        # Optional reward normalisation (recommended — reduces training variance)
        use_vecnorm = self._rl_cfg.get("use_vec_normalize", True)
        if use_vecnorm:
            self._env = VecNormalize(
                raw_env,
                norm_obs=False,       # We normalise obs manually in the env
                norm_reward=True,     # Normalise rewards to mean=0 std=1
                clip_reward=10.0,     # Clip extreme rewards (e.g. after a big kill streak)
                gamma=self._rl_cfg.get("gamma", 0.99),
            )
            logger.info("VecNormalize enabled — rewards will be normalised to mean=0 std=1")
        else:
            self._env = raw_env

        algo = self._rl_cfg.get("algorithm", "RecurrentPPO")
        self._model = self._make_model(algo)

        total_params = sum(p.numel() for p in self._model.policy.parameters())
        logger.info(
            "Model built: %s  total_policy_params=%s  vec_normalize=%s",
            algo, f"{total_params:,}", use_vecnorm,
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Resume training from a saved .zip checkpoint.

        Also restores the sibling `<name>_vecnorm.pkl` (VecNormalize running
        reward stats) if present.  Resuming a policy with RESET normalisation
        stats rescales every reward the value function sees — that mismatch is
        what collapsed the June 28 sim run.

        A corrupt/unreadable checkpoint no longer kills the trainer thread —
        we log the error and continue with the freshly built model instead.
        """
        algo = self._rl_cfg.get("algorithm", "RecurrentPPO")
        cls  = self._get_model_cls(algo)

        # 1. Restore VecNormalize stats if a sibling pkl exists.
        try:
            norm_path = Path(path).with_name(Path(path).stem + "_vecnorm.pkl")
            if norm_path.exists() and isinstance(self._env, VecNormalize):
                self._env = VecNormalize.load(str(norm_path), self._raw_env)
                self._env.training = True
                logger.info("VecNormalize stats restored from %s", norm_path)
        except Exception as exc:
            logger.warning("VecNormalize restore failed (%s) — continuing with fresh stats.", exc)

        # 2. Load the model itself.
        try:
            self._model = cls.load(
                path,
                env=self._env,
                tensorboard_log=self._storage["tensorboard_dir"],
            )
            logger.info("Resumed from checkpoint: %s", path)
        except Exception as exc:
            logger.error(
                "Could not load checkpoint %s (%s) — the file may be corrupt "
                "(e.g. an interrupted save). Continuing with a FRESH model.",
                path, exc,
            )
            # Re-point the already-built fresh model at the (possibly re-wrapped) env.
            if self._model is not None:
                self._model.set_env(self._env)

    def train(self) -> None:
        if self._model is None or self._env is None:
            raise RuntimeError("Call build() before train().")

        self._shared.update(is_training=True)
        callbacks = self._build_callbacks()

        try:
            self._model.learn(
                total_timesteps=self._train_cfg["total_timesteps"],
                callback=callbacks,
                reset_num_timesteps=False,
                progress_bar=True,
            )
        finally:
            self._shared.update(is_training=False)
            # Neutralise the gamepad — sticks/triggers now persist across steps,
            # so without this an interrupted run would leave Leon walking into a
            # wall until the process exits.
            try:
                if self._base_env is not None and hasattr(self._base_env, "_ctrl"):
                    self._base_env._ctrl.release_all()
                    logger.info("Gamepad released — all axes/buttons neutralised.")
            except Exception as exc:
                logger.warning("Gamepad release on stop failed (non-fatal): %s", exc)
            # Atomic save: write to a temp file then rename, so an interrupted
            # save can never leave a truncated (corrupt) re_agent_final.zip.
            final_path = Path(self._storage["checkpoint_dir"]) / "re_agent_final.zip"
            tmp_path   = final_path.with_name(final_path.stem + ".tmp.zip")
            self._model.save(str(tmp_path))
            os.replace(tmp_path, final_path)

            # If VecNormalize is active, save its running stats too so they can
            # be restored when resuming — otherwise the first steps after resume
            # will have wildly miscalibrated reward normalisation.
            if isinstance(self._env, VecNormalize):
                norm_path = str(final_path).replace(".zip", "_vecnorm.pkl")
                self._env.save(norm_path)
                logger.info("VecNormalize stats saved → %s", norm_path)

            logger.info("Training complete — model saved to %s", final_path)

    def stop(self) -> None:
        self._shared.update(stop_requested=True)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _make_model(self, algo: str) -> object:
        """
        Build the RL model with the Impala CNN feature extractor and tuned
        hyperparameters for RE4's visual complexity.

        HYPERPARAMETER NOTES:
          learning_rate: 2.5e-4 is the standard PPO value; lower for stability
          n_steps:       512  — shorter rollouts for faster curriculum feedback
                                (vs 2048 default which gives stale data)
          batch_size:    128  — larger batch smoother gradient estimates
          n_epochs:      4    — fewer epochs prevents over-fitting on-policy data
          ent_coef:      0.01 — entropy bonus keeps the policy exploring
          clip_range:    0.2  — PPO clipping (standard)
          lstm_hidden:   256  — LSTM state size; 512 is better but uses more VRAM
        """
        from feature_extractor import RE4FeaturesExtractor

        cls    = self._get_model_cls(algo)
        policy = "MultiInputLstmPolicy" if algo == "RecurrentPPO" else "MultiInputPolicy"

        features_dim = self._rl_cfg.get("features_dim", 512)

        policy_kwargs: dict = {
            # Replace the default NatureCNN with our Impala ResNet
            # Analogy: swapping out a basic flashlight for a professional-grade
            # searchlight — same batteries, much better at illuminating dark corners.
            "features_extractor_class":  RE4FeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": features_dim},

            # Policy and value function heads — 2-layer MLP with 256 units each.
            # SB3 ≥1.8 format: a plain dict (not a list containing a dict).
            # The old format [dict(...)] is deprecated and may emit warnings.
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        }

        if algo == "RecurrentPPO":
            # LSTM memory for temporal reasoning
            policy_kwargs["lstm_hidden_size"] = self._rl_cfg.get("lstm_hidden_size", 256)
            policy_kwargs["n_lstm_layers"]    = self._rl_cfg.get("n_lstm_layers", 1)

        # Learning-rate schedule — "linear" decays to 0 over the run, which
        # stabilises late training; "constant" preserves the old behaviour.
        lr_value = float(self._rl_cfg.get("learning_rate", 2.5e-4))
        if str(self._rl_cfg.get("lr_schedule", "constant")).lower() == "linear":
            learning_rate = lambda progress_remaining: lr_value * progress_remaining
        else:
            learning_rate = lr_value

        model = cls(
            policy=policy,
            env=self._env,
            learning_rate=learning_rate,
            target_kl=self._rl_cfg.get("target_kl", None),
            n_steps=self._rl_cfg.get("n_steps", 512),
            batch_size=self._rl_cfg.get("batch_size", 128),
            n_epochs=self._rl_cfg.get("n_epochs", 4),
            gamma=self._rl_cfg.get("gamma", 0.99),
            gae_lambda=self._rl_cfg.get("gae_lambda", 0.95),
            clip_range=self._rl_cfg.get("clip_range", 0.2),
            ent_coef=self._rl_cfg.get("ent_coef", 0.01),
            max_grad_norm=self._rl_cfg.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=self._storage["tensorboard_dir"],
            verbose=1,
        )
        return model

    @staticmethod
    def _get_model_cls(algo: str):
        """
        Lazy import of the model class.

        IMPORTANT — this must be a lazy import (inside the method, not at the
        top of the file).  Why?  The trainer module is imported before the
        environment is set up, and sb3-contrib's RecurrentPPO imports torch
        immediately.  If torch isn't on PYTHONPATH yet (e.g. during testing),
        a top-level import would crash the whole module.
        """
        if algo == "RecurrentPPO":
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO
        from stable_baselines3 import PPO
        return PPO

    def _build_callbacks(self) -> list:
        """
        Assemble the callback list.  Callbacks run in order every step.

        Order matters:
          1. CheckpointCallback  — save model first (before potential stop)
          2. DashboardCallback   — update metrics (needed for accurate stop detection)
          3. MemoryCallback      — log to DB (lowest priority)
          4. LLMCallback         — fire LLM if needed (expensive, last)
          5. CurriculumCallback  — check stage transitions (after metrics updated)
        """
        checkpoint_cb = CheckpointCallback(
            save_freq=self._train_cfg.get("checkpoint_freq", 50_000),
            save_path=self._storage["checkpoint_dir"],
            name_prefix="re_agent",
            verbose=1,
        )
        dashboard_cb  = DashboardCallback(self._shared)
        memory_cb     = MemoryCallback(self._memory, self._shared)

        cbs = [checkpoint_cb, dashboard_cb, memory_cb]

        # LLM callback — only if a consultant was provided
        if self._consultant is not None:
            llm_cfg = self._cfg.get("llm_settings", {})
            cbs.append(LLMCallback(
                consultant=self._consultant,
                memory=self._memory,
                shared=self._shared,
                call_every=llm_cfg.get("llm_every_n_steps", 60),
                cooldown_seconds=llm_cfg.get("call_cooldown_seconds", 3.0),
            ))

        # Curriculum callback — only if curriculum is enabled
        if self._cur_cfg.get("enabled", False):
            cbs.append(CurriculumCallback(
                env=self._base_env,
                shared=self._shared,
                curriculum_cfg=self._cur_cfg,
            ))

        return cbs
