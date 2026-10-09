"""
Multi-worker training pipeline for simulation mode.

KEY DIFFERENCE FROM trainer.py
────────────────────────────────
trainer.py uses DummyVecEnv — a single-process wrapper that runs one
environment at a time (sequentially in the main process).

sim_trainer.py uses SubprocVecEnv — one worker process per environment.
Each process runs its own SimResidentEvilEnv with a different random seed.
SB3 collects rollouts from all 8 workers simultaneously, then merges them
for the policy update.

Analogy: DummyVecEnv is like a single student doing all the practice problems
sequentially.  SubprocVecEnv is like 8 students each working through a
different set of problems at the same time, then pooling their answers.
The teacher (the policy gradient update) learns from all 8 sets at once.

EFFECTIVE BATCH SIZE
─────────────────────
With 8 workers and n_steps=512:
  Rollout size = 8 × 512 = 4,096 transitions before each policy update.
  This gives a much better gradient estimate than 512 from a single env.
  The learning rate may need to be slightly lower (≤2e-4) to compensate.

CALLBACK CHANGES
────────────────
SB3's callbacks receive vectorised arrays when using multiple envs:
  self.locals["rewards"]  → shape (8,) — one reward per worker per step
  self.locals["infos"]    → list of 8 dicts

SimDashboardCallback handles this correctly by aggregating across workers
and routing per-worker data to SharedState.update_worker().
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from shared_state import SharedState
from memory import MemorySystem

logger = logging.getLogger(__name__)


# ── Callbacks ─────────────────────────────────────────────────────────────────

class SimDashboardCallback(BaseCallback):
    """
    Aggregates multi-worker metrics into SharedState for the dashboard.

    Called every step with arrays of shape (n_workers,) for rewards and
    a list of n_workers dicts for infos.

    Per-worker snapshots (small dicts from sim_environment.py) are routed
    to SharedState.update_worker() whenever they appear in the info dict.
    The dashboard uses these snapshots to render the live village grid.

    Throughput tracking: counts total steps over the last 5 seconds and
    writes steps/second to shared.sim_steps_per_sec.  Like a tachometer —
    gives immediate feedback on how fast training is running.
    """

    def __init__(self, shared: SharedState, n_workers: int, verbose: int = 0):
        super().__init__(verbose)
        self._shared      = shared
        self._n_workers   = n_workers
        self._ep_rewards  = [0.0] * n_workers
        self._ep_lengths  = [0]   * n_workers

        # Throughput tracking
        self._step_times: List[float] = []
        self._last_throughput_calc    = time.monotonic()
        self._steps_since_last        = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(self._n_workers))
        infos   = self.locals.get("infos",   [{}] * self._n_workers)

        now = time.monotonic()
        self._steps_since_last += self._n_workers  # Each step = N_WORKERS transitions

        # ── Per-worker stats ──────────────────────────────────────────────────
        for i, (r, info) in enumerate(zip(rewards, infos)):
            self._ep_rewards[i] += float(r)
            self._ep_lengths[i] += 1

            # Route snapshot to SharedState if available
            if "sim_snapshot" in info:
                self._shared.update_worker(
                    worker_id=i,
                    snapshot=info["sim_snapshot"],
                    reward=self._ep_rewards[i],
                )

            # Episode boundary — log and reset accumulators
            if "episode" in info:
                ep   = info["episode"]
                ep_r = ep.get("r", self._ep_rewards[i])
                ep_l = ep.get("l", self._ep_lengths[i])

                self._shared.finalize_episode(total_reward=ep_r, length=ep_l)

                # Log notable events to the objectives log
                kills   = ep.get("kills", 0)
                items   = ep.get("items", 0)
                success = ep.get("success", False)
                shotgun = ep.get("shotgun", False)

                msg_parts = [f"W{i} ep#{self._shared.episode_count}"]
                if success:
                    msg_parts.append("✓ SURVIVED")
                if shotgun:
                    msg_parts.append("★ Shotgun collected!")
                if kills >= 5:
                    msg_parts.append(f"{kills} kills")
                if items >= 3:
                    msg_parts.append(f"{items} items")
                msg_parts.append(f"reward={ep_r:.1f}")

                self._shared.log_objective_event("  ".join(msg_parts))

                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0

                # Episode count is tracked below; keep last worker snapshot/reward intact.
                # Also increment episode counter safely via SharedState
                with self._shared._lock:
                    while len(self._shared.worker_episodes) <= i:
                        self._shared.worker_episodes.append(0)
                    self._shared.worker_episodes[i] += 1

        # ── Aggregate metrics ─────────────────────────────────────────────────
        mean_reward = float(np.mean(rewards))
        self._shared.update(
            total_steps=self.num_timesteps,
            last_reward=mean_reward,
        )
        self._shared.append_reward(mean_reward)

        # ── Throughput calculation (every 2 seconds) ──────────────────────────
        elapsed = now - self._last_throughput_calc
        if elapsed >= 2.0:
            sps = self._steps_since_last / elapsed
            self._shared.update(sim_steps_per_sec=sps)
            self._steps_since_last       = 0
            self._last_throughput_calc   = now

        return not self._shared.stop_requested


class SimMemoryCallback(BaseCallback):
    """
    Logs simulation episode summaries to the SQLite memory DB.

    Only logs episode-level data (not every step) to keep DB writes fast.
    Step-level logging is skipped — at 80k steps/second, per-step DB writes
    would immediately become the bottleneck.

    Analogy: a race track that records lap times but not every wheel rotation.
    """

    def __init__(self, memory: MemorySystem, shared: SharedState, verbose: int = 0):
        super().__init__(verbose)
        self._memory  = memory
        self._shared  = shared
        self._ep_ids  = {}  # worker_id → current episode id

    def _on_training_start(self) -> None:
        for i in range(self.training_env.num_envs):  # type: ignore[attr-defined]
            self._ep_ids[i] = self._memory.start_episode(
                curriculum=self._shared.curriculum_stage
            )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            if "episode" in info:
                ep = info["episode"]
                self._memory.end_episode(
                    total_reward=ep.get("r", 0.0),
                    steps=ep.get("l", 0),
                    death_count=int(not ep.get("success", False)),
                )
                self._ep_ids[i] = self._memory.start_episode(
                    curriculum=self._shared.curriculum_stage
                )
        return True

    def _on_training_end(self) -> None:
        """Clean up any open episodes on training end to avoid DB leaks."""
        for ep_id in list(self._ep_ids.values()):
            try:
                self._memory.end_episode(0.0, 0, 0)
            except Exception as e:
                logger.warning("Failed to end episode %s on training end: %s", ep_id, e)


class SimCurriculumCallback(BaseCallback):
    """
    Advances curriculum stage based on timestep thresholds.
    Identical logic to trainer.py's CurriculumCallback but updated to
    call set_curriculum_stage on each underlying SubprocVecEnv worker
    via env_method (the SubprocVecEnv API for calling env methods).
    """

    def __init__(self, shared: SharedState, curriculum_cfg: dict, verbose: int = 0):
        super().__init__(verbose)
        self._shared    = shared
        self._stages    = sorted(
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
                # Call set_curriculum_stage on all workers via SubprocVecEnv API
                self.training_env.env_method(  # type: ignore[attr-defined]
                    "set_curriculum_stage", next_stage
                )
                self._shared.update(curriculum_stage=next_stage)
                msg = f"Curriculum → {next_stage} at step {self.num_timesteps:,}"
                logger.info(msg)
                self._shared.log_objective_event(f"★ {msg}")
        return True


# ── Trainer ───────────────────────────────────────────────────────────────────

class SimRETrainer:
    """
    Builds a SubprocVecEnv (8 workers), creates the RL model, and runs training.

    Usage:
        trainer = SimRETrainer(config_path, shared, memory)
        trainer.build()
        trainer.train()   # blocks until done or stop_requested

    The public interface is identical to RETrainer in trainer.py so that
    sim_main.py and main.py share the same pattern.

    SUBPROCESS CAVEATS
    ───────────────────
    SubprocVecEnv spawns real OS processes — on Windows, each process
    reimports the entire module before __init__ runs.  That's why env
    factory functions must be simple callables (lambdas work) and why
    the simulation modules must be importable without side effects.

    if __name__ == "__main__" guards in sim_game_state.py and
    sim_environment.py ensure their test blocks don't run in subprocesses.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        shared: Optional[SharedState] = None,
        memory: Optional[MemorySystem] = None,
        consultant=None,
    ):
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        self._config_path = config_path
        self._shared      = shared or SharedState()
        self._memory      = memory or MemorySystem(self._cfg["storage"]["db_path"])
        self._consultant  = consultant

        self._rl_cfg    = self._cfg["rl_hyperparameters"]
        self._train_cfg = self._cfg["training"]
        self._cur_cfg   = self._cfg["curriculum"]
        self._sim_cfg   = self._cfg.get("simulation", {})
        self._storage   = self._cfg["storage"]

        self._n_workers: int = int(self._sim_cfg.get("n_workers", 8))

        Path(self._storage["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self._storage["tensorboard_dir"]).mkdir(parents=True, exist_ok=True)

        self._env:   Optional[object] = None
        self._model: Optional[object] = None

        logger.info("SimRETrainer: %d workers, config=%s", self._n_workers, config_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self) -> None:
        """
        Create the SubprocVecEnv with N worker processes and the RL model.

        Each worker gets a unique worker_id (0 through N-1) so their random
        seeds and dashboard snapshot slots differ.

        IMPORTANT — factory function pattern:
          The factory MUST use default-argument capture (worker_id=i) to avoid
          the Python closure-over-loop-variable gotcha.  Without it, all 8
          workers would get worker_id=7 (the final value of i).

          Analogy: a factory stamping parts with a serial number.  If the
          stamp isn't set before the next part comes down the line, every
          part gets the same number.
        """
        config_path = self._config_path
        # NOTE: do NOT capture `self._shared` in the factory closure.
        # SubprocVecEnv pickles the factory and everything it closes over
        # before shipping it to the worker process.  SharedState contains a
        # threading.RLock which is not picklable → OSError on Windows spawn.
        #
        # Workers are self-contained: they receive only config_path + worker_id
        # (both picklable strings/ints) and report back via the info dict.
        # The SimDashboardCallback in the main process writes to SharedState.
        #
        # Analogy: sending a letter (picklable data) to a field agent rather
        # than trying to hand them a live phone line (RLock) through the mail.

        def _make_env(worker_id: int):
            """Factory — returns a callable that constructs one sim env."""
            def _init():
                # Import inside the function so subprocesses reimport cleanly
                from sim_environment import SimResidentEvilEnv
                env = SimResidentEvilEnv(
                    config_path=config_path,
                    worker_id=worker_id,
                    # shared_state intentionally omitted — see note above
                )
                return Monitor(env)
            return _init

        # Build the list of factory callables — one per worker
        env_fns = [_make_env(i) for i in range(self._n_workers)]

        # SubprocVecEnv: "start_method='spawn'" is required on Windows.
        # On Linux/macOS 'fork' is the default and is faster, but 'spawn'
        # is safe on all platforms.
        raw_vec = SubprocVecEnv(env_fns, start_method="spawn")

        # Reward normalisation — same as trainer.py
        use_vecnorm = self._rl_cfg.get("use_vec_normalize", True)
        if use_vecnorm:
            self._env = VecNormalize(
                raw_vec,
                norm_obs=False,
                norm_reward=True,
                clip_reward=10.0,
                gamma=self._rl_cfg.get("gamma", 0.99),
            )
            logger.info("VecNormalize enabled across %d workers", self._n_workers)
        else:
            self._env = raw_vec

        algo = self._rl_cfg.get("algorithm", "RecurrentPPO")
        self._model = self._make_model(algo)

        total_params = sum(p.numel() for p in self._model.policy.parameters())
        logger.info(
            "Model: %s  params=%s  workers=%d  effective_batch=%d",
            algo, f"{total_params:,}", self._n_workers,
            self._n_workers * self._rl_cfg.get("n_steps", 512),
        )

        # Initialise SharedState worker slots to exact size.
        # This prevents "list assignment index out of range" when callbacks
        # fire during/after VecNormalize or at training end.
        self._shared.update(
            worker_snapshots=[{}] * self._n_workers,
            worker_rewards=[0.0] * self._n_workers,
            worker_episodes=[0] * self._n_workers,
            worker_positions=[(25.0, 42.0)] * self._n_workers,
        )

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a sim checkpoint."""
        algo = self._rl_cfg.get("algorithm", "RecurrentPPO")
        cls  = self._get_model_cls(algo)
        self._model = cls.load(
            path,
            env=self._env,
            tensorboard_log=self._storage["tensorboard_dir"],
        )
        logger.info("Resumed from checkpoint: %s", path)

    def train(self) -> None:
        if self._model is None or self._env is None:
            raise RuntimeError("Call build() before train().")

        self._shared.update(is_training=True)
        callbacks = self._build_callbacks()

        start_time = time.monotonic()
        logger.info("Simulation training started — %d workers.", self._n_workers)
        self._shared.log_objective_event(
            f"▶ Training started: {self._n_workers} workers, "
            f"{self._train_cfg['total_timesteps']:,} total steps"
        )

        try:
            self._model.learn(
                total_timesteps=self._train_cfg["total_timesteps"],
                callback=callbacks,
                reset_num_timesteps=False,
                progress_bar=True,
            )
        finally:
            elapsed = time.monotonic() - start_time
            self._shared.update(is_training=False)

            prefix = "sim_re_agent"
            final_path = Path(self._storage["checkpoint_dir"]) / f"{prefix}_final.zip"
            self._model.save(str(final_path))

            if isinstance(self._env, VecNormalize):
                norm_path = str(final_path).replace(".zip", "_vecnorm.pkl")
                self._env.save(norm_path)
                logger.info("VecNormalize stats → %s", norm_path)

            total_steps = self._shared.total_steps
            sps = total_steps / max(elapsed, 1.0)
            summary = (
                f"✓ Training complete in {elapsed/60:.1f} min | "
                f"{total_steps:,} steps | {sps:,.0f} steps/sec | "
                f"saved: {final_path.name}"
            )
            logger.info(summary)
            self._shared.log_objective_event(summary)

    def stop(self) -> None:
        self._shared.update(stop_requested=True)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _make_model(self, algo: str) -> object:
        """
        Build the RL model — identical to trainer.py._make_model.

        The policy architecture (Impala CNN + LSTM) is unchanged so checkpoints
        from sim training can be loaded directly into the real-game trainer.
        """
        from feature_extractor import RE4FeaturesExtractor

        cls    = self._get_model_cls(algo)
        policy = "MultiInputLstmPolicy" if algo == "RecurrentPPO" else "MultiInputPolicy"
        features_dim = self._rl_cfg.get("features_dim", 512)

        policy_kwargs: dict = {
            "features_extractor_class":  RE4FeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        }
        if algo == "RecurrentPPO":
            policy_kwargs["lstm_hidden_size"] = self._rl_cfg.get("lstm_hidden_size", 256)
            policy_kwargs["n_lstm_layers"]    = self._rl_cfg.get("n_lstm_layers", 1)

        model = cls(
            policy=policy,
            env=self._env,
            learning_rate=self._rl_cfg.get("learning_rate", 2.5e-4),
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
        if algo == "RecurrentPPO":
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO
        from stable_baselines3 import PPO
        return PPO

    def _build_callbacks(self) -> list:
        """
        Callback order:
          1. CheckpointCallback  — save model first
          2. SimDashboardCallback — update metrics + per-worker snapshots
          3. SimMemoryCallback   — log to DB (episode-level only)
          4. SimCurriculumCallback — advance curriculum stage
          5. LLMCallback         — optional; same as trainer.py
        """
        checkpoint_cb = CheckpointCallback(
            save_freq=max(
                self._train_cfg.get("checkpoint_freq", 50_000) // self._n_workers, 1
            ),
            save_path=self._storage["checkpoint_dir"],
            name_prefix="sim_re_agent",
            verbose=1,
        )
        dashboard_cb = SimDashboardCallback(self._shared, self._n_workers)
        memory_cb    = SimMemoryCallback(self._memory, self._shared)
        curric_cb    = SimCurriculumCallback(self._shared, self._cur_cfg)

        cbs = [checkpoint_cb, dashboard_cb, memory_cb, curric_cb]

        if self._consultant is not None:
            # Import LLMCallback from trainer.py (it works with simulation)
            from trainer import LLMCallback
            llm_cfg = self._cfg.get("llm_settings", {})
            cbs.append(LLMCallback(
                consultant=self._consultant,
                memory=self._memory,
                shared=self._shared,
                call_every=llm_cfg.get("llm_every_n_steps", 60),
                cooldown_seconds=llm_cfg.get("call_cooldown_seconds", 3.0),
            ))

        return cbs
