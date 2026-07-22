"""
train_sim_complete.py  —  All-in-one sim trainer with sim-to-real best practices
================================================================================

ONE COMMAND, EVERY FEATURE
--------------------------
This script wraps your existing SimRETrainer and adds the full sim-to-real
pipeline:

    [1]  Pre-flight  : validates that sim & real env interfaces match
    [2]  Training    : SubprocVecEnv with N workers + DomainRandomization
    [3]  Evaluation  : every K steps, runs N clean (un-randomised) eval episodes
    [4]  Growth      : metrics CSV + auto-plotted PNG learning curves
    [5]  Best-model  : saves the best-eval policy separately from latest
    [6]  Transfer    : converts the best policy into a real-game-ready file

USAGE
-----
    # Default 2M-step run, 8 workers, eval every 50k steps
    python train_sim_complete.py

    # Quick 200k-step shake-down to verify the pipeline
    python train_sim_complete.py --timesteps 200_000 --eval-every 25_000

    # Resume an existing checkpoint
    python train_sim_complete.py --resume models/checkpoints/sim_re_agent_500000_steps.zip

    # Disable randomization (debugging — should NOT be used for transfer)
    python train_sim_complete.py --no-randomization

OUTPUTS
-------
    models/checkpoints/sim_re_agent_*_steps.zip   — periodic checkpoints
    models/checkpoints/sim_re_agent_final.zip     — final policy (last step)
    models/checkpoints/sim_re_agent_best.zip      — best policy by eval reward
    models/checkpoints/sim_re_agent_for_real.zip  — transfer-ready (= best)
    runs/<timestamp>/eval_log.csv                 — per-eval metrics
    runs/<timestamp>/learning_curve.png           — auto-rendered plot
    runs/<timestamp>/best_checkpoint.txt          — best step + reward

MENTAL MODEL
------------
                            ┌─────────────────────────────┐
                            │  N x SubprocVecEnv worker   │
   training loop  ──────►   │  ┌───────────────────────┐  │
                            │  │ DomainRandomization   │  │
                            │  │  ↳ SimResidentEvilEnv │  │
                            │  └───────────────────────┘  │
                            └─────────────────────────────┘
                                        │
                                        ▼
                       every  --eval-every  steps:
                            ┌─────────────────────────────┐
                            │   Single CLEAN eval env     │
                            │   (eval_preset, no DR)      │
   evaluation     ──────►   │     run_evaluation()        │
                            │            │                │
                            │   GrowthTracker.record()    │
                            │     ↳ CSV + PNG + best.zip  │
                            └─────────────────────────────┘
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from dotenv import load_dotenv

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger_config import setup_logging
from shared_state  import SharedState
from memory        import MemorySystem

# Sim package — our new code
from sim.domain_randomization import (
    DomainRandomizationWrapper, RandomizationConfig,
)
from sim.sim2real_bridge import (
    validate_sim_real_compatibility,
    convert_sim_checkpoint_for_real,
)
from sim.training_metrics import GrowthTracker, EvalSummary, run_evaluation


# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RE4 sim trainer with full sim-to-real pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default="config.yaml",
                   help="Path to config.yaml.")
    p.add_argument("--workers",      type=int, default=None,
                   help="Override simulation.n_workers from config.")
    p.add_argument("--timesteps",    type=int, default=None,
                   help="Override training.total_timesteps from config.")
    p.add_argument("--resume",       default=None,
                   help="Path to a checkpoint .zip to resume from.")
    p.add_argument("--eval-every",   type=int, default=50_000,
                   help="Eval pass interval in training steps.")
    p.add_argument("--eval-episodes",type=int, default=5,
                   help="Number of clean eval episodes per pass.")
    p.add_argument("--no-randomization", action="store_true",
                   help="Disable domain randomization (debugging only — kills sim-to-real).")
    p.add_argument("--randomization-preset",
                   choices=["training", "gentle", "eval"],
                   default="training",
                   help="Which DR preset to use during training.")
    p.add_argument("--run-name",     default=None,
                   help="Subfolder name under runs/.  Defaults to timestamp.")
    p.add_argument("--no-validation", action="store_true",
                   help="Skip the static sim/real compatibility check.")
    p.add_argument("--log-level",    default="INFO")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Env factory (used by SubprocVecEnv)
# ──────────────────────────────────────────────────────────────────────────────

def _atomic_save(model, path: Path) -> None:
    """Save a checkpoint via temp-file + rename so an interrupted run can never
    leave a truncated (corrupt) .zip — the failure mode that produced the
    unloadable sim_re_agent_final.zip on June 28."""
    tmp = path.with_name(path.stem + ".tmp.zip")
    model.save(str(tmp))
    os.replace(tmp, path)


def make_env_factory(config_path: str, worker_id: int,
                     preset: str, enabled: bool):
    """
    Returns a callable that constructs ONE training env per worker process.

    SubprocVecEnv pickles this factory and ships it across process boundaries,
    so it must capture only picklable values (strings, ints, simple dicts).
    Importantly we do NOT capture SharedState or any thread-locked object.

    Analogy
    -------
    Sending a recipe (the factory) to remote kitchens (worker processes).
    The recipe must be self-contained — all measurements in plain numbers,
    no "use the same flour as last time" because the remote chef doesn't
    have that flour.
    """
    def _init():
        # Imports inside function so subprocesses can re-import cleanly
        from stable_baselines3.common.monitor import Monitor
        from sim_environment import SimResidentEvilEnv
        from sim.domain_randomization import (
            DomainRandomizationWrapper, RandomizationConfig,
        )

        env = SimResidentEvilEnv(config_path=config_path, worker_id=worker_id)

        if enabled:
            cfg = {
                "training": RandomizationConfig.training_preset(),
                "gentle":   RandomizationConfig.gentle_preset(),
                "eval":     RandomizationConfig.eval_preset(),
            }[preset]
            # Use a different seed per worker so DR samples differ
            cfg.seed = worker_id * 7919 + 1
            env = DomainRandomizationWrapper(env, cfg)

        return Monitor(env)
    return _init


# ──────────────────────────────────────────────────────────────────────────────
# Build evaluator
# ──────────────────────────────────────────────────────────────────────────────

def build_eval_env(config_path: str):
    """
    Build a single, non-vectorised eval env with NO randomization.

    Why a separate env?  We want eval to measure the policy's clean ability,
    not its randomized-training reward.  Sharing the train envs would mean
    eval sees the same random distractors / colour shifts the policy was
    trained against — making the metric an in-distribution-only estimate.

    Analogy
    -------
    Training is gym practice with weights, distractions, awkward angles.
    Eval is the actual performance on a clean stage.  Different equipment.
    """
    from sim_environment import SimResidentEvilEnv
    env = SimResidentEvilEnv(config_path=config_path, worker_id=999)
    return DomainRandomizationWrapper(env, RandomizationConfig.eval_preset())


# ──────────────────────────────────────────────────────────────────────────────
# Build model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(env, cfg: Dict, tensorboard_dir: str):
    """
    Constructs the SB3 model — same architecture as sim_trainer.py.SimRETrainer.

    The features-extractor (feature_extractor.RE4FeaturesExtractor) is shared
    between sim and real.  As long as the obs space matches, the same model
    class can be loaded into either env.
    """
    rl_cfg = cfg["rl_hyperparameters"]
    algo   = rl_cfg.get("algorithm", "RecurrentPPO")

    from feature_extractor import RE4FeaturesExtractor
    if algo == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO as ModelCls
        policy_name = "MultiInputLstmPolicy"
    else:
        from stable_baselines3 import PPO as ModelCls
        policy_name = "MultiInputPolicy"

    policy_kwargs: Dict[str, Any] = {
        "features_extractor_class":  RE4FeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": rl_cfg.get("features_dim", 512)},
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    }
    if algo == "RecurrentPPO":
        policy_kwargs["lstm_hidden_size"] = rl_cfg.get("lstm_hidden_size", 256)
        policy_kwargs["n_lstm_layers"]    = rl_cfg.get("n_lstm_layers", 1)

    # Linear LR decay + target_kl guard — both are stability measures against
    # the catastrophic policy collapse observed in the June 28 "capped_aim"
    # run (policy stopped shooting entirely and died every episode).
    lr_value = float(rl_cfg.get("learning_rate", 2.5e-4))
    if str(rl_cfg.get("lr_schedule", "constant")).lower() == "linear":
        learning_rate = lambda progress_remaining: lr_value * progress_remaining
    else:
        learning_rate = lr_value

    model = ModelCls(
        policy=policy_name,
        env=env,
        learning_rate=learning_rate,
        target_kl    = rl_cfg.get("target_kl", None),
        n_steps      = rl_cfg.get("n_steps", 512),
        batch_size   = rl_cfg.get("batch_size", 128),
        n_epochs     = rl_cfg.get("n_epochs", 4),
        gamma        = rl_cfg.get("gamma", 0.99),
        gae_lambda   = rl_cfg.get("gae_lambda", 0.95),
        clip_range   = rl_cfg.get("clip_range", 0.2),
        ent_coef     = rl_cfg.get("ent_coef", 0.01),
        max_grad_norm= rl_cfg.get("max_grad_norm", 0.5),
        policy_kwargs= policy_kwargs,
        tensorboard_log= tensorboard_dir,
        verbose=1,
    )
    return model, ModelCls


# ──────────────────────────────────────────────────────────────────────────────
# Training callbacks
# ──────────────────────────────────────────────────────────────────────────────

def _build_callbacks(shared, memory, n_workers: int, cfg: Dict, ckpt_dir: str):
    """
    Build the callback stack — reuses the existing simfunctional callbacks
    from sim_trainer.py so the dashboard wiring stays intact.
    """
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sim_trainer import (
        SimDashboardCallback,
        SimMemoryCallback,
        SimCurriculumCallback,
    )

    # Periodic save — pre-empts catastrophic crashes
    checkpoint_cb = CheckpointCallback(
        save_freq=max(cfg["training"].get("checkpoint_freq", 50_000) // n_workers, 1),
        save_path=ckpt_dir,
        name_prefix="sim_re_agent",
        verbose=1,
    )

    dashboard_cb = SimDashboardCallback(shared, n_workers)
    memory_cb    = SimMemoryCallback(memory, shared)  # memory param added for consistency
    curric_cb    = SimCurriculumCallback(shared, cfg["curriculum"])
    return [checkpoint_cb, dashboard_cb, memory_cb, curric_cb]


# ──────────────────────────────────────────────────────────────────────────────
# The training loop with periodic eval
# ──────────────────────────────────────────────────────────────────────────────

def train_with_eval(
    args, cfg: Dict, shared: SharedState, run_dir: Path, memory: MemorySystem,
) -> Optional[str]:
    """
    Returns: path to best-eval checkpoint, or None if training was interrupted.

    Strategy
    --------
    SB3's `model.learn(total_timesteps=N)` runs N steps and then returns.
    To interleave evaluation, we call `learn()` multiple times in a loop,
    each time advancing by `eval_every` steps, then run an eval pass.

    `reset_num_timesteps=False` is essential — without it, each call would
    reset the internal counter and tensorboard would graph N overlapping
    plots instead of one continuous curve.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    n_workers   = args.workers   or cfg.get("simulation", {}).get("n_workers", 8)
    total_steps = args.timesteps or cfg["training"]["total_timesteps"]
    eval_every  = args.eval_every
    ckpt_dir    = cfg["storage"]["checkpoint_dir"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── Build training env (SubprocVecEnv + VecNormalize) ────────────────────
    enabled = not args.no_randomization
    preset  = args.randomization_preset

    logger = logging.getLogger("train_sim_complete")
    logger.info("Building %d-worker SubprocVecEnv (DR=%s preset=%s)",
                n_workers, enabled, preset)

    env_fns = [
        make_env_factory(args.config, i, preset, enabled)
        for i in range(n_workers)
    ]
    raw_vec = SubprocVecEnv(env_fns, start_method="spawn")

    # When resuming, restore the sibling VecNormalize stats pkl if it exists.
    # Resuming a policy while RESETTING the reward-normalisation stats rescales
    # every reward the value function sees — that silent mismatch is what
    # collapsed the June 28 sim run.
    resume_norm_pkl: Optional[Path] = None
    if args.resume:
        cand = Path(args.resume).with_name(Path(args.resume).stem + "_vecnorm.pkl")
        if cand.exists():
            resume_norm_pkl = cand

    if resume_norm_pkl is not None:
        train_env = VecNormalize.load(str(resume_norm_pkl), raw_vec)
        train_env.training    = True
        train_env.norm_reward = True
        logger.info("VecNormalize stats restored from %s", resume_norm_pkl)
    else:
        if args.resume:
            logger.warning(
                "No _vecnorm.pkl found next to %s — reward normalisation "
                "starts fresh (expect a few noisy updates).", args.resume,
            )
        train_env = VecNormalize(
            raw_vec,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=cfg["rl_hyperparameters"].get("gamma", 0.99),
        )

    # ── Build model ───────────────────────────────────────────────────────────
    tb_dir = cfg["storage"]["tensorboard_dir"]
    model, ModelCls = build_model(train_env, cfg, tb_dir)
    if args.resume:
        try:
            model = ModelCls.load(args.resume, env=train_env, tensorboard_log=tb_dir)
            logger.info("Resumed from %s", args.resume)
        except Exception as exc:
            logger.error(
                "Could not load %s (%s) — file may be corrupt. "
                "Continuing with a FRESH model.", args.resume, exc,
            )

    # ── Build eval env (single, no DR) ────────────────────────────────────────
    logger.info("Building clean eval env (single-process, eval_preset, no DR)")
    eval_env = build_eval_env(args.config)

    # ── Tracker ──────────────────────────────────────────────────────────────
    tracker = GrowthTracker(str(run_dir), run_name=run_dir.name)

    # ── Initial eval (step 0) so the first PNG point is meaningful ───────────
    logger.info("Initial evaluation (untrained policy as baseline)…")
    summary = run_evaluation(eval_env, model,
                             n_episodes=args.eval_episodes, timestep=0)
    tracker.record(summary)
    logger.info("Init: %s", summary.pretty())

    # ── Main loop: alternate learn() and evaluate() ─────────────────────────
    callbacks = _build_callbacks(shared, memory, n_workers, cfg, ckpt_dir)
    shared.update(is_training=True)

    elapsed_steps = model.num_timesteps   # respects --resume's starting count
    best_ckpt_path: Optional[str] = None

    try:
        while elapsed_steps < total_steps:
            chunk = min(eval_every, total_steps - elapsed_steps)
            logger.info("Training chunk: %d → %d steps",
                        elapsed_steps, elapsed_steps + chunk)
            model.learn(
                total_timesteps=chunk,
                callback=callbacks,
                reset_num_timesteps=False,
                progress_bar=True,
            )
            elapsed_steps = model.num_timesteps

            # ── Eval pass ─────────────────────────────────────────────────────
            logger.info("Evaluating at step %d…", elapsed_steps)
            summary = run_evaluation(
                eval_env, model,
                n_episodes=args.eval_episodes,
                timestep=elapsed_steps,
            )
            is_best = tracker.record(summary)
            logger.info("Eval: %s", summary.pretty())

            # ── Save best policy separately ──────────────────────────────────
            if is_best:
                best_path = Path(ckpt_dir) / "sim_re_agent_best.zip"
                _atomic_save(model, best_path)
                if isinstance(train_env, VecNormalize):
                    train_env.save(str(best_path).replace(".zip", "_vecnorm.pkl"))
                best_ckpt_path = str(best_path)
                logger.info("★ Saved BEST policy → %s", best_path)

            if shared.stop_requested:
                logger.info("Stop requested via SharedState — exiting cleanly")
                break

    finally:
        # Always save final policy & VecNormalize stats
        final_path = Path(ckpt_dir) / "sim_re_agent_final.zip"
        _atomic_save(model, final_path)
        if isinstance(train_env, VecNormalize):
            train_env.save(str(final_path).replace(".zip", "_vecnorm.pkl"))
        logger.info("Saved FINAL policy → %s", final_path)

        try:
            train_env.close()
            eval_env.close()
        except Exception:
            pass

        shared.update(is_training=False)

    return best_ckpt_path


# ──────────────────────────────────────────────────────────────────────────────
# main()
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    load_dotenv()

    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("train_sim_complete")

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Pre-flight: sim/real compatibility ──────────────────────────────────
    if not args.no_validation:
        report = validate_sim_real_compatibility(args.config, raise_on_fail=False)
        print(report.fmt())
        if not report.ok:
            logger.error("Compatibility check failed — aborting.")
            sys.exit(1)

    # ── Run dir ──────────────────────────────────────────────────────────────
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name    = args.run_name or stamp
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    # ── Banner ──────────────────────────────────────────────────────────────
    n_workers = args.workers or cfg.get("simulation", {}).get("n_workers", 8)
    n_steps   = args.timesteps or cfg["training"]["total_timesteps"]
    print("=" * 70)
    print(" RE4 Agent  —  COMPLETE Sim Trainer")
    print("=" * 70)
    print(f"  Workers          : {n_workers}")
    print(f"  Total timesteps  : {n_steps:,}")
    print(f"  Eval every       : {args.eval_every:,} steps")
    print(f"  Eval episodes    : {args.eval_episodes}")
    print(f"  Domain rand      : {'ON' if not args.no_randomization else 'OFF'}  ({args.randomization_preset})")
    print(f"  Run dir          : {run_dir}")
    print(f"  Resume from      : {args.resume or '(fresh start)'}")
    print("=" * 70)

    # ── Shared state + memory ───────────────────────────────────────────────
    shared = SharedState()
    memory = MemorySystem(cfg["storage"]["db_path"])
    memory.start()

    # ── Train ────────────────────────────────────────────────────────────────
    try:
        best_ckpt = train_with_eval(args, cfg, shared, run_dir, memory)
    finally:
        memory.stop()

    # ── Convert best policy to transfer-ready file ──────────────────────────
    if best_ckpt:
        try:
            transfer_path = convert_sim_checkpoint_for_real(
                best_ckpt, config_path=args.config
            )
            print()
            print("─" * 70)
            print(f"  ✓ Best sim policy: {best_ckpt}")
            print(f"  ✓ Transfer-ready : {transfer_path}")
            print()
            print("To fine-tune on the real game:")
            print(f"  python main.py --resume {transfer_path}")
            print("─" * 70)
        except Exception as e:
            logger.error("Transfer conversion failed: %s", e)


if __name__ == "__main__":
    # Windows requires this for SubprocVecEnv (see sim_main.py for details)
    from multiprocessing import freeze_support
    freeze_support()
    main()
