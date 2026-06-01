"""
Resident Evil AI Agent — Simulation Mode Entry Point

Trains entirely in an abstract Python simulation — no game window needed.
All 8 workers run in separate OS processes; each simulates its own instance
of the RE4 village.

Usage:
    python sim_main.py                            # 8 workers, 1M steps
    python sim_main.py --workers 4                # fewer workers (less RAM)
    python sim_main.py --timesteps 5_000_000      # longer run
    python sim_main.py --resume models/checkpoints/sim_re_agent_final.zip
    python sim_main.py --no-llm                   # disable LLM advisor
    python sim_main.py --dashboard-only           # just the dashboard, no training

Outputs (in models/checkpoints/):
    sim_re_agent_final.zip          — final policy checkpoint
    sim_re_agent_NNNNNN_steps.zip   — intermediate checkpoints
    sim_re_agent_final_vecnorm.pkl  — VecNormalize running stats

TRANSFER TO REAL GAME
──────────────────────
After simulation training, load the checkpoint into the real-game trainer:
    python main.py --resume models/checkpoints/sim_re_agent_final.zip

The observation and action spaces are identical, so the policy weights
transfer directly.  Expect the real-game agent to start much more competent
than a randomly-initialised policy.

Dashboard opens at http://127.0.0.1:7861  (7861 to avoid conflict with main.py)
"""

import argparse
import logging
import os
import threading
import time

from dotenv import load_dotenv

from logger_config import setup_logging
from shared_state import SharedState
from memory import MemorySystem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RE4 Agent — Simulation Mode")
    p.add_argument("--config",         default="config.yaml",  help="Path to config.yaml")
    p.add_argument("--workers",        type=int, default=None,
                   help="Number of parallel simulation workers (overrides config)")
    p.add_argument("--timesteps",      type=int, default=None,
                   help="Total training timesteps (overrides config)")
    p.add_argument("--resume",         default=None,
                   help="Checkpoint .zip to resume from")
    p.add_argument("--no-llm",         action="store_true",
                   help="Disable LLM integration")
    p.add_argument("--dashboard-only", action="store_true",
                   help="Launch dashboard without starting training")
    p.add_argument("--port",           type=int, default=7861,
                   help="Dashboard port (default 7861)")
    p.add_argument("--log-level",      default="INFO",
                   help="Logging level (DEBUG/INFO/WARNING)")
    return p.parse_args()


def _build_llm(config_path: str, shared: SharedState, memory: MemorySystem):
    """
    Initialise the Claude LLM advisor (same as main.py).
    Only ANTHROPIC_API_KEY is required.  Gracefully disabled if missing.
    """
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ant_key:
        logging.warning(
            "ANTHROPIC_API_KEY not set — LLM advisor disabled. "
            "Set it in .env to enable Claude strategic guidance."
        )
        return None, None

    try:
        from llm_agent import ClaudeAdvisor, LLMConsultant
        advisor    = ClaudeAdvisor(config_path)
        consultant = LLMConsultant(advisor, shared, memory=memory)
        logging.info("Claude LLM advisor ready (simulation mode).")
        return advisor, consultant
    except Exception as exc:
        logging.error("LLM init failed: %s — continuing without LLM.", exc)
        return None, None


def _training_thread(
    config_path: str,
    shared: SharedState,
    memory: MemorySystem,
    consultant,
    resume_path: str,
    worker_override: int,
    timestep_override: int,
) -> None:
    """
    Background thread: builds the simulation trainer and waits for the
    dashboard Start button before training begins.

    Worker-count and timestep overrides (from CLI flags --workers / --timesteps)
    are applied by temporarily patching the loaded config dict.
    """
    log = logging.getLogger("sim_trainer_thread")
    try:
        import yaml
        from sim_trainer import SimRETrainer

        # Patch config if CLI overrides were given
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        if worker_override is not None:
            cfg.setdefault("simulation", {})["n_workers"] = worker_override
            log.info("Worker count overridden → %d", worker_override)

        if timestep_override is not None:
            cfg["training"]["total_timesteps"] = timestep_override
            log.info("Total timesteps overridden → %d", timestep_override)

        # Write the patched config to a temp file so SimRETrainer can read it
        import tempfile, json
        tmp_cfg = config_path  # Default: use original file if no overrides
        if worker_override is not None or timestep_override is not None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, dir="."
            ) as tf:
                yaml.dump(cfg, tf)
                tmp_cfg = tf.name
            log.debug("Patched config written to %s", tmp_cfg)

        trainer = SimRETrainer(tmp_cfg, shared, memory, consultant)
        trainer.build()

        if resume_path:
            trainer.load_checkpoint(resume_path)

        n_workers = cfg.get("simulation", {}).get("n_workers", 8)
        log.info(
            "Simulation trainer ready (%d workers) — press ▶ Start in dashboard.",
            n_workers,
        )

        # Wait for dashboard Start button (sets is_training=True)
        while not shared.is_training and not shared.stop_requested:
            time.sleep(0.5)

        if not shared.stop_requested:
            log.info("Simulation training started.")
            trainer.train()

        # Clean up temp config if created
        if tmp_cfg != config_path:
            try:
                os.unlink(tmp_cfg)
            except OSError:
                pass

    except Exception as exc:
        log.exception("Simulation training thread crashed: %s", exc)
        shared.update(is_training=False)


def main() -> None:
    args = parse_args()
    load_dotenv()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=log_level)
    logger = logging.getLogger("sim_main")

    # ── Banner ─────────────────────────────────────────────────────────────────
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_workers  = args.workers   or cfg.get("simulation", {}).get("n_workers", 8)
    n_steps    = args.timesteps or cfg["training"]["total_timesteps"]
    real_sps   = 12.0
    sim_est    = n_workers * 5_000   # conservative estimate
    speedup    = sim_est / real_sps
    real_hours = n_steps / real_sps / 3600
    sim_mins   = n_steps / sim_est / 60

    logger.info("=" * 60)
    logger.info("RE4 Agent — SIMULATION MODE")
    logger.info("  Workers:        %d", n_workers)
    logger.info("  Target steps:   %s", f"{n_steps:,}")
    logger.info("  Est. speedup:   ~%sx vs real game", f"{speedup:,.0f}")
    logger.info("  Real-game time: ~%.1f hours | Sim time: ~%.1f min", real_hours, sim_mins)
    logger.info("=" * 60)

    # ── Shared state and memory DB ─────────────────────────────────────────────
    shared = SharedState()
    memory = MemorySystem(cfg["storage"]["db_path"])
    memory.start()

    # ── LLM advisor ───────────────────────────────────────────────────────────
    llm, consultant = (None, None) if args.no_llm else _build_llm(
        args.config, shared, memory
    )

    # ── Training thread ────────────────────────────────────────────────────────
    if not args.dashboard_only:
        threading.Thread(
            target=_training_thread,
            args=(
                args.config, shared, memory, consultant,
                args.resume, args.workers, args.timesteps,
            ),
            daemon=True,
            name="SimTrainerThread",
        ).start()
        logger.info("Simulation trainer thread launched.")

    # ── Dashboard (blocking — runs in main thread) ─────────────────────────────
    from sim_dashboard import SimAgentDashboard
    dash = SimAgentDashboard(shared, memory, n_workers=n_workers)
    dash.build()
    logger.info("Launching simulation dashboard at http://127.0.0.1:%d", args.port)

    try:
        dash.launch(host="127.0.0.1", port=args.port)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    finally:
        shared.update(stop_requested=True)
        memory.stop()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    # CRITICAL on Windows: SubprocVecEnv spawns real OS processes.
    # Without the freeze_support() + if __name__ == "__main__" guard,
    # each spawned subprocess would re-run the training startup code,
    # creating an infinite fork-bomb of processes.
    #
    # Analogy: if a recipe says "make 8 copies of yourself to cook faster",
    # you only want the original chef to read that line — not each copy.
    from multiprocessing import freeze_support
    freeze_support()
    main()
