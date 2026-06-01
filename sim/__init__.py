"""
sim/  —  Sim-to-Real Toolkit for the RE4 RL Agent
=================================================

This package extends the existing `sim_environment.py` / `sim_game_state.py`
foundation with everything needed for **policy transfer to the real game**:

    domain_randomization.py  →  visual + dynamics randomisation wrapper
    pygame_renderer.py       →  richer Pygame-based rendering (more game-like)
    sim2real_bridge.py       →  validation + checkpoint transfer utilities
    training_metrics.py      →  growth tracking with PNG learning curves
    play_manual.py           →  WASD keyboard play for sanity-checking the sim

Why a separate package?
-----------------------
Your existing `sim_*.py` modules are stable and tested.  Putting the new code
in `sim/` keeps the original files untouched while letting you opt-in to the
new capabilities one at a time.

Analogy
-------
Your current sim is like a flight simulator with the cockpit, physics, and
controls all working.  This package adds the things you need before flying
a real plane: instrument-error injection (domain randomisation), realistic
weather rendering (Pygame renderer), and a flight-log analyser (metrics).

Quick-start
-----------
The single command that pulls everything together::

    python train_sim_complete.py --timesteps 2_000_000

That script (in the project root) imports from this package, builds a richly-
randomised env, trains, evaluates every N steps, plots learning curves, and
writes a final transfer-ready checkpoint.

Then::

    python main.py --resume models/checkpoints/sim_re_agent_final.zip
"""

__version__ = "1.0.0"

# Re-exports use LAZY imports (PEP 562 __getattr__) so that submodules with
# heavy or optional dependencies (gymnasium, pygame, matplotlib) only get
# imported when they're actually used.  This keeps `import sim` cheap and
# lets standalone tools (like the static compat check) run on minimal envs.
#
# Usage stays identical to direct imports:
#     from sim import DomainRandomizationWrapper       # lazy
#     from sim.domain_randomization import ...         # also fine

__all__ = [
    "DomainRandomizationWrapper",
    "RandomizationConfig",
    "validate_sim_real_compatibility",
    "SimToRealValidator",
    "convert_sim_checkpoint_for_real",
    "GrowthTracker",
    "EvalSummary",
    "run_evaluation",
]


def __getattr__(name):
    """Lazy import — defers heavy module loads until first access."""
    if name in ("DomainRandomizationWrapper", "RandomizationConfig"):
        from sim.domain_randomization import (
            DomainRandomizationWrapper, RandomizationConfig,
        )
        return {"DomainRandomizationWrapper": DomainRandomizationWrapper,
                "RandomizationConfig":        RandomizationConfig}[name]

    if name in ("validate_sim_real_compatibility", "SimToRealValidator",
                "convert_sim_checkpoint_for_real"):
        from sim.sim2real_bridge import (
            validate_sim_real_compatibility,
            SimToRealValidator,
            convert_sim_checkpoint_for_real,
        )
        return {
            "validate_sim_real_compatibility": validate_sim_real_compatibility,
            "SimToRealValidator":              SimToRealValidator,
            "convert_sim_checkpoint_for_real": convert_sim_checkpoint_for_real,
        }[name]

    if name in ("GrowthTracker", "EvalSummary", "run_evaluation"):
        from sim.training_metrics import GrowthTracker, EvalSummary, run_evaluation
        return {"GrowthTracker":  GrowthTracker,
                "EvalSummary":    EvalSummary,
                "run_evaluation": run_evaluation}[name]

    raise AttributeError(f"module 'sim' has no attribute {name!r}")
