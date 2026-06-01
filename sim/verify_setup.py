"""
verify_setup.py  —  End-to-end sanity check for the sim-to-real pipeline
========================================================================

Run this BEFORE training to catch problems early.  It checks:

    [1]  Imports — all required packages installed
    [2]  Compat  — sim and real envs have matching interfaces
    [3]  Sim     — base env constructs and steps cleanly
    [4]  DR      — domain randomization wrapper produces valid obs
    [5]  Render  — pygame renderer produces a valid frame
    [6]  Eval    — tracker writes CSV + PNG correctly
    [7]  Speed   — measured throughput (steps/second)

Usage:
    python -m sim.verify_setup

Exit code 0 = all checks passed.  Non-zero = something is broken.

Why a single verify script?  Because the pipeline has many moving parts and
when training fails at hour 3, you want to know "was it always broken or
did something regress?" — this gives you a baseline.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


GREEN = "\033[92m"
RED   = "\033[91m"
YEL   = "\033[93m"
END   = "\033[0m"


def check(label: str, fn) -> bool:
    print(f"  • {label:<48s}", end=" ", flush=True)
    try:
        result = fn()
        if result is False:
            print(f"{RED}FAIL{END}")
            return False
        if isinstance(result, str):
            print(f"{GREEN}OK{END}  ({result})")
        else:
            print(f"{GREEN}OK{END}")
        return True
    except Exception as e:
        print(f"{RED}FAIL{END}")
        traceback.print_exc()
        return False


def main() -> int:
    print("=" * 60)
    print(" RE4 Sim Setup Verifier")
    print("=" * 60)
    failures = 0

    # ── [1] Imports ────────────────────────────────────────────────────────
    print("\n[1] Imports")

    def _imp():
        import numpy, gymnasium, stable_baselines3, sb3_contrib, yaml, cv2, pygame
        return f"sb3 {stable_baselines3.__version__}, gym {gymnasium.__version__}"
    if not check("Required packages", _imp):
        failures += 1

    def _imp_local():
        import sim_environment, sim_game_state, environment, feature_extractor
        from sim import (
            DomainRandomizationWrapper, RandomizationConfig,
            validate_sim_real_compatibility, GrowthTracker,
        )
        return None
    if not check("Local sim modules", _imp_local):
        failures += 1

    # ── [2] Compatibility ─────────────────────────────────────────────────
    print("\n[2] Sim/Real compatibility")

    def _compat():
        from sim.sim2real_bridge import validate_sim_real_compatibility
        rep = validate_sim_real_compatibility(
            "config.yaml", raise_on_fail=False
        )
        return "all checks passed" if rep.ok else False
    if not check("Static config check", _compat):
        failures += 1

    # ── [3] Sim env ────────────────────────────────────────────────────────
    print("\n[3] Base simulation env")

    def _sim_construct():
        from sim_environment import SimResidentEvilEnv
        env = SimResidentEvilEnv(config_path="config.yaml", worker_id=0)
        obs, info = env.reset()
        for _ in range(10):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: env.reset()
        return f"obs frame {obs['frame'].shape}, hud {obs['hud'].shape}"
    if not check("Construct + 10 steps", _sim_construct):
        failures += 1

    # ── [4] Domain randomization ──────────────────────────────────────────
    print("\n[4] Domain randomization")

    def _dr():
        from sim_environment import SimResidentEvilEnv
        from sim.domain_randomization import (
            DomainRandomizationWrapper, RandomizationConfig
        )
        base = SimResidentEvilEnv(config_path="config.yaml", worker_id=0)
        wrapped = DomainRandomizationWrapper(base, RandomizationConfig.training_preset())
        obs, _ = wrapped.reset()
        # Same shape/dtype as base?
        assert obs["frame"].dtype == base.observation_space["frame"].dtype, "dtype drift"
        assert obs["frame"].shape == base.observation_space["frame"].shape, "shape drift"
        # Run a few steps
        for _ in range(20):
            a = wrapped.action_space.sample()
            wrapped.step(a)
        return f"randomized obs shape {obs['frame'].shape}"
    if not check("DR wrapper preserves obs space", _dr):
        failures += 1

    # ── [5] Pygame renderer ───────────────────────────────────────────────
    print("\n[5] Pygame renderer")

    def _renderer():
        import math
        from sim.pygame_renderer import PygameRenderer
        from sim_game_state import EnemyState, ItemType
        r = PygameRenderer(50, 50, (84, 84))
        img = r.render((25.0, 42.0, -math.pi/2), True,
                       [(20.0, 12.0, EnemyState.CHASE, 0.7)],
                       [(15.0, 30.0, ItemType.AMMO)])
        assert img.shape == (84, 84, 3), f"bad shape: {img.shape}"
        assert img.dtype.name == "uint8", f"bad dtype: {img.dtype}"
        return f"rendered {img.shape} ok"
    if not check("PygameRenderer.render()", _renderer):
        failures += 1

    # ── [6] Growth tracker ────────────────────────────────────────────────
    print("\n[6] Growth tracker")

    def _gt():
        import shutil
        from sim.training_metrics import GrowthTracker, EvalSummary
        d = "_verify_metrics_tmp"
        try:
            shutil.rmtree(d)
        except FileNotFoundError:
            pass
        gt = GrowthTracker(d, "verify")
        for step in range(0, 100_000, 20_000):
            gt.record(EvalSummary(timestep=step, n_episodes=3, mean_reward=step/10000))
        ok = (os.path.exists(os.path.join(d, "eval_log.csv")) and
              os.path.exists(os.path.join(d, "learning_curve.png")))
        shutil.rmtree(d, ignore_errors=True)
        return "csv + png written" if ok else False
    if not check("Tracker writes CSV + PNG", _gt):
        failures += 1

    # ── [7] Throughput benchmark ──────────────────────────────────────────
    print("\n[7] Throughput")

    def _speed():
        from sim_environment import SimResidentEvilEnv
        env = SimResidentEvilEnv(config_path="config.yaml", worker_id=0)
        env.reset()
        N = 2000
        t = time.perf_counter()
        for _ in range(N):
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc: env.reset()
        elapsed = time.perf_counter() - t
        sps = N / elapsed
        if sps < 500:
            print(f"  {YEL}WARNING:{END} {sps:,.0f} steps/sec is low — sim may be CPU-bound")
        return f"{sps:,.0f} steps/sec (single worker)"
    if not check("Sim base throughput", _speed):
        failures += 1

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures == 0:
        print(f" {GREEN}ALL CHECKS PASSED{END} — ready to run train_sim_complete.py")
        return 0
    else:
        print(f" {RED}{failures} CHECK(S) FAILED{END} — fix above before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
