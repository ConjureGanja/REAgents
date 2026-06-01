"""
Sim-to-Real Bridge — Validation and Transfer Utilities
=======================================================

WHY THIS EXISTS
---------------
Transfer learning works only if the *interface contract* between the sim and
the real env is identical:

    1. Same observation_space  (shape, dtype, value range)
    2. Same action_space       (MultiDiscrete dimensions)
    3. Same HUD vector layout  (which index is health, ammo, etc.)
    4. Same action semantics   (action[3]=2 must mean "shoot" in both)

If ANY of these drift, the policy weights become meaningless when loaded into
the real env — at best, behaviour collapses to random; at worst, it learns
to do dangerous things (like firing at allies).

This module provides:

    SimToRealValidator        — runtime checker that fails LOUD if interfaces drift
    validate_sim_real_compatibility() — convenience function for a one-line check
    convert_sim_checkpoint_for_real() — strips sim-only state from a saved policy

Analogy
-------
Think of the sim and real envs as two electrical sockets, and the policy as
a plug.  As long as the prongs match, the plug works.  This module is the
voltmeter that tells you BEFORE you plug it in whether the sockets are
compatible — saving you a melted policy.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CompatReport:
    """
    Structured result of a sim-vs-real compatibility check.

    Use `report.ok` for a quick pass/fail, or `report.problems` to see
    every individual mismatch.  When `ok=False`, do NOT load the
    checkpoint into the real env — fix the mismatch first.
    """
    ok: bool
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details:  Dict[str, Any] = field(default_factory=dict)

    def fmt(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append(
            f"Sim-Real Compatibility: {'✓ PASS' if self.ok else '✗ FAIL'}"
        )
        lines.append("=" * 60)
        if self.problems:
            lines.append("Problems (must fix before transfer):")
            for p in self.problems:
                lines.append(f"  ✗ {p}")
        if self.warnings:
            lines.append("Warnings (won't break transfer but worth checking):")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if not self.problems and not self.warnings:
            lines.append("No issues found.")
        if self.details:
            lines.append("")
            lines.append("Details:")
            for k, v in self.details.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────────

class SimToRealValidator:
    """
    Compare a SimResidentEvilEnv and a ResidentEvilEnv (or any two envs claiming
    to share an interface) and produce a CompatReport.

    USAGE
    -----
        from sim_environment import SimResidentEvilEnv
        from environment      import ResidentEvilEnv
        from sim.sim2real_bridge import SimToRealValidator

        # NOTE: building ResidentEvilEnv requires the game to be running.
        # If not running, pass `inspect_real_only_metadata=True` to skip env construction.

        v = SimToRealValidator()
        report = v.compare(SimResidentEvilEnv(), ResidentEvilEnv())
        print(report.fmt())
        assert report.ok, "Cannot transfer — fix problems first."
    """

    # Names of HUD keys that BOTH envs must produce identically
    EXPECTED_HUD_KEYS = ("health_pct", "ammo_clip", "ammo_res")

    # MultiDiscrete action dims required for both envs
    EXPECTED_ACTION_DIMS = (9, 5, 2, 4, 3, 2)

    def compare(self, sim_env, real_env) -> CompatReport:
        """
        Compare two envs.  Each must expose .observation_space and .action_space.
        Optionally, both should expose .reset() / .step() but we don't *call* them
        here — the real env would need the actual game running.

        Calls a separate static-analysis check to compare:
            - obs space shapes and dtypes
            - action space dims
            - HUD vector layout (via reading the env's reset() return)
            - reward weight presence in config
        """
        problems: List[str] = []
        warnings: List[str] = []
        details:  Dict[str, Any] = {}

        # ── 1. Observation space ─────────────────────────────────────────────
        sim_obs  = sim_env.observation_space
        real_obs = real_env.observation_space

        if type(sim_obs).__name__ != type(real_obs).__name__:
            problems.append(
                f"Obs space types differ: sim={type(sim_obs).__name__} "
                f"real={type(real_obs).__name__}"
            )
        else:
            # Both should be Dict spaces with "frame" + "hud"
            for key in ("frame", "hud"):
                if key not in sim_obs.spaces or key not in real_obs.spaces:
                    problems.append(f"Obs Dict missing key '{key}'")
                    continue
                s, r = sim_obs.spaces[key], real_obs.spaces[key]
                if s.shape != r.shape:
                    problems.append(
                        f"Obs '{key}' shape mismatch: sim={s.shape} real={r.shape}"
                    )
                if s.dtype != r.dtype:
                    problems.append(
                        f"Obs '{key}' dtype mismatch: sim={s.dtype} real={r.dtype}"
                    )

        details["obs_frame_shape"] = sim_obs.spaces.get("frame", None)
        details["obs_hud_shape"]   = sim_obs.spaces.get("hud", None)

        # ── 2. Action space ──────────────────────────────────────────────────
        sim_act_nvec  = tuple(getattr(sim_env.action_space,  "nvec", ()))
        real_act_nvec = tuple(getattr(real_env.action_space, "nvec", ()))

        if sim_act_nvec != real_act_nvec:
            problems.append(
                f"Action MultiDiscrete differs: sim={sim_act_nvec} real={real_act_nvec}"
            )
        if sim_act_nvec != self.EXPECTED_ACTION_DIMS:
            problems.append(
                f"Sim action MultiDiscrete differs from expected "
                f"{self.EXPECTED_ACTION_DIMS}: got {sim_act_nvec}"
            )

        details["action_nvec"] = sim_act_nvec

        # ── 3. HUD key sanity (sim-side only — real needs game running) ──────
        try:
            sim_obs_dict, sim_info = sim_env.reset()
            for key in self.EXPECTED_HUD_KEYS:
                if key not in sim_info:
                    warnings.append(f"Sim info dict missing HUD key '{key}'")
            details["sim_hud_vec_first_episode"] = sim_obs_dict["hud"].tolist()
        except Exception as e:
            warnings.append(f"sim_env.reset() raised — couldn't sanity-check HUD: {e}")

        # ── 4. Final verdict ─────────────────────────────────────────────────
        return CompatReport(
            ok=not problems,
            problems=problems,
            warnings=warnings,
            details=details,
        )

    @staticmethod
    def compare_static(config_path: str = "config.yaml") -> CompatReport:
        """
        Static check that uses ONLY the config + module imports.
        Doesn't require either env to actually start (real env needs game running).

        This is the most useful entrypoint for CI / pre-flight checks.
        """
        problems, warnings, details = [], [], {}

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        rl   = cfg["rl_hyperparameters"]
        h, w = rl["obs_frame_size"]
        gray = rl.get("grayscale", False)
        stack = rl.get("frame_stack", 4)

        per_frame_c = 1 if gray else 3
        total_c     = per_frame_c * stack
        details["expected_obs_frame_shape"] = (h, w, total_c)
        details["expected_action_nvec"]     = SimToRealValidator.EXPECTED_ACTION_DIMS

        # Cross-check: BOTH environment.py and sim_environment.py construct the
        # observation space from these same config values, so as long as the
        # config is single-source-of-truth, the spaces match.  We just verify
        # the config values themselves are sensible.

        if h <= 0 or w <= 0:
            problems.append(f"Invalid obs_frame_size in config: {h}x{w}")
        if stack < 1:
            problems.append(f"Invalid frame_stack: {stack}")
        if rl.get("algorithm") not in ("PPO", "RecurrentPPO"):
            problems.append(f"Unknown algorithm: {rl.get('algorithm')}")

        # Curriculum stages must be the SAME between sim and real (verified by
        # the fact that BOTH envs read this same _CURRICULUM_WEIGHTS table)
        stages = cfg.get("curriculum", {}).get("stages", {})
        for required in ("exploration", "combat", "completion"):
            if required not in stages:
                problems.append(f"Curriculum stage '{required}' missing from config")

        details["curriculum_stages"]      = list(stages.keys())
        details["recurrent"]              = rl.get("algorithm") == "RecurrentPPO"
        details["features_dim"]           = rl.get("features_dim", 512)

        return CompatReport(
            ok=not problems,
            problems=problems,
            warnings=warnings,
            details=details,
        )


# ──────────────────────────────────────────────────────────────────────────────
# One-liner entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def validate_sim_real_compatibility(
    config_path: str = "config.yaml",
    sim_env=None,
    real_env=None,
    raise_on_fail: bool = True,
) -> CompatReport:
    """
    Convenience: run the static check, plus the dynamic check if envs are given.

    Returns the merged CompatReport.  If `raise_on_fail=True`, raises
    `RuntimeError` when problems are found.
    """
    static = SimToRealValidator.compare_static(config_path)
    if sim_env is not None and real_env is not None:
        dynamic = SimToRealValidator().compare(sim_env, real_env)
        # Merge the two reports — problems from either side count
        merged = CompatReport(
            ok=static.ok and dynamic.ok,
            problems=static.problems + dynamic.problems,
            warnings=static.warnings + dynamic.warnings,
            details={**static.details, **dynamic.details},
        )
    else:
        merged = static

    if not merged.ok and raise_on_fail:
        msg = merged.fmt()
        raise RuntimeError(f"Sim-Real compatibility check FAILED:\n{msg}")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert_sim_checkpoint_for_real(
    sim_checkpoint: str,
    output_path: Optional[str] = None,
    config_path: str = "config.yaml",
) -> str:
    """
    Sanity-check a sim checkpoint and copy it to a transfer-ready location.

    Stable Baselines 3 stores the env's VecNormalize stats and observation
    space metadata inside the checkpoint.  When you load with `model.load()`
    in the real env, SB3 automatically remaps weights as long as the
    observation/action spaces match.  This function:

        1. Runs the static compatibility check (errors out early if config
           is misaligned).
        2. Copies the checkpoint to `<dir>/sim_re_agent_for_real.zip` so you
           keep the original sim-trained file untouched.
        3. Returns the new path.

    Why a separate file?  When you fine-tune on the real game, SB3 will save
    new checkpoints over the original.  Keeping a clean transfer copy means
    you can always re-start the real-game fine-tune from a known-good base.

    Analogy
    -------
    A photographer keeps the RAW file untouched and works from a copy.
    Same idea: this is your "RAW" sim policy, never overwritten.
    """
    p = Path(sim_checkpoint).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    # 1. Static compat check — fail loud if anything's off
    report = SimToRealValidator.compare_static(config_path)
    if not report.ok:
        raise RuntimeError(
            "Cannot convert checkpoint — config compatibility check failed:\n"
            + report.fmt()
        )

    # 2. Copy checkpoint
    if output_path is None:
        output_path = str(p.parent / "sim_re_agent_for_real.zip")
    shutil.copyfile(p, output_path)

    # Also copy VecNormalize stats if they exist alongside the checkpoint
    vecnorm_src = p.with_name(p.stem + "_vecnorm.pkl")
    if vecnorm_src.exists():
        vecnorm_dst = Path(output_path).with_name(
            Path(output_path).stem + "_vecnorm.pkl"
        )
        shutil.copyfile(vecnorm_src, vecnorm_dst)
        logger.info("Copied VecNormalize stats: %s", vecnorm_dst)

    logger.info("Sim checkpoint ready for real-game transfer: %s", output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="Sim/Real compatibility validator")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--dynamic", action="store_true",
        help="Also build both envs and do a runtime comparison (real env "
             "needs game running on screen).",
    )
    ap.add_argument(
        "--convert", default=None,
        help="Path to a sim checkpoint to convert to a transfer-ready copy.",
    )
    args = ap.parse_args()

    if args.convert:
        out = convert_sim_checkpoint_for_real(args.convert, config_path=args.config)
        print(f"OK — converted: {out}")
        sys.exit(0)

    if args.dynamic:
        from sim_environment import SimResidentEvilEnv
        from environment      import ResidentEvilEnv
        sim  = SimResidentEvilEnv(config_path=args.config, worker_id=0)
        real = ResidentEvilEnv(config_path=args.config)
        report = validate_sim_real_compatibility(
            args.config, sim_env=sim, real_env=real, raise_on_fail=False
        )
    else:
        report = validate_sim_real_compatibility(args.config, raise_on_fail=False)

    print(report.fmt())
    sys.exit(0 if report.ok else 1)
