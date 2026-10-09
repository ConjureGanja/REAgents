"""
Central configuration loader.

Parses config.yaml ONCE and hands out a shared dict to every subsystem.
Previously each of environment.py / perception.py / capture.py / llm_agent.py /
trainer.py / main.py re-opened and re-parsed the file in its own __init__ —
wasteful and a drift risk (two modules could disagree if the file changed on
disk mid-run).

GAME PROFILES:
  config.yaml has a `game_profiles` section with `active_profile` and per-game
  blocks (window_title_substring, guide_module, health_style).  The active
  profile's keys are merged OVER `game_settings`, so adding an RE2/RE3 profile
  later is a config-only change — no code edits.

Usage:
    cfg = load_config()                    # parse config.yaml once
    cap = ScreenCapture(config=cfg)        # pass the dict down
"""

import logging
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load config.yaml and apply the active game-profile overlay.

    Returns the merged config dict.  The profile merge only touches
    `game_settings` keys that the profile explicitly defines; everything
    else (perception, rl_hyperparameters, …) passes through untouched.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    profiles = cfg.get("game_profiles") or {}
    active = profiles.get("active_profile")
    if active and active in profiles:
        overlay = profiles[active] or {}
        # Only whitelisted per-game keys are merged; guide_module/health_style
        # are stored for consumers that want them but don't override anything.
        gs = cfg.setdefault("game_settings", {})
        for key in ("window_title_substring",):
            if key in overlay:
                gs[key] = overlay[key]
        # Stash non-game_settings profile metadata where consumers can find it
        # without re-reading the file (e.g. llm_agent picking the guide module).
        cfg["_active_profile"] = {
            "name": active,
            "guide_module": overlay.get("guide_module", "guide_data"),
            "health_style": overlay.get("health_style", "radial_ring"),
        }
        logger.info("Game profile active: %s", active)
    elif active:
        logger.warning(
            "game_profiles.active_profile=%r not defined — using base game_settings.",
            active,
        )

    return cfg


def ensure_config(
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """
    Backward-compat helper for subsystem constructors.

    Lets callers pass either a pre-loaded config dict (preferred) or a path
    (legacy).  Keeps the old `SomeClass("config.yaml")` call sites working
    while new code passes the shared dict.
    """
    if config is not None:
        return config
    return load_config(config_path)
