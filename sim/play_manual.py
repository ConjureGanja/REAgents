"""
Manual Play Mode  —  Human-in-the-loop sim debugger
====================================================

WHY THIS EXISTS
---------------
Before you spend hours training a policy, you want to verify the sim itself
behaves the way you expect:

    - Movement feels responsive
    - Enemies actually pursue and attack
    - Shooting connects when aimed
    - Items can be collected
    - The map layout makes sense

Trying to debug those by reading reward curves is like diagnosing a broken
car by listening to the engine through a wall.  This script lets you DRIVE
the sim with your keyboard so you can FEEL whether the dynamics are right.

KEYBINDINGS
-----------
    W / S / A / D            — move forward / back / left / right (relative to facing)
    ← / →   (or  Q / E)      — rotate camera
    Space                    — dodge
    Shift  (held)            — sprint
    R-mouse  (or  Right ctrl) — aim
    L-mouse  (or  Left ctrl)  — shoot
    F                        — interact (pick up items)
    Tab                      — toggle inventory
    Esc / window close       — quit

ON-SCREEN HUD shows:
    - Player health, ammo (clip / reserve)
    - Episode step / max step
    - Total kills, items collected
    - Current curriculum stage
    - Reward earned this step (small text)

USAGE
-----
    python -m sim.play_manual
    python -m sim.play_manual --config config.yaml --scale 8

`--scale` controls the on-screen pixel size (the sim renders at 84x84;
scale=8 → an 672x672 window).  Bigger = easier to play, smaller = closer
to what the agent actually sees.

DESIGN NOTE
-----------
This deliberately uses pygame for the window because (a) it's already a
dependency for any future renderer, and (b) it gives us pixel-perfect
control over how the agent's downsampled view is presented.  The sim's
own render_frame() is the source of truth — we just upscale it for display.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Tuple

import numpy as np
import yaml

# Add parent dir so we can import top-level modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Action mapping
# ──────────────────────────────────────────────────────────────────────────────

def keys_to_action(keys, mouse_buttons) -> Tuple[list, str]:
    """
    Translate the current keyboard / mouse state into a 6-element action.

    Returns (action, debug_label) where debug_label is a short string for the
    HUD text ("fwd+aim+shoot" etc.) so the player can see what the agent would
    have observed.

    Action indices match SimResidentEvilEnv's MultiDiscrete([9, 5, 2, 4, 3, 2]):
        [0] movement   0=stop 1=fwd 2=back 3=left 4=right 5–8=diagonals
        [1] camera     0=none 1=cam-left 2=cam-right 3=up 4=down
        [2] interact   0=none 1=interact
        [3] combat     0=none 1=aim 2=shoot 3=aim+shoot
        [4] evasion    0=none 1=dodge 2=sprint
        [5] inventory  0=none 1=toggle
    """
    import pygame
    K = pygame.key

    # ── Movement: 8 directions + stop ──────────────────────────────────────────
    fwd   = keys[K.K_w]
    back  = keys[K.K_s]
    left  = keys[K.K_a]
    right = keys[K.K_d]

    if   fwd   and left:  mv, mv_lbl = 5, "fwd-left"
    elif fwd   and right: mv, mv_lbl = 6, "fwd-right"
    elif back  and left:  mv, mv_lbl = 7, "back-left"
    elif back  and right: mv, mv_lbl = 8, "back-right"
    elif fwd:             mv, mv_lbl = 1, "fwd"
    elif back:            mv, mv_lbl = 2, "back"
    elif left:            mv, mv_lbl = 3, "left"
    elif right:           mv, mv_lbl = 4, "right"
    else:                 mv, mv_lbl = 0, "stop"

    # ── Camera rotation ────────────────────────────────────────────────────────
    if   keys[K.K_LEFT]  or keys[K.K_q]: cam = 1   # rotate CCW
    elif keys[K.K_RIGHT] or keys[K.K_e]: cam = 2   # rotate CW
    elif keys[K.K_UP]:                   cam = 3
    elif keys[K.K_DOWN]:                 cam = 4
    else:                                cam = 0

    # ── Interact ───────────────────────────────────────────────────────────────
    inter = 1 if keys[K.K_f] else 0

    # ── Combat ────────────────────────────────────────────────────────────────
    aim   = mouse_buttons[2] == 1 or keys[K.K_RCTRL]
    shoot = mouse_buttons[0] == 1 or keys[K.K_LCTRL]
    if   aim and shoot: comb, comb_lbl = 3, "aim+shoot"
    elif shoot:         comb, comb_lbl = 2, "shoot"
    elif aim:           comb, comb_lbl = 1, "aim"
    else:               comb, comb_lbl = 0, ""

    # ── Evasion ───────────────────────────────────────────────────────────────
    if   keys[K.K_SPACE]: ev, ev_lbl = 1, "dodge"
    elif keys[K.K_LSHIFT] or keys[K.K_RSHIFT]: ev, ev_lbl = 2, "sprint"
    else:                 ev, ev_lbl = 0, ""

    # ── Inventory ─────────────────────────────────────────────────────────────
    inv = 1 if keys[K.K_TAB] else 0

    label = "  ".join(p for p in [mv_lbl, comb_lbl, ev_lbl] if p)
    return [mv, cam, inter, comb, ev, inv], label


# ──────────────────────────────────────────────────────────────────────────────
# HUD overlay
# ──────────────────────────────────────────────────────────────────────────────

def draw_hud(screen, font, env, action_label: str, last_reward: float,
             total_reward: float, scale: int) -> None:
    """
    Render a sidebar HUD with player state, action label, and reward.
    Operates on the pygame screen directly.
    """
    import pygame
    sim = env._sim

    lines = [
        f"FPS step  : {env._episode_step}/{sim.max_steps}",
        f"Health    : {sim.player_health:>5.1f} / 100",
        f"Ammo clip : {sim.ammo_clip:>3}  reserve {sim.ammo_reserve:>3}",
        f"Aiming    : {'YES' if sim.player_aiming else 'no'}",
        f"Kills     : {sim.total_kills}",
        f"Items     : {sim.items_collected}  shotgun={sim.shotgun_collected}",
        f"Stage     : {env._curriculum_stage}",
        f"Action    : {action_label or '(idle)'}",
        f"Reward    : {last_reward:+.2f}  total {total_reward:+.1f}",
    ]
    bar_x = 84 * scale + 8
    bar_y = 8

    # Translucent background
    pygame.draw.rect(screen, (20, 20, 25), (bar_x - 4, 0, 360, 84 * scale + 8))
    for i, line in enumerate(lines):
        surf = font.render(line, True, (220, 220, 220))
        screen.blit(surf, (bar_x, bar_y + i * 22))

    # Health bar (graphical)
    hp_x, hp_y, hp_w, hp_h = bar_x, bar_y + len(lines) * 22 + 12, 320, 12
    pygame.draw.rect(screen, (60, 60, 60), (hp_x, hp_y, hp_w, hp_h), 1)
    fill = int(hp_w * sim.player_health / 100.0)
    color = (60, 200, 60) if sim.player_health > 50 else \
            (200, 160, 40) if sim.player_health > 20 else (220, 50, 50)
    pygame.draw.rect(screen, color, (hp_x + 1, hp_y + 1, fill - 2, hp_h - 2))


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Manual play mode for the RE4 sim")
    ap.add_argument("--config", default="config.yaml", help="Config path")
    ap.add_argument("--scale",  type=int, default=8,
                    help="Display upscaling factor (window = 84*scale wide)")
    ap.add_argument("--fps",    type=int, default=12,
                    help="Game tick rate; 12 matches the agent's action_hold_seconds")
    ap.add_argument("--randomize", action="store_true",
                    help="Apply training-preset domain randomization (see what the agent sees)")
    args = ap.parse_args()

    # Pygame must be imported AFTER we set the SDL video driver if needed
    import pygame
    pygame.init()
    pygame.display.set_caption("RE4 Sim — Manual Play (WASD + L/R-mouse, Esc to quit)")

    from sim_environment import SimResidentEvilEnv
    env = SimResidentEvilEnv(config_path=args.config, worker_id=0)

    if args.randomize:
        from sim.domain_randomization import (
            DomainRandomizationWrapper, RandomizationConfig
        )
        env = DomainRandomizationWrapper(env, RandomizationConfig.training_preset())
        print("Domain randomization ENABLED — this is the view the agent sees during training.")

    obs, _ = env.reset()
    h, w   = obs["frame"].shape[:2]
    scale  = max(1, args.scale)

    win_w  = w * scale + 360
    win_h  = max(h * scale, 320)
    screen = pygame.display.set_mode((win_w, win_h))
    font   = pygame.font.SysFont("consolas", 16)

    clock = pygame.time.Clock()
    last_reward  = 0.0
    total_reward = 0.0
    action_label = ""
    running = True

    while running:
        # ── Event pump ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # ── Build action from current input state ───────────────────────────
        keys           = pygame.key.get_pressed()
        mouse_buttons  = pygame.mouse.get_pressed(num_buttons=3)
        action, action_label = keys_to_action(keys, mouse_buttons)

        # ── Step the env ────────────────────────────────────────────────────
        obs, reward, term, trunc, info = env.step(action)
        last_reward   = float(reward)
        total_reward += last_reward

        if term or trunc:
            print(f"Episode end — terminated={term} truncated={trunc}  "
                  f"reward={total_reward:.1f}  resetting…")
            obs, _ = env.reset()
            total_reward = 0.0

        # ── Render frame to screen ──────────────────────────────────────────
        # The sim returns (H, W, C); we only use the most recent (or only) frame.
        frame = obs["frame"]
        if frame.shape[2] >= 3:
            # Take the most recent RGB triplet (last 3 channels of stack)
            rgb = frame[:, :, -3:]
        else:
            # Grayscale — replicate to RGB
            g = frame[:, :, -1:]
            rgb = np.repeat(g, 3, axis=2)

        # numpy (H, W, 3) → pygame surface and upscale
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (w * scale, h * scale))

        screen.fill((10, 10, 12))
        screen.blit(surf, (0, 0))

        # Use the wrapped-or-base env to access internals for HUD
        base_env = env.env if hasattr(env, "env") else env
        draw_hud(screen, font, base_env, action_label, last_reward, total_reward, scale)

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    print("Bye.")


if __name__ == "__main__":
    main()
