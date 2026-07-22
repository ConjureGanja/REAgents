"""
Gamepad diagnostic — does the virtual Xbox 360 pad actually move Leon?

This bypasses the RL policy entirely and drives the left stick directly, so we
can separate three failure modes:

  A) Pad works only when RE4 is FOCUSED  -> RE4 drops background controller input
  B) Pad works regardless of focus        -> controls fine; the policy was idle
  C) Pad never moves Leon                  -> game-side mapping / device issue

HOW TO RUN (from the re_agent env):
  C:/Users/wizar/miniconda3/envs/re_agent/python.exe test_gamepad_move.py

Follow the on-screen countdowns.  Make sure RE4 is in ACTUAL GAMEPLAY
(Leon standing in the world), not a menu or cutscene.
"""

import time
import vgamepad as vg


def countdown(msg: str, secs: int) -> None:
    for i in range(secs, 0, -1):
        print(f"  {msg} in {i}…", end="\r", flush=True)
        time.sleep(1)
    print(" " * 60, end="\r")


def hold_forward(pad: vg.VX360Gamepad, seconds: float) -> None:
    """Push left stick fully forward continuously for `seconds`."""
    pad.left_joystick_float(x_value_float=0.0, y_value_float=1.0)  # +1 y = up = forward
    pad.update()
    time.sleep(seconds)
    pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    pad.update()


def spin_camera(pad: vg.VX360Gamepad, seconds: float) -> None:
    pad.right_joystick_float(x_value_float=0.8, y_value_float=0.0)  # pan right
    pad.update()
    time.sleep(seconds)
    pad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
    pad.update()


def main() -> None:
    print("Creating virtual Xbox 360 pad (ViGEm)…")
    pad = vg.VX360Gamepad()
    pad.update()
    print("Pad created. RE4 should show 'Xbox 360 controller connected'.\n")

    # ── TEST 1: RE4 FOCUSED ──────────────────────────────────────────────
    print("TEST 1 — FOCUSED.  Click into the RE4 window NOW so it has focus.")
    countdown("Walking forward 4s", 6)
    print("  >> Leon should WALK FORWARD now (4s)…")
    hold_forward(pad, 4.0)
    print("  >> Panning camera RIGHT now (3s)…")
    spin_camera(pad, 3.0)
    print("  TEST 1 done.\n")

    # ── TEST 2: RE4 UNFOCUSED (background input) ─────────────────────────
    print("TEST 2 — UNFOCUSED.  Now click your OTHER monitor / browser so RE4")
    print("is NOT the active window (this is the 'use my PC freely' scenario).")
    countdown("Walking forward 4s (RE4 in background)", 8)
    print("  >> Leon should WALK FORWARD even though RE4 is not focused (4s)…")
    hold_forward(pad, 4.0)
    print("  TEST 2 done.\n")

    print("RESULTS:")
    print("  • Moved in BOTH tests      -> controls are perfect; the RL policy")
    print("    was just choosing to stand still. Nothing to fix in controls.")
    print("  • Moved ONLY in TEST 1     -> RE4 ignores controller input when its")
    print("    window is in the background. You must keep RE4 focused, OR we")
    print("    enable a 'keep RE4 foreground' helper.")
    print("  • Moved in NEITHER         -> game-side issue: check RE4 Options >")
    print("    Controls that a controller is enabled, and that you were in")
    print("    actual gameplay (not a menu/cutscene).")


if __name__ == "__main__":
    main()
