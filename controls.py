"""
GameControls — Virtual Xbox 360 gamepad via ViGEm / vgamepad.

SETUP (one-time, do this before launching the agent):
  1. Install ViGEm Bus Driver:
     https://github.com/nefarius/ViGEmBus/releases  → ViGEmBus_Setup_*.exe
  2. pip install vgamepad

WHY GAMEPAD INSTEAD OF KEYBOARD+MOUSE:
  pydirectinput.SendInput() sends keystrokes to the FOREGROUND window.  The
  agent had to keep RE4 focused at all times, blocking your entire PC.
  A virtual XInput controller is polled directly by RE4 regardless of which
  window is in front — you can freely use the browser dashboard, alt-tab,
  watch YouTube, etc. while the agent plays.

RE4 REMAKE DEFAULT XBOX BUTTON LAYOUT (keyboard equivalent in parens):
  Left Stick  = Move Leon        (WASD)
  Right Stick = Camera           (Mouse)
  LT          = Aim              (Right-click)
  RT          = Shoot            (Left-click)
  A           = Interact/Confirm (F)
  B           = Dodge/Run        (Space / Shift)
  View/Back   = Attaché Case     (Tab)
  Start/Menu  = Pause Menu       (Esc)
  D-pad Down  = Quick 180 turn
  D-pad L/R   = Weapon select
"""

import ctypes
import ctypes.wintypes
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── ViGEm / vgamepad import ───────────────────────────────────────────────────
try:
    import vgamepad as vg
    _VGAMEPAD_OK = True
except ImportError:
    _VGAMEPAD_OK = False

# ── Left-stick positions for each movement action index ──────────────────────
# x = strafe (left -1 / right +1), y = forward/back.
# IMPORTANT: in vgamepad / XInput, +Y = stick UP = FORWARD (verified in-game:
# y=-1 walked Leon BACKWARD).  So forward is +1.0, back is -1.0.
_D = 0.707   # 1/√2 — keeps diagonal moves on the unit circle
_MV_STICK = {
    0: ( 0.0,  0.0),  # stop
    1: ( 0.0,  1.0),  # fwd
    2: ( 0.0, -1.0),  # back
    3: (-1.0,  0.0),  # left
    4: ( 1.0,  0.0),  # right
    5: (-_D,   _D ),  # fwd_left
    6: ( _D,   _D ),  # fwd_right
    7: (-_D,  -_D ),  # back_left
    8: ( _D,  -_D ),  # back_right
}

# ── Right-stick deflection for each camera action index ──────────────────────
# 0.8 gives smooth panning without over-rotating; tune in RE4 sensitivity settings.
# Same axis convention as the left stick: +Y = up = "look up".
_CS = 0.8
_CAM_STICK = {
    0: ( 0.0,  0.0),   # no camera input
    1: (-_CS,  0.0),   # look left
    2: ( _CS,  0.0),   # look right
    3: ( 0.0,  _CS),   # look up
    4: ( 0.0, -_CS),   # look down
}

# ── Win32 helpers for window-focus queries ────────────────────────────────────
EnumWindowsProc = ctypes.WINFUNCTYPE(
    ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
)

def _foreground_title() -> str:
    hwnd   = ctypes.windll.user32.GetForegroundWindow()
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


class GameControls:
    """
    Virtual Xbox 360 gamepad controller for RE4 Remake.

    All training-step inputs are applied atomically via execute_step():
    every axis, trigger, and button is set simultaneously, held for
    action_hold_seconds, then reset to neutral — no stacking sleeps.

    Menu navigation (reset_game, escape_to_gameplay) uses gamepad buttons
    that work regardless of window focus, so the user never needs to switch
    to RE4 manually.
    """

    def __init__(
        self,
        window_title_substring: str = "RESIDENT EVIL 4",
        smoothing: Optional[dict] = None,
    ):
        self._title_sub = window_title_substring.upper()

        # ── Human-like input smoothing ────────────────────────────────────────
        # When enabled, the analog sticks/triggers are low-pass filtered toward
        # their target each step instead of snapping, so the on-screen motion
        # looks like a person playing rather than a robot twitching.  Discrete
        # buttons (shoot/interact/dodge) are unaffected and stay crisp.
        sm = smoothing or {}
        self._smooth_enabled  = bool(sm.get("enabled", False))
        self._move_alpha      = float(sm.get("move_alpha",      0.35))
        self._camera_alpha    = float(sm.get("camera_alpha",    0.25))
        self._camera_max_rate = float(sm.get("camera_max_rate", 0.20))
        self._aim_alpha       = float(sm.get("aim_alpha",       0.50))
        # Smoothed analog state — persists across steps; reset in release_all().
        self._cur_lx = self._cur_ly = 0.0   # left stick (movement)
        self._cur_rx = self._cur_ry = 0.0   # right stick (camera)
        self._cur_lt = self._cur_rt = 0.0   # triggers (aim / shoot)

        if not _VGAMEPAD_OK:
            raise RuntimeError(
                "vgamepad not installed.\n"
                "Run:  pip install vgamepad\n"
                "Then install ViGEm Bus Driver: "
                "https://github.com/nefarius/ViGEmBus/releases"
            )

        try:
            self._pad = vg.VX360Gamepad()
            self._pad.update()
            logger.info("Virtual Xbox 360 gamepad ready via ViGEm.")
            if self._smooth_enabled:
                logger.info(
                    "Input smoothing ON (move=%.2f cam=%.2f cap=%.2f aim=%.2f) "
                    "— sticks/camera/aim ease in for human-like motion.",
                    self._move_alpha, self._camera_alpha,
                    self._camera_max_rate, self._aim_alpha,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create virtual gamepad ({exc}).\n"
                "Ensure ViGEm Bus Driver is installed and the service is running:\n"
                "https://github.com/nefarius/ViGEmBus/releases"
            ) from exc

    # ── Focus helpers (used by the focus monitor — not by training steps) ─────

    def is_game_focused(self) -> bool:
        return self._title_sub in _foreground_title().upper()

    def focus_game_window(self) -> bool:
        """Bring the RE4 window to the foreground. Returns True on success."""
        found: list = []

        @EnumWindowsProc
        def _cb(hwnd, _):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                if self._title_sub in buf.value.upper():
                    found.append(hwnd)
                    return False
            return True

        ctypes.windll.user32.EnumWindows(_cb, 0)
        if not found:
            logger.error("focus_game_window: '%s' not found", self._title_sub)
            return False
        ctypes.windll.user32.SetForegroundWindow(found[0])
        time.sleep(0.3)
        return self.is_game_focused()

    def release_all(self) -> None:
        """Zero all axes and release all buttons — called when training pauses."""
        try:
            self._pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self._pad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self._pad.left_trigger_float(value_float=0.0)
            self._pad.right_trigger_float(value_float=0.0)
            # Reset smoothed analog state so motion restarts from neutral after a
            # pause/stop instead of easing out from a stale deflection.
            self._cur_lx = self._cur_ly = 0.0
            self._cur_rx = self._cur_ry = 0.0
            self._cur_lt = self._cur_rt = 0.0
            for btn in (
                vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
                vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
            ):
                self._pad.release_button(button=btn)
            self._pad.update()
        except Exception:
            pass

    # ── Core training-step input ──────────────────────────────────────────────

    def execute_step(
        self,
        mv:   int,
        cam:  int,
        inter: int,
        comb: int,
        ev:   int,
        inv:  int,
        hold: float = 0.08,
    ) -> None:
        """
        Apply one complete RL action frame atomically.

        All six action dimensions are applied in a single gamepad update:
          mv   — movement    (0–8): left-stick position
          cam  — camera      (0–4): right-stick position
          inter— interact    (0–1): A button tap
          comb — combat      (0–3): LT (aim) / RT (shoot) triggers
          ev   — evasion     (0–2): B button (1=dodge tap, 2=run hold)
          inv  — inventory   (0–1): View/Back button tap

        Stick / trigger PERSISTENCE (critical for movement to register):
          The left stick (movement), right stick (camera), and triggers are
          set to the commanded values and then LEFT in place — they are NOT
          zeroed at the end of the step.  They keep their deflection until the
          next execute_step overwrites them (or movement/camera index 0 sets
          them back to neutral).

          Why?  A real player holds the stick down continuously to walk.  The
          old code snapped the stick back to neutral after only `hold` (0.08s)
          every step, so with neutral gaps between steps Leon barely twitched —
          the dashboard showed movement commands but he didn't actually move.
          Persisting the stick makes a run of "forward" actions one smooth walk.

          Only the BUTTONS (A/B/Back) are pulsed: pressed, held for `hold`,
          then released within this call — they are discrete events, not held.
        """
        tgt_lx, tgt_ly = _MV_STICK.get(mv,  (0.0, 0.0))
        tgt_rx, tgt_ry = _CAM_STICK.get(cam, (0.0, 0.0))
        tgt_lt = 1.0 if comb in (1, 3) else 0.0   # aim
        tgt_rt = 1.0 if comb in (2, 3) else 0.0   # shoot

        if self._smooth_enabled:
            # Low-pass the movement stick toward its target — Leon accelerates
            # into a direction and decelerates out of it instead of snapping.
            ma = self._move_alpha
            self._cur_lx += ma * (tgt_lx - self._cur_lx)
            self._cur_ly += ma * (tgt_ly - self._cur_ly)

            # Camera: low-pass AND clamp the per-step change (turn-rate cap) so
            # even a full-swing target can't snap — it pans like a thumb on a stick.
            cap = self._camera_max_rate
            d_rx = max(-cap, min(cap, self._camera_alpha * (tgt_rx - self._cur_rx)))
            d_ry = max(-cap, min(cap, self._camera_alpha * (tgt_ry - self._cur_ry)))
            self._cur_rx += d_rx
            self._cur_ry += d_ry

            # Aim trigger ramps in/out smoothly (gun raises naturally); the SHOOT
            # trigger stays crisp (a human pulls the trigger immediately).
            self._cur_lt += self._aim_alpha * (tgt_lt - self._cur_lt)
            self._cur_rt = tgt_rt

            lx, ly, rx, ry, lt, rt = (
                self._cur_lx, self._cur_ly, self._cur_rx, self._cur_ry,
                self._cur_lt, self._cur_rt,
            )
        else:
            lx, ly, rx, ry, lt, rt = (
                tgt_lx, tgt_ly, tgt_rx, tgt_ry, tgt_lt, tgt_rt,
            )

        # Continuous controls — set and LEAVE (overwritten next step).
        self._pad.left_joystick_float(x_value_float=lx, y_value_float=ly)
        self._pad.right_joystick_float(x_value_float=rx, y_value_float=ry)
        self._pad.left_trigger_float(value_float=lt)
        self._pad.right_trigger_float(value_float=rt)

        # Discrete controls — press now, release after the hold.
        if inter == 1:
            self._pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        if ev in (1, 2):
            self._pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        if inv == 1:
            self._pad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)

        self._pad.update()
        time.sleep(hold)

        # Release ONLY the buttons — leave sticks and triggers deflected so
        # continuous movement/aim carries across steps.
        self._pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self._pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self._pad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
        self._pad.update()

    # ── Menu navigation helpers ───────────────────────────────────────────────

    def _tap(self, button, hold: float = 0.15) -> None:
        """Press and release one button, then wait 50 ms for RE4 to respond."""
        self._pad.press_button(button=button)
        self._pad.update()
        time.sleep(hold)
        self._pad.release_button(button=button)
        self._pad.update()
        time.sleep(0.05)

    def escape_to_gameplay(self, max_presses: int = 8) -> None:
        """
        Back out of any open menu to return to active gameplay.

        Alternates B (cancel/back) and Start (pause toggle) so that:
          - B    closes sub-menus (Challenges, Tutorials, Results, Options)
          - Start closes the root pause menu if B doesn't dismiss it
        The pattern is B B Start B B Start B B (8 presses with 0.3 s gaps).
        """
        # CRITICAL: zero the sticks/triggers first.  execute_step deliberately
        # leaves the sticks deflected between steps (continuous walking), but a
        # deflected left stick CONTINUOUSLY SCROLLS RE4's menus — the D-pad/A
        # taps below then land on random items ("Challenges", "Results", …).
        # This was the menu-wandering bug during resets.
        self.release_all()
        time.sleep(0.2)

        logger.info("escape_to_gameplay: pressing B/Start to clear menus…")
        for i in range(max_presses):
            btn = (
                vg.XUSB_BUTTON.XUSB_GAMEPAD_START
                if i % 3 == 2
                else vg.XUSB_BUTTON.XUSB_GAMEPAD_B
            )
            self._tap(btn, hold=0.12)
            time.sleep(0.3)

    def reset_game(self, cfg: dict = None) -> None:
        """
        Reload the most recent save via the in-game pause menu.

        RE4 Remake controller sequence:
          Start → pause menu
          D-pad Down × nav_down_count → highlight "Load Game"
          A → open save-list
          A → select top save slot (most recent)
          D-pad Left × confirm_yes_count → move from "No" to "Yes"
          A → confirm load

        Adjust nav_down_count in config.yaml if your pause menu has a
        different number of items above "Load Game" (e.g. DLC adds entries),
        and confirm_yes_dir / confirm_yes_count if the Yes/No dialog differs.
        """
        cfg              = cfg or {}
        menu_open_wait   = float(cfg.get("menu_open_wait",   1.5))
        nav_between_wait = float(cfg.get("nav_between_wait", 0.4))
        select_wait      = float(cfg.get("select_wait",      1.0))
        confirm_wait     = float(cfg.get("confirm_wait",     0.5))
        nav_down_count   = int(cfg.get("nav_down_count",     1))

        # The "Load this checkpoint? Yes / No" dialog defaults to NO (a safety
        # default).  Pressing A on No CANCELS and dumps us back into the menus to
        # wander — that was the reset bug.  We must first move the cursor onto
        # "Yes".  In RE4 Remake this dialog is horizontal with Yes on the LEFT,
        # so one D-pad LEFT lands on Yes.  Both the direction and the number of
        # presses are configurable in case your build differs.
        confirm_yes_dir   = str(cfg.get("confirm_yes_dir", "left")).lower()
        confirm_yes_count = int(cfg.get("confirm_yes_count", 1))
        _DPAD = {
            "left":  vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
            "right": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
            "up":    vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
            "down":  vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
        }
        yes_btn = _DPAD.get(confirm_yes_dir, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)

        logger.info("reset_game: opening pause menu and reloading save…")

        # Zero sticks/triggers BEFORE touching the menu — a deflected left
        # stick (persisted from the last training step) auto-scrolls the menu
        # cursor and desyncs every D-pad count below.
        self.release_all()
        time.sleep(0.2)

        # 1. Open pause menu
        self._tap(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        time.sleep(menu_open_wait)

        # 2. Navigate to "Load Game"
        for _ in range(nav_down_count):
            self._tap(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
            time.sleep(nav_between_wait)

        # 3. Open the save list
        self._tap(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        time.sleep(select_wait)

        # 4. Select the top (most recent) save slot
        self._tap(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        time.sleep(confirm_wait)

        # 5. Move from "No" to "Yes" on the confirmation dialog, then confirm.
        logger.info("reset_game: moving to 'Yes' (%s ×%d) then confirming load.",
                    confirm_yes_dir, confirm_yes_count)
        for _ in range(confirm_yes_count):
            self._tap(yes_btn)
            time.sleep(nav_between_wait)
        self._tap(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        # Caller sleeps through the loading screen (loading_wait in config)

    # ── Thin wrappers (keep environment.py working without changes) ───────────

    def move(self, direction: str, duration: float = 0.1):
        mv_map = {
            'fwd': 1, 'back': 2, 'left': 3, 'right': 4,
            'fwd_left': 5, 'fwd_right': 6, 'back_left': 7, 'back_right': 8,
        }
        self.execute_step(mv_map.get(direction, 0), 0, 0, 0, 0, 0, hold=duration)

    def look(self, dx: int, dy: int):
        cam = 2 if dx > 0 else (1 if dx < 0 else (4 if dy > 0 else (3 if dy < 0 else 0)))
        self.execute_step(0, cam, 0, 0, 0, 0, hold=0.08)

    def interact(self):
        self.execute_step(0, 0, 1, 0, 0, 0, hold=0.1)

    def combat(self, action: str):
        comb = {'aim': 1, 'shoot': 2, 'aim+shoot': 3, 'stop_aim': 0}.get(action, 0)
        self.execute_step(0, 0, 0, comb, 0, 0, hold=0.08)

    def evade(self, sprint: bool = False):
        self.execute_step(0, 0, 0, 0, 2 if sprint else 1, 0, hold=0.1)

    def inventory(self):
        self.execute_step(0, 0, 0, 0, 0, 1, hold=0.1)


if __name__ == "__main__":
    print("Creating virtual gamepad… (ViGEm Bus Driver must be installed)")
    ctrl = GameControls()
    print("Connected. Moving forward for 1 second in 3 s — switch to RE4 if open.")
    time.sleep(3)
    ctrl.execute_step(1, 0, 0, 0, 0, 0, hold=1.0)
    print("Done.")
