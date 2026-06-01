import ctypes
import ctypes.wintypes
import logging
import pydirectinput
import time

logger = logging.getLogger(__name__)

# Moving mouse to any corner kills the script immediately
pydirectinput.FAILSAFE = True

_HELD_KEYS = ['w', 's', 'a', 'd', 'shift', 'space']

EnumWindowsProc = ctypes.WINFUNCTYPE(
    ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
)


def _get_foreground_window_title() -> str:
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


class GameControls:
    """
    DirectX-compatible input control using pydirectinput.
    RE games typically ignore standard Win32 PostMessage calls (pyautogui).

    All input methods are no-ops when the game window does not have focus,
    preventing the agent from sending keystrokes/mouse events to other apps.
    """

    def __init__(self, window_title_substring: str = "RESIDENT EVIL 4"):
        pydirectinput.PAUSE = 0.02
        self._title_sub = window_title_substring.upper()

    # ── Focus helpers ─────────────────────────────────────────────────────────

    def is_game_focused(self) -> bool:
        """Return True only when the RE4 window currently has focus."""
        return self._title_sub in _get_foreground_window_title().upper()

    def release_all(self) -> None:
        """
        Release every input that could be left held — call this when focus
        is lost so the agent does not leave right-mouse (aim) or WASD stuck
        in whatever window the user switched to.
        """
        for button in ('right', 'left'):
            try:
                pydirectinput.mouseUp(button=button)
            except Exception:
                pass
        for key in _HELD_KEYS:
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass

    def focus_game_window(self) -> bool:
        """
        Find the game window by title and bring it to the foreground.
        Returns True if the window is focused after the call.
        """
        found: list = []

        @EnumWindowsProc
        def _cb(hwnd, _):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                if self._title_sub in buf.value.upper():
                    found.append(hwnd)
                    return False  # stop enumeration
            return True

        ctypes.windll.user32.EnumWindows(_cb, 0)
        if not found:
            logger.error("focus_game_window: could not find window '%s'", self._title_sub)
            return False

        ctypes.windll.user32.SetForegroundWindow(found[0])
        time.sleep(0.3)
        return self.is_game_focused()

    # ── Input methods — each is a no-op when the game is not focused ──────────

    def move(self, direction: str, duration: float = 0.1):
        if not self.is_game_focused():
            return
        keys = []
        if 'fwd'   in direction: keys.append('w')
        if 'back'  in direction: keys.append('s')
        if 'left'  in direction: keys.append('a')
        if 'right' in direction: keys.append('d')
        for key in keys: pydirectinput.keyDown(key)
        time.sleep(duration)
        for key in keys: pydirectinput.keyUp(key)

    def look(self, dx: int, dy: int):
        if not self.is_game_focused():
            return
        pydirectinput.moveRel(dx, dy, relative=True)

    def interact(self):
        if not self.is_game_focused():
            return
        pydirectinput.press('f')

    def combat(self, action: str):
        if not self.is_game_focused():
            return
        if action == 'aim':
            pydirectinput.mouseDown(button='right')
        elif action == 'shoot':
            pydirectinput.click(button='left')
        elif action == 'reload':
            pydirectinput.press('r')
        elif action == 'stop_aim':
            pydirectinput.mouseUp(button='right')

    def evade(self, sprint: bool = False):
        if not self.is_game_focused():
            return
        if sprint:
            pydirectinput.keyDown('shift')
            time.sleep(0.1)
            pydirectinput.keyUp('shift')
        else:
            pydirectinput.press('space')

    def inventory(self):
        if not self.is_game_focused():
            return
        pydirectinput.press('tab')

    def reset_game(self, cfg: dict = None) -> None:
        """
        Reloads the most recent save through the in-game pause menu.

        RE4 Remake saves at typewriters only — there is no quick-load shortcut.
        The only way to reload is:
          ESC  → pause menu opens
          DOWN → highlight "Load Game"  (nav_down_count presses)
          F    → open the save-list screen
          F    → select the most recent save (top slot, pre-highlighted)
          F    → confirm the load dialog

        Adjust nav_down_count in config.yaml if your menu layout differs
        (e.g. if a DLC adds items above "Load Game").
        """
        cfg = cfg or {}
        menu_open_wait   = cfg.get("menu_open_wait",   1.5)
        nav_between_wait = cfg.get("nav_between_wait", 0.4)
        select_wait      = cfg.get("select_wait",      1.0)
        confirm_wait     = cfg.get("confirm_wait",     0.5)
        nav_down_count   = int(cfg.get("nav_down_count", 1))
        confirm_key      = cfg.get("confirm_key",      "f")

        # Ensure the game window has focus before sending menu inputs
        if not self.is_game_focused():
            logger.info("reset_game: game not focused — attempting to refocus…")
            if not self.focus_game_window():
                logger.error("reset_game: cannot reach game window; aborting reset")
                return

        # 1. Open the pause menu
        pydirectinput.press('escape')
        time.sleep(menu_open_wait)

        # 2. Navigate to "Load Game"
        for _ in range(nav_down_count):
            pydirectinput.press('down')
            time.sleep(nav_between_wait)

        # 3. Select "Load Game"
        pydirectinput.press(confirm_key)
        time.sleep(select_wait)

        # 4. Select the most recent save (top slot, should be pre-highlighted)
        pydirectinput.press(confirm_key)
        time.sleep(confirm_wait)

        # 5. Confirm the load dialog (if one appears)
        pydirectinput.press(confirm_key)
        # Caller is responsible for sleeping through the loading screen


if __name__ == "__main__":
    # Test sequence - ensure game is windowed and focused!
    print("Testing controls in 3 seconds... SWITCH TO GAME WINDOW")
    time.sleep(3)
    ctrl = GameControls()
    ctrl.move('fwd', duration=0.5)
    ctrl.look(100, 0)
    print("Test complete.")
