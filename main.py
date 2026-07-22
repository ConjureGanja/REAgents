"""
Resident Evil AI Agent — Entry Point

Usage:
    python main.py                            # start fresh
    python main.py --resume path/to/ckpt.zip  # resume training
    python main.py --no-llm                   # disable LLM (RL only)
    python main.py --record                    # ENABLE gameplay recording (off by default)
    python main.py --dashboard-only           # just show the dashboard, no training

Architecture:
    Main thread    → Gradio dashboard (blocking)
    TrainerThread  → RL training loop (RETrainer.train)
    LLMThread      → Claude advisor (fire-and-forget, via LLMConsultant)
    MemoryThread   → SQLite writer (MemorySystem internal)
    RecorderThread → Gameplay recorder (writes footage/*.mp4)
    FailsafeThread → Mouse-corner watchdog (sets stop_requested)

LLM REQUIREMENT:
    Only ANTHROPIC_API_KEY is required for the LLM advisor.
    Set it in your .env file.  To run without LLM at all, use --no-llm.
"""

import argparse
import ctypes
import logging
import os
import threading
import time

# Set per-monitor DPI awareness before any mss or Win32 screen calls.
# Without this, Windows DPI virtualisation causes mss to report scaled-up
# monitor coordinates on high-DPI monitors, producing an offset/black capture.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

from dotenv import load_dotenv

from logger_config import setup_logging
from shared_state import SharedState
from memory import MemorySystem
from recorder import GameplayRecorder
from dashboard import AgentDashboard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resident Evil AI Agent")
    p.add_argument("--config",         default="config.yaml",  help="Path to config.yaml")
    p.add_argument("--resume",         default=None,           help="Checkpoint .zip to resume from")
    p.add_argument("--no-llm",         action="store_true",    help="Disable LLM integration")
    p.add_argument("--record",         action="store_true",    help="Enable gameplay video recording (OFF by default — 1080p MP4 encoding costs real CPU/FPS)")
    p.add_argument("--dashboard-only", action="store_true",    help="Launch dashboard without training")
    p.add_argument("--log-level",      default="INFO",         help="Logging level (DEBUG/INFO/WARNING)")
    return p.parse_args()


def _failsafe_watchdog(shared: SharedState) -> None:
    """
    Polls mouse position at 20 Hz and stops the agent if the cursor reaches
    the top-left corner of the virtual desktop (covers all monitors).

    Why virtual desktop bounds instead of pyautogui.size()?
    pyautogui.size() returns only the PRIMARY monitor dimensions. On a
    multi-monitor setup the secondary monitor has x-coordinates beyond that
    width, so any mouse position on monitor 2 would falsely trigger the
    right-edge check (x >= sw - 3). We use SM_CXVIRTUALSCREEN / SM_XVIRTUALSCREEN
    to get the true bounds of the full desktop across all monitors, and only
    check the top-left corner so the right/bottom edges of monitor 1 (which are
    the middle of the desktop) never fire.
    """
    import ctypes
    import pyautogui
    log = logging.getLogger("failsafe")
    try:
        user32 = ctypes.windll.user32
        # Full virtual desktop origin and size (all monitors combined)
        vx  = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
        vy  = user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
        vsw = user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
        vsh = user32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN
        x_min, y_min = vx + 2,       vy + 2
        x_max, y_max = vx + vsw - 3, vy + vsh - 3
        log.info(
            "Failsafe watchdog active — virtual desktop %dx%d at (%d,%d). "
            "Move mouse to any corner to stop the agent.",
            vsw, vsh, vx, vy,
        )
        # DWELL requirement: the cursor must stay in a corner CONTINUOUSLY for
        # this many seconds before the failsafe fires.  Why?  With the virtual
        # gamepad the agent never controls the real mouse, so the old instant
        # "mouse-to-corner = kill" panic button is obsolete AND harmful: corners
        # are exactly where you click during normal PC use (Start menu, window
        # close buttons, the seam between monitors).  A dwell window lets you use
        # your PC freely while still preserving a deliberate emergency stop —
        # just park the cursor in any corner and hold it there.
        DWELL_SECONDS = 1.5
        corner_since: float | None = None
        while not shared.stop_requested:
            x, y = pyautogui.position()
            # Only trigger on CORNERS — not on edges.  Moving the mouse along
            # the bottom or right edge of the primary monitor is normal; only
            # landing in one of the four corner zones should stop the agent.
            in_left   = x <= x_min
            in_right  = x >= x_max
            in_top    = y <= y_min
            in_bottom = y >= y_max
            in_corner = (in_left or in_right) and (in_top or in_bottom)
            if in_corner:
                now = time.time()
                if corner_since is None:
                    corner_since = now
                elif now - corner_since >= DWELL_SECONDS:
                    shared.update(stop_requested=True)
                    log.warning(
                        "FAILSAFE triggered at (%d, %d) — cursor held in corner "
                        "%.1fs — agent stopped.", x, y, DWELL_SECONDS,
                    )
                    return
            else:
                corner_since = None   # left the corner — reset the dwell timer
            time.sleep(0.05)
    except Exception as exc:
        log.error("Failsafe watchdog crashed: %s", exc)


def _focus_monitor(shared: SharedState, game_title_sub: str) -> None:
    """
    Polls the foreground window title at 10 Hz.

    Behaviour when auto_pause_on_focus_loss is True (the default):
      • RE4 window loses focus  →  paused = True  (all inputs blocked instantly)
      • RE4 window regains focus →  paused = False (training resumes seamlessly)

    This means you can freely click the browser dashboard, alt-tab to check
    something, or adjust settings — the agent simply freezes in place while
    you do so, then picks up exactly where it left off when you come back.

    Behaviour when auto_pause_on_focus_loss is False:
      • Only updates game_focused flag (for the status bar); no auto-pause.
      • Use this if you prefer to manage pausing manually via the dashboard button.

    Why poll rather than hook into Windows WM_ACTIVATE?
    Win32 SetWindowsHookEx requires a message pump on the same thread, which
    conflicts with pydirectinput's SendInput timing.  Polling at 100 ms is
    imperceptible to the user and safe from any other thread.
    """
    import ctypes
    log = logging.getLogger("focus_monitor")
    title_upper = game_title_sub.upper()
    log.info("Focus monitor active — auto-pause on focus loss: %s",
             shared.auto_pause_on_focus_loss)

    def _foreground_title() -> str:
        hwnd   = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value.upper()

    while not shared.stop_requested:
        try:
            focused = title_upper in _foreground_title()
            shared.update(game_focused=focused)

            if shared.auto_pause_on_focus_loss and shared.is_training:
                if not focused and not shared.paused:
                    # Game lost focus — auto-pause to protect browser interaction
                    shared.update(paused=True)
                    log.debug("Auto-paused: RE4 lost foreground focus.")
                elif focused and shared.paused:
                    # Game regained focus — auto-resume
                    shared.update(paused=False)
                    log.debug("Auto-resumed: RE4 regained foreground focus.")
        except Exception as exc:
            log.error("Focus monitor error: %s", exc)

        time.sleep(0.1)   # 10 Hz poll — fast enough to feel instant, cheap on CPU


def _build_llm(config_path: str, shared: SharedState, memory: MemorySystem):
    """
    Initialise the Claude LLM advisor.

    Only ANTHROPIC_API_KEY is required.  If it's missing, the LLM is
    gracefully disabled and training continues in RL-only mode.

    Analogy: the LLM is like a GPS navigator — very helpful, but the car can
    still drive without it.  You just won't get spoken turn-by-turn directions.
    """
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ant_key:
        logging.warning(
            "ANTHROPIC_API_KEY not set — LLM advisor disabled. "
            "Set it in .env to enable Claude vision guidance."
        )
        return None, None

    try:
        from llm_agent import ClaudeAdvisor, LLMConsultant
        advisor    = ClaudeAdvisor(config_path)
        consultant = LLMConsultant(advisor, shared, memory=memory)
        logging.info("Claude LLM advisor ready.")
        return advisor, consultant
    except Exception as exc:
        logging.error("LLM init failed: %s — continuing without LLM.", exc)
        return None, None


def _training_thread(
    config_path: str,
    shared: SharedState,
    memory: MemorySystem,
    consultant,
    resume_path,
) -> None:
    """
    Runs in a background daemon thread.
    Blocks on RETrainer.build() then waits for the dashboard Start button.
    """
    log = logging.getLogger("trainer_thread")
    try:
        from trainer import RETrainer
        trainer = RETrainer(config_path, shared, memory, consultant)
        trainer.build()

        if resume_path:
            trainer.load_checkpoint(resume_path)

        log.info("Trainer ready — press ▶ Start in the dashboard to begin training.")
        while not shared.is_training and not shared.stop_requested:
            time.sleep(0.5)

        if not shared.stop_requested:
            log.info("Training started — escaping any open menus before first episode…")
            try:
                from controls import GameControls
                import yaml as _yaml2
                with open(config_path) as _f2:
                    _cfg2 = _yaml2.safe_load(_f2)
                _title2 = _cfg2["game_settings"].get("window_title_substring", "RESIDENT EVIL 4")
                _ctrl_tmp = GameControls(_title2)
                _ctrl_tmp.escape_to_gameplay()
            except Exception as _esc_exc:
                log.warning("Pre-training menu escape failed (non-fatal): %s", _esc_exc)
            trainer.train()
    except Exception as exc:
        log.exception("Training thread crashed: %s", exc)
        shared.update(is_training=False)


def main() -> None:
    args = parse_args()

    # 1. Load .env (API keys — only ANTHROPIC_API_KEY required for LLM)
    load_dotenv()

    # 2. Logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=log_level)
    logger = logging.getLogger("main")
    logger.info("RE Agent starting up…")

    # 3. Shared state (thread-safe dataclass) and memory DB
    shared = SharedState()
    memory = MemorySystem()
    memory.start()

    # 4a. Failsafe watchdog (move mouse to any screen corner to emergency-stop)
    threading.Thread(
        target=_failsafe_watchdog,
        args=(shared,),
        daemon=True,
        name="FailsafeWatchdog",
    ).start()

    # 4b. Focus monitor — auto-pauses when the game window loses focus so the
    #     user can interact with the browser dashboard freely without the agent
    #     sending stray keypresses or mouse movements to the wrong window.
    import yaml as _yaml
    with open(args.config) as _f:
        _cfg = _yaml.safe_load(_f)
    _game_title = _cfg["game_settings"].get("window_title_substring", "RESIDENT EVIL 4")
    threading.Thread(
        target=_focus_monitor,
        args=(shared, _game_title),
        daemon=True,
        name="FocusMonitor",
    ).start()

    # 5. Gameplay recorder (saves .mp4 footage for review)
    #    OFF by default: software-encoding a 1080p MP4 at 20 fps competes with
    #    capture + perception + the policy forward pass and noticeably lowers the
    #    agent's decision/vision FPS.  Opt in with --record only when you actually
    #    want footage to review.
    recorder = None
    if args.record:
        recorder = GameplayRecorder(shared, output_dir="footage", fps=20)
        logger.info("Gameplay recording ENABLED (--record) — expect some FPS cost.")
    else:
        logger.info("Gameplay recording disabled (default). Pass --record to enable.")

    # 6. LLM advisor (Claude-only — only ANTHROPIC_API_KEY needed)
    llm, consultant = (None, None) if args.no_llm else _build_llm(
        args.config, shared, memory
    )

    # 7. Trainer thread (starts paused; unblocked by the dashboard Start button)
    if not args.dashboard_only:
        threading.Thread(
            target=_training_thread,
            args=(args.config, shared, memory, consultant, args.resume),
            daemon=True,
            name="TrainerThread",
        ).start()
        logger.info("Trainer thread launched.")

    # 8. Start recorder
    if recorder:
        recorder.start()
        logger.info("Gameplay recorder started → footage/")

    # 9. Gradio dashboard (runs in main thread — blocking)
    dash = AgentDashboard(shared, memory, recorder=recorder)
    dash.build()
    logger.info("Launching Gradio dashboard at http://127.0.0.1:7860")

    try:
        dash.launch()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    finally:
        shared.update(stop_requested=True)
        if recorder:
            recorder.stop()
        memory.stop()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
