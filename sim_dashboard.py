"""
Gradio 5 monitoring dashboard for simulation-mode training.

LAYOUT (four tabs)
──────────────────
  📺 Live View   — 2×4 grid of all 8 worker village maps, updated every second.
                   Each cell shows one worker's current game state rendered as
                   a top-down 2D image.  You can see all agents simultaneously.

  📈 Progress    — Per-worker episode reward curves + aggregate mean.
                   Shows how each worker is learning over time.
                   Analogy: eight learning curves on the same chart — you can
                   see when different workers "click" and start improving.

  🎯 Objectives  — Scrolling log of notable events (kills, item pickups,
                   curriculum changes, episode outcomes).  Shows WHAT the
                   agent is achieving, not just numbers.

  ⚡ Speed       — Steps/second, sim-vs-real speedup ratio, ETA to target
                   timesteps.  Lets you know how much faster the sim is vs
                   waiting for the real game.

  🎮 Controls    — Start/Stop/Pause, curriculum stage selector.

RENDERING
──────────
The live-view grid renders worker snapshots (small dicts) using the same
render_frame() logic as the simulation.  This avoids shipping large numpy
arrays across processes — only the compact snapshot dict crosses the pipe.
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from shared_state import SharedState
from memory import MemorySystem
from sim_game_state import (
    SimGameState, EnemyState, ItemType, ItemPickup, EnemyAgent,
    OBSTACLES, BARN_ZONE, BELL_ZONE, ITEM_SPAWNS, MAP_W, MAP_H,
)

logger = logging.getLogger(__name__)

# ── Colours ───────────────────────────────────────────────────────────────────
_COLOUR_LEGEND = """
**Map legend**
🟩 Green = player  🔴 Red = enemy (chasing)  🟠 Orange = stunned
🟡 Yellow = ammo  🔵 Cyan = herb  ⭐ Gold = shotgun  🔷 Blue tint = objective
"""


# ── Snapshot → frame renderer ──────────────────────────────────────────────────

def _render_snapshot(snap: Optional[Dict], obs_h: int = 120, obs_w: int = 120) -> np.ndarray:
    """
    Render a worker snapshot dict into an RGB image for the live grid.

    This mirrors SimGameState.render_frame() but takes the lightweight
    snapshot dict (not the full game state) — no subprocess pipes needed.

    If snap is None (worker not yet started), returns a dark placeholder frame.
    """
    img = np.full((obs_h, obs_w, 3), 18, dtype=np.uint8)  # Dark background

    if not snap:
        # Draw "WAITING" text area using a white rectangle placeholder
        img[obs_h // 2 - 2: obs_h // 2 + 3, obs_w // 4: obs_w * 3 // 4] = 60
        return img

    sx = obs_w / MAP_W
    sy = obs_h / MAP_H

    def _px(wx: float, wy: float) -> Tuple[int, int]:
        return (
            int(min(max(wx * sx, 0), obs_w - 1)),
            int(min(max(wy * sy, 0), obs_h - 1)),
        )

    def _fill(x1, y1, x2, y2, colour):
        px1, py1 = _px(x1, y1)
        px2, py2 = _px(x2, y2)
        img[py1: py2 + 1, px1: px2 + 1] = colour

    def _dot(wx, wy, r_world, colour):
        px_, py_ = _px(wx, wy)
        r = max(1, int(r_world * sx))
        img[max(0, py_ - r): py_ + r + 1, max(0, px_ - r): px_ + r + 1] = colour

    # Objective zone tints
    for zone, tint in [(BARN_ZONE, [10, 10, 40]), (BELL_ZONE, [10, 30, 10])]:
        x1, y1, x2, y2 = zone
        px1, py1 = _px(x1, y1)
        px2, py2 = _px(x2, y2)
        img[py1: py2 + 1, px1: px2 + 1] = np.clip(
            img[py1: py2 + 1, px1: px2 + 1].astype(np.int16) + tint,
            0, 255
        ).astype(np.uint8)

    # Obstacles
    for (x1, y1, x2, y2) in OBSTACLES:
        _fill(x1, y1, x2, y2, [45, 45, 45])

    # Items from snapshot
    items_remaining = snap.get("items_remaining", 0)
    if items_remaining > 0:
        # Re-render default item positions (we don't ship positions in snapshot)
        for (ix, iy, itype) in ITEM_SPAWNS:
            colour = {
                "ammo":    [220, 200,  0],
                "herb":    [  0, 200, 200],
                "shotgun": [220, 180,  50],
            }[itype]
            _dot(ix, iy, 0.5, colour)

    # Enemies
    _ECOL = {
        0: [160,  40,  40],  # PATROL
        1: [220,   0,   0],  # CHASE
        2: [255,  60,  60],  # ATTACK
        3: [220, 120,   0],  # STUNNED
        4: [ 50,   0,   0],  # DEAD
    }
    for (ex, ey, estate) in snap.get("enemy_positions", []):
        _dot(ex, ey, 0.7, _ECOL.get(estate, [180, 0, 0]))

    # Player
    px_, py_ = snap.get("player_pos", (25.0, 42.0))
    health   = snap.get("player_health", 100.0)

    # Player colour shifts from green → yellow → red with health
    health_frac = max(0.0, min(1.0, health / 100.0))
    pcol = [
        int((1.0 - health_frac) * 220),
        int(health_frac * 220),
        0,
    ]
    _dot(px_, py_, 1.0, pcol)

    # Barn-visited indicator: white tick on barn zone
    if snap.get("barn_visited", False):
        bx, by = 40.0, 10.0
        ppx, ppy = _px(bx, by)
        img[max(0, ppy - 1): ppy + 2, max(0, ppx - 1): ppx + 2] = [255, 255, 255]

    # Health bar at top of cell
    bar_w = int(obs_w * health_frac)
    img[0:3, 0:bar_w] = pcol
    img[0:3, bar_w:]  = [60, 0, 0]

    # Step progress bar at bottom
    step    = snap.get("step", 0)
    max_s   = snap.get("max_steps", 600)
    prog_w  = int(obs_w * min(step / max(max_s, 1), 1.0))
    img[-3:, 0:prog_w] = [0, 80, 200]
    img[-3:, prog_w:]  = [20, 20, 20]

    return img


def _make_grid(
    snapshots: List[Optional[Dict]],
    n_workers: int = 8,
    cell_h: int = 120,
    cell_w: int = 120,
    cols: int = 4,
) -> np.ndarray:
    """
    Arrange N worker frames into a (rows × cols) grid image.

    With 8 workers and 4 columns → a 2×4 grid of 120×120 cells.
    Border pixels separate cells for readability.

    Analogy: a security monitor bank showing multiple camera feeds side-by-side.
    """
    rows  = math.ceil(n_workers / cols)
    border = 2
    total_h = rows * cell_h + (rows + 1) * border
    total_w = cols * cell_w + (cols + 1) * border
    grid = np.full((total_h, total_w, 3), 8, dtype=np.uint8)  # Dark border colour

    for idx in range(n_workers):
        row = idx // cols
        col = idx % cols

        snap = snapshots[idx] if idx < len(snapshots) else None
        cell = _render_snapshot(snap, cell_h, cell_w)

        y0 = border + row * (cell_h + border)
        x0 = border + col * (cell_w + border)
        grid[y0: y0 + cell_h, x0: x0 + cell_w] = cell

        # Worker label (white pixels approximating "W0"–"W7" in top-left corner)
        # Gradient bar based on recent reward — simple quality-of-life indicator
        ep_r = snap.get("episode_reward", 0.0) if snap else 0.0
        norm_r = min(max((ep_r + 50) / 150.0, 0.0), 1.0)  # Normalise roughly -50..100
        grid[y0 + cell_h - 1, x0: x0 + int(cell_w * norm_r)] = [0, 180, 60]

    return grid


# ── Dashboard class ────────────────────────────────────────────────────────────

class SimAgentDashboard:
    """
    Gradio 5 dashboard for monitoring simulation training.

    build() constructs the UI components.
    launch() starts the Gradio server (blocking).

    All data flows from SharedState.get_snapshot() → display components.
    The timer-driven _update_*() methods are called by gr.Timer at 1 Hz.
    """

    def __init__(
        self,
        shared: SharedState,
        memory: MemorySystem,
        n_workers: int = 8,
    ):
        self._shared    = shared
        self._memory    = memory
        self._n_workers = n_workers
        self._app: Optional[gr.Blocks] = None

        # Cached history for charts
        self._reward_history:   List[float] = []
        self._worker_ep_rewards: Dict[int, List[float]] = {i: [] for i in range(n_workers)}
        self._steps_history:    List[int]   = []
        self._sps_history:      List[float] = []

    def build(self) -> None:
        """Construct all Gradio components."""
        with gr.Blocks(title="RE4 Sim Training Monitor") as self._app:

            gr.Markdown(
                "# 🎮 RE4 Agent — Simulation Training Monitor\n"
                "Training in parallel simulation mode — no game window required."
            )

            with gr.Tabs():
                # ── Tab 1: Live View ──────────────────────────────────────────
                with gr.Tab("📺 Live View"):
                    gr.Markdown(
                        "All 8 workers rendered simultaneously. "
                        "**Green dot** = Leon (brightness = health). "
                        "**Red dots** = enemies. "
                        "**Top bar** = health. **Bottom bar** = episode progress."
                    )
                    live_grid = gr.Image(
                        label="Worker Grid (8 agents)",
                        height=None,
                        show_label=True,
                    )
                    with gr.Row():
                        live_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            max_lines=1,
                        )
                    gr.Markdown(_COLOUR_LEGEND)

                # ── Tab 2: Progress ───────────────────────────────────────────
                with gr.Tab("📈 Progress"):
                    gr.Markdown(
                        "Episode reward over time, averaged across workers. "
                        "The agent improves as the curve trends upward."
                    )
                    reward_plot = gr.LinePlot(
                        x="Episode",
                        y="Reward",
                        title="Mean Episode Reward",
                        height=300,
                        tooltip=["Episode", "Reward"],
                    )
                    with gr.Row():
                        best_reward_box  = gr.Textbox(label="Best Reward",  interactive=False)
                        ep_count_box     = gr.Textbox(label="Total Episodes", interactive=False)
                        total_steps_box  = gr.Textbox(label="Total Steps",  interactive=False)
                    with gr.Row():
                        sps_plot = gr.LinePlot(
                            x="Sample",
                            y="Steps/sec",
                            title="Simulation Throughput (steps/sec)",
                            height=200,
                        )

                # ── Tab 3: Objectives ─────────────────────────────────────────
                with gr.Tab("🎯 Objectives"):
                    gr.Markdown(
                        "Live log of notable events across all workers.\n"
                        "Shows curriculum milestones, kills, item pickups, and episode outcomes."
                    )
                    with gr.Row():
                        curriculum_box = gr.Textbox(
                            label="Current Curriculum Stage",
                            interactive=False,
                        )
                        llm_obj_box = gr.Textbox(
                            label="Active LLM Objective",
                            interactive=False,
                        )
                    objectives_log = gr.Textbox(
                        label="Event Log (newest at top)",
                        lines=20,
                        interactive=False,
                        max_lines=25,
                    )
                    with gr.Row():
                        worker_stats = gr.Dataframe(
                            headers=["Worker", "Episodes", "Best Reward",
                                     "Total Kills", "Shotgun?"],
                            label="Per-Worker Summary",
                            interactive=False,
                        )

                # ── Tab 4: Speed ──────────────────────────────────────────────
                with gr.Tab("⚡ Speed"):
                    gr.Markdown(
                        "Measures how much faster simulation is compared to real-game training.\n\n"
                        "Real game: ~12 steps/sec (limited by action_hold_seconds=0.08).\n"
                        "Simulation with 8 workers: typically 40,000–100,000 steps/sec."
                    )
                    with gr.Row():
                        sps_box       = gr.Textbox(label="Steps / Second",       interactive=False)
                        speedup_box   = gr.Textbox(label="Speedup vs Real Game", interactive=False)
                        eta_box       = gr.Textbox(label="ETA to Target Steps",  interactive=False)
                    with gr.Row():
                        wall_time_box = gr.Textbox(label="Wall Time Elapsed", interactive=False)
                        sim_steps_box = gr.Textbox(label="Sim Steps Completed", interactive=False)
                        equiv_real_box = gr.Textbox(
                            label="Equivalent Real-Game Time",
                            interactive=False,
                        )

                    self._start_time_ref = [time.monotonic()]  # mutable container for closure

                # ── Tab 5: Controls ───────────────────────────────────────────
                with gr.Tab("🎮 Controls"):
                    gr.Markdown(
                        "Controls for the simulation training run.\n"
                        "Press **▶ Start** to begin.  "
                        "Press **⏸ Pause** to freeze workers without losing progress.  "
                        "Press **⏹ Stop** to end the session and save the checkpoint."
                    )
                    with gr.Row():
                        start_btn  = gr.Button("▶ Start",  variant="primary")
                        pause_btn  = gr.Button("⏸ Pause",  variant="secondary")
                        stop_btn   = gr.Button("⏹ Stop",   variant="stop")

                    curriculum_dd = gr.Dropdown(
                        choices=["exploration", "combat", "completion"],
                        value="exploration",
                        label="Override Curriculum Stage",
                        interactive=True,
                    )
                    force_llm_btn = gr.Button("🤖 Force LLM Consultation", variant="secondary")

                    ctrl_status = gr.Textbox(
                        label="Control Status",
                        interactive=False,
                        max_lines=2,
                    )

            # ── Timers ────────────────────────────────────────────────────────
            # gr.Timer fires callbacks at a set interval.
            # 1 Hz for live view and charts; 0.5 Hz for objectives (less critical).
            fast_timer = gr.Timer(value=1.0)
            slow_timer = gr.Timer(value=2.0)

            # ── Timer callbacks ────────────────────────────────────────────────

            @fast_timer.tick
            def _update_live():
                snap  = self._shared.get_snapshot()
                snaps = snap.get("worker_snapshots", [])

                grid_img = _make_grid(snaps, self._n_workers)

                sps       = snap.get("sim_steps_per_sec", 0.0)
                paused    = snap.get("paused", False)
                training  = snap.get("is_training", False)
                steps     = snap.get("total_steps", 0)
                episodes  = snap.get("episode_count", 0)

                state_str = "PAUSED" if paused else ("TRAINING" if training else "IDLE")
                status_str = (
                    f"[{state_str}]  {steps:,} steps  |  {episodes:,} episodes  "
                    f"|  {sps:,.0f} steps/sec"
                )

                return grid_img, status_str

            fast_timer.tick(
                fn=_update_live,
                inputs=[],
                outputs=[live_grid, live_status],
            )

            @slow_timer.tick
            def _update_progress():
                snap = self._shared.get_snapshot()

                # Reward history
                hist = snap.get("reward_history", [])
                if hist:
                    self._reward_history = hist

                reward_data = {
                    "Episode": list(range(len(self._reward_history))),
                    "Reward":  self._reward_history,
                }

                # Throughput history
                sps = snap.get("sim_steps_per_sec", 0.0)
                self._sps_history.append(sps)
                if len(self._sps_history) > 120:
                    self._sps_history.pop(0)
                sps_data = {
                    "Sample": list(range(len(self._sps_history))),
                    "Steps/sec": self._sps_history,
                }

                best_r  = snap.get("best_reward", float("-inf"))
                best_s  = f"{best_r:.2f}" if best_r > float("-inf") else "—"
                ep_cnt  = snap.get("episode_count", 0)
                steps   = snap.get("total_steps", 0)

                return (
                    reward_data,
                    best_s,
                    f"{ep_cnt:,}",
                    f"{steps:,}",
                    sps_data,
                )

            slow_timer.tick(
                fn=_update_progress,
                inputs=[],
                outputs=[reward_plot, best_reward_box, ep_count_box,
                         total_steps_box, sps_plot],
            )

            @slow_timer.tick
            def _update_objectives():
                snap = self._shared.get_snapshot()
                log_entries = snap.get("objectives_log", [])
                log_text    = "\n".join(reversed(log_entries)) if log_entries else "(No events yet)"
                curriculum  = snap.get("curriculum_stage", "exploration")
                llm_obj     = snap.get("llm_objective", "(none)")

                # Worker summary table
                worker_snaps   = snap.get("worker_snapshots", [])
                worker_eps     = snap.get("worker_episodes", [])
                worker_rewards = snap.get("worker_rewards", [])
                rows = []
                for i in range(self._n_workers):
                    ws  = worker_snaps[i] if i < len(worker_snaps) else {}
                    ep  = worker_eps[i] if i < len(worker_eps) else 0
                    rwd = worker_rewards[i] if i < len(worker_rewards) else 0.0
                    kills  = ws.get("total_kills", 0)
                    shotgun = "★" if ws.get("shotgun_collected", False) else "—"
                    rows.append([f"W{i}", ep, f"{rwd:.1f}", kills, shotgun])

                return log_text, curriculum, llm_obj, rows

            slow_timer.tick(
                fn=_update_objectives,
                inputs=[],
                outputs=[objectives_log, curriculum_box, llm_obj_box, worker_stats],
            )

            @slow_timer.tick
            def _update_speed():
                snap  = self._shared.get_snapshot()
                sps   = snap.get("sim_steps_per_sec", 0.0)
                steps = snap.get("total_steps", 0)

                # Reference start time
                st = self._start_time_ref[0]
                elapsed = time.monotonic() - st
                h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
                wall_str = f"{h:02d}:{m:02d}:{s:02d}"

                # Speedup vs real game (12 steps/sec baseline)
                real_sps = 12.0
                speedup  = sps / real_sps if sps > 0 else 0.0

                # ETA to total_timesteps
                target = 1_000_000  # default; could read from config
                remaining = max(0, target - steps)
                eta_sec = remaining / sps if sps > 1 else float("inf")
                if math.isinf(eta_sec):
                    eta_str = "—"
                else:
                    eh, em, es = int(eta_sec // 3600), int((eta_sec % 3600) // 60), int(eta_sec % 60)
                    eta_str = f"{eh:02d}:{em:02d}:{es:02d}"

                # Equivalent real-game time at 12 steps/sec
                equiv_real_sec = steps / real_sps
                er_h = int(equiv_real_sec // 3600)
                er_m = int((equiv_real_sec % 3600) // 60)
                equiv_str = f"{er_h:,}h {er_m:02d}m  (vs {h:02d}h {m:02d}m actual)"

                return (
                    f"{sps:,.0f} steps/sec",
                    f"{speedup:,.0f}× faster than real game",
                    eta_str,
                    wall_str,
                    f"{steps:,}",
                    equiv_str,
                )

            slow_timer.tick(
                fn=_update_speed,
                inputs=[],
                outputs=[sps_box, speedup_box, eta_box, wall_time_box,
                         sim_steps_box, equiv_real_box],
            )

            # ── Control button callbacks ───────────────────────────────────────

            def _start_training():
                if not self._shared.is_training:
                    self._shared.update(is_training=True, stop_requested=False, paused=False)
                    self._start_time_ref[0] = time.monotonic()
                    return "Training started."
                return "Already training."

            def _pause_training():
                new_paused = not self._shared.paused
                self._shared.update(paused=new_paused)
                return "Paused." if new_paused else "Resumed."

            def _stop_training():
                self._shared.update(stop_requested=True)
                return "Stop requested — saving checkpoint…"

            def _set_curriculum(stage: str):
                self._shared.update(curriculum_stage=stage)
                self._shared.log_objective_event(f"★ Manual curriculum override → {stage}")
                return f"Curriculum set to: {stage}"

            def _force_llm():
                self._shared.update(force_llm_call=True)
                return "LLM consultation triggered."

            start_btn.click(fn=_start_training, outputs=ctrl_status)
            pause_btn.click(fn=_pause_training, outputs=ctrl_status)
            stop_btn.click(fn=_stop_training,  outputs=ctrl_status)
            curriculum_dd.change(fn=_set_curriculum, inputs=curriculum_dd, outputs=ctrl_status)
            force_llm_btn.click(fn=_force_llm, outputs=ctrl_status)

    def launch(self, host: str = "127.0.0.1", port: int = 7861) -> None:
        """Launch the Gradio server (blocking)."""
        if self._app is None:
            raise RuntimeError("Call build() before launch().")
        logger.info("Simulation dashboard → http://%s:%d", host, port)
        self._app.launch(
            server_name=host,
            server_port=port,
            share=False,
            theme=gr.themes.Soft(primary_hue="blue"),
        )
