"""
Gradio 5 dashboard for the RE Agent.

Layout:
  Tab 1 – Live Feed      : Annotated game frame + HUD
  Tab 2 – Agent Brain    : Expandable panels per LLM + RL action
  Tab 3 – Training       : Reward / episode-length plots + stats
  Tab 4 – Memory         : Recent LLM decisions + knowledge
  Tab 5 – Controls       : Start/stop, curriculum, force-LLM

Refresh rate: 1 second (gr.Timer).  All data comes from SharedState snapshots;
no direct DB reads happen in the UI thread.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go

from memory import MemorySystem
from shared_state import SharedState

logger = logging.getLogger(__name__)

_ACTION_LABELS = {
    0: ["stop","fwd","back","left","right","fwd-left","fwd-right","back-left","back-right"],
    1: ["no-cam","cam-left","cam-right","cam-up","cam-down"],
    2: ["no-interact","interact"],
    3: ["no-combat","aim","shoot","aim+shoot"],
    4: ["no-evasion","dodge","sprint"],
    5: ["no-inv","inv-toggle"],
}

_CURRICULUM_STAGES = ["exploration", "combat", "completion"]


def _action_to_readable(action: List[int]) -> str:
    if not action or len(action) < 6:
        return "—"
    labels = [_ACTION_LABELS[i][min(v, len(_ACTION_LABELS[i])-1)] for i, v in enumerate(action)]
    return " | ".join(labels)


def _make_reward_plot(reward_history: List[float]) -> go.Figure:
    fig = go.Figure()
    if reward_history:
        fig.add_trace(go.Scatter(
            y=reward_history,
            mode="lines",
            name="Episode Reward",
            line=dict(color="#00d4ff", width=2),
        ))
        # Rolling average
        if len(reward_history) >= 10:
            import statistics
            window = 10
            avg = [
                statistics.mean(reward_history[max(0, i-window):i+1])
                for i in range(len(reward_history))
            ]
            fig.add_trace(go.Scatter(
                y=avg,
                mode="lines",
                name="10-ep avg",
                line=dict(color="#ff6b35", width=2, dash="dash"),
            ))
    fig.update_layout(
        template="plotly_dark",
        title="Episode Reward History",
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _make_length_plot(length_history: List[int]) -> go.Figure:
    fig = go.Figure()
    if length_history:
        fig.add_trace(go.Bar(
            y=length_history,
            name="Episode Length",
            marker_color="#7c4dff",
        ))
    fig.update_layout(
        template="plotly_dark",
        title="Episode Length (steps)",
        xaxis_title="Episode",
        yaxis_title="Steps",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _annotate_frame(frame: Optional[np.ndarray], detections: List[Dict], hud: Dict) -> Optional[np.ndarray]:
    if frame is None:
        return None
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        label = f"{det.get('label','?')} {det.get('confidence', 0):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(y1-6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # HUD overlay (health_pct is None while the ring isn't visible)
    _hp  = hud.get("health_pct", None)
    hp_txt = f"{float(_hp):.0%}" if _hp is not None else "--"
    clip = hud.get("ammo_clip", "?")
    res  = hud.get("ammo_res",  "?")
    cv2.putText(annotated, f"HP: {hp_txt}  Ammo: {clip}/{res}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


class AgentDashboard:
    def __init__(self, shared: SharedState, memory: MemorySystem, recorder=None):
        self._shared   = shared
        self._memory   = memory
        self._recorder = recorder   # Optional[GameplayRecorder]
        self._demo: Optional[gr.Blocks] = None

    # ── UI construction ───────────────────────────────────────────────────────

    def build(self) -> gr.Blocks:
        self._theme = gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="violet",
            neutral_hue="slate",
        )

        with gr.Blocks(title="RE Agent Dashboard") as demo:
            gr.Markdown("# 🎮 Resident Evil AI Agent — Live Dashboard")
            status_md = gr.Markdown("**Status:** Idle")

            with gr.Tabs():

                # ── Tab 1: Live Feed ─────────────────────────────────────────
                with gr.Tab("📺 Live Feed"):
                    with gr.Row():
                        live_img = gr.Image(
                            label="Annotated Game View",
                            height=480,
                            show_label=True,
                        )
                        with gr.Column(scale=1):
                            health_slider = gr.Slider(
                                label="Health", minimum=0, maximum=1,
                                value=1.0, interactive=False,
                            )
                            ammo_box = gr.Textbox(label="Ammo (clip / reserve)", interactive=False)
                            det_json = gr.JSON(label="YOLO Detections")
                            action_box = gr.Textbox(label="Current RL Action", interactive=False)
                            reward_box = gr.Textbox(label="Last Step Reward", interactive=False)

                # ── Tab 2: Agent Brain ───────────────────────────────────────
                with gr.Tab("🧠 Agent Brain"):
                    gr.Markdown(
                        "_Powered by Claude (claude-sonnet-4-6 with vision). "
                        "All three panels are from a single Claude API call every ~60 RL steps._"
                    )
                    with gr.Accordion("📷 Claude Vision Analysis", open=True):
                        gpt_box = gr.Textbox(
                            label="What Claude sees on screen",
                            lines=6,
                            interactive=False,
                            placeholder="Waiting for first LLM call — requires ANTHROPIC_API_KEY in .env…",
                        )

                    with gr.Accordion("🎯 Claude Strategic Plan", open=False):
                        claude_box = gr.Textbox(
                            label="30-second strategic goal",
                            lines=8,
                            interactive=False,
                            placeholder="Waiting for first LLM call…",
                        )

                    with gr.Accordion("⚡ Claude Tactical Decision", open=False):
                        grok_box = gr.Textbox(
                            label="Immediate action recommendation",
                            lines=5,
                            interactive=False,
                            placeholder="Waiting for first LLM call…",
                        )

                    with gr.Accordion("🗺️ Active Objective (reward shaping)", open=False):
                        obj_box = gr.Textbox(
                            label="Current objective phrase",
                            interactive=False,
                            placeholder="None — set automatically when Claude calls",
                        )

                # ── Tab 3: Training ──────────────────────────────────────────
                with gr.Tab("📈 Training"):
                    with gr.Row():
                        ep_count  = gr.Number(label="Episodes",     interactive=False)
                        total_st  = gr.Number(label="Total Steps",  interactive=False)
                        best_rew  = gr.Number(label="Best Reward",  interactive=False)
                        deaths    = gr.Number(label="Total Deaths",  interactive=False)
                        stage_box = gr.Textbox(label="Curriculum Stage", interactive=False)

                    reward_plot = gr.Plot(label="Reward History")
                    length_plot = gr.Plot(label="Episode Length")

                # ── Tab 4: Memory ────────────────────────────────────────────
                with gr.Tab("💾 Memory"):
                    with gr.Row():
                        refresh_mem_btn = gr.Button("↻ Refresh", size="sm")
                    llm_table = gr.Dataframe(
                        headers=["Timestamp", "Model", "Decision", "Response"],
                        label="Recent LLM Decisions",
                        interactive=False,
                        wrap=True,
                    )
                    summary_json = gr.JSON(label="Training Summary")

                # ── Tab 5: Controls ──────────────────────────────────────────
                with gr.Tab("🕹️ Controls"):
                    gr.Markdown("### Training Control")
                    with gr.Row():
                        start_btn    = gr.Button("▶ Start Training",   variant="primary")
                        stop_btn     = gr.Button("⏹ Stop Training",    variant="stop")
                        force_llm    = gr.Button("🤖 Force LLM Call",  variant="secondary")

                    gr.Markdown("### Pause / Resume")
                    gr.Markdown(
                        "_Use **Pause** to freely interact with this browser dashboard "
                        "without the agent sending inputs to the game. The training loop "
                        "keeps running — only keypresses and mouse movements are blocked. "
                        "Click **Resume** (or just click back on the game window if "
                        "Auto-Pause is ON) to continue._"
                    )
                    with gr.Row():
                        pause_btn    = gr.Button("⏸ Pause Agent",      variant="secondary")
                        resume_btn   = gr.Button("▶ Resume Agent",     variant="primary")
                    auto_pause_chk = gr.Checkbox(
                        label="🔒 Auto-Pause when game loses focus (recommended)",
                        value=True,
                        info="When checked, the agent pauses automatically whenever "
                             "you click on the browser or any other window, and resumes "
                             "when you click back on RE4.",
                    )

                    gr.Markdown("### Recording")
                    with gr.Row():
                        _init_rec_label = (
                            "⏹ Stop Recording"
                            if (self._recorder and self._recorder.is_recording)
                            else "⏺ Start Recording"
                        )
                        record_btn = gr.Button(_init_rec_label, variant="secondary")
                    rec_info = gr.Markdown(
                        "_Footage saved to `footage/` as timestamped `.mp4` files (auto-split every 10 min)._"
                    )

                    gr.Markdown("### Curriculum Stage")
                    curr_radio = gr.Radio(
                        choices=_CURRICULUM_STAGES,
                        value="exploration",
                        label="Active Stage",
                        interactive=True,
                    )

                    gr.Markdown("### Status")
                    ctrl_status = gr.Textbox(label="", interactive=False, lines=3)

            # ── Timer-driven auto-refresh ────────────────────────────────────
            # SLOW timer (1 s): plots, LLM text, counters — expensive to rebuild.
            timer = gr.Timer(value=1.0)
            timer.tick(
                fn=self._refresh_live,
                inputs=[],
                outputs=[
                    status_md,
                    live_img, health_slider, ammo_box, det_json,
                    action_box, reward_box,
                    gpt_box, claude_box, grok_box, obj_box,
                    ep_count, total_st, best_rew, deaths, stage_box,
                    reward_plot, length_plot,
                ],
            )

            # FAST timer: streams ONLY the live video frame + cheap HUD fields so
            # the game view updates ~10×/sec instead of 1×/sec.  It deliberately
            # does NOT touch the Plotly charts (those stay on the 1 s timer) so we
            # get a smooth feed without rebuilding plots on every tick.
            img_timer = gr.Timer(value=0.1)   # 10 fps dashboard video
            img_timer.tick(
                fn=self._refresh_image,
                inputs=[],
                outputs=[live_img, health_slider, ammo_box, action_box, reward_box],
            )

            # ── Memory tab refresh ───────────────────────────────────────────
            refresh_mem_btn.click(
                fn=self._refresh_memory,
                inputs=[],
                outputs=[llm_table, summary_json],
            )

            # ── Control buttons ──────────────────────────────────────────────
            start_btn.click(
                fn=lambda: self._handle_start(),
                outputs=[ctrl_status],
            )
            stop_btn.click(
                fn=lambda: self._handle_stop(),
                outputs=[ctrl_status],
            )
            force_llm.click(
                fn=lambda: self._handle_force_llm(),
                outputs=[ctrl_status],
            )
            pause_btn.click(
                fn=lambda: self._handle_pause(),
                outputs=[ctrl_status],
            )
            resume_btn.click(
                fn=lambda: self._handle_resume(),
                outputs=[ctrl_status],
            )
            auto_pause_chk.change(
                fn=self._handle_auto_pause_toggle,
                inputs=[auto_pause_chk],
                outputs=[ctrl_status],
            )
            curr_radio.change(
                fn=self._handle_curriculum,
                inputs=[curr_radio],
                outputs=[ctrl_status],
            )
            record_btn.click(
                fn=self._handle_record,
                outputs=[record_btn, ctrl_status],
            )

        self._demo = demo
        return demo

    def launch(self, **kwargs) -> None:
        if self._demo is None:
            self.build()
        self._demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            theme=self._theme,
            css=self._css(),
            **kwargs,
        )

    # ── Refresh callbacks ─────────────────────────────────────────────────────

    def _refresh_image(self) -> Tuple:
        """
        Lightweight high-frequency refresh — just the annotated game frame and
        the cheap HUD readouts.  Runs ~10×/sec so the live view looks like video.
        Deliberately avoids any plot/LLM/DB work so it stays fast.
        """
        try:
            snap = self._shared.get_snapshot()
            img      = _annotate_frame(snap["frame"], snap["detections"], snap["hud"])
            hud      = snap["hud"]
            health   = float(hud.get("health_pct", 1.0) or 0.0)
            ammo_txt = f"{hud.get('ammo_clip','?')} / {hud.get('ammo_res','?')}"
            act_txt  = _action_to_readable(snap["current_action"])
            rew_txt  = f"{snap['last_reward']:+.3f}"
            return img, health, ammo_txt, act_txt, rew_txt
        except Exception as exc:
            logger.debug("Image refresh error: %s", exc)
            return None, 1.0, "? / ?", "", "+0.000"

    def _refresh_live(self) -> Tuple:
        try:
            snap = self._shared.get_snapshot()

            # ── Status badge ──────────────────────────────────────────────────
            # Priority order: paused > training > idle
            # Each state gets a distinct colour and message so the user always
            # knows at a glance what the agent is doing.
            if snap.get("paused", False):
                # Yellow — agent is paused; no inputs are being sent to the game
                status = "🟡 **Paused** — inputs blocked, safe to use this browser"
            elif snap["is_training"]:
                status = "🟢 **Training**"
            else:
                status = "🔴 **Idle**"

            focus_warn = ""
            auto_pause = snap.get("auto_pause_on_focus_loss", False)
            if auto_pause and not snap.get("game_focused", True) and snap["is_training"] and not snap.get("paused", False):
                focus_warn = "   ⚠️ **GAME NOT FOCUSED** — click RE4 window to resume"

            auto_tag = "🔒 auto-pause ON" if auto_pause else "🔓 gamepad mode — focus not required"
            status_md = (
                f"**Status:** {status}   "
                f"**Stage:** `{snap['curriculum_stage']}`   "
                f"_{auto_tag}_{focus_warn}"
            )

            img = _annotate_frame(snap["frame"], snap["detections"], snap["hud"])

            hud      = snap["hud"]
            _hp_raw  = hud.get("health_pct", None)
            health   = float(_hp_raw) if _hp_raw is not None else 1.0
            ammo_txt = f"{hud.get('ammo_clip','?')} / {hud.get('ammo_res','?')}"

            act_txt = _action_to_readable(snap["current_action"])
            rew_txt = f"{snap['last_reward']:+.3f}"

            reward_fig = _make_reward_plot(snap["reward_history"])
            length_fig = _make_length_plot(snap["episode_length_history"])

            return (
                status_md,
                img, health, ammo_txt, snap["detections"],
                act_txt, rew_txt,
                snap["gpt_analysis"], snap["claude_plan"], snap["grok_tactical"],
                snap["llm_objective"],
                float(snap["episode_count"]),
                float(snap["total_steps"]),
                float(snap["best_reward"]) if snap["best_reward"] != float("-inf") else 0.0,
                float(snap["death_count"]),
                snap["curriculum_stage"],
                reward_fig, length_fig,
            )
        except Exception as exc:
            logger.error("Dashboard refresh error: %s", exc)
            empty_fig = go.Figure()
            return (
                "**Status:** ⚠️ Refresh error — check logs",
                None, 1.0, "—", [],
                "—", "0.000",
                "", "", "", "",
                0.0, 0.0, 0.0, 0.0, "exploration",
                empty_fig, empty_fig,
            )

    def _refresh_memory(self) -> Tuple:
        decisions = self._memory.get_recent_llm_decisions(limit=20)
        rows = [
            [
                d.get("timestamp", ""),
                d.get("model", ""),
                d.get("decision", ""),
                (d.get("response", "") or "")[:120] + "…",
            ]
            for d in decisions
        ]
        summary = self._memory.get_training_summary()
        return rows, summary

    # ── Control handlers ──────────────────────────────────────────────────────

    def _handle_start(self) -> str:
        if self._shared.is_training:
            return "Already training."
        # Clear paused flag so training doesn't immediately freeze on start
        self._shared.update(stop_requested=False, is_training=True, paused=False)
        logging.getLogger("dashboard").info("START button pressed — training enabled.")
        return "▶ Training started."

    def _handle_stop(self) -> str:
        self._shared.update(stop_requested=True, paused=False)
        logging.getLogger("dashboard").warning(
            "STOP button pressed — stop_requested set; training will halt after "
            "the current rollout."
        )
        return "⏹ Stop requested — will complete current rollout then halt."

    def _handle_force_llm(self) -> str:
        self._shared.update(force_llm_call=True)
        return "🤖 LLM call queued — results will appear in Agent Brain tab."

    def _handle_pause(self) -> str:
        """
        Pause all game inputs without stopping the training loop.

        The RL model keeps computing actions and the reward function keeps
        tracking state — we just block the _take_action() method from sending
        anything to the game.  This means you can use the browser dashboard,
        check TensorBoard, type in a terminal, etc. with zero risk of stray
        inputs landing in the game or in whatever window you're using.
        """
        if not self._shared.is_training:
            return "Nothing is training — nothing to pause."
        if self._shared.paused:
            return "Already paused."
        self._shared.update(paused=True)
        return "⏸ Agent paused — all game inputs blocked. Click ▶ Resume when ready."

    def _handle_resume(self) -> str:
        """
        Resume game inputs after a manual or auto pause.

        Important: only resume when the RE4 window is active (or is about to
        become active).  If you click Resume while the browser is still in
        front, the focus monitor will immediately re-pause (if auto-pause is ON).
        """
        if not self._shared.paused:
            return "Not currently paused."
        self._shared.update(paused=False)
        return "▶ Agent resumed — inputs will fire on the next training step."

    def _handle_auto_pause_toggle(self, enabled: bool) -> str:
        """
        Toggle the auto-pause-on-focus-loss behaviour.

        ON  (recommended) — agent auto-pauses whenever RE4 loses focus;
            click back on the game to auto-resume.  Safe for dashboard use.
        OFF — agent ignores focus changes; you must pause/resume manually.
            Use this only if you have a specific reason (e.g. testing input
            robustness or running a headless capture setup).
        """
        self._shared.update(auto_pause_on_focus_loss=enabled)
        state = "enabled ✅" if enabled else "disabled ⚠️"
        return f"Auto-pause on focus loss: {state}"

    def _handle_curriculum(self, stage: str) -> str:
        self._shared.update(curriculum_stage=stage)
        return f"Curriculum stage set to `{stage}`."

    def _handle_record(self) -> tuple:
        if self._recorder is None:
            return gr.update(), "Recorder not available (launched with --no-record)."
        if self._recorder.is_recording:
            self._recorder.stop()
            return gr.update(value="⏺ Start Recording"), "⏹ Recording stopped — file saved to footage/."
        else:
            self._recorder.start()
            return gr.update(value="⏹ Stop Recording"), "⏺ Recording started — saving to footage/."

    # ── CSS ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _css() -> str:
        return """
        .gradio-container { font-family: 'Segoe UI', sans-serif; }
        .gr-button-primary { background: linear-gradient(90deg, #00d4ff, #7c4dff) !important; }
        .gr-accordion { border-left: 3px solid #00d4ff !important; margin-bottom: 8px; }
        """
