"""
Claude-only AI advisor for the RE4 agent.

WHY single-model instead of 3?
  The original design used GPT-4o (vision) → Claude (planning) → Grok (tactical)
  in a LangGraph pipeline.  In practice this was fragile: all three API keys
  had to be valid, each call could fail independently, and the three-hop
  latency was 8–15 s per consultation (blocking the dashboard).

  The new design makes ONE Claude call that does everything:
    1. Vision analysis   – what's on screen right now
    2. Strategic plan    – what the agent should do over the next 30 s
    3. Tactical decision – the exact action to take this step
    4. Action override   – optional 6-tuple to hard-steer the RL agent

  Single model → single key required → much more reliable.
  Claude claude-sonnet-4-6 supports vision (images in the user message),
  so we can send the current game frame directly.

  Analogy: instead of three specialists who must phone each other in sequence,
  we have one generalist doctor who can see the patient, read the chart, and
  write the prescription in one visit.

REQUIRED ENV VAR:
  ANTHROPIC_API_KEY  — get one at console.anthropic.com

OPTIONAL:
  Nothing else. OpenAI and xAI keys are no longer required.
"""

import asyncio
import base64
import logging
import os
import re
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml
from anthropic import AsyncAnthropic
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from guide_data import WALKTHROUGH, GENERAL_TIPS, MERCHANT_UPGRADES_PRIORITY
from shared_state import SharedState

logger = logging.getLogger(__name__)


# ── Frame encoding ────────────────────────────────────────────────────────────

def _encode_frame(frame: np.ndarray, quality: int = 70, max_side: int = 512) -> str:
    """
    Encode a BGR numpy frame to a base64 JPEG string for the Anthropic API.

    We downscale to max_side pixels on the longest edge before encoding.
    Rationale: sending a full 1920×1080 frame costs ~10× more tokens and adds
    latency with no meaningful gain in understanding — Claude can read the HUD
    and detect enemies from a 512-px image just as well.
    Example: 1920×1080 → 512×288 → ~35 KB JPEG vs ~500 KB → ~10× cheaper.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    hud: Dict[str, Any],
    detections: List[Dict],
    chapter: str = "",
) -> str:
    """
    Build the full system + user prompt for Claude.

    We inject:
      - HUD state (health, ammo)
      - YOLO detections (enemies, items)
      - Walkthrough context for the current chapter
      - General RE4 tips (knife parry, ammo conservation, etc.)

    Output format (strictly structured so _parse_response works reliably):
      VISION: <2-3 sentences>
      PLAN: <1-2 sentences>
      DECISION: <one sentence>
      OVERRIDE: [mv, cam, inter, comb, ev, inv]  OR  null
      OBJECTIVE: <short phrase>
    """
    # Chapter-specific guidance
    chapter_ctx = ""
    if chapter and chapter in WALKTHROUGH:
        entry = WALKTHROUGH[chapter]
        chapter_ctx = (
            f"\nChapter guide for '{chapter}':\n"
            f"  Objective: {entry.get('objective', 'unknown')}\n"
            f"  Strategy:  {entry.get('optimal_strategy', '')}\n"
            f"  Key items: {', '.join(entry.get('key_items', []))}\n"
            f"  Threats:   {', '.join(entry.get('threats', []))}\n"
        )
    else:
        # Provide a condensed summary of ALL chapters so the model can infer the stage
        summaries = "\n".join(
            f"  {ch}: {data.get('objective', '')}"
            for ch, data in list(WALKTHROUGH.items())[:4]   # first 4 chapters cover the village
        )
        chapter_ctx = f"\nEarly-game chapter objectives:\n{summaries}\n"

    general = "  • " + "\n  • ".join(GENERAL_TIPS[:6])   # top 6 most important tips
    merchant = "  • " + "\n  • ".join(MERCHANT_UPGRADES_PRIORITY[:4])

    # Detected objects summary
    det_summary = ", ".join(
        f"{d['label']}({d['confidence']:.0%})" for d in detections[:8]
    ) or "none"

    prompt = f"""You are the AI brain of a Resident Evil 4 Remake agent.
You receive the current game frame and sensor data. Provide concise, actionable guidance.

=== SENSOR DATA ===
Health:    {float(hud.get('health_pct', 1.0)):.0%}
Ammo clip: {hud.get('ammo_clip', '?')}
Ammo res:  {hud.get('ammo_res', '?')}
Detections: {det_summary}
{chapter_ctx}
=== RE4 TIPS ===
{general}

=== UPGRADE PRIORITIES ===
{merchant}

=== ACTION SPACE ===
mv:   0=stop 1=fwd 2=back 3=left 4=right 5=fwd-left 6=fwd-right 7=back-left 8=back-right
cam:  0=none 1=left 2=right 3=up 4=down
inter: 0=none 1=interact(F)
comb: 0=none 1=aim(RMB) 2=shoot(LMB) 3=aim+shoot
ev:   0=none 1=dodge(space) 2=sprint(shift)
inv:  0=none 1=toggle(tab)

=== INSTRUCTIONS ===
Analyse the frame and sensor data. Reply with EXACTLY these 5 lines, no extra text:

VISION: <2-3 sentences: what is on screen, immediate threats, item locations>
PLAN: <1-2 sentences: 30-second strategic goal>
DECISION: <one sentence: immediate action this step>
OVERRIDE: [mv,cam,inter,comb,ev,inv]  OR  null
OBJECTIVE: <short phrase: active goal for reward shaping, e.g. "reach shotgun house">

Example:
VISION: Leon is in the village square. Three Ganados are approaching from the south. The well and a green herb are visible to the east.
PLAN: Retreat north toward the two-storey house to grab the shotgun. Avoid the Chainsaw Man until the bell rings.
DECISION: Sprint north-forward away from the approaching Ganados.
OVERRIDE: [1,0,0,0,2,0]
OBJECTIVE: reach shotgun house north of village
"""
    return prompt


# ── Response parser ────────────────────────────────────────────────────────────

class _ParsedResponse:
    __slots__ = ("vision", "plan", "decision", "override", "objective")

    def __init__(self):
        self.vision    = ""
        self.plan      = ""
        self.decision  = ""
        self.override: Optional[List[int]] = None
        self.objective = ""


def _parse_response(text: str) -> _ParsedResponse:
    """
    Parse Claude's structured response into fields.

    We use simple line-prefix matching rather than regex to be forgiving
    of minor formatting variations.  The OVERRIDE line is the only one that
    needs numeric parsing; everything else is free text.
    """
    result = _ParsedResponse()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("VISION:"):
            result.vision = stripped[len("VISION:"):].strip()
        elif stripped.startswith("PLAN:"):
            result.plan = stripped[len("PLAN:"):].strip()
        elif stripped.startswith("DECISION:"):
            result.decision = stripped[len("DECISION:"):].strip()
        elif stripped.startswith("OVERRIDE:"):
            raw = stripped[len("OVERRIDE:"):].strip()
            if raw.lower() not in ("null", "none", ""):
                nums = [int(x) for x in re.findall(r"\d+", raw)]
                if len(nums) == 6:
                    result.override = nums
        elif stripped.startswith("OBJECTIVE:"):
            result.objective = stripped[len("OBJECTIVE:"):].strip()
    return result


# ── Main LLM class ─────────────────────────────────────────────────────────────

class ClaudeAdvisor:
    """
    Wraps the Anthropic async client and the structured prompt/parse logic.

    Usage:
        advisor = ClaudeAdvisor("config.yaml")
        result  = advisor.consult(shared_state, episode_step)
        # result is a dict ready to pass to shared.update(**result)
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self._llm_cfg = cfg["llm_settings"]

        ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not ant_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set.  Add it to your .env file.\n"
                "  Get a key at: https://console.anthropic.com"
            )

        self._client = AsyncAnthropic(api_key=ant_key)
        self._model  = self._llm_cfg.get("claude_model", "claude-sonnet-4-6")
        self._max_tokens = int(self._llm_cfg.get("claude_max_tokens", 600))
        self._temperature = float(self._llm_cfg.get("temperature", 0.5))

        logger.info("ClaudeAdvisor ready — model=%s max_tokens=%d",
                    self._model, self._max_tokens)

    # ── Async API call (with retry on transient errors) ───────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
    async def _call(self, frame_b64: str, prompt: str) -> str:
        """
        Single async call to Claude with a game frame + text prompt.

        The image is sent as a base64 JPEG in the user message.  Claude can
        see the frame and the HUD data simultaneously.
        """
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.content[0].text if resp.content else ""

    # ── Synchronous public interface ──────────────────────────────────────────

    def consult(self, shared: SharedState, episode_step: int) -> Dict[str, Any]:
        """
        Run a full consultation and return a dict of SharedState fields to update.
        Returns empty dict if the frame is missing or if all retries fail.

        This method creates a fresh event loop each call so it can be safely
        called from any background thread (not just the main asyncio thread).
        """
        frame = shared.frame
        if frame is None:
            logger.debug("consult: no frame available yet — skipping.")
            return {}

        t0 = time.perf_counter()

        # Encode frame
        frame_b64 = _encode_frame(frame)

        # Build prompt with current sensor data
        prompt = _build_prompt(
            hud=dict(shared.hud),
            detections=list(shared.detections),
            chapter="",   # Chapter detection is vision-based; Claude infers from context
        )

        # Run async call in a fresh event loop
        loop = asyncio.new_event_loop()
        try:
            raw = loop.run_until_complete(self._call(frame_b64, prompt))
        except Exception as exc:
            logger.error("ClaudeAdvisor.consult failed: %s", exc)
            return {}
        finally:
            loop.close()

        elapsed = time.perf_counter() - t0
        logger.debug("Claude consult completed in %.1fs (step=%d)", elapsed, episode_step)

        # Parse structured response
        parsed = _parse_response(raw)

        # We repurpose the SharedState LLM fields:
        #   gpt_analysis  → Claude's vision analysis (what's on screen)
        #   claude_plan   → Claude's strategic plan
        #   grok_tactical → Claude's tactical decision text
        return {
            "gpt_analysis":        parsed.vision,      # labelled as "Vision" in dashboard
            "claude_plan":         parsed.plan,
            "grok_tactical":       parsed.decision,    # "Tactical Decision" in dashboard
            "llm_objective":       parsed.objective,
            "llm_action_override": parsed.override,    # may be None
        }


# ── Backward-compatibility aliases ────────────────────────────────────────────

class MultiModelLLM(ClaudeAdvisor):
    """
    Thin alias so trainer.py and other callers that import MultiModelLLM
    continue to work without modification.

    Why: trainer.py imports `from llm_agent import MultiModelLLM` — renaming
    would require touching every callsite.  An alias costs nothing and keeps
    the codebase stable during the transition.
    """
    pass


# ── Background consultant thread ──────────────────────────────────────────────

class LLMConsultant:
    """
    Wraps ClaudeAdvisor in a reusable fire-and-forget background thread.

    Call trigger() from the training callback; results land in SharedState.
    A second trigger() while a call is in-flight is silently dropped — this
    keeps the RL training loop running at full speed even if Claude is slow.

    Analogy: the consultant is like a radio operator on a submarine.  The
    captain (RL agent) keeps doing their job; the operator sends a message
    when the timing is right and delivers the reply when it arrives, without
    ever blocking the captain.
    """

    def __init__(self, llm: ClaudeAdvisor, shared: SharedState, memory=None):
        self._llm    = llm
        self._shared = shared
        self._memory = memory   # Optional[MemorySystem]
        self._thread: Optional[threading.Thread] = None

    def trigger(self, episode_step: int) -> None:
        """Fire-and-forget: launch LLM consult if no call is already running."""
        if self._thread is not None and self._thread.is_alive():
            return   # Previous call still in-flight — skip
        self._thread = threading.Thread(
            target=self._run,
            args=(episode_step,),
            daemon=True,
            name="LLMConsult",
        )
        self._thread.start()

    def _run(self, episode_step: int) -> None:
        t0 = time.perf_counter()
        try:
            result = self._llm.consult(self._shared, episode_step)
            if not result:
                return
            self._shared.update(**result)
            elapsed = time.perf_counter() - t0

            # Persist decision to memory DB (shows up in the Memory tab)
            if self._memory:
                self._memory.log_llm(
                    step=episode_step,
                    model=self._llm._model,
                    response=(
                        f"[Vision] {result.get('gpt_analysis', '')[:200]} "
                        f"[Plan] {result.get('claude_plan', '')[:200]}"
                    ),
                    decision=result.get("grok_tactical", ""),
                    prompt_summary=f"step={episode_step} t={elapsed:.1f}s",
                )
        except Exception as exc:
            logger.error("LLMConsultant._run error: %s", exc)
