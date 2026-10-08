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

from collections import deque

from guide_data import WALKTHROUGH, GENERAL_TIPS, MERCHANT_UPGRADES_PRIORITY, tips_for_stage
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

def _build_system_text(chapter: str = "", stage: str = "exploration") -> str:
    """
    Static instruction block — identical on every consult for a given
    (chapter, stage) pair.

    Sent as a cached `system` block (cache_control: ephemeral) so Anthropic
    only bills full input-token price on the FIRST call in each 5-minute TTL
    window; consults run every ~5 s, so every later call is a cache hit at
    ~10% cost.  Anything that changes per call (health, ammo, detections)
    lives in the user message instead — see _build_sensor_text.

    Output contract: Claude answers via the report_tactical_guidance tool
    (schema-validated JSON); the free-text line format is only a fallback.
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

    # Stage-ranked tips: exploration/combat/completion each surface the most
    # relevant advice instead of a hardcoded first-6 slice.
    general = "  • " + "\n  • ".join(tips_for_stage(stage, 4))
    merchant = "  • " + "\n  • ".join(MERCHANT_UPGRADES_PRIORITY[:4])

    return f"""You are the AI brain of a Resident Evil 4 Remake agent.
You receive the current game frame and sensor data. Provide concise, actionable guidance.
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
Analyse the frame and sensor data, then call the report_tactical_guidance tool:
  vision    — 2-3 sentences: what is on screen, immediate threats, item locations
  plan      — 1-2 sentences: 30-second strategic goal
  decision  — one sentence: immediate action this step
  override  — [mv,cam,inter,comb,ev,inv] integer array, or null
  objective — short phrase: active goal for reward shaping, e.g. "reach shotgun house"

Only set override when you are confident a specific action is required RIGHT NOW
(e.g. fleeing the Chainsaw Man, grabbing the shotgun). Otherwise use null and let
the RL policy act freely.
"""


def _build_sensor_text(hud: Dict[str, Any], detections: List[Dict]) -> str:
    """Per-call dynamic content: current HUD state + detections only."""
    # Detected objects summary
    det_summary = ", ".join(
        f"{d['label']}({d['confidence']:.0%})" for d in detections[:8]
    ) or "none"

    # health_pct is None when the HUD ring isn't visible (Leon not aiming)
    _hp = hud.get("health_pct", None)
    health_str = f"{float(_hp):.0%}" if _hp is not None else "unknown (HUD hidden — Leon not aiming)"

    return f"""=== SENSOR DATA ===
Health:    {health_str}
Ammo clip: {hud.get('ammo_clip', '?')}
Ammo res:  {hud.get('ammo_res', '?')}
Detections: {det_summary}"""


# ── Response parser ────────────────────────────────────────────────────────────

# Single source of truth lives in constants.py — the env builds its
# MultiDiscrete action space from the same tuple.
from constants import ACTION_SPACE_SIZES as _ACTION_SPACE_SIZES

# Structured-output contract: Claude is forced (tool_choice) to answer through
# this tool, so responses arrive as schema-validated JSON instead of free text
# that a line-prefix parser has to guess at.
_GUIDANCE_TOOL = {
    "name": "report_tactical_guidance",
    "description": "Report vision analysis, strategic plan, tactical decision, optional action override, and current objective for the RE4 agent.",
    "input_schema": {
        "type": "object",
        "properties": {
            "vision":    {"type": "string", "description": "2-3 sentences: what is on screen, immediate threats, item locations"},
            "plan":      {"type": "string", "description": "1-2 sentences: 30-second strategic goal"},
            "decision":  {"type": "string", "description": "One sentence: immediate action this step"},
            "override":  {
                "anyOf": [
                    {"type": "array", "items": {"type": "integer"}, "minItems": 6, "maxItems": 6},
                    {"type": "null"},
                ],
                "description": "Optional [mv,cam,inter,comb,ev,inv] action override, or null to let the RL policy act freely",
            },
            "objective": {"type": "string", "description": "Short phrase: active goal for reward shaping, e.g. 'reach shotgun house'"},
        },
        "required": ["vision", "plan", "decision", "override", "objective"],
    },
}

class _ParsedResponse:
    __slots__ = ("vision", "plan", "decision", "override", "objective")

    def __init__(self):
        self.vision    = ""
        self.plan      = ""
        self.decision  = ""
        self.override: Optional[List[int]] = None
        self.objective = ""


def _parse_tool_input(inp: Dict[str, Any]) -> _ParsedResponse:
    """
    Build a _ParsedResponse from a schema-validated tool-use input dict.

    This is the primary parse path — the JSON schema guarantees field
    presence and types, so there is no guessing.  The override array is still
    clamped to the MultiDiscrete action-space bounds (the schema enforces
    length/integer-ness, not the per-slot ranges).
    """
    result = _ParsedResponse()
    result.vision    = str(inp.get("vision") or "")
    result.plan      = str(inp.get("plan") or "")
    result.decision  = str(inp.get("decision") or "")
    result.objective = str(inp.get("objective") or "")
    ov = inp.get("override")
    if isinstance(ov, list) and len(ov) == 6:
        try:
            result.override = [
                max(0, min(int(n), hi - 1))
                for n, hi in zip(ov, _ACTION_SPACE_SIZES)
            ]
        except (TypeError, ValueError):
            result.override = None
    return result


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
                    # Clamp each element to the MultiDiscrete([9,5,2,4,3,2])
                    # action-space bounds.  LLMs occasionally hallucinate an
                    # out-of-range index (e.g. mv=9); unclamped, that silently
                    # mapped to "no input" via dict .get defaults deep inside
                    # controls — the override looked accepted but did nothing.
                    # Analogy: a GPS telling you to take "exit 9" on a highway
                    # that only has exits 0-8 — better to snap to the nearest
                    # real exit than drive straight past everything.
                    result.override = [
                        max(0, min(n, hi - 1))
                        for n, hi in zip(nums, _ACTION_SPACE_SIZES)
                    ]
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

    def __init__(self, config_path: str = "config.yaml", config: Optional[Dict[str, Any]] = None):
        from config_loader import ensure_config
        cfg = ensure_config(config, config_path)
        self._llm_cfg = cfg["llm_settings"]

        ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not ant_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set.  Add it to your .env file.\n"
                "  Get a key at: https://console.anthropic.com"
            )

        self._client = AsyncAnthropic(api_key=ant_key)
        self._model  = self._llm_cfg.get("claude_model", "claude-sonnet-4-6")
        self._fast_model: Optional[str] = self._llm_cfg.get("fast_model")  # two-tier advisor
        self._escalate_hp   = float(self._llm_cfg.get("escalate_health_below", 0.50))
        self._escalate_enemy = bool(self._llm_cfg.get("escalate_on_enemies", True))
        self._max_tokens = int(self._llm_cfg.get("claude_max_tokens", 600))
        self._temperature = float(self._llm_cfg.get("temperature", 0.5))

        # Rolling window of recent objectives — used to infer which chapter the
        # agent is in, so the chapter guide injected into the system prompt
        # tracks the game instead of always summarising the first four.
        self._objective_history: deque = deque(maxlen=8)

        # Cache of built system blocks, keyed by (chapter, stage) — built lazily.
        self._system_cache: Dict[Any, List[Dict[str, Any]]] = {}
        self._last_usage: Optional[Dict[str, int]] = None

        logger.info("ClaudeAdvisor ready — model=%s max_tokens=%d",
                    self._model, self._max_tokens)

    # ── Cached system prompt ────────────────────────────────────────────────────

    def _system_blocks(self, chapter: str = "", stage: str = "exploration") -> List[Dict[str, Any]]:
        """
        Static instruction blocks for the Anthropic `system` parameter.

        Built once per (chapter, stage) pair and reused; the block carries
        cache_control=ephemeral so repeated consults (every ~5 s) hit the
        prompt cache instead of re-billing the full instruction prefix.
        """
        key = (chapter, stage)
        if key not in self._system_cache:
            self._system_cache[key] = [{
                "type": "text",
                "text": _build_system_text(chapter, stage),
                "cache_control": {"type": "ephemeral"},
            }]
        return self._system_cache[key]

    # ── Chapter inference ─────────────────────────────────────────────────────

    def _infer_chapter(self, detections: List[Dict]) -> str:
        """
        Guess the current WALKTHROUGH chapter from recent objectives + current
        detection labels, using keyword overlap.  Returns "" (generic guide)
        when evidence is too weak — a wrong chapter is worse than none.
        """
        if not self._objective_history:
            return ""
        haystack = " ".join(self._objective_history).lower()
        haystack += " " + " ".join(d.get("label", "") for d in detections).lower()

        best, best_score = "", 0
        for ch, data in WALKTHROUGH.items():
            words = set()
            for field in ("objective", "optimal_strategy"):
                words |= {w.strip(".,;:()—") for w in str(data.get(field) or "").lower().split()}
            for item in list(data.get("key_items", [])) + list(data.get("threats", [])):
                words |= {w.strip(".,;:()—") for w in str(item).lower().split()}
            # Require 5+ chars so "the"/"and"-class words never score.
            score = sum(1 for w in words if len(w) >= 5 and w in haystack)
            if score > best_score:
                best, best_score = ch, score
        return best if best_score >= 2 else ""

    # ── Async API call (with retry on transient errors) ───────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
    async def _call(
        self,
        frame_b64: str,
        user_text: str,
        system: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Any:
        """
        Single async call to Claude with a game frame + per-call sensor text.

        Static instructions ride in `system` with ephemeral prompt caching —
        after the first consult in a 5-minute window the instruction prefix is
        served from cache (~10% of full input-token cost).  `model` overrides
        the default (two-tier advisor: cheap model for routine consults).
        Responses are forced through the report_tactical_guidance tool, so the
        payload is schema-validated JSON, not free text.

        Returns (text, usage, tool_input) — tool_input is None only if Claude
        unexpectedly answered with plain text (fallback parser handles it).
        """
        resp = await self._client.messages.create(
            model=model or self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            tools=[_GUIDANCE_TOOL],
            tool_choice={"type": "tool", "name": "report_tactical_guidance"},
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
                    {"type": "text", "text": user_text},
                ],
            }],
        )
        text_parts: List[str] = []
        tool_input: Optional[Dict[str, Any]] = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                tool_input = dict(block.input)
            elif getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "\n".join(text_parts)
        u = resp.usage
        usage = {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        self._last_usage = usage
        return text, usage, tool_input

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

        hud_now  = dict(shared.hud)
        dets_now = list(shared.detections)

        # Two-tier advisor: routine consults (calm, healthy, no enemies) go to
        # the cheap fast model; the strong model is reserved for escalations —
        # low health, enemies on screen, or a suspected death screen.
        model = self._model
        if self._fast_model:
            hp = hud_now.get("health_pct")
            low_hp = hp is not None and float(hp) < self._escalate_hp
            enemies = any(
                d.get("label") in ("person", "zombie", "enemy") for d in dets_now
            )
            escalate = (
                low_hp
                or (enemies and self._escalate_enemy)
                or bool(getattr(shared, "death_screen", False))
            )
            model = self._model if escalate else self._fast_model

        # Chapter inference + stage-aware tips ride in the cached system block.
        chapter = self._infer_chapter(dets_now)
        stage   = getattr(shared, "curriculum_stage", "exploration") or "exploration"
        system    = self._system_blocks(chapter=chapter, stage=stage)
        user_text = _build_sensor_text(hud=hud_now, detections=dets_now)

        # Run async call in a fresh event loop
        loop = asyncio.new_event_loop()
        try:
            raw, usage, tool_input = loop.run_until_complete(
                self._call(frame_b64, user_text, system, model=model)
            )
        except Exception as exc:
            logger.error("ClaudeAdvisor.consult failed: %s", exc)
            return {}
        finally:
            loop.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Claude consult %.1fs (step=%d, model=%s, chapter=%r) tokens in=%d out=%d cache_read=%d cache_write=%d",
            elapsed, episode_step, model, chapter or "?",
            usage["input_tokens"], usage["output_tokens"],
            usage["cache_read_input_tokens"], usage["cache_creation_input_tokens"],
        )

        # Parse: schema-validated tool input is primary; line-prefix text parse
        # is the fallback if Claude ignored the tool (shouldn't happen with
        # tool_choice forced, but never trust an LLM's formatting).
        parsed = _parse_tool_input(tool_input) if tool_input else _parse_response(raw)
        if parsed.objective:
            self._objective_history.append(parsed.objective)

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
