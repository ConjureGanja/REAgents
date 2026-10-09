# REAgents Optimization Plan

**Date:** 2026-10-03
**Target project:** `V:\AI-ML\REAgents` (Resident Evil RL + LLM agent)
**Mode:** Optimization of existing codebase (not greenfield, not full rewrite)

## Summary

Optimize the REAgents codebase across four axes requested by the user: **performance/speed**, **code quality/refactor**, **agent instructions/prompts**, and **features**.

### Note on process deviation

The `creating-implementation-plan` workflow normally consumes a feature spec with `REQ-XXX` IDs plus upstream design artifacts (`constitution.md`, `knowledge-graph.json`, design docs). None exist for this project — the spec lives only in `RE_AI_Agent_Spec.pdf` (prose) and there is no coordinator task metadata. Rather than blocking, this plan **derives its own requirement set** (`REQ-P*`, `REQ-Q*`, `REQ-R*`, `REQ-F*`) directly from a code read of the repository and the user's four stated optimization goals. Every requirement below is traceable to concrete code. No project-topology file exists, so G-group labels are omitted.

### Requirements (derived)

| ID | Goal axis | Requirement |
|----|-----------|-------------|
| REQ-P1 | Performance | Speed up YOLO detection (input downscale, FP16, explicit device) |
| REQ-P2 | Performance | Speed up / harden EasyOCR ammo reads (allowlist, preprocessing) |
| REQ-P3 | Performance | Decouple perception from the RL step loop (dedicated thread) |
| REQ-P4 | Performance | Cut LLM token cost and latency via prompt caching + `system` param |
| REQ-P5 | Performance | Eliminate redundant full-frame copies in capture path |
| REQ-Q1 | Quality | Decompose the `environment.py` god object into focused modules |
| REQ-Q2 | Quality | Single source of truth for the action-space sizes |
| REQ-Q3 | Quality | Load `config.yaml` once and inject; stop re-parsing in 4+ modules |
| REQ-Q4 | Quality | Remove `MultiModelLLM` back-compat alias; update call sites |
| REQ-R1 | Prompts | Structured LLM output via tool-use/JSON instead of line-prefix parsing |
| REQ-R2 | Prompts | Static prompt sections moved to cached `system` block |
| REQ-R3 | Prompts | Deliberate prompt-context budget (chapter inference, tips selection) |
| REQ-F1 | Features | Combat-metrics tracking (`combat_metrics.py` from NEXT_STEPS.md) |
| REQ-F2 | Features | Wire the `game_profiles` config merge loader (RE2/RE3 readiness) |
| REQ-F3 | Features | Two-tier LLM advisor (cheap routine consults, strong model on events) |

## Technical Context

- **Stack:** Python 3.11, Windows 10/11, CUDA PyTorch, Stable-Baselines3 RecurrentPPO + sb3-contrib, Gymnasium, ultralytics YOLO11, EasyOCR, mss, OpenCV, Gradio 5, Anthropic SDK (`claude-sonnet-4-6`), SQLite (stdlib).
- **Thread model (real game):** Main = Gradio; `TrainerThread` = SB3 loop; `LLMConsultant` = fire-and-forget Claude calls; `MemoryThread` = SQLite queue writer; capture grabber thread at 30 fps; focus monitor; failsafe watchdog. All cross-module data flows through `SharedState` (lock-copied snapshots — never mutate snapshots).
- **Obs/action contract:** `Dict(frame=(84,84,12) uint8 RGB×4 stack, hud=7×float32)`; `MultiDiscrete([9,5,2,4,3,2])`. Identical between `environment.py` and `sim_environment.py` — this contract is what makes sim-to-real transfer work. **Any refactor must preserve it exactly.**
- **Perception throttle:** YOLO+OCR re-runs at `perception_fps=10` inside `env.step()`; raw frames are always fresh via the capture thread.
- **Key measured constraint (from code comments):** YOLO + EasyOCR together cost ~150 ms per invocation; LLM consult ~seconds, non-blocking, every `llm_every_n_steps=60` steps with a 5 s cooldown.

## Constitution Check

No `constitution.md` exists. The following project invariants were extracted from `CLAUDE.md` / `README.md` and are treated as binding constraints for every task:

1. **Obs/action contract freeze** — `environment.py` and `sim_environment.py` spaces must stay identical, or sim-to-real transfer silently breaks.
2. **`SharedState` is the only bus** — no new direct module-to-module references; snapshots are lock-copied and read-only.
3. **Training loop never blocks on I/O** — LLM calls and SQLite writes stay off the step hot path (existing design; P3 extends it to perception).
4. **Failsafe integrity** — corner-dwell watchdog and focus-monitor auto-pause must keep working after any refactor.
5. **`hud["health_pct"]` may be `None`** — every consumer must keep guarding for it (phantom-death regression risk).

## Applied Guidelines

No matching entries under `skills/guidelines/` apply (this is a Python RL system, not a framework-to-framework migration). General principles applied instead: SB3 callback conventions, Gymnasium env API stability, Anthropic prompt-caching best practices, and YOLO/EasyOCR inference optimization practice.

## Implementation Steps

### Phase 1 — Measurement baseline (do first)

You cannot optimize what you don't measure. Build a small benchmark harness that records: step-loop FPS, perception latency (YOLO / OCR / health-mask separately), LLM consult latency + token usage, and end-to-end decision rate. Re-run after each later phase.

- **P1.1** Benchmark harness script producing `runs/benchmark_<ts>.json` (REQ-P1..P5 evidence)
- **P1.2** One-time GPU/util sanity check (is YOLO actually on CUDA? is EasyOCR?)

### Phase 2 — Foundational refactors (unblock the rest)

Low-risk structural changes that the perf and prompt work builds on.

- **P2.1** Central config loader — parse `config.yaml` once, pass a config object down (REQ-Q3)
- **P2.2** Shared `ACTION_SPACE_SIZES` constant consumed by both env and `llm_agent.py` (REQ-Q2)
- **P2.3** Profile-merge loader for `game_profiles` (REQ-F2; config-only change, RE4 stays default)

### Phase 3 — Performance

- **P3.1** YOLO: `imgsz` downscale, `half=True` on CUDA, explicit `device`, warmup call at init (REQ-P1)
- **P3.2** OCR: digit/`/` allowlist, grayscale+threshold preprocessing, skip read when region unchanged (REQ-P2)
- **P3.3** Perception thread: move YOLO/OCR off `env.step()` into a worker that publishes to `SharedState`; env reads latest cached results (REQ-P3; honors invariant #3)
- **P3.4** LLM prompt caching: static blocks (action space, tips, upgrades, chapter guide) in a cached `system` message; per-call user message carries only frame + HUD + detections (REQ-P4, REQ-R2)
- **P3.5** Capture path: remove the redundant `.copy()` in the hot read path (return the buffer under lock semantics that are already copy-on-write safe) (REQ-P5)

### Phase 4 — Prompt & LLM quality

- **P4.1** Structured outputs via Anthropic tool-use (JSON schema for VISION/PLAN/DECISION/OVERRIDE/OBJECTIVE) replacing `_parse_response` line-prefix parsing (REQ-R1)
- **P4.2** Deliberate context budget: chapter auto-inference from detections/objective history; top-N tips ranked by curriculum stage instead of hardcoded `[:6]` (REQ-R3)
- **P4.3** Two-tier advisor: fast/cheap model for routine consults, `claude-sonnet-4-6` reserved for combat/death/low-health events (REQ-F3)

### Phase 5 — Code quality

- **P5.1** Split `environment.py` (~1000 lines) into `env/obs_builder.py`, `env/rewards.py`, `env/death_detection.py`, keeping `ResidentEvilEnv` as the thin Gymnasium shell (REQ-Q1; must preserve contract — invariant #1)
- **P5.2** Remove `MultiModelLLM` alias; update `trainer.py` import to `ClaudeAdvisor` (REQ-Q4)

### Phase 6 — Features

- **P6.1** `combat_metrics.py` per NEXT_STEPS.md §1.3, wired into `MemoryCallback` and the dashboard (REQ-F1)

### Phase 7 — Validation & polish

- **P7.1** Re-run P1.1 benchmark; compare against baseline; update README performance notes
- **P7.2** Sim-mode regression check: `python sim/verify_setup.py` + short `sim_main.py` run to prove the obs/action contract survived

## Project Structure (after refactor)

```
REAgents/
├── config_loader.py          # NEW — single config parse + profile merge (P2.1, P2.3)
├── constants.py              # NEW — ACTION_SPACE_SIZES and friends (P2.2)
├── env/
│   ├── __init__.py
│   ├── obs_builder.py        # NEW — frame stack + HUD vector (P5.1)
│   ├── rewards.py            # NEW — reward computation (P5.1)
│   └── death_detection.py    # NEW — death-screen + health-streak logic (P5.1)
├── perception.py             # MODIFIED — YOLO FP16/imgsz, OCR allowlist (P3.1, P3.2)
├── perception_worker.py      # NEW — threaded perception publisher (P3.3)
├── llm_agent.py              # MODIFIED — caching, tool-use, two-tier (P3.4, P4.x)
├── capture.py                # MODIFIED — hot-path copy removal (P3.5)
├── combat_metrics.py         # NEW — per NEXT_STEPS.md (P6.1)
├── benchmarks/
│   └── benchmark.py          # NEW — perf harness (P1.1)
└── (existing files unchanged unless listed)
```

## Tasks

### Phase 1: Setup & Measurement

- [x] T001 [Plan:1.1] Create `benchmarks/benchmark.py` measuring step FPS, YOLO ms, OCR ms, health-mask ms, LLM consult latency and input/output tokens; writes `runs/benchmark_<timestamp>.json` [Source: perception.py#detect_objects,read_hud,get_health_percentage] 
- [x] T002 [P] [Plan:1.2] Add GPU sanity checks at startup in `perception.py` (log `torch.cuda.is_available()`, YOLO model device, EasyOCR device) and warn loudly if CUDA is unused
- [ ] T003 [Plan:1.1] Run baseline benchmark with the game open: `python benchmarks/benchmark.py --duration 120` and commit the JSON as `runs/benchmark_baseline.json`

### Phase 2: Foundational

- [x] T004 [Plan:2.1] Create `config_loader.py` with `load_config(path) -> Config` that parses `config.yaml` once, applies `game_profiles` merge, and exposes typed section accessors [Source: config.yaml]
- [x] T005 [Plan:2.3] Implement profile merge in `config_loader.py`: `active_profile` overlays `window_title_substring`, `guide_module`, `health_style` onto `game_settings`/`perception` [Source: config.yaml#game_profiles]
- [x] T006 [Plan:2.1] Refactor `main.py`, `environment.py`, `perception.py`, `capture.py`, `llm_agent.py`, `trainer.py` to accept the shared config object instead of re-opening `config.yaml` (keep `config_path` fallback for backward compat) [Source: environment.py#__init__, capture.py#__init__, perception.py#__init__, llm_agent.py#__init__]
- [x] T007 [P] [Plan:2.2] Create `constants.py` with `ACTION_SPACE_SIZES = (9, 5, 2, 4, 3, 2)`; make `environment.py` build `spaces.MultiDiscrete` from it and `llm_agent.py` import it (delete `_ACTION_SPACE_SIZES`) [Source: environment.py, llm_agent.py#_ACTION_SPACE_SIZES]

### Phase 3: Performance

- [x] T008 [Plan:3.1] In `perception.py` `detect_objects`, pass `imgsz` (config `perception.yolo_imgsz`, default 640), `half=True` when CUDA, and explicit `device`; add a one-time warmup inference in `__init__` [Source: perception.py#detect_objects]
- [x] T009 [P] [Plan:3.1] Add `perception.yolo_imgsz`, `perception.yolo_half`, `perception.yolo_device` keys to `config.yaml` with comments
- [x] T010 [Plan:3.2] In `perception.py` `read_hud`, call `reader.readtext(roi, detail=0, allowlist="0123456789/")`, preprocess ROI (grayscale → upscale 2× → Otsu threshold), and skip OCR when the ROI's pixel hash is unchanged from last read [Source: perception.py#read_hud]
- [x] T011 [Plan:3.3] Create `perception_worker.py`: background thread running YOLO+OCR+health at `perception_fps`, publishing via `shared.update(detections=..., hud=..., death_screen=...)` [Source: perception.py]
- [x] T012 [Plan:3.3] Change `environment.py` `step()` to read cached perception from `SharedState` instead of calling `self._eyes` inline; keep a synchronous fallback for the first N steps after reset [Source: environment.py#step]
- [x] T013 [Plan:3.4] In `llm_agent.py`, move static prompt blocks (ACTION SPACE, RE4 TIPS, UPGRADE PRIORITIES, chapter guide) into a `system` parameter list with `cache_control: {"type": "ephemeral"}` on the final static block; per-call user message carries only SENSOR DATA + frame [Source: llm_agent.py#_build_prompt,_call]
- [x] T014 [P] [Plan:3.5] In `capture.py` `get_frame()`, return the latest buffer without the extra `.copy()` when a per-call copy is provably unnecessary (recorder/annotator already copy); document the ownership rule in the docstring [Source: capture.py#get_frame]

### Phase 4: Prompts & LLM Quality

- [x] T015 [Plan:4.1] Define an Anthropic tool `report_tactical_guidance` with a JSON schema (vision, plan, decision, override: 6-int array|null, objective) in `llm_agent.py`; call with `tool_choice` forcing the tool; replace `_parse_response` with schema-validated tool-input extraction (keep `_parse_response` as a fallback for one release) [Source: llm_agent.py#_parse_response,_call]
- [x] T016 [Plan:4.2] Implement chapter inference in `llm_agent.py`: maintain a small objective-history window; pick the `WALKTHROUGH` entry whose keywords best match recent objectives/detections instead of always using the first-4-chapters summary [Source: llm_agent.py#_build_prompt, guide_data.py]
- [x] T017 [P] [Plan:4.2] Rank `GENERAL_TIPS` per curriculum stage (exploration/combat/completion) in `guide_data.py` and inject the top 4 for the active stage rather than a hardcoded `[:6]` [Source: guide_data.py#GENERAL_TIPS]
- [x] T018 [Plan:4.3] Add two-tier routing to `LLMConsultant`: routine consults use `llm_settings.fast_model` (e.g. claude-haiku), escalate to `claude_model` when `health_pct` drops, enemies appear, or a death screen is suspected; add config keys `fast_model` and `escalation_rules` [Source: llm_agent.py#LLMConsultant]

### Phase 5: Code Quality

- [x] T019 [Plan:5.1] Create `env/obs_builder.py` with `ObsBuilder` (frame stack + HUD vector) extracted from `environment.py` `_build_obs`; preserve exact dtypes/shapes (84×84×12 uint8, 7×float32) [Source: environment.py#_build_obs]
- [x] T020 [Plan:5.1] Create `env/rewards.py` with the reward computation extracted from `environment.py` `_calculate_reward`, parameterized by curriculum stage weights [Source: environment.py#_calculate_reward]
- [x] T021 [Plan:5.1] Create `env/death_detection.py` with the death-screen streak + low-health-streak + grace-window logic extracted from `environment.py` [Source: environment.py#step,_calculate_reward]
- [x] T022 [Plan:5.1] Slim `ResidentEvilEnv` to orchestration only (reset/step wiring the three new modules); verify `observation_space`/`action_space` are byte-identical to before via a comparison script against `sim_environment.py` [Source: environment.py]
- [x] T023 [P] [Plan:5.2] Delete the `MultiModelLLM` alias in `llm_agent.py` and update the import in `trainer.py` to `ClaudeAdvisor` [Source: llm_agent.py#MultiModelLLM, trainer.py]

### Phase 6: Features

- [x] T024 [Plan:6.1] Create `combat_metrics.py` with `CombatMetrics`/`CombatLogger` per NEXT_STEPS.md §1.3 (accuracy, efficiency, rolling stats)
- [x] T025 [Plan:6.1] Wire `CombatLogger` into `MemoryCallback` (log shots/kills/damage from HUD deltas and detection changes) and persist per-episode combat stats in `memory.py` [Source: trainer.py#MemoryCallback, memory.py]
- [x] T026 [P] [Plan:6.1] Add a Combat tab panel in `dashboard.py` showing rolling accuracy/efficiency/kills [Source: dashboard.py]

### Phase 7: Validation & Polish

- [x] T027 [Plan:7.1] Re-run `benchmarks/benchmark.py --duration 120` and diff against `runs/benchmark_baseline.json`; record deltas in README's performance notes
- [x] T028 [P] [Plan:7.2] Run `python sim/verify_setup.py` and a 5k-step `sim_main.py --no-llm` run to prove the sim/real obs-action contract still matches
- [ ] T029 [Plan:7.2] Manual smoke: real-game 10-minute session with LLM on; verify failsafe corner-dwell, focus auto-pause, and dashboard counters all behave (invariants #4, #5)

## Requirement Mapping

| REQ ID | Description | Plan Items | Implementation Evidence |
|--------|-------------|------------|-------------------------|
| REQ-P1 | YOLO speedup (imgsz/FP16/device) | P3.1 | perception.py (`detect_objects`, `__init__` warmup), config.yaml keys, benchmark JSON |
| REQ-P2 | OCR speedup/hardening | P3.2 | perception.py (`read_hud` allowlist + preprocessing + skip-unchanged) |
| REQ-P3 | Perception off step loop | P3.3 | perception_worker.py, environment.py (`step` reads SharedState) |
| REQ-P4 | LLM token/latency reduction | P3.4 | llm_agent.py (`system` + `cache_control`), lower `usage.input_tokens` in logs |
| REQ-P5 | Remove redundant frame copies | P3.5 | capture.py (`get_frame` ownership rule) |
| REQ-Q1 | Decompose environment.py | P5.1 | env/obs_builder.py, env/rewards.py, env/death_detection.py, slim environment.py |
| REQ-Q2 | Single action-space source | P2.2 | constants.py imported by environment.py and llm_agent.py |
| REQ-Q3 | Config parsed once | P2.1 | config_loader.py; no `yaml.safe_load` in hot modules |
| REQ-Q4 | Remove MultiModelLLM alias | P5.2 | llm_agent.py (alias gone), trainer.py imports ClaudeAdvisor |
| REQ-R1 | Structured LLM outputs | P4.1 | llm_agent.py tool-use schema + validated parsing |
| REQ-R2 | Cached system prompt | P4.2, P3.4 | llm_agent.py system blocks with cache_control |
| REQ-R3 | Deliberate context budget | P4.2 | llm_agent.py chapter inference, guide_data.py ranked tips |
| REQ-F1 | Combat metrics | P6.1 | combat_metrics.py, MemoryCallback wiring, dashboard Combat panel |
| REQ-F2 | Game-profile loader | P2.3 | config_loader.py profile merge; RE4 default unchanged |
| REQ-F3 | Two-tier LLM advisor | P4.3 | llm_agent.py fast/strong routing + config keys |

**Coverage check:** 15/15 requirements mapped to plan items and tasks. No orphan requirements, no orphan plan items.
