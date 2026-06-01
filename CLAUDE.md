# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A hybrid RL + LLM agent that plays Resident Evil 4 Remake in real time. It combines:
- **Stable Baselines3 RecurrentPPO** for low-level motor control (movement, camera, combat)
- **LangGraph pipeline** (GPT-4o → Claude Sonnet → Grok-2-Vision) for strategic/tactical reasoning
- **Gradio 5 dashboard** for live monitoring and control

Requires Python 3.11.x, Windows 10/11, CUDA 11.8+, and the game running in Borderless Windowed mode.

## Running the Agent

```bash
# Fresh start
python main.py

# Resume from checkpoint
python main.py --resume models/checkpoints/re_agent_200000_steps.zip

# RL only (no LLM calls)
python main.py --no-llm

# Dashboard without training
python main.py --dashboard-only
```

Dashboard opens at **http://127.0.0.1:7860**. Training does not start until you press Start in the Controls tab.

```bash
# View training metrics
tensorboard --logdir logs/tensorboard
```

## Setup

```bash
pip install -r requirements.txt
copy env.example .env   # then fill in API keys
```

Required keys in `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`

## Architecture

### Thread model
- **Main thread** — Gradio dashboard (blocking)
- **TrainerThread** — `RETrainer.train()` runs `model.learn()`, blocks until done or stopped
- **LLM threads** — fire-and-forget via `LLMConsultant`; skipped if a call is already in-flight
- **MemoryThread** — background SQLite writer inside `MemorySystem`

### Data flow
All modules communicate through **`SharedState`** (a thread-safe dataclass). Nothing holds direct references to other modules. The dashboard calls `shared.get_snapshot()` which copies under a lock — never mutate the snapshot.

### Stopping training
`shared.stop_requested = True` → `DashboardCallback._on_step()` returns `False` → SB3 halts. The Stop button in the dashboard sets this flag.

### LLM pipeline (`llm_agent.py`)
Three-node LangGraph graph: `vision_node` (GPT-4o) → `plan_node` (Claude Sonnet) → `tactical_node` (Grok). Results are written to `shared.gpt_analysis`, `shared.claude_plan`, `shared.grok_tactical`. Grok uses the OpenAI-compatible xAI endpoint (`https://api.x.ai/v1`).

### Observation & action spaces (`environment.py`)
- **Obs:** Dict with `"frame"` (84×84×1 uint8 grayscale) and `"hud"` (6-element float32 vector)
- **Action:** `MultiDiscrete([9, 5, 2, 4, 3, 2])` — movement / camera / interact / combat / evasion / inventory

### Curriculum (`trainer.py`, `config.yaml`)
Three stages — `exploration` → `combat` → `completion` — each with different reward weights. `CurriculumCallback` advances stages automatically at configured timestep thresholds. The current stage can also be switched manually from the dashboard.

### Memory (`memory.py`)
SQLite at `data/re_agent_memory.db`. All writes go through a queue to avoid blocking the training loop. Schema: episode records, per-step records, LLM decision logs.

## Key Configuration (`config.yaml`)

| Section | Important fields |
|---------|-----------------|
| `game_settings` | `monitor_index` (1=primary, 2=secondary) |
| `perception.ocr_regions` | Pixel coords for ammo/health HUD — **must be calibrated to your monitor** |
| `rl_hyperparameters.algorithm` | `"RecurrentPPO"` or `"PPO"` |
| `llm_settings` | `llm_every_n_steps`, `call_cooldown_seconds` |
| `training.resume_path` | Alternative to `--resume` flag |
| `curriculum.stages` | Timestep thresholds and reward weights per stage |

OCR regions use `[y1, x1, y2, x2]` in full 1920×1080 pixel space. If capture returns a black frame, check `monitor_index`.

## Perception (`perception.py`)

- YOLO11 model at `yolo11n.pt` detects enemies and items
- EasyOCR reads ammo counter from the HUD region
- Health is parsed from a colour-mask of the health ring (`health_colors` HSV ranges in config)

## Controls (`controls.py`)

Uses `pydirectinput` for DirectX-compatible input injection. **Mouse corner = immediate failsafe kill** (pyautogui built-in). F9 triggers a manual quick-load sequence (navigates RE4's pause menu programmatically via `reset` config).
