# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A hybrid RL + LLM agent that plays Resident Evil 4 Remake in real time. It combines:
- **Stable Baselines3 RecurrentPPO** for low-level motor control (movement, camera, combat)
- **Claude claude-sonnet-4-6 vision advisor** (`llm_agent.py`) for strategic/tactical reasoning via a single API call
- **Gradio 5 dashboard** for live monitoring and control

Requires Python 3.11.x, Windows 10/11, CUDA 11.8+, and the game running in Borderless Windowed mode.

## Running the Agent

```bash
# Real-game mode
python main.py                                              # fresh start
python main.py --resume models/checkpoints/re_agent_200000_steps.zip
python main.py --no-llm                                    # RL only
python main.py --dashboard-only                            # dashboard without training

# Simulation mode (no game window required — ~400× faster per worker)
python sim_main.py                                         # 8 workers, 1M steps
python sim_main.py --workers 4 --timesteps 500_000
python sim_main.py --resume models/checkpoints/sim_re_agent_final.zip

# Full sim-to-real pipeline (training + eval + best-model export)
python train_sim_complete.py
python train_sim_complete.py --timesteps 200_000 --eval-every 25_000 --no-llm
```

Real-game dashboard: **http://127.0.0.1:7860** — sim dashboard: **http://127.0.0.1:7861**. Training does not start until you press Start in the Controls tab.

```bash
tensorboard --logdir logs/tensorboard
```

## Setup

```bash
pip install -r requirements.txt
copy env.example .env   # then fill in API keys
```

Only `ANTHROPIC_API_KEY` is required (for the LLM advisor). The old multi-model design using `OPENAI_API_KEY` and `XAI_API_KEY` has been replaced by a single Claude call. Run with `--no-llm` to skip the LLM entirely.

## Architecture

### Thread model (real-game mode)
- **Main thread** — Gradio dashboard (blocking)
- **TrainerThread** — `RETrainer.train()` runs `model.learn()`, blocks until done or stopped
- **LLMThread** — fire-and-forget via `LLMConsultant`; skipped if a call is already in-flight
- **MemoryThread** — background SQLite writer inside `MemorySystem`
- **RecorderThread** — writes `footage/*.mp4` via `GameplayRecorder` (disabled with `--no-record`)
- **FocusMonitor** — polls foreground window at 10 Hz; auto-pauses training when RE4 loses focus so browser interaction doesn't send stray inputs

### Simulation mode threading
`sim_main.py` / `sim_trainer.py` use **SubprocVecEnv** — each worker is a separate OS process running `SimResidentEvilEnv`. The `if __name__ == "__main__": freeze_support()` guard in `sim_main.py` is mandatory on Windows to prevent fork-bomb on process spawn.

### Data flow
All modules communicate through **`SharedState`** (a thread-safe dataclass in `shared_state.py`). Nothing holds direct references to other modules. The dashboard calls `shared.get_snapshot()` which copies under a lock — never mutate the snapshot.

### Stopping training
`shared.stop_requested = True` → `DashboardCallback._on_step()` returns `False` → SB3 halts. The Stop button sets this flag; moving the mouse to any screen corner also triggers it via `FailsafeWatchdog`.

### LLM advisor (`llm_agent.py`)
Single `ClaudeAdvisor` call using `claude-sonnet-4-6` with vision. One call produces four outputs written to `SharedState`: vision analysis, strategic plan, tactical decision, and optional action override (a 6-tuple that hard-steers the RL agent for one step). Uses `tenacity` for retries. `guide_data.py` provides walkthrough, tips, and merchant upgrade context injected into the system prompt.

### Observation & action spaces (`environment.py` / `sim_environment.py`)
Both environments expose **identical** spaces — this is the contract that makes sim-to-real transfer work:
- **Obs:** Dict with `"frame"` (84×84×12 uint8 — RGB × 4-frame stack) and `"hud"` (7-element float32 vector: health, clip, reserve, enemies, in_combat, time-pressure, ammo-delta). `hud["health_pct"]` from perception can be **None** (ring not visible) — every consumer must guard for it.
- **Action:** `MultiDiscrete([9, 5, 2, 4, 3, 2])` — movement / camera / interact / combat / evasion / inventory

### Feature extractor (`feature_extractor.py`)
Impala ResNet CNN processes the `"frame"` observation before the LSTM. Configured via `features_dim` in `config.yaml`.

### Curriculum (`trainer.py`, `config.yaml`)
Three stages — `exploration` → `combat` → `completion` — each with different reward weights. `CurriculumCallback` advances stages automatically at configured timestep thresholds. The current stage can also be switched manually from the dashboard.

### Memory (`memory.py`)
SQLite at `data/re_agent_memory.db`. All writes go through a queue to avoid blocking the training loop. Schema: episode records, per-step records, LLM decision logs.

### Sim-to-real transfer (`sim/`)
- `sim/domain_randomization.py` — wraps `SimResidentEvilEnv` with visual/physics perturbations each episode so the policy learns invariant features
- `sim/sim2real_bridge.py` — `SimToRealValidator` verifies obs/action space compatibility between sim and real envs before transfer; `convert_sim_checkpoint_for_real()` strips sim-only state from saved policies
- `sim/play_manual.py` — lets you manually drive the sim env with keyboard to sanity-check it before training
- `sim/verify_setup.py` — pre-flight checks for the simulation pipeline
- `train_sim_complete.py` outputs: `sim_re_agent_final.zip`, `sim_re_agent_best.zip`, `sim_re_agent_for_real.zip` (transfer-ready), `runs/<timestamp>/eval_log.csv`, `runs/<timestamp>/learning_curve.png`

## Key Configuration (`config.yaml`)

| Section | Important fields |
|---------|-----------------|
| `game_settings` | `monitor_index` (1=primary, 2=secondary), `window_title_substring` |
| `perception.ocr_regions` | Pixel coords for ammo/health HUD — **must be calibrated to your monitor** |
| `rl_hyperparameters.algorithm` | `"RecurrentPPO"` or `"PPO"` |
| `rl_hyperparameters.obs_frame_size` | Default `[84, 84]`; `[128, 128]` improves combat at ~15% speed cost |
| `llm_settings` | `llm_every_n_steps`, `call_cooldown_seconds` |
| `training.resume_path` | Alternative to `--resume` flag |
| `curriculum.stages` | Timestep thresholds and reward weights per stage |
| `simulation.n_workers` | Parallel workers for sim mode (default 8) |

OCR regions use `[y1, x1, y2, x2]` in full 1920×1080 pixel space. If capture returns a black frame, check `monitor_index`.

## Perception (`perception.py`)

- YOLO11 model at `yolo11n.pt` detects enemies and items
- EasyOCR reads ammo counter from the HUD region
- Health is parsed from a colour-mask of the health ring (`health_colors` HSV ranges in config)

## Controls (`controls.py`)

Uses `pydirectinput` for DirectX-compatible input injection. **Mouse corner = immediate failsafe kill** (custom watchdog, fires within 50 ms). F9 triggers a manual quick-load sequence (navigates RE4's pause menu programmatically via `reset` config).
