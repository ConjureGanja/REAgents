# REAgents — Resident Evil RL + LLM Agent

A hybrid **reinforcement learning + large language model** system that learns to play *Resident Evil 4 Remake* (with a multi-game roadmap for RE2/RE3 Remakes). It trains low-level motor control with **Stable-Baselines3 RecurrentPPO**, uses **Claude vision** for strategic advice, and supports a fast **2D simulation** for sim-to-real transfer before fine-tuning on the real game.

| Mode | Entry point | Needs game window? | Dashboard |
|------|-------------|--------------------|-----------|
| **Real game** | `python main.py` | Yes (Borderless 1080p) | http://127.0.0.1:7860 |
| **Simulation** | `python sim_main.py` | No | http://127.0.0.1:7861 |
| **Full sim pipeline** | `python train_sim_complete.py` | No | (metrics under `runs/`) |

> **Legal / safety note:** This project is for research and personal learning. You must own the game. Input injection can interfere with your PC — use the **mouse-corner failsafe** (hold cursor in any screen corner ~1.5s) to stop the agent immediately.

---

## Table of contents

1. [What this is](#what-this-is)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Requirements](#requirements)
5. [Quick start (conda)](#quick-start-conda)
6. [Configuration](#configuration)
7. [Running the agent](#running-the-agent)
8. [Simulation & sim-to-real](#simulation--sim-to-real)
9. [Observation & action spaces](#observation--action-spaces)
10. [Curriculum & rewards](#curriculum--rewards)
11. [LLM advisor](#llm-advisor)
12. [Dashboard](#dashboard)
13. [Checkpoints, logs & memory](#checkpoints-logs--memory)
14. [Safety & failsafe](#safety--failsafe)
15. [Troubleshooting](#troubleshooting)
16. [Further docs](#further-docs)
17. [Roadmap](#roadmap)
18. [License / credits](#license--credits)

---

## What this is

REAgents is a full training stack for an AI that plays Resident Evil:

- **RL policy (RecurrentPPO + Impala CNN + LSTM)** — decides movement, camera, aim/shoot, dodge, interact, inventory every ~40–80 ms.
- **Perception** — screen capture (`mss`), YOLO11 for enemies/items, EasyOCR for ammo, HSV colour-mask for the RE4 health ring (visible while aiming).
- **Controls** — DirectX-compatible input via `pydirectinput` with human-like stick smoothing and gamepad-style menu reset (Load Game).
- **LLM advisor (Claude)** — optional background vision consult for analysis, plan, tactics, objective text (reward shaping), and optional hard action overrides.
- **Simulation** — abstract top-down village siege with identical obs/action spaces, domain randomization, multi-process workers, eval metrics, and transfer-ready checkpoints.
- **Dashboards** — Gradio 5 UIs for live feed, training curves, LLM brain, memory, and start/stop controls.

Training does **not** auto-start: open the dashboard and press **Start Training** in the Controls tab.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Real-game mode (main.py)                          │
│                                                                           │
│  Game (Borderless 1080p)                                                  │
│       │ mss capture                                                       │
│       ▼                                                                   │
│  Perception (YOLO + OCR + health ring) ──► SharedState                    │
│       │                                        ▲                          │
│       ▼                                        │                          │
│  ResidentEvilEnv (Gymnasium)                   │                          │
│   obs: frame stack + HUD                       │                          │
│       │                                        │                          │
│       ▼                                        │                          │
│  RecurrentPPO (Impala CNN → LSTM → MultiDiscrete action)                  │
│       │                                        │                          │
│       ▼                                        │                          │
│  GameControls (smoothed sticks + buttons) ─────┘                          │
│                                                                           │
│  Claude advisor thread (every ~N steps) → plan / override / objective     │
│  Memory (SQLite queue) · Recorder (optional MP4) · Focus monitor          │
│  Gradio dashboard (main thread)                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    Simulation mode (sim_main / train_sim_complete)        │
│                                                                           │
│  N × SubprocVecEnv workers                                                │
│    DomainRandomization → SimResidentEvilEnv → SimGameState (2D village)   │
│                                                                           │
│  Same obs/action contract as real env → checkpoint transfers to main.py   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Thread model (real game)

| Thread | Role |
|--------|------|
| **Main** | Gradio dashboard (blocking) |
| **TrainerThread** | `RETrainer.train()` → `model.learn()` |
| **LLMThread** | Fire-and-forget Claude consult (skipped if call in-flight) |
| **MemoryThread** | Async SQLite writes |
| **RecorderThread** | Optional `footage/*.mp4` |
| **FocusMonitor** | Pauses input when RE4 loses foreground |
| **FailsafeWatchdog** | Corner-dwell emergency stop |

Modules talk only through **`SharedState`** (`shared_state.py`). Dashboard snapshots are lock-copied; never mutate the snapshot object.

### Stopping training

`shared.stop_requested = True` → `DashboardCallback._on_step()` returns `False` → SB3 stops. Sources: dashboard **Stop**, failsafe corner dwell, or process kill.

---

## Repository layout

```
REAgents/
├── main.py                 # Real-game entry
├── sim_main.py             # Simulation entry + dashboard
├── train_sim_complete.py   # Full sim pipeline (train/eval/best/transfer)
├── trainer.py              # Real-game SB3 trainer + curriculum callbacks
├── sim_trainer.py          # Sim SubprocVecEnv trainer
├── environment.py          # Real Gymnasium env (capture + reward)
├── sim_environment.py      # Sim Gymnasium env (identical spaces)
├── sim_game_state.py       # 2D village physics / combat sim
├── feature_extractor.py    # Impala ResNet CNN + HUD MLP
├── capture.py              # Screen grab (mss, DPI-aware)
├── perception.py           # YOLO + OCR + health ring
├── controls.py             # Input injection + smoothing + menu reset
├── llm_agent.py            # Claude advisor + consultant
├── guide_data.py           # Walkthrough / tips for LLM context
├── memory.py               # SQLite episode / step / LLM logs
├── shared_state.py         # Thread-safe shared bus
├── dashboard.py            # Real-game Gradio UI (:7860)
├── sim_dashboard.py        # Sim Gradio UI (:7861)
├── recorder.py             # Optional episode video
├── logger_config.py        # Logging setup
├── config.yaml             # All tunables (game, RL, LLM, curriculum, sim)
├── requirements.txt
├── env.example             # Copy to .env
├── sim/                    # Domain randomization, sim2real bridge, metrics
│   ├── domain_randomization.py
│   ├── sim2real_bridge.py
│   ├── training_metrics.py
│   ├── play_manual.py      # Drive sim with keyboard
│   ├── verify_setup.py     # Pre-flight checks
│   └── pygame_renderer.py
├── models/checkpoints/     # (gitignored) policy .zip + VecNormalize .pkl
├── data/                   # (gitignored) SQLite DB
├── logs/                   # (gitignored) app + TensorBoard
├── footage/                # (gitignored) recorded episodes
├── runs/                   # (gitignored) sim eval CSVs / learning curves
└── docs (various .md)      # Guides, troubleshooting, roadmap
```

---

## Requirements

### Hardware

- **GPU:** NVIDIA RTX 3070 or better (8 GB+ VRAM recommended)
- **RAM:** 16 GB+ (32 GB comfortable for multi-worker sim)
- **Storage:** ~10 GB free for checkpoints and recordings
- **OS:** Windows 10/11 (input stack is Windows-oriented)

### Software

- **Python 3.11.x** (recommended; SB3/contrib compatibility)
- **CUDA 11.8+** (or 12.x) with matching cuDNN for GPU PyTorch
- **Resident Evil 4 Remake** (Steam) for real-game mode
- **Conda/Miniconda** (optional but recommended)

### API keys

| Key | Required? | Purpose |
|-----|-----------|---------|
| `ANTHROPIC_API_KEY` | Only if using LLM | Claude vision advisor |
| — | No | Run RL-only with `--no-llm` (no API key) |

---

## Quick start (conda)

This matches a typical local layout: env at  
`C:\Users\wizar\miniconda3\envs\re_agent` and project at `V:\AI-ML\REAgents`.

### 1. Activate env and go to the project

```powershell
conda activate C:\Users\wizar\miniconda3\envs\re_agent
cd V:\AI-ML\REAgents
python --version   # expect 3.11.x
```

### 2. Install PyTorch (GPU) first, then dependencies

```powershell
# Adjust cu121 → cu118 if you use CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> Install GPU torch **before** `requirements.txt` so you do not silently get a CPU-only wheel from PyPI.

### 3. Environment file

```powershell
copy env.example .env
# Edit .env and set ANTHROPIC_API_KEY if you want the LLM advisor
```

### 4. Game settings (real mode only)

In RE4:

- Display Mode → **Borderless Windowed**
- Resolution → **1920×1080**
- V-Sync → **OFF**
- Brightness → default (extreme values break health-ring detection)

Save at the **Village typewriter** (Chapter 1 siege). Resets load the most recent save via the pause menu.

In `config.yaml`, set:

```yaml
game_settings:
  monitor_index: 2   # 1 = primary, 2 = secondary (where the game is)
  window_title_substring: "RESIDENT EVIL 4"
```

### 5. Launch

```powershell
# Simulation (fastest way to train first)
python sim_main.py --no-llm

# Or full sim pipeline with eval + best-model export
python train_sim_complete.py --timesteps 200_000 --eval-every 25_000 --no-llm

# Real game (RE4 running, village save loaded)
python main.py --no-llm
# With Claude:
python main.py
```

Open the dashboard URL printed in the console, then press **▶ Start Training**.

---

## Configuration

Almost everything is in **`config.yaml`** (no code edits required for most tuning).

| Section | What it controls |
|---------|------------------|
| `game_profiles` | Scaffolding for RE2/RE3/RE4 profiles (RE4 active today) |
| `game_settings` | Monitor index, window title, capture region/FPS |
| `controls.smoothing` | Stick low-pass filters (human-like motion) |
| `perception` | YOLO model, OCR/health pixel boxes, HSV health colours |
| `rl_hyperparameters` | RecurrentPPO vs PPO, frame size/stack, LR, KL, death detection, anti-freeze |
| `llm_settings` | Claude model, consult interval, cooldown, override hold |
| `storage` | DB, checkpoint, log, TensorBoard paths |
| `reset` | Pause-menu Load Game navigation timings |
| `training` | Total timesteps, checkpoint/eval frequency |
| `curriculum` | Stage thresholds and reward weights |
| `simulation` | Workers, map, enemy waves, physics, vision FOV |

### Calibrate HUD regions

OCR and health boxes must match **your** 1920×1080 layout:

```powershell
python perception.py
```

Adjust `perception.ocr_regions.ammo` and `perception.health_circle` (`[y1, x1, y2, x2]` in full-frame pixels) until the overlays sit on the aim radial HUD.

---

## Running the agent

### Real game (`main.py`)

```powershell
python main.py                                              # RL + LLM
python main.py --resume models/checkpoints/re_agent_final.zip
python main.py --no-llm                                     # RL only
python main.py --dashboard-only                             # UI only
python main.py --record                                     # enable MP4 footage
python main.py --config config.yaml --log-level DEBUG
```

Dashboard: **http://127.0.0.1:7860**

### Simulation (`sim_main.py`)

```powershell
python sim_main.py                                          # config defaults (often 8 workers, 1M steps)
python sim_main.py --workers 4 --timesteps 500_000
python sim_main.py --resume models/checkpoints/sim_re_agent_final.zip
python sim_main.py --no-llm
python sim_main.py --dashboard-only --port 7861
```

Dashboard: **http://127.0.0.1:7861**

> On Windows, `if __name__ == "__main__": freeze_support()` is required so SubprocVecEnv does not spawn recursively.

### Complete sim pipeline (`train_sim_complete.py`)

One script for train + periodic clean eval + growth plots + best/transfer checkpoints:

```powershell
python train_sim_complete.py
python train_sim_complete.py --timesteps 200_000 --eval-every 25_000 --no-llm
python train_sim_complete.py --resume models/checkpoints/sim_re_agent_500000_steps.zip
python train_sim_complete.py --no-randomization   # debug only — bad for transfer
```

**Outputs:**

| Path | Meaning |
|------|---------|
| `models/checkpoints/sim_re_agent_*_steps.zip` | Periodic checkpoints |
| `models/checkpoints/sim_re_agent_final.zip` | Last step |
| `models/checkpoints/sim_re_agent_best.zip` | Best eval reward |
| `models/checkpoints/sim_re_agent_for_real.zip` | Transfer-ready (= best) |
| `runs/<name>/eval_log.csv` | Eval metrics over time |
| `runs/<name>/learning_curve.png` | Auto plot |
| `runs/<name>/best_checkpoint.txt` | Best step + reward |

### TensorBoard

```powershell
tensorboard --logdir logs/tensorboard
```

Open http://localhost:6006

### Manual sim / pre-flight

```powershell
python sim/verify_setup.py
python sim/play_manual.py
```

---

## Simulation & sim-to-real

Real-game training is ~12–20 decisions/sec. With 8 parallel sim workers you can reach tens of thousands of steps/sec — orders of magnitude faster.

**Transfer recipe:**

1. Train in sim with **domain randomization** (visual/physics noise so the CNN learns invariants, not exact sim pixels).
2. Pick the **best eval** policy (`sim_re_agent_for_real.zip`).
3. Fine-tune on the real game:

```powershell
python main.py --resume models/checkpoints/sim_re_agent_for_real.zip
```

Obs/action spaces match by design (`sim/sim2real_bridge.py` validates this). See **`SIM_TO_REAL_TUTORIAL.md`** for a step-by-step walkthrough.

**Sim village snapshot:**

- 50×50 top-down map (barn, house, well, fence)
- Enemy waves, aim cone, ammo, dodge cooldown, “bell rings” success at max steps
- Symbolic RGB frames + same 7-D HUD vector as real env

---

## Observation & action spaces

These contracts are shared by **real** and **sim** envs (required for transfer).

### Observation (`Dict`)

| Key | Shape / dtype | Meaning |
|-----|---------------|---------|
| `frame` | `(H, W, C×stack)` uint8, default **84×84×12** (RGB×4) | Stacked frames for motion |
| `hud` | `(7,)` float32 | health, clip, reserve, enemies, in_combat, time_pressure, ammo_delta |

`health_pct` from perception can be **`None`** when the ring is not visible (not aiming). All consumers must treat that as *unknown*, never as “dead.”

### Action (`MultiDiscrete([9, 5, 2, 4, 3, 2])`)

| Index | Dim | Values |
|-------|-----|--------|
| 0 | Movement | stop, N/S/E/W and diagonals (9) |
| 1 | Camera | none / left / right / up / down |
| 2 | Interact | none / interact |
| 3 | Combat | none / aim / shoot / aim+shoot |
| 4 | Evasion | none / dodge / sprint |
| 5 | Inventory | none / toggle |

### Feature extractor

`feature_extractor.py` implements an **Impala-style ResNet CNN** on `frame`, a small MLP on `hud`, concat → `features_dim` (default 512) → LSTM (RecurrentPPO) or policy head (PPO).

---

## Curriculum & rewards

Three stages (thresholds and weights live in `config.yaml`):

| Stage | Default until | Focus |
|-------|---------------|--------|
| **exploration** | 200k steps | Survive, move, explore, find tools |
| **combat** | 600k steps | Efficient kills, ammo discipline |
| **completion** | 1M steps | Objectives / LLM-shaped goals |

Reward components include survival, exploration diversity, combat (aim→shoot, kills), objective progress, and penalties for freeze / inventory spam / damage. Stage can also be switched from the dashboard.

---

## LLM advisor

Implemented in `llm_agent.py` as a single **Claude** vision call (`claude-sonnet-4-6` by default):

- Vision analysis of the current frame
- Strategic plan (~30 s horizon)
- Tactical decision
- Optional **action override** (hard-steers RL for `override_hold_steps`)
- Objective phrase for reward shaping

`guide_data.py` injects walkthrough/tips/merchant context into the system prompt. Calls are rate-limited (`llm_every_n_steps`, `call_cooldown_seconds`) with `tenacity` retries.

```powershell
python main.py --no-llm   # pure RL; no Anthropic key needed
```

---

## Dashboard

### Real game (`dashboard.py` — :7860)

| Tab | Contents |
|-----|----------|
| Live Feed | Annotated frame, YOLO boxes, HUD |
| Agent Brain | Claude vision / plan / tactics |
| Training | Rewards, lengths, deaths, stage |
| Memory | Recent LLM decisions from SQLite |
| Controls | Start / Stop / pause / force LLM / stage |

### Simulation (`sim_dashboard.py` — :7861)

Worker grid snapshots, steps/sec, rewards, same control pattern for sim training.

---

## Checkpoints, logs & memory

| Artifact | Location | Git? |
|----------|----------|------|
| Policy `.zip` + VecNormalize `.pkl` | `models/checkpoints/` | Ignored |
| SQLite memory | `data/re_agent_memory.db` | Ignored |
| App + TensorBoard logs | `logs/` | Ignored |
| Episode videos | `footage/` | Ignored |
| Sim eval runs | `runs/` | Ignored |
| YOLO weights | `yolo11n.pt` | Ignored (download via Ultralytics on first use if missing) |

Resume examples:

```powershell
python main.py --resume models/checkpoints/re_agent_final.zip
python sim_main.py --resume models/checkpoints/sim_re_agent_best.zip
```

Or set `training.resume_path` in `config.yaml`.

---

## Safety & failsafe

- **Corner dwell (~1.5 s):** hold mouse in any **corner** of the virtual desktop → sets `stop_requested`.
- **Focus auto-pause:** when RE4 loses foreground, input is paused so browser/dashboard clicks do not leak keys into other apps.
- **Release-all** before menu navigation so stuck sticks do not scroll menus.
- Do not leave the agent unattended on a machine you care about without understanding the failsafe.

---

## Troubleshooting

Quick hits (full guide: **`TROUBLESHOOTING.md`**):

| Symptom | Things to try |
|---------|----------------|
| Status stuck Idle | Press **Start Training** in Controls |
| Black capture | Wrong `monitor_index`; not Borderless; test `capture.py` / mss monitors |
| Phantom deaths | Health ring misread — raise `health_min_visible_frac`, confirm death-screen path |
| CUDA OOM | Lower `batch_size` / `n_steps` / `obs_frame_size` |
| Menu reset fails | Tune `reset.nav_down_count` / confirm-yes direction for your menu layout |
| Policy stops shooting | Check `target_kl`, LR schedule; inspect eval curves for collapse |
| DPI offset capture | `main.py` sets per-monitor DPI awareness; run without DPI scaling hacks if needed |

```powershell
# Smoke-test capture
python -c "from capture import ScreenCapture; import yaml; ..."
```

---

## Further docs

| File | Topic |
|------|--------|
| `LAUNCH_GUIDE.md` | First real-game training session |
| `RUN_GUIDE.md` | Setup, run, dashboard, curriculum |
| `SIM_TO_REAL_TUTORIAL.md` | End-to-end sim → transfer |
| `TROUBLESHOOTING.md` | Common failures |
| `QUICK_START_COMBAT_STAGE.md` | Combat curriculum focus |
| `NEXT_STEPS.md` | Historical status / recommendations |
| `RE_MULTI_GAME_ROADMAP.md` | RE2/RE3 profile plan |
| `CLAUDE.md` | Contributor / agent coding guide |
| `RE_AI_Agent_Spec.pdf` / `.docx` | Design specification |

---

## Roadmap

Long-term goal: **one policy stack, multiple Resident Evil Remakes** via `game_profiles` in `config.yaml` (window title, health style, guide module, reset sequence). RE4 is the working baseline; RE2/RE3 need HUD calibration (vignette/ECG health), guide modules, and optional sim maps. Observation and action spaces stay fixed so the policy does not fork per game.

See **`RE_MULTI_GAME_ROADMAP.md`**.

---

## License / credits

- **Author / GitHub:** [ConjureGanja/REAgents](https://github.com/ConjureGanja/REAgents)
- Built with: Stable-Baselines3, sb3-contrib (RecurrentPPO), Gymnasium, Ultralytics YOLO, EasyOCR, Anthropic Claude, Gradio, PyTorch.
- Game content © Capcom. This project is an independent research tool and is not affiliated with Capcom.

---

## Contributing

1. Use Python 3.11 and the project conda env when possible.
2. Keep real/sim **observation and action spaces identical**.
3. Prefer config-driven tuning over hardcoding reward weights or HUD boxes.
4. Do not commit `.env`, checkpoints, logs, footage, or large binaries (see `.gitignore`).
5. Document non-obvious RL or perception trade-offs in comments or the relevant guide.

```powershell
# Suggested local workflow
conda activate C:\Users\wizar\miniconda3\envs\re_agent
cd V:\AI-ML\REAgents
git checkout -b feature/my-change
# ... edit, test with sim_main or train_sim_complete smoke run ...
git push -u origin HEAD
# open a PR against main
```
