# Resident Evil AI Agent — Run Guide

Complete setup and operation guide for the RE4 RL + LLM agent.

---

## 1. Prerequisites

### Hardware
- **GPU:** NVIDIA RTX 3070 or better (8 GB+ VRAM recommended)
- **RAM:** 16 GB+ (32 GB recommended for comfortable multi-app training)
- **Storage:** 10 GB free (model checkpoints + footage recordings)

### Software
- **OS:** Windows 10 or 11 (64-bit)
- **Python:** 3.11.x — use exactly this version for full library compatibility
- **CUDA:** 11.8 or 12.x with matching cuDNN
- **Game:** Resident Evil 4 Remake (Steam) — save your game at the **Village** typewriter before starting

---

## 2. Setup

### Step 1 — Install PyTorch (GPU build first)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*(Replace `cu121` with your CUDA version — e.g. `cu118` for CUDA 11.8)*

### Step 2 — Install all other dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Configure your API key
```bash
copy env.example .env
```
Then open `.env` and fill in your Anthropic key:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```
Get a key at **https://console.anthropic.com** — only this one key is needed.
The LLM is optional; use `--no-llm` to run RL-only without any API key.

### Step 4 — Configure the game
In `config.yaml`, check these two settings match your setup:
- `monitor_index: 2` — change to `1` if RE4 is on your primary monitor
- `capture_region` — leave as 1920×1080 unless you use a different resolution

In-game settings:
- Display Mode → **Borderless Windowed** (required for screen capture)
- Resolution → **1920×1080**
- V-Sync → **OFF** (reduces input latency)
- Brightness → default (extreme values break the health ring OCR)

---

## 3. Running the Agent

```bash
# Standard start (RL + LLM advisor)
python main.py

# Resume training from a checkpoint
python main.py --resume models/checkpoints/re_agent_200000_steps.zip

# RL only — no LLM, no API key required
python main.py --no-llm

# Dashboard only — inspect state without training
python main.py --dashboard-only
```

The Gradio dashboard opens at **http://127.0.0.1:7860**.
Training does **not** start automatically — press **▶ Start Training** in the Controls tab.

---

## 4. Calibrate HUD Regions (important!)

The agent reads health and ammo from specific screen pixel regions. If your
monitor resolution or HUD position differs from 1920×1080, you must calibrate:

```bash
python perception.py
```

This opens a live view with blue rectangles showing the current OCR regions.
Adjust `ocr_regions.ammo` and `health_circle` coordinates in `config.yaml`
until the boxes align with the actual HUD elements.

---

## 5. Dashboard Overview

| Tab | What you see |
|-----|-------------|
| **📺 Live Feed** | Annotated game frame with YOLO bounding boxes, health, ammo |
| **🧠 Agent Brain** | Claude's vision analysis, strategic plan, and tactical decision |
| **📈 Training** | Reward curve, episode length, deaths, curriculum stage |
| **💾 Memory** | Recent LLM decisions logged to the SQLite DB |
| **🕹️ Controls** | Start / Stop training, force LLM call, switch curriculum stage |

---

## 6. How the AI Works

```
Game Frame (1920×1080)
        │
        ▼
ScreenCapture + YOLO + OCR
  → frame, detections, hud
        │
        ├──► RecurrentPPO policy (fast, every 80ms)
        │       Impala CNN → LSTM → action [mv, cam, inter, comb, ev, inv]
        │
        └──► Claude LLM advisor (every ~5s, background thread)
                Sends: frame + HUD + YOLO data
                Returns: vision analysis, strategic plan, tactical suggestion,
                         optional action override, objective for reward shaping
```

### RL Algorithm: RecurrentPPO with Impala CNN
- **RecurrentPPO** — PPO with an LSTM layer for memory (remembers events from
  ~2-3 seconds ago, e.g. "heard a chainsaw but don't see it yet")
- **Impala CNN** — residual network feature extractor (3 stages with skip
  connections), much better than the default NatureCNN for 3D games
- **Frame stacking** — 4 consecutive frames stacked along the channel axis,
  giving the CNN motion information without needing a second LSTM
- **VecNormalize** — rewards normalised to mean=0 std=1 for training stability

### LLM Advisor: Claude (claude-sonnet-4-6)
- Runs in a background thread every 60 RL steps (~5 seconds)
- Sends the current game frame (downscaled to 512px) + HUD data
- Returns structured output: vision / plan / decision / action override / objective
- Only `ANTHROPIC_API_KEY` required — no OpenAI or xAI keys needed

---

## 7. Curriculum Stages

The reward function re-weights automatically based on training progress:

| Stage | Steps | Focus |
|-------|-------|-------|
| **exploration** | 0 – 200k | Survive, move around, find the shotgun |
| **combat** | 200k – 600k | Kill enemies efficiently, conserve ammo |
| **completion** | 600k – 1M | Follow LLM objectives, complete the village |

You can also manually switch stages from the Controls tab at any time.

---

## 8. TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```
Open **http://localhost:6006** for live training curves.

---

## 9. Checkpoints

Saved every 50,000 steps to `models/checkpoints/`.
Resume at any time with `--resume`:
```bash
python main.py --resume models/checkpoints/re_agent_50000_steps.zip
```

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| **Black capture frame** | Wrong monitor — change `monitor_index` in config.yaml |
| **OCR reads wrong numbers** | Run `python perception.py` and recalibrate `ocr_regions` |
| **"ANTHROPIC_API_KEY not set"** | Add it to your `.env` file, or run with `--no-llm` |
| **LLM calls slow / timing out** | Increase `call_cooldown_seconds` in config.yaml |
| **Out of VRAM** | Reduce `obs_frame_size` to `[84, 84]` and `features_dim` to `256` |
| **Game loses focus / inputs stop** | Click the game window; the agent auto-resumes when focused |
| **Failsafe triggered** | Move mouse away from screen corners to re-arm |
| **Training not starting** | Press ▶ Start in the Controls tab of the dashboard |
