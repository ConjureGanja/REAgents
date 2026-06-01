# RE4 AI Agent — First Training Session Guide

Everything you need to launch, monitor, and understand your agent's progress.

---

## Pre-Launch Checklist (do these once)

### 1. Install PyTorch with GPU support (do this FIRST, before requirements.txt)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ Common mistake: running `pip install -r requirements.txt` before PyTorch often installs
> the CPU-only torch from PyPI. Always do the GPU install first.
> Replace `cu121` with your CUDA version — check with `nvcc --version`.
> For CUDA 11.8: use `cu118`.

### 2. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

```bash
copy env.example .env
```

Open `.env` and fill in your Anthropic key:

```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

Get a key at https://console.anthropic.com — only this key is needed.
To run completely without the LLM: add `--no-llm` to any command below.

### 4. Set your monitor index in config.yaml

```yaml
game_settings:
  monitor_index: 2   # ← change to 1 if RE4 is on your primary screen
```

### 5. Set RE4 in-game display settings

- Display Mode → **Borderless Windowed** (required for screen capture)
- Resolution → **1920×1080**
- V-Sync → **OFF**
- Brightness → default (extreme values break the health ring colour detection)

### 6. Save your game at the Village typewriter

The agent resets by loading your most-recent save via the pause menu.
Make sure you have a save at the start of the village siege (Chapter 1, right after
the opening cabin). This is where training starts and is the optimal starting point.

---

## Running the Agent — Step by Step

### Step 1: Open a terminal in the REAgents folder

```bash
cd path\to\REAgents
```

### Step 2: Start RE4, load your village save, and leave Leon standing there

The agent doesn't launch RE4 for you — it captures whatever is on the screen.
Leave the game running and in focus before you start the agent.

### Step 3: Launch the agent

```bash
# Standard run (RL + Claude LLM advisor)
python main.py

# First time or no API key? Run RL-only — no API key needed
python main.py --no-llm

# Resume from a checkpoint
python main.py --resume models/checkpoints/re_agent_50000_steps.zip
```

### Step 4: Open the dashboard

Navigate to **http://127.0.0.1:7860** in your browser.

> ⚠️ Common mistake: clicking the browser window causes the agent to detect focus-loss
> and auto-pause. This is intentional — it protects your browser from stray keystrokes.
> The agent will auto-resume when you click back on the game window.
> You can also toggle the Pause / Resume buttons in the Controls tab manually.

### Step 5: Press ▶ Start Training in the Controls tab

Training does NOT start automatically — you must press the button.
Watch the console: you'll see `Training started.` and the first reset sequence begin.

---

## What to Watch During Training

### The Live Feed tab (📺)

- **Frame view**: shows the annotated game frame in near real-time (1 s refresh)
- **YOLO boxes**: green rectangles around detected enemies/items — if you see no boxes
  during active combat, the YOLO model may need retraining on RE4-specific classes
- **HUD overlay**: health %, ammo clip / reserve in the top-left corner
- **Status bar**: shows 🟢 Training / 🟡 Paused / 🔴 Idle + focus indicator

### The Training tab (📈)

This is your primary progress monitor. Look at:

| Metric | What it means | Healthy sign |
|--------|---------------|--------------|
| **Episode Reward** | Total score for the last episode | Rising trend over 10-20 episodes |
| **Reward curve** | Per-episode total over time | Upward slope, even if noisy |
| **Episode Length** | How long each episode lasted | Getting longer (agent surviving longer) |
| **Death Count** | Total deaths across all training | Rate slows down over time |

> The rewards will be very noisy early on — single episodes can swing wildly.
> Look at the **trend over 20-30 episodes**, not individual episodes.

### The Agent Brain tab (🧠)

Shows Claude's real-time analysis (if LLM is enabled):

- **Claude Vision Analysis**: what Claude sees on screen — should mention enemies, health, shotgun house
- **Claude Strategic Plan**: the 30-second goal — early training should show "get to shotgun house"
- **Claude Tactical Decision**: the immediate recommended action

### The Memory tab (💾)

Shows a log of Claude's decisions stored in the SQLite database.
Useful for reviewing what the LLM was thinking during important moments.

### TensorBoard (detailed charts)

```bash
tensorboard --logdir logs/tensorboard
```

Open **http://localhost:6006** in another browser tab. Key charts:

- `train/reward` — episode reward over time (most important)
- `train/ep_len_mean` — average episode length
- `train/value_loss` — should decrease and stabilise (indicates the value network is learning)
- `train/entropy_loss` — should stay above ~-0.5 (if it collapses to 0, the agent is stuck)
- `train/approx_kl` — should stay below ~0.05 (high KL = unstable policy updates)

---

## How to Tell the Agent is Getting Better

### Phase 1: First 0–10,000 steps — Chaotic

**What you'll see**: Leon spinning in place, walking into walls, dying immediately to the
first Ganado, completely ignoring the shotgun house. Rewards are negative or near zero.

**This is normal.** Think of it like a newborn learning to walk — total chaos at first.
The agent is randomly exploring the action space and building its first reward signals.

**Green flags at this stage**:
- Any movement at all (not standing completely still)
- Occasional survival for 20+ seconds
- Episode rewards occasionally going slightly positive

### Phase 2: 10,000–50,000 steps — First Signs of Learning

**What you'll see**: Leon starts surviving longer, maybe 30–60 seconds per episode.
He'll probably find the "run north" behaviour since it has the highest exploration reward.
Deaths per episode start decreasing.

**Green flags**:
- Average episode length creeping up (was 10 s, now 30 s)
- Reward curve has an upward slope (even if noisy)
- Agent is moving — not getting stuck in one spot

### Phase 3: 50,000–200,000 steps — Exploration Stage Learning

**What you'll see**: Agent learns to move toward the shotgun house with some regularity.
It may start attempting combat, though inaccurately. You may see it survive the full
3-minute bell ring occasionally.

**Green flags**:
- Some episodes lasting 60+ seconds
- TensorBoard value_loss decreasing and stabilising
- Occasional kills (combat reward spikes)

### Phase 4: 200,000+ steps — Combat Stage

The curriculum automatically advances to the `combat` stage, increasing combat reward weights.
You should see more deliberate aiming, fewer hip-fires, and more efficient use of ammo.

---

## Common Problems and Fixes

| Problem | What you see | Fix |
|---------|-------------|-----|
| **Black capture frame** | Solid black in Live Feed | Change `monitor_index` in config.yaml (1↔2) |
| **Agent not moving** | Leon standing still every episode | Check the Controls tab — might still be paused |
| **OCR reads wrong ammo** | Ammo shows `0/0` even when loaded | Run `python perception.py`, recalibrate `ocr_regions.ammo` |
| **"ANTHROPIC_API_KEY not set"** | Warning in console, no LLM analysis | Add key to `.env`, or use `--no-llm` |
| **Agent inputs going to browser** | Typing/clicking in the browser | Agent auto-pauses on focus loss — click the game window to resume |
| **Reward stuck near 0** | Flat line in training chart | Check `python tensorboard` entropy_loss — if near 0, training may need restart |
| **Out of VRAM** | CUDA OOM error | Set `obs_frame_size: [84, 84]` and `features_dim: 256` in config.yaml |
| **Training not starting** | Nothing happening after launch | Press ▶ Start Training in the Controls tab — it doesn't auto-start |
| **Failsafe triggered** | Agent suddenly stops | Your mouse hit a screen corner — restart with `python main.py` |
| **Episodes too short (<5s)** | Agent dying instantly | Verify RE4 is running at village save point, health ring calibrated |

---

## Resuming After a Break

Checkpoints are saved every 50,000 steps to `models/checkpoints/`. To resume:

```bash
python main.py --resume models/checkpoints/re_agent_50000_steps.zip
```

> **VecNormalize stats**: When resuming, the agent also needs the reward normalisation stats
> (`_vecnorm.pkl` file saved alongside the `.zip`). If you see rewards that look wildly
> different after resuming, the `.pkl` may be missing — in that case, start fresh or
> set `use_vec_normalize: false` in config.yaml for the resumed run.

---

## What "Getting Better" Looks Like — Summary

Think of it in three milestones, not absolute numbers:

**Milestone 1** — Agent survives more than 60 seconds consistently.
*Proof: episode_length_mean > 60 in TensorBoard.*

**Milestone 2** — Agent reaches the shotgun house in most episodes.
*Proof: item_pickup reward spikes visible in step_reward_history; ammo reserve sometimes jumps.*

**Milestone 3** — Agent rings the bell (survives 3 full minutes).
*Proof: episode_length_mean > 180 seconds; reward curve has a clear upward trend.*

Each milestone typically takes 50,000–100,000 additional steps with the current setup.
Full competence at the village (Chapter 1) usually emerges around 300,000–500,000 steps.

---

## Stopping Safely

To stop without losing progress:
1. Press **⏹ Stop Training** in the Controls tab — this saves a final checkpoint immediately
2. Wait for `Training complete — model saved` in the console
3. Press Ctrl+C to close the dashboard

Or: move your mouse to any screen corner — the failsafe watchdog will stop the agent
within 50 ms (then manually save via the Controls tab if you want to keep the weights).
