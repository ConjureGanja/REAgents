# Troubleshooting Guide - RE4 RL Agent

Quick fixes for common issues during training.

---

## 🚨 Training Issues

### Issue: Training won't start / Stuck at "Waiting to start"

**Symptoms:**
- Dashboard opens but shows "Status: Idle"
- No frame updates in Live Feed tab
- Total steps remain at 0

**Fixes:**
1. Did you press "▶ Start Training" in the Controls tab?
   - Training does NOT auto-start
   
2. Is RE4 running and visible on the correct monitor?
   ```yaml
   # Check config.yaml
   game_settings:
     monitor_index: 2  # Try changing to 1
   ```

3. Check if capture is working:
   ```bash
   python -c "from capture import ScreenCapture; cap = ScreenCapture(); print(cap.get_frame().shape)"
   ```
   Should output: `(1080, 1920, 3)` or similar

---

### Issue: Black screen in dashboard / No frame captured

**Symptoms:**
- Live Feed shows black rectangle
- Detection count always 0

**Fixes:**
1. **Wrong monitor index:**
   ```yaml
   game_settings:
     monitor_index: 1  # Try 1 if primary, 2 if secondary
   ```

2. **Game not in Borderless Windowed:**
   - In RE4: Options → Display → Display Mode → **Borderless Windowed**
   - Fullscreen exclusive mode blocks screen capture

3. **UAC/Permission issues:**
   - Run terminal as Administrator
   - Some capture APIs require elevated permissions

4. **Test capture directly:**
   ```python
   from mss import mss
   with mss() as sct:
       monitor = sct.monitors[2]  # Change to your monitor index
       img = sct.grab(monitor)
       print(f"Captured: {img.width}x{img.height}")
   ```

---

### Issue: CUDA out of memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Quick Fixes:**

**Option 1: Reduce batch size**
```yaml
rl_hyperparameters:
  batch_size: 64   # Was 128
  n_steps: 256     # Was 512 (must still divide batch_size evenly)
```

**Option 2: Reduce model size**
```yaml
rl_hyperparameters:
  lstm_hidden_size: 192  # Was 256
  features_dim: 384      # Was 512
```

**Option 3: Reduce observation size**
```yaml
rl_hyperparameters:
  obs_frame_size: [84, 84]  # Don't use 128×128 if VRAM limited
```

**Option 4: Reduce frame stack**
```yaml
rl_hyperparameters:
  frame_stack: 2  # Was 4 (less temporal info but saves VRAM)
```

**Check VRAM usage:**
```bash
nvidia-smi
```
Look for "Memory-Usage" column

---

### Issue: Training extremely slow (<5 steps/sec)

**Expected:** ~12 steps/sec for real game training

**Possible Causes:**

**1. YOLO/OCR taking too long**
Check perception time:
```python
import time
from perception import PerceptionSystem
from capture import ScreenCapture

cap = ScreenCapture()
eyes = PerceptionSystem()

frame = cap.get_frame()
start = time.time()
detections = eyes.detect_objects(frame)
hud = eyes.read_hud(frame)
elapsed = time.time() - start
print(f"Perception took {elapsed*1000:.1f}ms")
```
Should be <100ms. If >200ms:
- Lower YOLO confidence threshold
- Use smaller YOLO model (yolo11n → yolo11s)
- Disable EasyOCR if not critical

**2. Recording enabled**
```bash
# Disable video recording
python main.py --no-record
```

**3. Multiple LLM calls stacking up**
```yaml
llm_settings:
  llm_every_n_steps: 120  # Increase from 60
  call_cooldown_seconds: 10.0  # Increase from 5.0
```

**4. Memory DB writes blocking**
Check `memory.py` — writes should be async via queue

---

### Issue: Reward stays flat / No learning progress

**Symptoms:**
- Mean reward doesn't change after 50k+ steps
- Episode length constant
- Policy loss not decreasing

**Diagnostics:**

1. **Check if agent is actually exploring:**
   ```bash
   tensorboard --logdir logs/tensorboard
   ```
   Look at "ep_reward_mean" and "entropy"
   - If entropy is near 0 → policy collapsed to deterministic
   - If reward flat + entropy high → reward function might be broken

2. **Check reward values are reasonable:**
   Watch dashboard live feed — reward per step should be in range [-5, +5]
   If all zeros or all same value → bug in reward calculation

**Fixes:**

**If entropy too low (< 0.01):**
```yaml
rl_hyperparameters:
  ent_coef: 0.02  # Was 0.01 — encourages more exploration
```

**If reward seems broken:**
```python
# Test environment manually
from environment import ResidentEvilEnv
env = ResidentEvilEnv()
obs, _ = env.reset()

for i in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {i}: action={action}, reward={reward:.2f}")
    if done or truncated:
        break
```
Rewards should vary — if all same, check `environment.py` reward calculation

**If learning rate too high/low:**
```yaml
rl_hyperparameters:
  learning_rate: 0.0001  # Try half the current value
```

---

### Issue: Agent keeps dying / Survival rate dropping

**Symptoms:**
- Death count increasing over time
- Mean survival time decreasing
- Health always low in HUD

**Fixes:**

**1. Increase survival reward weight:**
```yaml
curriculum:
  stages:
    combat:  # Or whichever stage you're in
      reward_weights:
        survival: 0.6  # Increase from default
```

**2. Penalize damage more heavily:**
In `environment.py`, find the damage penalty:
```python
# Increase this multiplier
damage_penalty = health_delta * -20  # Was -10
```

**3. Check if health detection is working:**
```python
from perception import PerceptionSystem
from capture import ScreenCapture

cap = ScreenCapture()
eyes = PerceptionSystem()
frame = cap.get_frame()
hud = eyes.read_hud(frame)
print(f"Health: {hud['health_pct']}")
```
Should return 0.0-1.0. If always 1.0 or 0.0 → calibrate health circle

---

## 🤖 LLM / API Issues

### Issue: "ANTHROPIC_API_KEY not found"

**Fix:**
1. Check `.env` file exists (not `.env.example`)
   ```bash
   # Windows
   copy env.example .env
   ```

2. Edit `.env` and add your key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   ```

3. Restart the agent (`.env` only loaded on startup)

**Alternative:** Run without LLM
```bash
python main.py --no-llm
```

---

### Issue: LLM calls failing / Timeout errors

**Symptoms:**
```
anthropic.APIConnectionError: Connection timeout
```

**Fixes:**

**1. Increase timeout:**
In `llm_agent.py`, find the API call:
```python
# Add timeout parameter
response = await self._client.messages.create(
    model=self._model,
    max_tokens=self._max_tokens,
    temperature=self._temperature,
    messages=[...],
    timeout=30.0,  # Add this — default is 10s
)
```

**2. Reduce call frequency:**
```yaml
llm_settings:
  llm_every_n_steps: 120  # Increase from 60
```

**3. Check API status:**
Visit https://status.anthropic.com

---

### Issue: LLM giving nonsense advice / Action overrides breaking agent

**Symptoms:**
- Agent makes obviously bad decisions when LLM is active
- Performance better with `--no-llm`

**Fixes:**

**1. Disable action overrides:**
In `llm_agent.py`, find where overrides are applied and comment out:
```python
# Only use LLM for advisory, not hard control
# if parsed_result.get('override'):
#     self._shared.update(action_override=parsed_result['override'])
```

**2. Reduce LLM influence:**
```yaml
llm_settings:
  llm_every_n_steps: 300  # Only consult occasionally
```

**3. Improve prompts:**
Review `guide_data.py` and `llm_agent.py` prompt construction
- Add more specific combat guidelines
- Include recent failure patterns
- Emphasize ammo conservation

---

## 📊 Dashboard / UI Issues

### Issue: Dashboard not opening / Connection refused

**Symptoms:**
```
Could not connect to http://127.0.0.1:7860
```

**Fixes:**

**1. Check if port is already in use:**
```powershell
# Windows
netstat -ano | findstr :7860
```
If occupied, kill the process or change port in `dashboard.py`

**2. Firewall blocking:**
Allow Python through Windows Firewall

**3. Try different browser:**
Chrome/Edge work best with Gradio

**4. Check Gradio version:**
```bash
pip show gradio
# Should be >=5.0
pip install --upgrade gradio
```

---

### Issue: Dashboard shows stale data / Not updating

**Symptoms:**
- Frame frozen
- Metrics not changing
- "Last updated" timestamp old

**Fixes:**

**1. Refresh browser:**
- Hard refresh: Ctrl+Shift+R

**2. Check training is actually running:**
- Look at terminal output
- Check if GPU is active (nvidia-smi)
- Verify "Status: Training" in dashboard

**3. Check SharedState updates:**
In `trainer.py`, verify callbacks are firing:
```python
# Add debug print in DashboardCallback._on_step()
logger.info(f"Step {self.num_timesteps}: reward={reward}")
```

---

## 🎮 Game Interaction Issues

### Issue: Inputs not registering in game

**Symptoms:**
- Agent starts but Leon doesn't move
- Dashboard shows actions but no in-game response

**Fixes:**

**1. Check game window has focus:**
- Click on RE4 window before starting training
- Don't click away during training (auto-pause will trigger)

**2. Verify controls are working:**
```python
from controls import GameControls
ctrl = GameControls()

# Test movement
ctrl.move_forward(0.5)  # Should move for 0.5s
ctrl.stop_all()
```
If no movement → pydirectinput may not be working

**3. Run as Administrator:**
Some DirectX input requires elevated permissions

**4. Check keyboard layout:**
- Controls assume QWERTY with WASD movement
- If different layout, may need to remap in `controls.py`

---

### Issue: Game resets failing / "Load Game" not working

**Symptoms:**
- Episode end triggers but game doesn't reload
- Stuck in pause menu
- Agent continues on dead screen

**Fixes:**

**1. Calibrate menu navigation:**
In `config.yaml`:
```yaml
reset:
  nav_down_count: 1  # Might need 2 or 3 depending on your menu
```

**2. Increase wait times:**
```yaml
reset:
  menu_open_wait: 2.0   # Was 1.5
  loading_wait: 12.0    # Was 8.0 — slow HDDs need more time
```

**3. Manual test:**
```python
from controls import GameControls
from config import yaml

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

ctrl = GameControls()
ctrl.reset_game(cfg['reset'])
# Watch if it navigates correctly
```

---

## 💾 Checkpoint / Save Issues

### Issue: "Failed to load checkpoint"

**Symptoms:**
```
FileNotFoundError: models/checkpoints/re_agent_final.zip
```

**Fixes:**

**1. Check file exists:**
```bash
dir models\checkpoints\
```

**2. Use absolute path:**
```bash
python main.py --resume "V:\AI-ML\REAgents\models\checkpoints\re_agent_final.zip"
```

**3. Check for typos:**
- File extension must be `.zip`
- Filenames are case-sensitive on some systems

---

### Issue: Loading checkpoint but performance is random

**Symptoms:**
- Checkpoint loads without errors
- But agent behaves randomly / worse than expected

**Likely Cause:** Missing VecNormalize stats

**Fix:**
Always load BOTH files:
- Model: `re_agent_final.zip`
- Stats: `re_agent_final_vecnorm.pkl`

In `trainer.py`, ensure VecNormalize is loaded:
```python
env = VecNormalize.load("models/checkpoints/re_agent_final_vecnorm.pkl", env)
```

---

## 📈 Performance / Metrics Issues

### Issue: eval_log.csv shows NaN or weird values

**Check:**
1. Is evaluation actually running?
   ```yaml
   training:
     eval_freq: 10_000  # Every 10k steps
     eval_episodes: 3   # Should be ≥3
   ```

2. Are episodes completing during eval?
   - If episodes timeout before finishing → increase max_episode_steps
   - If agent dies immediately → reward function may be broken

---

### Issue: TensorBoard not showing curves

**Fixes:**

**1. Check log directory:**
```bash
dir logs\tensorboard\RecurrentPPO_0\
# Should contain .tfevents files
```

**2. Correct command:**
```bash
tensorboard --logdir logs/tensorboard
# NOT --logdir logs/tensorboard/RecurrentPPO_0
```

**3. Refresh browser:**
TensorBoard caches aggressively — use Ctrl+Shift+R

**4. Check TensorBoard version:**
```bash
pip install --upgrade tensorboard
```

---

## 🔧 Python / Dependencies Issues

### Issue: ModuleNotFoundError

**Common Missing Modules:**

**1. EasyOCR:**
```bash
pip install easyocr
```

**2. Ultralytics:**
```bash
pip install ultralytics
```

**3. sb3-contrib:**
```bash
pip install sb3-contrib
```

**Full reinstall:**
```bash
pip install --upgrade -r requirements.txt
```

---

### Issue: "CUDA not available" despite having GPU

**Fix:**

**1. Check PyTorch installation:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

**2. Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Replace `cu121` with your CUDA version

**3. Verify CUDA toolkit installed:**
```bash
nvcc --version
```

---

### Issue: "Python 3.12 not supported"

**SB3 known issue with Python 3.12**

**Fix:** Use Python 3.11.x
```bash
python --version
# Should show 3.11.x

# If not, install Python 3.11 and use:
py -3.11 -m pip install -r requirements.txt
py -3.11 main.py
```

---

## 🆘 Emergency Recovery

### Nuclear Option: Reset Everything

If nothing works, start fresh:

```bash
# 1. Backup important files
copy models\checkpoints\re_agent_final.zip backup\
copy _test_metrics\eval_log.csv backup\
copy data\re_agent_memory.db backup\

# 2. Delete generated files
rmdir /s /q logs
rmdir /s /q models\checkpoints
del data\re_agent_memory.db

# 3. Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# 4. Start fresh
python main.py
```

---

## 📞 Still Stuck?

1. **Check GitHub Issues:** Similar problems may be documented
2. **Enable debug logging:**
   ```bash
   python main.py --log-level DEBUG
   ```
   Review `logs/re_agent.log` for detailed errors

3. **Share error context:**
   - Full error traceback
   - Relevant config.yaml sections
   - GPU/Python/CUDA versions
   - What you tried already

**System info command:**
```powershell
python --version
nvidia-smi
pip show stable-baselines3 sb3-contrib torch
```
