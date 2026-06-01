# Quick Start: Combat Stage Training

**Current Status:** 180k steps ✅ | Entering Combat Stage (200k-600k)  
**Goal:** Achieve 600k steps with combat mastery

---

## 🚀 Start Training NOW (3 Commands)

### Option A: Real Game Training (~10 hours)
```bash
# 1. Resume from best checkpoint
python main.py --resume models/checkpoints/re_agent_final.zip

# 2. Open dashboard
# Navigate to: http://127.0.0.1:7860

# 3. Press "▶ Start Training" in Controls tab
```

### Option B: Simulation Training (~5 minutes + transfer)
```bash
# 1. Train in simulation (super fast)
python sim_main.py --total-timesteps 600000

# 2. Transfer to real game for fine-tuning
python main.py --resume models/checkpoints/sim_re_agent_final.zip
```

---

## 📊 What to Watch

### Dashboard Metrics (Combat Stage)
| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Mean Kills** | 10+ | Combat effectiveness |
| **Accuracy** (kills/shots) | >0.5 | Ammo efficiency |
| **Survival Rate** | >95% | Avoiding deaths |
| **Inventory Spam** | <3 | Not opening suitcase mid-fight |
| **Mean Episode Reward** | >25 | Overall improvement |

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```
Watch for:
- ✅ Smooth upward reward curve
- ✅ Decreasing policy loss
- ⚠️ Sudden reward drops (reload checkpoint if this happens)

---

## ⚙️ Quick Optimizations

### 1. Higher Resolution (Better Aim)
**Edit `config.yaml`:**
```yaml
rl_hyperparameters:
  obs_frame_size: [128, 128]  # Was [84, 84]
```
**Trade-off:** +15% slower, +20-30% better combat

### 2. More Frequent LLM Advice
**Edit `config.yaml`:**
```yaml
llm_settings:
  llm_every_n_steps: 30  # Was 60 — now every 2.5s
```
**Trade-off:** 2× API costs, better tactical decisions

### 3. Adjust Combat Rewards (if needed)
**Edit `config.yaml`:**
```yaml
curriculum:
  stages:
    combat:
      reward_weights:
        combat: 5.0  # Increase from 4.0 if kills aren't improving
```

---

## 🔍 Checkpoints & Evaluation

### Save Checkpoints Every 50k Steps
Checkpoints auto-save to: `models/checkpoints/re_agent_*_steps.zip`

### Run Manual Evaluation
```bash
# Test current policy for 10 episodes
python -c "
from stable_baselines3 import RecurrentPPO
from environment import ResidentEvilEnv

model = RecurrentPPO.load('models/checkpoints/re_agent_final.zip')
env = ResidentEvilEnv()

for ep in range(10):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        done = done or truncated
    print(f'Episode {ep+1}: Reward = {ep_reward:.2f}')
"
```

---

## 🛑 Emergency Fixes

### Problem: Agent stuck in loop
**Fix:** Increase exploration
```yaml
rl_hyperparameters:
  ent_coef: 0.02  # Was 0.01
```

### Problem: Too many deaths
**Fix:** Increase survival weight
```yaml
curriculum:
  stages:
    combat:
      reward_weights:
        survival: 0.4  # Was 0.2
```

### Problem: Training very slow
**Fix:** Use simulation
```bash
python sim_main.py --total-timesteps 420000  # Remaining steps
```

### Problem: CUDA out of memory
**Fix:** Reduce batch size
```yaml
rl_hyperparameters:
  batch_size: 64      # Was 128
  n_steps: 256        # Was 512
```

---

## 📈 Progression Milestones

### 300k Steps (~6 hours from now)
- **Expected:** 8-9 kills/episode, 85%+ bell rate
- **Check:** Are kills increasing? Is health conservation improving?

### 400k Steps (~10 hours from now)
- **Expected:** 9-10 kills/episode, ammo efficiency >0.4
- **Action:** Run full evaluation, compare to 300k

### 600k Steps (Combat Complete, ~20 hours total)
- **Expected:** 10+ kills/episode, 95%+ survival, shotgun 80%+
- **Next:** Transition to Completion Stage (objectives)

---

## 💾 Backup Your Progress

### Critical Files to Save
```bash
# Copy these to a safe location every 100k steps
models/checkpoints/re_agent_*_steps.zip
models/checkpoints/re_agent_final_vecnorm.pkl
_test_metrics/eval_log.csv
data/re_agent_memory.db
logs/tensorboard/RecurrentPPO_0/*
```

### Cloud Backup Command (Windows)
```powershell
# One-time setup
$BackupDir = "D:\RE_Agent_Backups\$(Get-Date -Format 'yyyy-MM-dd_HHmm')"
New-Item -ItemType Directory -Path $BackupDir

# Copy important files
Copy-Item -Recurse models\checkpoints\* $BackupDir\checkpoints\
Copy-Item -Recurse _test_metrics\* $BackupDir\metrics\
Copy-Item data\re_agent_memory.db $BackupDir\
```

---

## 🎯 Combat Stage Success Criteria

At 600k steps, you should achieve:
- [x] **10+ mean kills** per episode
- [x] **>0.5 kill/shot ratio** (accuracy proxy)
- [x] **95%+ survival rate** (bell reached)
- [x] **80%+ shotgun acquisition**
- [x] **<5 inventory opens** during combat
- [x] **>30 mean reward**

**If achieved:** Proceed to Completion Stage  
**If not:** Review `NEXT_STEPS.md` → Priority 1 troubleshooting

---

## ⏱️ Time Estimates

| Method | Time to 600k | Pros | Cons |
|--------|--------------|------|------|
| **Real game only** | ~20 hours | No transfer gap | Very slow |
| **Sim → Real** | ~5 min + 3 hours | Super fast sim | Small transfer gap |
| **Sim parallel** | ~2 min + 2 hours | Fastest overall | Need multi-core CPU |

**Recommendation:** Use simulation for first 400k, then real game for final 200k.

---

**Ready to train? Run one of the commands at the top and let it cook! 🔥**
