# Next Steps & Recommendations for RE4 RL Agent

**Current Status:** 180,000 steps completed | Reward: +19.46 | Kill rate: 7.86 | Survival: 593s  
**Training Phase:** Just transitioned into **Combat Stage** (200k–600k steps)  
**Date Generated:** May 31, 2026

---

## 📊 Performance Summary

Your agent has shown **excellent progress** through the exploration stage:

| Metric | Initial (0 steps) | Current (180k) | Improvement |
|--------|------------------|----------------|-------------|
| **Mean Reward** | -9.80 | +19.46 | **+29.26 points** |
| **Mean Kills** | 0.05 | 7.86 | **157x increase** |
| **Survival Time** | 203s | 593s | **192% increase** |
| **Deaths** | 4/5 episodes | 0/5 episodes | **100% survival** |
| **Bell Rate** | 0.67% | 98.2% | **Near perfect** |
| **Shotgun Rate** | 0.47% | 68.7% | **Strong objective progress** |

**Key Achievement:** The agent has learned basic combat, survival, and objective-seeking behaviors.

---

## 🎯 Priority 1: Complete Combat Stage Training (180k → 600k steps)

**Why this matters:** You're now in the **Combat Curriculum Stage** where reward weights shift to prioritize efficient fighting:
- Survival weight: 0.4 → **0.2** (less important)
- Combat weight: 1.5 → **4.0** (primary focus)
- Exploration weight: 2.0 → **0.5** (already learned)

### Action Items

#### 1.1 Resume Training from Best Checkpoint
```bash
python main.py --resume models/checkpoints/re_agent_final.zip
```

**Expected Timeline:**
- Combat stage completes at 600k steps
- From 180k → 600k = 420k additional steps
- At ~12 steps/sec = **~9.7 hours** of real-game training
- Recommendation: Run overnight or split into 2-3 sessions

#### 1.2 Monitor Combat-Specific Metrics

Watch these in the dashboard during combat training:
- **Accuracy** — shots fired vs. kills (should improve)
- **Ammo efficiency** — kills per magazine (target: >3 kills/15 rounds)
- **Health conservation** — mean HP at episode end (should increase)
- **Inventory spam** — should drop to near zero in combat

#### 1.3 Enable Enhanced Logging for Combat Analysis

Add combat-specific tracking to understand tactical improvements:

**Create:** `v:\AI-ML\REAgents\combat_metrics.py`

```python
"""Combat-specific metric tracking for the Combat curriculum stage."""

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class CombatMetrics:
    shots_fired: int = 0
    kills: int = 0
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    headshots: int = 0  # if detectable
    melee_kills: int = 0
    
    @property
    def accuracy(self) -> float:
        """Kill/shot ratio (proxy for accuracy)."""
        return self.kills / max(1, self.shots_fired)
    
    @property
    def efficiency(self) -> float:
        """Damage dealt per damage taken ratio."""
        return self.damage_dealt / max(0.1, self.damage_taken)

class CombatLogger:
    def __init__(self):
        self.episode_metrics: List[CombatMetrics] = []
        self.current = CombatMetrics()
    
    def log_shot(self):
        self.current.shots_fired += 1
    
    def log_kill(self, is_melee: bool = False):
        self.current.kills += 1
        if is_melee:
            self.current.melee_kills += 1
    
    def log_damage(self, damage: float, is_player: bool):
        if is_player:
            self.current.damage_taken += damage
        else:
            self.current.damage_dealt += damage
    
    def finalize_episode(self):
        self.episode_metrics.append(self.current)
        self.current = CombatMetrics()
        return self.episode_metrics[-1]
    
    def get_rolling_stats(self, window: int = 10):
        recent = self.episode_metrics[-window:]
        if not recent:
            return None
        return {
            'mean_accuracy': np.mean([m.accuracy for m in recent]),
            'mean_efficiency': np.mean([m.efficiency for m in recent]),
            'mean_kills': np.mean([m.kills for m in recent]),
        }
```

---

## 🎯 Priority 2: Leverage Simulation for Faster Iteration (OPTIONAL)

Your codebase includes a **simulation environment** that trains **~6000× faster** than the real game. This is ideal for:
- Rapid hyperparameter tuning
- Testing reward function changes
- Pre-training a policy before real-game fine-tuning

### Why Sim-to-Real?

**Real Game Training:**
- Speed: ~12 steps/sec
- 420k steps remaining = ~9.7 hours

**Simulation Training:**
- Speed: ~80,000 steps/sec (8 workers)
- 420k steps = **~5 minutes**
- Then transfer to real game for fine-tuning

### Action Items

#### 2.1 Run Simulation Training (Now Fixed)

**The simulation pipeline has been improved:**
- Fixed `list assignment index out of range` crash in `SimDashboardCallback`
- Robust `SharedState` initialization for all workers
- Better error handling in memory and curriculum callbacks
- `train_sim_complete.py` now correctly wires `SimMemoryCallback`

**Recommended command (full sim-to-real pipeline):**
```bash
# Quick verification run
python train_sim_complete.py --timesteps 20000 --eval-every 5000 --no-llm

# Full pre-training run (recommended)
python train_sim_complete.py --timesteps 600000 --eval-every 50000 --eval-episodes 20
```

This produces `models/checkpoints/sim_re_agent_for_real.zip` — ready for transfer.

#### 2.2 Transfer to Real Game
```bash
# Load the sim-to-real converted policy
python main.py --resume models/checkpoints/sim_re_agent_for_real.zip
```

The `train_sim_complete.py` script automatically creates this transfer-ready checkpoint with domain-randomization-aware weights.

**Expected Results:**
- Sim training completes in minutes
- Real game fine-tuning: ~2-3 hours to adapt to photorealistic visuals
- Total time saved: **~7 hours**

#### 2.3 Monitor Transfer Quality

Look for these signs of successful transfer:
- ✅ Initial performance in real game > 50% of sim performance
- ✅ Improvement curve resumes upward within 10k real steps
- ❌ If performance drops to near-random: domain randomization may need tuning

---

## 🎯 Priority 3: Optimize Observation Space for Combat

Currently using: `obs_frame_size: [84, 84]`

### Recommendation: Upgrade to 128×128 for Combat Stage

**Why?** Combat requires precise aim and target tracking. Higher resolution helps the CNN:
- Distinguish Ganados at distance
- Read health ring color more accurately
- Track enemy movement for lead shots

**Trade-off:**
- VRAM usage: ~1.8× increase (likely fine on RTX 3070+)
- Training speed: ~15% slower
- Combat performance: potentially **+20-30% improvement**

### Action Items

#### 3.1 Test with a Short Run
```yaml
# In config.yaml
rl_hyperparameters:
  obs_frame_size: [128, 128]  # Upgrade from [84, 84]
```

```bash
# Test for 10k steps to check VRAM usage
python main.py --no-llm
```

**Monitor:** GPU memory in Task Manager. If <90% VRAM usage, safe to continue.

#### 3.2 If VRAM is Limited

Alternative optimizations:
```yaml
rl_hyperparameters:
  obs_frame_size: [96, 96]     # Compromise resolution
  lstm_hidden_size: 192        # Reduce from 256 to save VRAM
  features_dim: 384            # Reduce from 512
```

---

## 🎯 Priority 4: Enhance LLM Integration for Tactical Guidance

Your Claude advisor currently runs every 60 steps (~5 seconds). During the **Combat Stage**, the LLM should provide more tactical guidance.

### Action Items

#### 4.1 Increase LLM Consultation Frequency in Combat

```yaml
# In config.yaml
llm_settings:
  llm_every_n_steps: 30      # Was 60 — now consult every 2.5s during combat
  call_cooldown_seconds: 3.0 # Was 5.0 — allow faster back-to-back calls
```

**Why?** Combat situations change rapidly. More frequent LLM input helps:
- Prioritize targets (weak enemies, dangerous chainsaw wielders)
- Suggest tactical retreats when surrounded
- Recommend ammo conservation strategies

#### 4.2 Add Combat-Specific Context to LLM Prompts

Enhance the prompt in `llm_agent.py` to include combat metrics:

**Find this section in `llm_agent.py`:**
```python
def _build_prompt(
    hud: Dict[str, Any],
    detections: List[Dict],
    chapter: str = "",
) -> str:
```

**Add combat context:**
```python
# Add after HUD info
combat_context = f"""
COMBAT STATUS:
- Enemies visible: {len([d for d in detections if 'enemy' in d['label'].lower()])}
- Ammo remaining: {hud.get('ammo_clip', 0)}/{hud.get('ammo_res', 0)}
- Health: {hud.get('health_pct', 1.0) * 100:.0f}%
- Tactical advice needed: immediate threat assessment
"""
```

This helps Claude make better action override decisions.

---

## 🎯 Priority 5: Implement Advanced Evaluation Metrics

Your `eval_log.csv` tracks high-level metrics. Add **granular combat analytics** for deeper insights.

### Action Items

#### 5.1 Create Combat Evaluation Suite

**Create:** `v:\AI-ML\REAgents\eval_combat.py`

```python
"""Combat-focused evaluation runner."""

import yaml
import numpy as np
from stable_baselines3 import RecurrentPPO
from environment import ResidentEvilEnv
from typing import Dict, List

def evaluate_combat_performance(
    model_path: str,
    n_episodes: int = 10,
    config_path: str = "config.yaml"
) -> Dict:
    """
    Run evaluation focused on combat metrics.
    
    Returns:
        dict with keys:
            - mean_kills, std_kills
            - mean_accuracy (kills/shots)
            - mean_health_conservation
            - mean_ammo_efficiency
            - tactical_diversity (action entropy)
    """
    model = RecurrentPPO.load(model_path)
    env = ResidentEvilEnv(config_path)
    
    results = {
        'kills': [],
        'shots': [],
        'final_health': [],
        'final_ammo': [],
        'action_counts': [],
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_kills = 0
        ep_shots = 0
        actions_taken = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Track combat actions
            if action[3] in [2, 3]:  # shoot or aim+shoot
                ep_shots += 1
            
            # Track kills from info if available
            if 'enemy_killed' in info:
                ep_kills += info['enemy_killed']
            
            actions_taken.append(action.copy())
            done = done or truncated
        
        results['kills'].append(ep_kills)
        results['shots'].append(ep_shots)
        results['final_health'].append(obs['hud'][0])  # health is index 0
        results['action_counts'].append(actions_taken)
    
    # Compute statistics
    stats = {
        'mean_kills': np.mean(results['kills']),
        'std_kills': np.std(results['kills']),
        'mean_shots': np.mean(results['shots']),
        'accuracy': np.mean([k/max(s,1) for k,s in zip(results['kills'], results['shots'])]),
        'mean_final_health': np.mean(results['final_health']),
        'survival_rate': np.mean([h > 0 for h in results['final_health']]),
    }
    
    return stats

if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "models/checkpoints/re_agent_final.zip"
    results = evaluate_combat_performance(checkpoint)
    
    print("\n" + "="*60)
    print("COMBAT EVALUATION RESULTS")
    print("="*60)
    for key, value in results.items():
        print(f"{key:25s}: {value:.3f}")
```

**Run after each major checkpoint:**
```bash
python eval_combat.py models/checkpoints/re_agent_final.zip
```

---

## 🎯 Priority 6: Prepare for Completion Stage (600k–1M steps)

The final curriculum stage shifts focus to **objective completion**:
- Objective weight: 1.5 → **3.0** (primary focus)
- Combat weight: 4.0 → **1.5** (secondary)

### Action Items

#### 6.1 Define Clear Objectives for the LLM

Update `guide_data.py` with specific village objectives:

```python
VILLAGE_OBJECTIVES = [
    "Survive until the church bell rings",
    "Collect the shotgun from the second floor",
    "Stock up on ammo and herbs",
    "Unlock the door to the barn",
    "Regroup at the safe house",
]
```

#### 6.2 Implement Objective Tracking

Add objective detection to `perception.py`:
- Template matching for key items (shotgun, keys, herbs)
- Distance estimation to known landmarks
- Completion state tracking

#### 6.3 Tune Reward Function for Objectives

In `environment.py`, boost rewards for objective-relevant actions:
```python
# Proximity to objective locations
obj_reward = 0.0
if 'shotgun_nearby' in info:
    obj_reward += 5.0
if 'bell_rang' in info:
    obj_reward += 50.0  # Episode success

total_reward = (
    survival_reward * weights['survival'] +
    exploration_reward * weights['exploration'] +
    combat_reward * weights['combat'] +
    obj_reward * weights['objective']
)
```

---

## 🎯 Priority 7: Memory and Knowledge Retention

Your SQLite memory system is underutilized. Enable the agent to **learn from past mistakes**.

### Action Items

#### 7.1 Implement Death Analysis

**Add to `memory.py`:**

```python
def get_death_patterns(self, last_n: int = 50) -> List[Dict]:
    """Retrieve circumstances of recent deaths."""
    cursor = self._conn.execute("""
        SELECT 
            e.id,
            e.total_reward,
            e.steps,
            s.health_pct,
            s.ammo_clip,
            s.enemy_count,
            s.action
        FROM episodes e
        JOIN step_log s ON s.episode_id = e.id
        WHERE e.death_count > 0
        AND s.step = (
            SELECT MAX(step) FROM step_log WHERE episode_id = e.id
        )
        ORDER BY e.end_time DESC
        LIMIT ?
    """, (last_n,))
    
    deaths = []
    for row in cursor.fetchall():
        deaths.append({
            'episode_id': row[0],
            'reward': row[1],
            'steps_survived': row[2],
            'final_health': row[3],
            'final_ammo': row[4],
            'enemies_nearby': row[5],
            'last_action': row[6],
        })
    return deaths
```

#### 7.2 Feed Death Patterns to LLM

In `llm_agent.py`, add context from recent failures:

```python
from memory import MemorySystem

memory = MemorySystem(db_path="data/re_agent_memory.db")
death_patterns = memory.get_death_patterns(last_n=10)

# Add to Claude prompt
if death_patterns:
    prompt += f"""
RECENT FAILURE PATTERNS:
{_format_death_patterns(death_patterns)}

Suggest strategies to avoid repeating these mistakes.
"""
```

---

## 🔧 Technical Improvements (Secondary Priority)

### 1. Implement Gradient Monitoring

Track training health via gradient norms:

```python
# Add to trainer.py
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class GradientMonitorCallback(BaseCallback):
    def _on_step(self) -> bool:
        if hasattr(self.model.policy, 'optimizer'):
            total_norm = 0
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            self._shared.update(gradient_norm=total_norm)
            
            # Warning if gradients vanish or explode
            if total_norm < 1e-6:
                logger.warning("Vanishing gradients detected: %f", total_norm)
            elif total_norm > 100:
                logger.warning("Exploding gradients detected: %f", total_norm)
        
        return True
```

### 2. Add Learning Rate Scheduling

For the completion stage, consider annealing the learning rate:

```python
from stable_baselines3.common.callbacks import BaseCallback

class LRScheduler(BaseCallback):
    def __init__(self, initial_lr=0.00025, final_lr=0.00001, total_steps=1_000_000):
        super().__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
    
    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_steps
        new_lr = self.initial_lr * (1 - progress) + self.final_lr * progress
        
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return True
```

### 3. Implement Checkpoint Ensemble

At 1M steps, create an ensemble of best checkpoints:

```python
def evaluate_ensemble(checkpoints: List[str], env, n_episodes=10):
    """Ensemble evaluation using majority voting."""
    models = [RecurrentPPO.load(ckpt) for ckpt in checkpoints]
    
    total_reward = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # Get action from each model
            actions = [m.predict(obs, deterministic=True)[0] for m in models]
            
            # Majority vote per action dimension
            final_action = np.array([
                np.bincount(actions[:, i]).argmax() for i in range(6)
            ])
            
            obs, reward, done, truncated, info = env.step(final_action)
            ep_reward += reward
            done = done or truncated
        
        total_reward += ep_reward
    
    return total_reward / n_episodes
```

---

## 📋 Weekly Training Schedule

### Week 1: Combat Mastery (180k → 400k steps)
- **Day 1-2:** Resume training, monitor combat metrics
- **Day 3:** Evaluate at 300k steps, check combat accuracy
- **Day 4-5:** Continue training to 400k
- **Day 6:** Full evaluation suite
- **Day 7:** Review tensorboard, adjust if needed

### Week 2: Combat Completion (400k → 600k steps)
- **Day 1-3:** Train to 600k
- **Day 4:** Major evaluation checkpoint
- **Day 5:** Test 128×128 observation upgrade
- **Day 6-7:** If successful, retrain final 50k with higher res

### Week 3: Transition to Completion Stage (600k → 800k)
- **Day 1:** Implement objective tracking
- **Day 2:** Update reward weights (auto via curriculum)
- **Day 3-5:** Train with objective focus
- **Day 6:** Evaluate objective completion rate
- **Day 7:** Fine-tune LLM prompts for objectives

### Week 4: Final Polish (800k → 1M steps)
- **Day 1-4:** Complete training to 1M
- **Day 5:** Full evaluation battery
- **Day 6:** Ensemble testing
- **Day 7:** Final analysis and documentation

---

## 🎓 Learning Resources & Best Practices

### RL Best Practices Applied to This Project

1. **Curriculum Learning** ✅ — Your staged approach is optimal
2. **Reward Shaping** ✅ — Multi-component rewards with stage-specific weights
3. **Memory Architecture** ✅ — RecurrentPPO's LSTM handles partial observability
4. **Feature Extraction** ✅ — Impala ResNet is state-of-the-art for visual RL
5. **Exploration** ✅ — Entropy bonus maintains policy diversity

### Additional Reading

- [Proximal Policy Optimization (Schulman et al.)](https://arxiv.org/abs/1707.06347)
- [IMPALA: Scalable Distributed Deep-RL (Espeholt et al.)](https://arxiv.org/abs/1802.01561)
- [Learning Dexterity (OpenAI)](https://openai.com/blog/learning-dexterity) — excellent sim-to-real case study
- [Emergent Tool Use from Multi-Agent Autocurricula (OpenAI)](https://openai.com/blog/emergent-tool-use/)

---

## 🚨 Red Flags to Watch For

Monitor for these issues during combat training:

### 1. Policy Collapse
**Symptoms:**
- Sudden reward drop of >50%
- Agent repeats one action continuously
- Loss spikes in tensorboard

**Fix:**
- Reload previous checkpoint
- Reduce learning rate by 50%
- Check for NaN gradients

### 2. Overfitting to Village Layout
**Symptoms:**
- Excellent performance in village
- Poor generalization to new areas

**Fix:**
- Add domain randomization to real env
- Train on multiple save points
- Increase exploration bonus

### 3. LLM Override Dominance
**Symptoms:**
- RL policy becomes passive
- Agent only acts when LLM suggests

**Fix:**
- Reduce LLM call frequency
- Use LLM as advisor only (no hard overrides)
- Track % of steps with LLM override (<10% ideal)

### 4. Memory Bottleneck
**Symptoms:**
- Training slows over time
- High disk I/O usage

**Fix:**
- Clean up old episode logs (keep last 100)
- Implement log rotation in `memory.py`
- Move DB to SSD if on HDD

---

## 🎯 Success Metrics for 1M Steps

### Minimum Viable Performance
- **Survival rate:** >95% (consistently reach the bell)
- **Mean reward:** >25
- **Kill rate:** >10 enemies per episode
- **Shotgun acquisition:** >90%

### Stretch Goals
- **Perfect village runs:** 5+ consecutive zero-death episodes
- **Ammo efficiency:** <2 shots per kill
- **Speed:** Complete village in <300 steps
- **Objective completion:** 100% bell + shotgun

---

## 📞 Next Steps Summary (TL;DR)

1. ✅ **Resume training** to 600k steps (combat focus)
2. ✅ **Monitor combat metrics** — accuracy, efficiency, ammo use
3. ⚡ **Optional:** Train in simulation for 6000× speedup
4. 🔍 **Upgrade to 128×128** observations for better aim
5. 🤖 **Enhance LLM** with combat-specific prompts
6. 📊 **Add combat evaluation** suite for granular insights
7. 🎯 **Prepare objectives** for 600k+ completion stage
8. 🧠 **Leverage memory** system for death pattern analysis
9. 📈 **Track gradients** and implement LR scheduling
10. 🏆 **Evaluate at 1M** with ensemble methods

**Estimated time to completion:** 3-4 weeks of continuous training

---

## 💬 Questions?

If you encounter issues or want to discuss strategy:
1. Check tensorboard for training curves
2. Review `eval_log.csv` for performance trends
3. Inspect recent LLM decisions in the Memory tab
4. Share any error messages or unexpected behaviors

**Good luck, and may your agent master the village!** 🎮🧟
