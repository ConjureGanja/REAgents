"""
Abstract 2D simulation of the Resident Evil 4 village opening.

WHY AN ABSTRACT SIMULATION?
─────────────────────────────
The real RE4 game runs at ~12 decisions/second (limited by action_hold_seconds).
This simulation has no sleep() calls and no screen captures — a single step
takes microseconds instead of 80 ms.  With 8 parallel workers, total throughput
reaches ~80,000 steps/second vs ~12 steps/second on the real game.

Analogy: the difference between practising chess against a physical opponent
(slow, requires both parties to be present) vs running a computer engine
(thousands of games per second).  Both are chess, but the speed of feedback
is orders of magnitude different.

MAP LAYOUT  (50×50 world units, each ≈ 2 m in the real game)
────────────────────────────────────────────────────────────
  ┌───────────────────────────────────────────────────┐  y=0
  │ [HOUSE 3–15]      [open]       [BARN 34–47]       │
  │                                                   │
  │                   [WELL 21–29]                    │
  │                                                   │
  │   [open]                         [open]           │
  │                                                   │
  │                  [GATE 48–50]                     │  y=50
  └───────────────────────────────────────────────────┘

Player starts at (25, 42) — just inside the gate, facing north.
Barn interior contains the shotgun objective.
Well area is the bell objective zone.

ENEMY WAVES
───────────
  Wave 1 (step 0):   8 villagers spawn from the north half
  Wave 2 (step ~50 kills): 8 more spawn from edges
  Wave 3 (final):    8 more — relentless pressure
After all waves, surviving until step 600 wins the episode.

RENDERING
──────────
render_frame() returns a 84×84×3 uint8 numpy array — a top-down view of the
village.  This is the "frame" input to the CNN, playing the same role as the
real game's screen capture.  The CNN learns abstract spatial reasoning from
these symbolic images.

Color legend (matching the dashboard's legend panel):
  ■ Dark-grey  — obstacle / wall
  ■ Green      — player  (bright dot + direction line)
  ■ Red        — active enemy
  ■ Orange     — stunned enemy
  ■ Dark-red   — dead enemy (fades after a few steps)
  ■ Yellow     — ammo pickup
  ■ Cyan       — health herb
  ■ Gold       — shotgun (special item)
  ■ Blue tint  — objective zone
  ■ White line — perimeter fence
"""

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

MAP_W: float = 50.0
MAP_H: float = 50.0

# Obstacle AABB list — (x1, y1, x2, y2).
# Collision check: point is inside if x1 <= x <= x2 AND y1 <= y <= y2.
# Analogy: invisible invisible walls — the agent can't walk through them
# but there's no art, just rectangles.
OBSTACLES: List[Tuple[float, float, float, float]] = [
    (3.0,  3.0, 15.0, 18.0),   # House (top-left)
    # Barn (top-right) — WALLS ONLY so the interior is walkable and the shotgun
    # at (41,10) is reachable.  Previously the barn was one solid block, which
    # meant the player was clipped ~6 units outside it and could NEVER collect
    # the shotgun or trigger barn_visited — the episode's main objective was
    # impossible.  Now it's a room with a south doorway gap at x∈[39,42].
    (34.0,  3.0, 47.0,  4.5),  # Barn north wall
    (34.0,  3.0, 35.5, 18.0),  # Barn west wall
    (45.5,  3.0, 47.0, 18.0),  # Barn east wall
    (34.0, 16.5, 39.0, 18.0),  # Barn south wall (left of door)
    (42.0, 16.5, 47.0, 18.0),  # Barn south wall (right of door)
    (21.0, 21.0, 29.0, 29.0),  # Well  (centre)
    # Perimeter fence segments (thin, treated as solid)
    (0.0,  0.0, 50.0,  1.5),   # North wall
    (0.0,  0.0,  1.5, 50.0),   # West wall
    (48.5, 0.0, 50.0, 50.0),   # East wall
    # South wall with gate gap (10–40) — players/enemies can enter/exit
    (0.0, 48.5, 10.0, 50.0),
    (40.0, 48.5, 50.0, 50.0),
]

# Objective zones (AABB): enter to trigger objective progress
BARN_ZONE = (34.0, 3.0, 47.0, 18.0)   # Inside barn → collect shotgun
BELL_ZONE = (19.0, 19.0, 31.0, 31.0)  # Near the well → bell trigger

# Item spawn positions: (x, y, type)
ITEM_SPAWNS: List[Tuple[float, float, str]] = [
    (10.0, 30.0, "ammo"),
    (40.0, 25.0, "ammo"),
    (15.0, 40.0, "ammo"),
    (44.0, 40.0, "ammo"),
    (25.0, 38.0, "ammo"),   # Close to start — reward early exploration
    ( 6.0, 25.0, "herb"),
    (30.0, 40.0, "herb"),
    (41.0, 10.0, "shotgun"),  # Inside barn — major reward
]

# Enemy spawn points per wave: (x, y)
WAVE_SPAWNS: List[List[Tuple[float, float]]] = [
    # Wave 1 — spread across north half
    [(8.0, 8.0), (20.0, 6.0), (32.0, 8.0), (45.0, 8.0),
     (5.0, 20.0), (25.0, 15.0), (40.0, 15.0), (46.0, 20.0)],
    # Wave 2 — edges, more flanking
    [(2.0, 30.0), (48.0, 30.0), (10.0, 5.0), (40.0, 5.0),
     (18.0, 12.0), (35.0, 12.0), (5.0, 35.0), (45.0, 35.0)],
    # Wave 3 — heavy pressure from all directions
    [(25.0, 3.0), (3.0, 15.0), (47.0, 15.0), (3.0, 38.0),
     (47.0, 38.0), (15.0, 5.0), (35.0, 5.0), (25.0, 45.0)],
]

# ── Enumerations ──────────────────────────────────────────────────────────────

class EnemyState(IntEnum):
    """
    State machine for each enemy villager.

    Transitions:
      PATROL  → CHASE   when player enters detection_range
      CHASE   → ATTACK  when player enters attack_range
      ATTACK  → CHASE   after attack completes
      CHASE/ATTACK → STUNNED  when hit by player
      STUNNED → CHASE   after stun_timer expires
      any     → DEAD    when hp <= 0
    """
    PATROL  = 0
    CHASE   = 1
    ATTACK  = 2
    STUNNED = 3
    DEAD    = 4


class ItemType(IntEnum):
    AMMO    = 0
    HERB    = 1
    SHOTGUN = 2   # One-time special pickup — large reward


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EnemyAgent:
    """
    A single enemy villager in the simulation.

    hp:            health out of 100 (starts at 100).
    state:         current FSM state (EnemyState).
    stun_timer:    steps remaining in STUNNED; 0 = not stunned.
    attack_timer:  cooldown steps before next attack.
    patrol_angle:  patrol direction in radians (changes slowly).
    patrol_timer:  steps until next direction change.
    dead_timer:    steps since death (body fades after 30 steps).
    """
    x: float
    y: float
    hp: float = 100.0
    state: EnemyState = EnemyState.PATROL
    stun_timer: int = 0
    attack_timer: int = 0
    patrol_angle: float = 0.0
    patrol_timer: int = 0
    dead_timer: int = 0

    # Assigned at spawn — index in the enemies list (used for rendering colour)
    enemy_id: int = 0


@dataclass
class ItemPickup:
    """A collectible item on the map."""
    x: float
    y: float
    item_type: ItemType
    collected: bool = False


# ── Core simulation ───────────────────────────────────────────────────────────

class SimGameState:
    """
    Manages the full state of one simulated RE4 village episode.

    One SimGameState instance lives inside each SimResidentEvilEnv.
    When the Gym env calls reset(), this class re-initialises everything.
    When it calls step(action), this class advances physics, AI, and combat
    by one discrete timestep and returns the new state.

    Analogy: this is the "game engine" — everything in environment.py that
    previously came from the real game (screen capture, OCR, DirectX input)
    is now computed here in pure Python.
    """

    def __init__(self, cfg: dict, rng_seed: Optional[int] = None):
        """
        cfg: the `simulation` sub-dict from config.yaml.
        rng_seed: for reproducibility per worker.  Each worker gets a different
                  seed so their episodes explore different random paths.
        """
        self._cfg = cfg
        self._rng = random.Random(rng_seed)
        self._np_rng = np.random.default_rng(rng_seed)

        # Cache frequently-accessed config values
        self.player_speed     = float(cfg.get("player_speed",      0.6))
        self.sprint_mult      = float(cfg.get("sprint_multiplier", 1.8))
        self.dodge_cooldown   = int(cfg.get("dodge_cooldown",       8))
        self.enemy_speed      = float(cfg.get("enemy_speed",        0.28))
        self.enemy_chase_spd  = float(cfg.get("enemy_chase_speed",  0.40))
        self.detect_range     = float(cfg.get("enemy_detection_range", 14.0))
        self.attack_range     = float(cfg.get("enemy_attack_range",    1.8))
        self.attack_damage    = float(cfg.get("enemy_attack_damage",   12.0))
        self.attack_cd        = int(cfg.get("enemy_attack_cooldown",   15))
        self.shoot_range      = float(cfg.get("shoot_range",  20.0))
        self.shoot_cone       = math.radians(cfg.get("shoot_cone_deg", 18.0))
        self.ammo_clip_start  = int(cfg.get("ammo_start_clip",    15))
        self.ammo_res_start   = int(cfg.get("ammo_start_reserve", 50))
        self.ammo_pickup_amt  = int(cfg.get("ammo_pickup_amount", 10))
        self.n_waves          = int(cfg.get("enemy_waves",         3))
        self.wave_kill_thresh = int(cfg.get("wave_trigger_kills",  5))
        self.max_steps        = int(cfg.get("max_episode_steps",   600))

        # Episode state — reset in reset()
        self.player_x:      float = 25.0
        self.player_y:      float = 42.0
        self.player_dir:    float = -math.pi / 2   # Facing north (up)
        self.player_health: float = 100.0
        self.ammo_clip:     int   = self.ammo_clip_start
        self.ammo_reserve:  int   = self.ammo_res_start
        self.player_aiming: bool  = False
        self.invincibility_timer: int = 0  # I-frames after taking a hit
        self.dodge_timer:   int   = 0      # Cooldown after dodge

        self.enemies:   List[EnemyAgent] = []
        self.items:     List[ItemPickup] = []

        self.sim_step:          int   = 0
        self.wave_index:        int   = 0  # Which wave has been spawned last
        self.wave_kills:        int   = 0  # Kills since last wave trigger
        self.total_kills:       int   = 0
        self.items_collected:   int   = 0
        self.shotgun_collected: bool  = False
        self.barn_visited:      bool  = False
        self.bell_area_reached: bool  = False

        # Episode stats (returned in info dict for reward calculation)
        self.kills_this_step:    int   = 0
        self.damage_this_step:   float = 0.0
        self.items_this_step:    int   = 0
        self.shotgun_this_step:  bool  = False
        self.dodge_success:      bool  = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reinitialise the simulation for a new episode."""
        self.player_x     = 25.0 + self._rng.uniform(-2.0, 2.0)
        self.player_y     = 42.0
        self.player_dir   = -math.pi / 2
        self.player_health = 100.0
        self.ammo_clip    = self.ammo_clip_start
        self.ammo_reserve = self.ammo_res_start
        self.player_aiming     = False
        self.invincibility_timer = 0
        self.dodge_timer         = 0

        self.sim_step          = 0
        self.wave_index        = 0
        self.wave_kills        = 0
        self.total_kills       = 0
        self.items_collected   = 0
        self.shotgun_collected = False
        self.barn_visited      = False
        self.bell_area_reached = False

        # Spawn all items at their fixed positions (randomise ammo amounts slightly)
        self.items = []
        for (x, y, t) in ITEM_SPAWNS:
            itype = {"ammo": ItemType.AMMO, "herb": ItemType.HERB,
                     "shotgun": ItemType.SHOTGUN}[t]
            # Small random offset so each episode's item layout is slightly different
            ox = self._rng.uniform(-1.0, 1.0) if itype != ItemType.SHOTGUN else 0.0
            oy = self._rng.uniform(-1.0, 1.0) if itype != ItemType.SHOTGUN else 0.0
            self.items.append(ItemPickup(x + ox, y + oy, itype))

        # Spawn Wave 1
        self.enemies = []
        self._spawn_wave(0)

    def step(self, action: List[int]) -> None:
        """
        Advance the simulation by one step given a MultiDiscrete action.

        action indices (same as real environment.py):
          [0] movement:  0=stop 1=fwd 2=back 3=left 4=right 5–8=diagonals
          [1] camera:    0=none 1=cam-left 2=cam-right (rotates facing direction)
          [2] interact:  0=none 1=interact (collect nearby items)
          [3] combat:    0=none 1=aim 2=shoot 3=aim+shoot
          [4] evasion:   0=none 1=dodge 2=sprint (modifies movement)
          [5] inventory: 0=none 1=toggle  (in sim: does nothing, just tracked)

        Everything is computed in pure Python — no game, no sleep, no I/O.
        """
        mv, cam, inter, comb, ev, inv = action

        # Reset per-step events
        self.kills_this_step   = 0
        self.damage_this_step  = 0.0
        self.items_this_step   = 0
        self.shotgun_this_step = False
        self.dodge_success     = False

        # ── 1. Player rotation (camera action) ───────────────────────────────
        # In 2D top-down, "camera left/right" rotates the player's facing
        # direction.  This is how the agent aims at enemies.
        if cam == 1:
            self.player_dir -= 0.15   # rotate CCW
        elif cam == 2:
            self.player_dir += 0.15   # rotate CW
        # Normalise angle to [-π, π]
        self.player_dir = (self.player_dir + math.pi) % (2 * math.pi) - math.pi

        # ── 2. Combat: aim / shoot ────────────────────────────────────────────
        # Aim  = player enters aim stance (affects shoot accuracy)
        # Shoot = fire one bullet in facing direction if anything in cone
        if comb == 1 or comb == 3:
            self.player_aiming = True
        elif comb == 0:
            self.player_aiming = False

        if comb == 2 or comb == 3:
            kills = self._do_shoot()
            self.kills_this_step += kills

        # ── 3. Movement ───────────────────────────────────────────────────────
        # Direction vectors relative to player facing direction
        # (fwd = along player_dir, left = player_dir - π/2, etc.)
        speed = self.player_speed
        if ev == 2:   # Sprint
            speed *= self.sprint_mult
        # Aiming ROOTS you in RE4 Remake — Leon can only shuffle while the gun
        # is raised.  Without this cost the sim policy learns to hold aim+shoot
        # permanently (it's free), and that habit transfers to the real game as
        # "frozen Leon".  Matching the physics makes constant-aim genuinely
        # expensive, so the policy learns to aim only when it wants to fire.
        if self.player_aiming:
            speed *= float(self._cfg.get("aim_move_mult", 0.3))

        # Is this step a dodge?  Dodge = brief i-frame burst in a direction.
        is_dodge = (ev == 1 and self.dodge_timer <= 0)
        if is_dodge:
            self.dodge_timer = self.dodge_cooldown
            speed *= 2.5   # Fast dodge lunge
            self.invincibility_timer = 5   # 5 I-frames

        # Build movement vector from action index
        # Analogy: a joystick has 8 directions + stop — we map them to (dx, dy)
        _MV_DIR: Dict[int, Tuple[float, float]] = {
            # fwd/back/left/right are relative to player facing (player_dir)
            1: ( math.cos(self.player_dir),          math.sin(self.player_dir)),
            2: (-math.cos(self.player_dir),         -math.sin(self.player_dir)),
            3: ( math.cos(self.player_dir - math.pi/2), math.sin(self.player_dir - math.pi/2)),
            4: ( math.cos(self.player_dir + math.pi/2), math.sin(self.player_dir + math.pi/2)),
            5: ( math.cos(self.player_dir - math.pi/4), math.sin(self.player_dir - math.pi/4)),
            6: ( math.cos(self.player_dir + math.pi/4), math.sin(self.player_dir + math.pi/4)),
            7: (-math.cos(self.player_dir - math.pi/4),-math.sin(self.player_dir - math.pi/4)),
            8: (-math.cos(self.player_dir + math.pi/4),-math.sin(self.player_dir + math.pi/4)),
        }
        if mv in _MV_DIR:
            dx, dy = _MV_DIR[mv]
            new_x = self.player_x + dx * speed
            new_y = self.player_y + dy * speed
            self.player_x, self.player_y = self._try_move(
                self.player_x, self.player_y, new_x, new_y
            )

        # ── 4. Interact: collect items ────────────────────────────────────────
        if inter == 1:
            self._try_collect()

        # ── 5. Update enemy AI ────────────────────────────────────────────────
        for enemy in self.enemies:
            self._update_enemy(enemy)

        # ── 6. Enemy attacks / damage ─────────────────────────────────────────
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        else:
            for enemy in self.enemies:
                if enemy.state == EnemyState.ATTACK and enemy.attack_timer <= 0:
                    dist = self._dist(self.player_x, self.player_y, enemy.x, enemy.y)
                    if dist <= self.attack_range:
                        self.player_health -= self.attack_damage
                        self.damage_this_step += self.attack_damage
                        enemy.attack_timer = self.attack_cd
                        self.invincibility_timer = 8  # Brief i-frames after hit

        # ── 7. Check objective zones ──────────────────────────────────────────
        if _point_in_aabb(self.player_x, self.player_y, *BARN_ZONE) and not self.barn_visited:
            self.barn_visited = True
        if _point_in_aabb(self.player_x, self.player_y, *BELL_ZONE):
            self.bell_area_reached = True

        # ── 8. Wave progression ───────────────────────────────────────────────
        # When enough enemies die, spawn the next wave (up to n_waves)
        active = sum(1 for e in self.enemies if e.state != EnemyState.DEAD)
        if (active <= 3 and self.wave_index < self.n_waves - 1):
            self.wave_index += 1
            self._spawn_wave(self.wave_index)

        # ── 9. Ammo reload ───────────────────────────────────────────────────
        # Auto-reload when clip is empty and reserve is available.
        # In the real game the player manually reloads; here we auto-reload
        # to keep the action space focused on movement/combat strategy.
        if self.ammo_clip == 0 and self.ammo_reserve > 0:
            reload_amount = min(self.ammo_clip_start, self.ammo_reserve)
            self.ammo_clip     = reload_amount
            self.ammo_reserve -= reload_amount

        # ── 10. Timers ────────────────────────────────────────────────────────
        if self.dodge_timer > 0:
            self.dodge_timer -= 1

        self.sim_step += 1

    # ── Derived properties ─────────────────────────────────────────────────────

    @property
    def player_dead(self) -> bool:
        return self.player_health <= 0.0

    @property
    def episode_success(self) -> bool:
        """True when the survival timer expires — analogous to the bell ringing."""
        return self.sim_step >= self.max_steps

    @property
    def enemy_count_active(self) -> int:
        return sum(1 for e in self.enemies if e.state not in (EnemyState.DEAD,))

    @property
    def health_pct(self) -> float:
        return max(0.0, min(1.0, self.player_health / 100.0))

    def get_hud(self) -> Dict:
        """
        Return a HUD dict in the exact format expected by environment.py's _build_obs.
        Keys match the real game's PerceptionSystem.read_hud() output.
        """
        return {
            "health_pct": self.health_pct,
            "ammo_clip":  self.ammo_clip,
            "ammo_res":   self.ammo_reserve,
            "enemy_count": self.enemy_count_active,
        }

    def get_detections(self) -> List[Dict]:
        """
        Return a detections list in the format expected by environment.py.
        Each entry mimics a YOLO detection dict (label, confidence, bbox).

        VISION-LIMITED, like the real game: YOLO only sees enemies that are ON
        SCREEN, so the sim only reports enemies within `vision_range` AND
        within a ~140° cone around the player's facing direction.  The old
        omniscient version returned every living enemy map-wide, which (a) fed
        the policy an enemy count the real game can never produce (sim-to-real
        distribution shift in the HUD vector) and (b) made "no enemy in sight"
        essentially never true, neutering the aim-at-nothing penalty.
        """
        vision_range = float(self._cfg.get("vision_range", 22.0))
        half_fov     = math.radians(float(self._cfg.get("vision_fov_deg", 140.0)) / 2.0)
        out = []
        for enemy in self.enemies:
            if enemy.state in (EnemyState.DEAD,):
                continue
            dx, dy = enemy.x - self.player_x, enemy.y - self.player_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > vision_range:
                continue
            ang_diff = abs(
                (math.atan2(dy, dx) - self.player_dir + math.pi) % (2 * math.pi) - math.pi
            )
            if ang_diff > half_fov:
                continue
            out.append({
                "label": "enemy",
                "confidence": 0.95,
                "bbox": [enemy.x, enemy.y, enemy.x + 2.0, enemy.y + 2.0],
            })
        return out

    def get_snapshot(self) -> Dict:
        """
        Lightweight serialisable snapshot for the dashboard.
        Sent from worker processes to the main process every N steps
        via the info dict.  Intentionally small — no numpy arrays.
        """
        return {
            "player_pos":      (round(self.player_x, 1), round(self.player_y, 1)),
            "player_health":   round(self.player_health, 1),
            "player_aiming":   self.player_aiming,
            "ammo_clip":       self.ammo_clip,
            "ammo_reserve":    self.ammo_reserve,
            "enemy_positions": [
                (round(e.x, 1), round(e.y, 1), int(e.state))
                for e in self.enemies
                if e.state != EnemyState.DEAD
            ],
            "items_remaining": sum(1 for it in self.items if not it.collected),
            "shotgun_collected": self.shotgun_collected,
            "barn_visited":    self.barn_visited,
            "bell_reached":    self.bell_area_reached,
            "total_kills":     self.total_kills,
            "step":            self.sim_step,
            "max_steps":       self.max_steps,
            "wave":            self.wave_index + 1,
        }

    # ── Rendering ──────────────────────────────────────────────────────────────

    def render_frame(self, obs_h: int = 84, obs_w: int = 84) -> np.ndarray:
        """
        Render the current state as a top-down pixel image.

        Returns an (obs_h, obs_w, 3) uint8 RGB array.
        This is the "frame" observation fed to the CNN — exactly the same
        role as the real game's screen capture.

        Performance: ~20–50 µs per call using numpy slice operations.
        No OpenCV required.

        Colour key:
          [40,40,40]  — obstacle (dark grey)
          [0,220,0]   — player (green)
          [0,160,0]   — player direction dot
          [220,0,0]   — active enemy (red)
          [220,120,0] — stunned enemy (orange)
          [150,0,0]   — dead enemy (dark red, fades)
          [220,200,0] — ammo pickup (yellow)
          [0,200,200] — herb pickup (cyan)
          [220,180,50]— shotgun pickup (gold)
          [20,20,70]  — objective zone tint (blue overlay)
        """
        img = np.zeros((obs_h, obs_w, 3), dtype=np.uint8)

        sx = obs_w / MAP_W
        sy = obs_h / MAP_H

        def _px(world_x: float, world_y: float) -> Tuple[int, int]:
            """World → pixel coordinates, clamped to image bounds."""
            px = int(min(max(world_x * sx, 0), obs_w - 1))
            py = int(min(max(world_y * sy, 0), obs_h - 1))
            return px, py

        def _fill_rect(x1, y1, x2, y2, colour):
            """Fill a world-coordinate rectangle with a colour."""
            px1, py1 = _px(x1, y1)
            px2, py2 = _px(x2, y2)
            img[py1:py2 + 1, px1:px2 + 1] = colour

        def _draw_dot(x, y, radius, colour):
            """Draw a filled square 'dot' of given world-unit radius."""
            r = max(1, int(radius * sx))
            px_, py_ = _px(x, y)
            img[max(0, py_ - r): py_ + r + 1,
                max(0, px_ - r): px_ + r + 1] = colour

        # ── Background (ground) ──────────────────────────────────────────────
        img[:] = [15, 15, 15]  # Very dark background

        # ── Objective zone tint ───────────────────────────────────────────────
        for zone in [BARN_ZONE, BELL_ZONE]:
            x1, y1, x2, y2 = zone
            px1, py1 = _px(x1, y1)
            px2, py2 = _px(x2, y2)
            img[py1:py2 + 1, px1:px2 + 1] = np.clip(
                img[py1:py2 + 1, px1:px2 + 1].astype(np.int16) + [10, 10, 40],
                0, 255
            ).astype(np.uint8)

        # ── Obstacles ─────────────────────────────────────────────────────────
        for (x1, y1, x2, y2) in OBSTACLES:
            _fill_rect(x1, y1, x2, y2, [40, 40, 40])

        # ── Items ─────────────────────────────────────────────────────────────
        _ITEM_COLOURS = {
            ItemType.AMMO:    [220, 200,  0],
            ItemType.HERB:    [  0, 200, 200],
            ItemType.SHOTGUN: [220, 180,  50],
        }
        for item in self.items:
            if not item.collected:
                _draw_dot(item.x, item.y, 0.6, _ITEM_COLOURS[item.item_type])

        # ── Enemies ───────────────────────────────────────────────────────────
        _ENEMY_COLOURS = {
            EnemyState.PATROL:  [160, 40,  40],
            EnemyState.CHASE:   [220,  0,   0],
            EnemyState.ATTACK:  [255, 50,  50],
            EnemyState.STUNNED: [220, 120,  0],
            EnemyState.DEAD:    [ 60,  0,   0],
        }
        for enemy in self.enemies:
            if enemy.state == EnemyState.DEAD and enemy.dead_timer > 20:
                continue  # Fully faded — skip render
            colour = _ENEMY_COLOURS.get(enemy.state, [180, 0, 0])
            _draw_dot(enemy.x, enemy.y, 0.8, colour)

        # ── Player ────────────────────────────────────────────────────────────
        # Main dot
        _draw_dot(self.player_x, self.player_y, 1.0, [0, 220, 0])

        # Direction indicator: small pixel at facing direction
        dir_world = 3.0  # 3 units ahead in facing direction
        dx_ = self.player_x + math.cos(self.player_dir) * dir_world
        dy_ = self.player_y + math.sin(self.player_dir) * dir_world
        _draw_dot(dx_, dy_, 0.4, [0, 160, 0])

        # Aim line when aiming (longer, brighter)
        if self.player_aiming:
            for t in [4.0, 7.0, 10.0]:
                lx = self.player_x + math.cos(self.player_dir) * t
                ly = self.player_y + math.sin(self.player_dir) * t
                if 0 < lx < MAP_W and 0 < ly < MAP_H:
                    ppx, ppy = _px(lx, ly)
                    img[ppy, ppx] = [0, 200, 80]

        return img

    # ── Private helpers ────────────────────────────────────────────────────────

    def _spawn_wave(self, wave_idx: int) -> None:
        """
        Spawn one wave of enemies at their predefined positions.
        Positions are jittered slightly so every episode is different.
        """
        if wave_idx >= len(WAVE_SPAWNS):
            return
        for (sx, sy) in WAVE_SPAWNS[wave_idx]:
            jx = sx + self._rng.uniform(-1.5, 1.5)
            jy = sy + self._rng.uniform(-1.5, 1.5)
            jx, jy = self._clip_to_map(jx, jy)
            enemy = EnemyAgent(
                x=jx, y=jy,
                hp=100.0,
                state=EnemyState.PATROL,
                patrol_angle=self._rng.uniform(0, 2 * math.pi),
                patrol_timer=self._rng.randint(10, 30),
                enemy_id=len(self.enemies),
            )
            self.enemies.append(enemy)

    def _update_enemy(self, enemy: EnemyAgent) -> None:
        """Advance enemy AI by one step."""
        if enemy.state == EnemyState.DEAD:
            enemy.dead_timer += 1
            return

        # Cooldown timers
        if enemy.stun_timer > 0:
            enemy.stun_timer -= 1
            if enemy.stun_timer == 0:
                enemy.state = EnemyState.CHASE
            return  # No movement while stunned

        if enemy.attack_timer > 0:
            enemy.attack_timer -= 1

        dist = self._dist(self.player_x, self.player_y, enemy.x, enemy.y)

        if enemy.state == EnemyState.PATROL:
            if dist <= self.detect_range:
                enemy.state = EnemyState.CHASE
            else:
                self._patrol_step(enemy)

        elif enemy.state == EnemyState.CHASE:
            if dist <= self.attack_range:
                enemy.state = EnemyState.ATTACK
            else:
                self._move_toward_player(enemy, speed=self.enemy_chase_spd)
                # Lose the player if too far away (re-patrol)
                if dist > self.detect_range * 1.5:
                    enemy.state = EnemyState.PATROL

        elif enemy.state == EnemyState.ATTACK:
            if dist > self.attack_range * 1.5:
                enemy.state = EnemyState.CHASE

    def _patrol_step(self, enemy: EnemyAgent) -> None:
        """Move enemy on a random patrol path."""
        if enemy.patrol_timer <= 0:
            # Pick a new patrol direction
            enemy.patrol_angle = self._rng.uniform(0, 2 * math.pi)
            enemy.patrol_timer = self._rng.randint(15, 40)

        enemy.patrol_timer -= 1
        nx = enemy.x + math.cos(enemy.patrol_angle) * self.enemy_speed
        ny = enemy.y + math.sin(enemy.patrol_angle) * self.enemy_speed
        new_x, new_y = self._try_move(enemy.x, enemy.y, nx, ny)

        # If blocked (hit an obstacle), change direction
        if abs(new_x - nx) > 0.01 or abs(new_y - ny) > 0.01:
            enemy.patrol_angle = self._rng.uniform(0, 2 * math.pi)
            enemy.patrol_timer = 0

        enemy.x, enemy.y = new_x, new_y

    def _move_toward_player(self, enemy: EnemyAgent, speed: float) -> None:
        """Move enemy directly toward the player."""
        dx = self.player_x - enemy.x
        dy = self.player_y - enemy.y
        dist = max(0.001, math.sqrt(dx * dx + dy * dy))
        nx, ny = dx / dist, dy / dist
        new_x = enemy.x + nx * speed
        new_y = enemy.y + ny * speed
        enemy.x, enemy.y = self._try_move(enemy.x, enemy.y, new_x, new_y)

    def _do_shoot(self) -> int:
        """
        Fire one bullet in the player's facing direction.
        Returns the number of enemies killed this shot.

        Hit detection uses an aim cone: any enemy within shoot_range and within
        ±shoot_cone radians of the player's facing direction is a valid target.
        The closest valid target is hit first (simulating aimed fire).

        Analogy: a flashlight beam — enemies caught in the beam get hit.
        Wider cone = more lenient aiming (hip-fire); narrower = precise (aimed).
        """
        if self.ammo_clip <= 0:
            return 0

        self.ammo_clip -= 1

        # Effective cone: aiming = full precision; hip-fire = wider cone
        effective_cone = self.shoot_cone if self.player_aiming else self.shoot_cone * 2.0

        # Find the closest enemy in the cone
        best_enemy = None
        best_dist  = self.shoot_range

        for enemy in self.enemies:
            if enemy.state == EnemyState.DEAD:
                continue
            dx = enemy.x - self.player_x
            dy = enemy.y - self.player_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.shoot_range:
                continue

            angle_to_enemy = math.atan2(dy, dx)
            angle_diff = abs(
                (angle_to_enemy - self.player_dir + math.pi) % (2 * math.pi) - math.pi
            )
            if angle_diff <= effective_cone and dist < best_dist:
                best_dist  = dist
                best_enemy = enemy

        if best_enemy is None:
            return 0  # Miss

        # Damage calculation:
        #   Aimed + close range = potential headshot (instant kill)
        #   Aimed + medium range = 40-60 damage
        #   Hip-fire = 15-30 damage (inaccurate in RE4 lore)
        if self.player_aiming:
            if best_dist < 4.0:
                damage = 100.0   # Close headshot — instant kill
            else:
                damage = self._rng.uniform(35.0, 60.0)
        else:
            damage = self._rng.uniform(15.0, 30.0)

        best_enemy.hp -= damage

        if best_enemy.hp <= 0:
            best_enemy.state   = EnemyState.DEAD
            best_enemy.dead_timer = 0
            self.total_kills  += 1
            self.kills_this_step += 1
            return 1
        else:
            best_enemy.state      = EnemyState.STUNNED
            best_enemy.stun_timer = 6 if self.player_aiming else 3
            return 0

    def _try_collect(self) -> None:
        """Collect any item within interaction range of the player."""
        COLLECT_RANGE = 2.5  # World units
        for item in self.items:
            if item.collected:
                continue
            dist = self._dist(self.player_x, self.player_y, item.x, item.y)
            if dist <= COLLECT_RANGE:
                item.collected = True
                self.items_collected += 1
                if item.item_type == ItemType.AMMO:
                    self.ammo_reserve += self.ammo_pickup_amt
                    self.ammo_reserve = min(self.ammo_reserve, 99)
                elif item.item_type == ItemType.HERB:
                    self.player_health = min(100.0, self.player_health + 35.0)
                elif item.item_type == ItemType.SHOTGUN:
                    self.shotgun_collected = True
                    self.shotgun_this_step = True
                self.items_this_step += 1

    @staticmethod
    def _is_free(x: float, y: float) -> bool:
        """True if (x, y) is inside map bounds and outside every obstacle."""
        if not (0.3 <= x <= MAP_W - 0.3 and 0.3 <= y <= MAP_H - 0.3):
            return False
        for (x1, y1, x2, y2) in OBSTACLES:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return False
        return True

    def _try_move(self, old_x: float, old_y: float,
                  new_x: float, new_y: float) -> Tuple[float, float]:
        """
        Axis-separated slide movement with REJECTING collision.

        The old `_clip_to_map` "push out of the nearest edge" logic could push
        an entity THROUGH a perimeter wall to a position OUTSIDE the map
        (e.g. x=50.2 beyond the east wall).  The trained policy discovered
        this and camped out-of-bounds where enemies couldn't reach — an
        exploit, not gameplay.  Rejecting invalid moves (with axis slide so
        walls can still be skimmed along) makes wall-embedding impossible.
        """
        if self._is_free(new_x, new_y):
            return new_x, new_y
        if self._is_free(new_x, old_y):      # slide along x
            return new_x, old_y
        if self._is_free(old_x, new_y):      # slide along y
            return old_x, new_y
        return old_x, old_y                  # fully blocked — stay put

    def _free_spawn(self, x: float, y: float) -> Tuple[float, float]:
        """Nudge a spawn position toward the map centre until it's collision-free."""
        for _ in range(60):
            if self._is_free(x, y):
                return x, y
            cx, cy = MAP_W / 2.0, MAP_H / 2.0
            dx, dy = cx - x, cy - y
            dist = max(0.001, math.sqrt(dx * dx + dy * dy))
            x += dx / dist
            y += dy / dist
        return MAP_W / 2.0, MAP_H / 2.0

    def _clip_to_map(self, x: float, y: float) -> Tuple[float, float]:
        """Legacy alias kept for spawn call-sites — resolves to a free position."""
        return self._free_spawn(x, y)

    @staticmethod
    def _dist(ax: float, ay: float, bx: float, by: float) -> float:
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _point_in_aabb(px: float, py: float,
                   x1: float, y1: float, x2: float, y2: float) -> bool:
    return x1 <= px <= x2 and y1 <= py <= y2


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    cfg = {
        "player_speed": 0.6, "sprint_multiplier": 1.8, "dodge_cooldown": 8,
        "enemy_speed": 0.28, "enemy_chase_speed": 0.40,
        "enemy_detection_range": 14.0, "enemy_attack_range": 1.8,
        "enemy_attack_damage": 12.0, "enemy_attack_cooldown": 15,
        "shoot_range": 20.0, "shoot_cone_deg": 18.0,
        "ammo_start_clip": 15, "ammo_start_reserve": 50,
        "ammo_pickup_amount": 10, "enemy_waves": 3,
        "wave_trigger_kills": 5, "max_episode_steps": 600,
    }

    sim = SimGameState(cfg, rng_seed=42)
    sim.reset()

    start = time.perf_counter()
    N = 10_000
    for i in range(N):
        action = [
            random.randint(0, 8), random.randint(0, 4), random.randint(0, 1),
            random.randint(0, 3), random.randint(0, 2), 0,
        ]
        sim.step(action)
        if sim.player_dead or sim.episode_success:
            sim.reset()

    elapsed = time.perf_counter() - start
    print(f"SimGameState: {N:,} steps in {elapsed:.3f}s  ({N/elapsed:,.0f} steps/sec)")

    frame = sim.render_frame()
    print(f"render_frame shape: {frame.shape}, dtype: {frame.dtype}")
    print("Smoke test PASSED.")
