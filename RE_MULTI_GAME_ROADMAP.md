# Multi-Game Roadmap — RE2 / RE3 / RE4 Remakes

Goal: one agent architecture that plays all three Remakes. RE4 is the working
baseline; RE2 and RE3 slot in as *profiles*, not rewrites. This works because
the two contracts the RL policy depends on never change per game:

- **Observation space** — `Dict{"frame": (84,84,12) uint8, "hud": (7,) float32}`
- **Action space** — `MultiDiscrete([9, 5, 2, 4, 3, 2])` (all three games share
  the same controller layout: move / camera / interact / aim+shoot / dodge-run / inventory)

Everything game-specific lives behind those contracts: HUD parsing, menu reset
sequences, guide data, and reward shaping details.

Analogy: the policy is a licensed driver; each game is a different rental car.
Same pedals and wheel — you only re-learn where the fuel gauge and door
handles are.

## What changes per game

| Layer | RE4 (done) | RE2 / RE3 (to do) |
|---|---|---|
| Window title | `RESIDENT EVIL 4` | `RESIDENT EVIL 2` / `RESIDENT EVIL 3` |
| Health read | Radial ring (colour mask, aim-only) | No on-screen bar — screen-edge red vignette + posture; needs a new detector |
| Ammo read | OCR on aim radial | OCR on bottom-right ammo counter (always visible while armed) |
| Death screen | "YOU ARE DEAD" red-on-black | Same style — likely reusable with re-tuned thresholds |
| Menu reset | Pause → Load Game (gamepad) | Same pattern, different item counts (`nav_down_count`) |
| Guide data | `guide_data.py` | `guide_data_re2.py` / `guide_data_re3.py` |
| Reward shaping | Village mob survival | RE2: exploration/backtracking-heavy; RE3: dodge-timing-heavy |

## Phases

### Phase 1 — Wire the profile loader (small code change)
Make config loading merge `game_profiles.<active_profile>` over
`game_settings`/`perception`/`reset`. One function, touched in one place
(`config` load in env/main). The scaffolding section already exists in
`config.yaml`.

- **You:** decide profile field names you want to tune per game.
- **AI (Claude):** write the merge function + tests. Trivial, low-risk.

### Phase 2 — RE2 perception calibration
RE2 has no persistent health HUD; health is inferred from the red screen
vignette and the inventory ECG. This is the hardest part of the port.

- **You (must be you — needs the game running):** capture reference frames at
  Fine/Caution/Danger states, screenshot menu layouts, record the pause-menu
  reset path. Click-by-click capture guide: run the game borderless 1080p →
  `python perception.py` → screenshot each health state → save to
  `calibration/re2/`.
- **AI:** build the vignette-based health estimator from your reference frames,
  tune HSV ranges, write the RE2 death-screen thresholds.
- **Hire? No.** Nothing here needs outside help; it's calibration labour.

### Phase 3 — RE2/RE3 guide data
- **AI:** draft `guide_data_re2.py` / `guide_data_re3.py` in the same dict
  format as `guide_data.py` (objective / strategy / key_items / threats per
  section). You review for accuracy against your own playthrough knowledge.

### Phase 4 — Sim generalisation (optional but recommended)
The 2D sim currently models the RE4 village. Add per-profile maps (RCPD
corridors for RE2 = narrow + backtracking; RE3 streets = open + dodge windows)
so sim-to-real transfer works per game.

- **AI:** new map layouts in `sim_game_state.py` behind the same interface.

### Phase 5 — Per-game training runs
Same pipeline, one profile flag. Expect RE2 to need heavier `exploration`
curriculum weighting (config-tunable now — the env reads
`curriculum.stages.*.reward_weights` from config as of this revision).

## Common pitfalls to avoid (learned from the RE4 build)

1. **Phantom deaths from HUD misreads** — RE4 cost weeks on this. For RE2/RE3,
   treat "health signal absent" as *unknown*, never as zero, from day one.
2. **Menu drift during reset** — always `release_all()` before menu navigation
   (persisted stick deflection scrolls menus). Already handled in `controls.py`;
   keep that pattern.
3. **Don't fork the env per game.** If you find yourself copying
   `environment.py` to `environment_re2.py`, stop — the difference belongs in a
   profile or in `perception.py`, not a parallel file that will rot.
