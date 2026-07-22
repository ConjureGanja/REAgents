# Resident Evil 4 Remake — Strategic Knowledge Base
# Used by the Claude planning node for high-level guidance.

WALKTHROUGH: dict = {
    "Chapter 1": {
        "objective": "Survive the Village Square siege. The bell will ring after ~3 minutes; all enemies retreat automatically.",
        "optimal_strategy": (
            "PRIORITY 1 — Get the Shotgun: Run north through the village to the two-storey farmhouse. "
            "Climb the ladder or stairs inside; the W-870 shotgun is on the wall upstairs. Do NOT fight on the way — sprint past enemies. "
            "PRIORITY 2 — Survive Chainsaw Man (Dr. Salvador): He appears from the south. Keep maximum distance. "
            "If cornered, sprint to a different area. Do NOT waste handgun ammo on him (he regenerates). "
            "Use the shotgun at close range for a knockback. "
            "PRIORITY 3 — Use the environment: windows upstairs allow you to jump out and break line-of-sight. "
            "The well area (centre of village) has a yellow herb — grab it but don't stop moving. "
            "PRIORITY 4 — Knife parry: Tap the knife button (LBumper then RBumper quickly) when an enemy lunges — "
            "this deflects the attack for zero damage. Saves enormous ammo against the mob. "
            "Once the bell rings, ALL enemies retreat. Stop fighting and explore for items."
        ),
        "puzzle": None,
        "key_items": ["W-870 Shotgun (north farmhouse, upstairs wall)", "Yellow Herb (near the well)", "Green Herb (south fences)"],
        "threats": [
            "Chainsaw Man / Dr. Salvador — extremely dangerous, keep distance, never aim long",
            "Village mob (El Ganados) — numerous, knife-parry to conserve ammo",
        ],
        "rl_hints": (
            "For the RL agent: the most important early behaviour is to move NORTH (forward+left from spawn). "
            "Reward navigation toward the farmhouse over fighting. "
            "Rewarding survival time is key — the bell rings at ~180 s into the episode. "
            "After the bell, reward exploration of the full village map."
        ),
    },
    "Chapter 2": {
        "objective": "Retrieve the Hexagonal Emblem from the Valley.",
        "optimal_strategy": (
            "Use stealth in the Valley; avoid alerting the full camp. "
            "Grab the emblem and exit immediately. Avoid combat where possible to save resources."
        ),
        "puzzle": "Village Chief Manor: symbol combination is Crop → Pig → Baby (Up, Left, Right).",
        "key_items": ["Hexagonal Emblem", "Spinel (sell)", "Elegant Mask (sell)"],
        "threats": ["Armoured Ganados", "Dog pack"],
    },
    "Chapter 3": {
        "objective": "Navigate to the Farm and reach the Village Chief's Manor.",
        "optimal_strategy": (
            "Combine all coloured gems with their matching jewellery before selling. "
            "Unlock the church before heading to the Manor — saves backtracking."
        ),
        "puzzle": "Farm well contains an Emerald; use crank at the windmill to drain the water.",
        "key_items": ["Church Insignia Key", "Emerald (combine)"],
        "threats": ["El Gigante — use dog whistle if found; aim for Plaga on its back"],
    },
    "Chapter 4": {
        "objective": "Cross the Lake and reach the Church.",
        "optimal_strategy": (
            "Get the Red9 handgun from the sunken shipwreck at lake centre — best handgun in the game. "
            "Del Lago boss: keep the boat moving to dodge tentacles, harpoon when surfaced."
        ),
        "puzzle": (
            "Large Cave Shrine: Two-Hands, Fish-like, Three-pointed. "
            "Small Cave Shrine: Triple-snake, Umbrella, Square-cross."
        ),
        "key_items": ["Red9 Handgun", "Church Grail", "Lake Map"],
        "threats": ["Del Lago (lake monster)"],
    },
    "Chapter 5": {
        "objective": "Rescue Ashley from the Church.",
        "optimal_strategy": (
            "Church laser puzzle: rotate mirrors to align beams with targets. "
            "After rescue, prioritise keeping Ashley behind you and away from Ganados."
        ),
        "puzzle": "Church laser puzzle — rotate blue, green, red mirrors to match target symbols.",
        "key_items": ["Blue, Green, Red Stones (church puzzle)"],
        "threats": ["Illuminados cultists", "Novistadores (insects in sewers)"],
    },
    "Chapter 6": {
        "objective": "Escape the Village with Ashley through the lake.",
        "optimal_strategy": (
            "Upgrade Red9 stock and firepower at the merchant before this chapter. "
            "Cabin siege: use the upstairs window to funnel enemies; grenades at stairs."
        ),
        "puzzle": None,
        "key_items": ["TMP (submachine gun)", "Shotgun upgrade"],
        "threats": ["Cabin siege — large Ganado wave", "Bella Sisters (chainsaw duo)"],
    },
    "Chapter 7": {
        "objective": "Infiltrate Salazar's Castle.",
        "optimal_strategy": (
            "Throw a Heavy Grenade at the AA cannon to skip the manual sequence. "
            "Use the TMP on the armoured knights — aim for the neck joint."
        ),
        "puzzle": "Treasury Swords: place Iron, Golden, Stripped, Rusted left to right.",
        "key_items": ["Salazar's Castle Key", "Golden Sword"],
        "threats": ["Armoured knights", "Garrador (blind clawed enemy — be silent)"],
    },
    "Chapter 8": {
        "objective": "Navigate the Castle interior.",
        "optimal_strategy": (
            "Garrador encounter: ring the bells on each side to stun, then knife its Plaga. "
            "The lava room: prioritise timing, not combat."
        ),
        "puzzle": "Lava room pedestal — rotate platform segments in correct order.",
        "key_items": ["Castle Gate Key", "Purple Gem"],
        "threats": ["Garrador", "Iron Maiden"],
    },
    "Chapter 9": {
        "objective": "Find the two crests for the double-door.",
        "optimal_strategy": (
            "Two-crest puzzle: the shooting gallery mini-game earns the Punisher (free magnum). "
            "Save Flash Grenades — Plaga heads die instantly to flash."
        ),
        "puzzle": "Two crests found via library maze and the waterway lever sequence.",
        "key_items": ["Serpent Ornament", "Goat Ornament"],
        "threats": ["Plaga-headed Ganados", "Regenerator (thermal scope required)"],
    },
    "Chapter 10": {
        "objective": "Defeat Salazar's right hand and progress through the maze.",
        "optimal_strategy": (
            "Verdugo boss: buy Rocket Launcher from merchant — one-shot kill when frozen by liquid nitrogen. "
            "Freeze with the nitrogen pipes first, then rocket."
        ),
        "puzzle": "Hedge maze — follow the torches; statue activation order shown on the map.",
        "key_items": ["Rocket Launcher (for Verdugo)", "Castle Insignia"],
        "threats": ["Verdugo (regenerating armoured enemy)"],
    },
    "Chapter 11": {
        "objective": "Escape the water treatment area.",
        "optimal_strategy": (
            "Sewers: conserve ammo — most enemies can be kited. "
            "Regenerator: mandatory thermal scope; shoot all Plagas parasites on its body."
        ),
        "puzzle": "Sewage gate valve sequence — turn valves in numbered order on wall diagram.",
        "key_items": ["Thermal Scope", "Sewage Key"],
        "threats": ["Regenerator", "Iron Maiden (spiked Regenerator)"],
    },
    "Chapter 12": {
        "objective": "Defeat Salazar (boss).",
        "optimal_strategy": (
            "Throw a Golden Chicken Egg at Salazar's face for ~70% instant damage — huge time saver. "
            "Then shoot the eye of the giant Ramon hand. "
            "Avoid the tentacle sweeps by moving to the safe platform segments."
        ),
        "puzzle": None,
        "key_items": ["Golden Chicken Egg", "Rocket Launcher (backup)"],
        "threats": ["Salazar (stage 1: tentacles; stage 2: hand-eye)"],
    },
    "Chapter 13": {
        "objective": "Board the military island via cargo plane.",
        "optimal_strategy": (
            "Stock up at the merchant before departure — island merchants are sparse. "
            "Upgrade the shotgun spread and handgun firepower."
        ),
        "puzzle": None,
        "key_items": ["Island Key", "Yellow Herb", "First Aid Spray"],
        "threats": ["Soldier-class Ganados with rifles and grenades"],
    },
    "Chapter 14": {
        "objective": "Rescue Ashley from the island lab.",
        "optimal_strategy": (
            "Laser hallway: crouch and time movements to laser pulses. "
            "Jet-ski escort: keep Ashley behind and use grenades on clusters."
        ),
        "puzzle": "Lab keycard doors — keypads show 4-digit codes found on nearby whiteboards.",
        "key_items": ["Lab Keycard A/B", "Plaga Sample"],
        "threats": ["U-3 boss — shoot weak points; avoid acid spray"],
    },
    "Chapter 15": {
        "objective": "Reach Saddler's lair across the island.",
        "optimal_strategy": (
            "U-3 (It) boss: bait to open, shoot the tentacle core. "
            "Saddler approach: use the rocket launcher from crate — saves huge ammo."
        ),
        "puzzle": "Island radar dish alignment — press buttons to align dish to specific frequency.",
        "key_items": ["Rocket Launcher", "Flash Grenade"],
        "threats": ["Saddler's mini-bosses", "U-3"],
    },
    "Chapter 16": {
        "objective": "Final boss — defeat Saddler.",
        "optimal_strategy": (
            "Saddler: shoot the exposed eye on his legs and torso to expose his main eye. "
            "Use the special Rocket Launcher Ada throws — instant kill when eye exposed. "
            "Jet-ski escape: follow the on-screen prompts; no combat required."
        ),
        "puzzle": None,
        "key_items": ["Ada's Rocket Launcher (scripted)"],
        "threats": ["Saddler (final form — multiple eyes)"],
    },
}

GENERAL_TIPS: list[str] = [
    "Knife parrying saves enormous ammo — tap knife (LB+RB quickly) at the last second before a melee hit. Works on almost every enemy melee attack.",
    "Shoot enemy LEGS to trip them, then run up and press melee (A or X) for a free kick — zero ammo cost, significant damage.",
    "Sprint (L31) away from crowds rather than shooting — conserve ammo for the shotgun house run.",
    "Combine treasures with matching coloured gems (5 gems = maximum value multiplier — never sell unmatched gems).",
    "Flash grenades instantly kill exposed Plagas parasites — extremely efficient (one grenade = multiple kills).",
    "Always buy a Rocket Launcher before Verdugo and Saddler fights; saves minutes of frustrating combat.",
    "Upgrade Red9 firepower and stock first — it outperforms most handguns throughout the game.",
    "Yellow herbs permanently increase max health cap — NEVER sell them.",
    "The merchant's shooting gallery earns spinels for free upgrades — complete it early.",
    "Regenerators require the thermal scope; never waste bullets without it equipped.",
    "Keep at least one Flash Grenade in reserve for Plaga-headed enemies.",
    "In the village siege (Chapter 1): don't try to kill everyone. Survive until the bell rings (~3 min), then enemies retreat.",
    "The W-870 shotgun is in the north farmhouse (upstairs on the wall). Getting it early makes the siege survivable.",
    "At Critical / Danger health (red ring), healing is the ONLY priority — stop fighting, find a herb.",
]

# ── Village-specific tactical knowledge (Chapter 1) ───────────────────────────
VILLAGE_TACTICS: dict = {
    "spawn_area": {
        "description": "Leon spawns in the south of the village square.",
        "immediate_action": "Run north immediately. Do not engage the crowd.",
        "nearby_items": ["Green Herb near south fence", "Handgun ammo in barn"],
    },
    "shotgun_house": {
        "description": "Two-storey farmhouse in the north-west of the village.",
        "how_to_reach": "From spawn: run north, veer left past the well. Look for a ladder on the side of the house.",
        "shotgun_location": "Climb the ladder inside. Shotgun is on the wall of the upper floor.",
        "why_important": "Shotgun one-shots most village enemies at close range and can knock back Chainsaw Man.",
    },
    "chainsaw_man": {
        "description": "Dr. Salvador — appears ~60s into the siege. Brown bag over his head, carrying a chainsaw.",
        "behaviour": "Slow but unstoppable. Instant-kill grab if he reaches you. Does NOT respond to the bell.",
        "counter": "Keep 10+ metres distance. Sprint away. Shotgun blast to face = knockback + stun.",
        "avoid": "Never let him corner you. Never stand still within his chainsaw reach.",
    },
    "bell_tower": {
        "description": "Tower at the north end of the village with a large bell.",
        "trigger": "The bell rings automatically at ~180 seconds into the siege (scripted event).",
        "effect": "ALL normal Ganados stop attacking and return to their positions.",
        "post_bell": "After the bell: explore the whole village for items — grenades, herbs, handgun ammo.",
    },
    "well": {
        "description": "Well in the centre of the village square.",
        "items": ["Yellow Herb on the ground nearby — PRIORITISE picking this up"],
        "tactical": "Good central position to see approaching enemies from all sides.",
    },
}

MERCHANT_UPGRADES_PRIORITY: list[str] = [
    "Red9 — Firepower (highest DPS handgun in the game)",
    "Red9 — Stock (reduces aim sway significantly)",
    "Shotgun — Firepower (critical for crowd control)",
    "TMP — Firepower (best DPS vs armoured enemies)",
    "First Aid Spray x2 (emergency survival)",
    "Rocket Launcher (required for Verdugo/Saddler — buy fresh each time)",
]
