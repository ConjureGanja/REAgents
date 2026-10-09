"""
Shared constants for the RE agent.

Single source of truth for values that multiple modules must agree on.
"""

# Action space: movement / camera / interact / combat / evasion / inventory.
# environment.py builds spaces.MultiDiscrete from this; llm_agent.py clamps
# Claude's action overrides against it.  Changing these numbers changes the
# contract with any trained checkpoint — treat as a breaking change.
ACTION_SPACE_SIZES = (9, 5, 2, 4, 3, 2)
