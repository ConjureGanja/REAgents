"""
Impala-style ResNet CNN feature extractor for Resident Evil 4.

WHY IMPALA CNN vs NatureCNN?
─────────────────────────────
NatureCNN (the SB3 default) was designed for classic Atari 84×84 games —
simple 2D sprites with flat colour backgrounds.  RE4 Remake is a full 3D game
with:
  • Complex lighting and shadows (Ganados blend into dark corners)
  • Fine detail you need to see (health ring colour, ammo counter, doorknobs)
  • Moving backgrounds (foliage, fire, rain in the village)

NatureCNN uses 3 plain conv layers with no skip connections.  Its gradients
shrink exponentially through each layer (vanishing gradient problem), so the
early layers barely learn.

Impala ResNet (Espeholt et al. 2018 "IMPALA: Scalable Distributed Deep-RL")
adds skip connections (residual shortcuts) that let gradients flow back to
the first layer unchanged — like a direct highway through the network.  Each
"ConvSequence" stage is:
    Conv → MaxPool → ResBlock → ResBlock
with the ResBlocks looking like:
    x → ReLU → Conv → ReLU → Conv → (+x) → output

This architecture achieves 2–3× better mean scores on complex visual games
compared to NatureCNN in published benchmarks.

WHAT THIS EXTRACTOR DOES
────────────────────────
1.  Processes the "frame" key (H × W × C) through the Impala CNN
2.  Processes the "hud" key (6-element float vector) through a small MLP
3.  Concatenates both outputs
4.  Projects through a final Linear → ReLU to `features_dim`

The output feeds into the LSTM (RecurrentPPO) or directly to the policy head
(plain PPO).

SHAPES (defaults: 84×84 grayscale, frame_stack=4  →  C=4)
──────────────────────────────────────────────────────────
  Input frame:            (B, 4, 84, 84)    [BCHW after permute]
  After stage 1 (ch=16):  (B, 16, 42, 42)   Conv(3,1,pad1) + MaxPool(3,2,pad1)
  After stage 2 (ch=32):  (B, 32, 21, 21)
  After stage 3 (ch=32):  (B, 32, 11, 11)
  After flatten:          (B, 3872)
  After concat with HUD:  (B, 3878)
  After linear:           (B, features_dim=512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ── Building blocks ───────────────────────────────────────────────────────────

class _ResidualBlock(nn.Module):
    """
    One Impala residual block:  skip + (ReLU → Conv3×3 → ReLU → Conv3×3)

    The skip connection means the block learns the *residual* (correction)
    on top of the identity.  Analogy: instead of learning from scratch, the
    block only learns 'what to add or subtract' from the existing signal.
    This makes gradients much easier to propagate.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Two 3×3 conv layers, padding=1 preserves spatial dimensions
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                           # Save skip connection
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual                    # Add skip — gradients flow freely here


class _ConvSequence(nn.Module):
    """
    One Impala stage:  Conv(3×3) → MaxPool(3×3, stride=2) → ResBlock → ResBlock

    The MaxPool halves the spatial dimensions while the Conv changes the
    channel count.  The two ResBlocks refine features at the new resolution.

    Example (input 84×84, out_channels=16):
        Conv(3×3, pad=1): 84×84 → 84×84 @ 16 ch   (no change in H,W)
        MaxPool(3,2,pad1): 84×84 → 42×42 @ 16 ch   (halved)
        ResBlock × 2:      42×42 @ 16 ch             (no change)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ResidualBlock(out_channels)
        self.res2 = _ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


# ── Feature extractor ─────────────────────────────────────────────────────────

class RE4FeaturesExtractor(BaseFeaturesExtractor):
    """
    Impala CNN + HUD MLP combined feature extractor.

    Compatible with SB3's RecurrentPPO (MultiInputLstmPolicy) and plain PPO
    (MultiInputPolicy).  Both accept a custom `features_extractor_class` in
    `policy_kwargs`.

    Parameters
    ──────────
    observation_space : spaces.Dict
        Must contain:
          "frame" — Box(H, W, C) uint8   [channels last, as gym convention]
          "hud"   — Box(7,)       float32  [health, clip, res, enemies, combat, time, ammo_delta]
    features_dim : int
        Output dimension fed to the LSTM / policy head.  Default 512.
    hud_hidden : int
        Hidden size of the HUD branch MLP.  Default 64.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 512,
        hud_hidden: int = 64,
    ):
        # We pass features_dim to the parent so SB3 knows the output size
        super().__init__(observation_space, features_dim=features_dim)

        # ── Frame branch (Impala CNN) ──────────────────────────────────────────
        frame_space = observation_space.spaces["frame"]
        # frame_space.shape is (H, W, C) — channels last (gym convention)
        h, w, n_channels = frame_space.shape

        self.cnn = nn.Sequential(
            # Stage 1: in_ch → 16, halve spatial dims
            # Example: (N, C, 84, 84) → (N, 16, 42, 42)
            _ConvSequence(n_channels, 16),

            # Stage 2: 16 → 32, halve again
            # Example: (N, 16, 42, 42) → (N, 32, 21, 21)
            _ConvSequence(16, 32),

            # Stage 3: 32 → 32, halve again
            # Example: (N, 32, 21, 21) → (N, 32, 11, 11)
            _ConvSequence(32, 32),

            # Final activation before flatten
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened CNN output size by doing a dummy forward pass
        with torch.no_grad():
            dummy_frame = torch.zeros(1, n_channels, h, w)
            cnn_out_dim = self.cnn(dummy_frame).shape[1]

        # ── HUD branch (small MLP) ─────────────────────────────────────────────
        hud_dim = observation_space.spaces["hud"].shape[0]  # 7 elements
        self.hud_mlp = nn.Sequential(
            # 6 raw HUD scalars → 64-dim embedding
            # This lets the network learn non-linear interactions between health,
            # ammo, enemy count, etc. rather than treating them independently.
            nn.Linear(hud_dim, hud_hidden),
            nn.ReLU(),
            nn.Linear(hud_hidden, hud_hidden),
            nn.ReLU(),
        )

        # ── Combined projection ────────────────────────────────────────────────
        combined_dim = cnn_out_dim + hud_hidden
        self.projection = nn.Sequential(
            # Project concatenated features down to features_dim
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"[RE4FeaturesExtractor] Built Impala CNN\n"
            f"  Frame input:  (H={h}, W={w}, C={n_channels})\n"
            f"  CNN output:   {cnn_out_dim}\n"
            f"  HUD output:   {hud_hidden}\n"
            f"  Combined:     {combined_dim}\n"
            f"  Features dim: {features_dim}\n"
            f"  Total params: {total_params:,}"
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass: frame (uint8 BHWC) + hud (float32) → feature vector (B, features_dim)

        Key steps:
          1. Permute frame from BHWC → BCHW  (PyTorch conv expects channels first)
          2. Normalise pixel values 0–255 → 0.0–1.0
          3. Pass through Impala CNN
          4. Pass HUD through small MLP
          5. Concatenate and project
        """
        # Frame: (B, H, W, C) → (B, C, H, W), then normalise to [0, 1]
        frame = observations["frame"].float() / 255.0
        frame = frame.permute(0, 3, 1, 2)          # BHWC → BCHW

        cnn_features = self.cnn(frame)              # (B, cnn_out_dim)

        # HUD: already float32 in [0, 1] from the environment
        hud_features = self.hud_mlp(observations["hud"].float())   # (B, hud_hidden)

        # Concatenate and project
        combined = torch.cat([cnn_features, hud_features], dim=1)  # (B, combined_dim)
        return self.projection(combined)            # (B, features_dim)
