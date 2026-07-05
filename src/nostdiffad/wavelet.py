from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletHighFrequency(nn.Module):
    """Haar high-frequency patch features W_wave(x_i)."""

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        kernels = torch.stack([lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("kernels", kernels)

    def forward(self, images: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        return self.patch_features(self.patches(images, grid_hw))

    def patches(self, images: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        patch_maps = F.adaptive_avg_pool2d(images, (grid_hw[0] * self.patch_size, grid_hw[1] * self.patch_size))
        patches = patch_maps.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            images.shape[0],
            grid_hw[0] * grid_hw[1],
            images.shape[1],
            self.patch_size,
            self.patch_size,
        )
        return patches

    def patch_features(self, patches: torch.Tensor) -> torch.Tensor:
        bsz, num_nodes, channels, patch_h, patch_w = patches.shape
        flat = patches.reshape(bsz * num_nodes, channels, patch_h, patch_w)
        filters = self.kernels.repeat(channels, 1, 1, 1)
        coeff = F.conv2d(flat, filters, stride=2, groups=channels).abs()
        pooled = coeff.mean(dim=(-2, -1))
        return pooled.view(bsz, num_nodes, channels, 3).reshape(bsz, num_nodes, channels * 3)
