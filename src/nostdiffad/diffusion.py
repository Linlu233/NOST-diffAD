from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        exponent = -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        freqs = torch.exp(exponent)
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(
        self,
        h0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(h0)
        alpha_bar_t = self.alpha_bar[t].to(device=h0.device, dtype=h0.dtype)
        while alpha_bar_t.ndim < h0.ndim:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        ht = alpha_bar_t.sqrt() * h0 + (1.0 - alpha_bar_t).sqrt() * noise
        return ht, noise

    def score_target(self, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bar[t].to(device=noise.device, dtype=noise.dtype)
        while alpha_bar_t.ndim < noise.ndim:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        return -noise / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-6)
