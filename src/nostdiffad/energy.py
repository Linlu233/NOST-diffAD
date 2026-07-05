from __future__ import annotations

import torch
import torch.nn.functional as F

from .diffusion import DiffusionSchedule
from .losses import PrototypeBank


def topk_mean_score(patch_energy: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    num_nodes = patch_energy.shape[1]
    k = max(1, int(round(num_nodes * topk_ratio)))
    return torch.topk(patch_energy, k=k, dim=1).values.mean(dim=1)


def patch_energy_to_pixel_map(
    patch_energy: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    bsz = patch_energy.shape[0]
    patch_map = patch_energy.view(bsz, 1, grid_hw[0], grid_hw[1])
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(patch_map, size=image_hw, mode=mode, align_corners=False)
    return F.interpolate(patch_map, size=image_hw, mode=mode)


class EnergyComputer:
    def __init__(
        self,
        schedule: DiffusionSchedule,
        prototype_bank: PrototypeBank,
        alpha: float,
        beta: float,
        gamma: float,
        eta: float,
        t_min: int,
        t_max: int,
        energy_steps: int,
        topk_ratio: float,
    ) -> None:
        self.schedule = schedule
        self.prototype_bank = prototype_bank
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.t_min = t_min
        self.t_max = t_max
        self.energy_steps = energy_steps
        self.topk_ratio = topk_ratio

    @torch.no_grad()
    def score_energy(self, model, h: torch.Tensor, adjacency: torch.Tensor, category_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.alpha == 0.0 or self.energy_steps <= 0:
            zero = h.new_zeros(h.shape[:2])
            return zero, h.new_zeros(h.shape)
        return score_energy_integral(
            model=model,
            schedule=self.schedule,
            h=h,
            adjacency=adjacency,
            category_id=category_id,
            t_min=self.t_min,
            t_max=self.t_max,
            energy_steps=self.energy_steps,
        )

    @torch.no_grad()
    def prototype_energy(self, h: torch.Tensor, category_id: torch.Tensor) -> torch.Tensor:
        return self.prototype_bank.min_distance(h, category_id)

    @staticmethod
    @torch.no_grad()
    def topology_energy(adjacency: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        diff = (residual[:, :, None, :] - residual[:, None, :, :]).abs().sum(dim=-1)
        return (adjacency * diff).sum(dim=-1)

    @staticmethod
    @torch.no_grad()
    def wave_energy(wave: torch.Tensor, wave_hat: torch.Tensor) -> torch.Tensor:
        if wave.shape[-1] == 0:
            return wave.new_zeros(wave.shape[:2])
        return (wave - wave_hat).abs().sum(dim=-1)

    @torch.no_grad()
    def total_energy(self, model, output, category_id: torch.Tensor) -> dict[str, torch.Tensor]:
        zero_patch = output.h.new_zeros(output.h.shape[:2])
        e_score, residual = self.score_energy(model, output.h, output.adjacency, category_id)
        e_proto = zero_patch if self.beta == 0.0 else self.prototype_energy(output.h, category_id)
        e_topo = zero_patch if self.gamma == 0.0 else self.topology_energy(output.adjacency, residual)
        e_wave = zero_patch if self.eta == 0.0 else self.wave_energy(output.wave, output.wave_hat)
        total = self.alpha * e_score + self.beta * e_proto + self.gamma * e_topo + self.eta * e_wave
        image_score = topk_mean_score(total, self.topk_ratio)
        return {
            "total": total,
            "score": e_score,
            "proto": e_proto,
            "topo": e_topo,
            "wave": e_wave,
            "image_score": image_score,
            "residual": residual,
        }


def score_energy_integral(
    model,
    schedule: DiffusionSchedule,
    h: torch.Tensor,
    adjacency: torch.Tensor,
    category_id: torch.Tensor,
    t_min: int,
    t_max: int,
    energy_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.linspace(t_min, t_max, energy_steps, device=h.device).round().long().clamp(0, schedule.timesteps - 1)
    values: list[torch.Tensor] = []
    residual_accum = h.new_zeros(h.shape)
    for t_scalar in times:
        t = torch.full((h.shape[0],), int(t_scalar.item()), device=h.device, dtype=torch.long)
        ht, noise = schedule.q_sample(h, t)
        score = model.score(ht, adjacency, t, category_id)
        target = schedule.score_target(noise, t)
        values.append(score.pow(2).sum(dim=-1))
        alpha_bar_t = schedule.alpha_bar[t].to(device=h.device, dtype=h.dtype)
        while alpha_bar_t.ndim < h.ndim:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        sigma_t = (1.0 - alpha_bar_t).sqrt().clamp_min(1e-6)
        sqrt_alpha_bar_t = alpha_bar_t.sqrt().clamp_min(1e-6)
        predicted_noise = -sigma_t * score
        denoised_h0 = (ht - sigma_t * predicted_noise) / sqrt_alpha_bar_t
        residual_accum = residual_accum + (denoised_h0 - h)
    if len(values) == 1:
        integral = values[0]
    else:
        stacked = torch.stack(values, dim=0)
        dt = (float(t_max) - float(t_min)) / max(len(values) - 1, 1)
        integral = torch.trapezoid(stacked, dx=dt, dim=0)
    residual_accum = residual_accum / max(len(values), 1)
    return integral, residual_accum
