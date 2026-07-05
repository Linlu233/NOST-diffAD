from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import DiffusionSchedule


class PrototypeBank(nn.Module):
    def __init__(self, num_classes: int, num_prototypes: int, hidden_dim: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, hidden_dim) * 0.02)

    def distances(self, h: torch.Tensor, category_id: torch.Tensor) -> torch.Tensor:
        proto = self.prototypes[category_id]
        diff = h[:, :, None, :] - proto[:, None, :, :]
        return diff.pow(2).sum(dim=-1)

    def min_distance(self, h: torch.Tensor, category_id: torch.Tensor) -> torch.Tensor:
        return self.distances(h, category_id).min(dim=-1).values

    def compactness_loss(self, h: torch.Tensor, category_id: torch.Tensor) -> torch.Tensor:
        return self.min_distance(h, category_id).sum()


class NMFConstraint(nn.Module):
    def __init__(self, num_classes: int, max_nodes: int, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.u_raw = nn.Parameter(torch.randn(num_classes, max_nodes, rank) * 0.02)
        self.v_raw = nn.Parameter(torch.randn(num_classes, hidden_dim, rank) * 0.02)

    @staticmethod
    def normal_structure_matrix(h_struct: torch.Tensor) -> torch.Tensor:
        return F.softplus(h_struct)

    def forward(self, h_c_plus: torch.Tensor, category_id: torch.Tensor, lambda_u: float, lambda_v: float) -> torch.Tensor:
        bsz, num_nodes, _ = h_c_plus.shape
        if num_nodes > self.max_nodes:
            raise ValueError(f"NMF max_nodes={self.max_nodes} is smaller than num_nodes={num_nodes}.")
        u = F.softplus(self.u_raw[category_id, :num_nodes, :])
        v = F.softplus(self.v_raw[category_id])
        recon = torch.bmm(u, v.transpose(1, 2))
        frob = (h_c_plus - recon).pow(2).sum()
        sparse = lambda_u * u.abs().sum() + lambda_v * v.abs().sum()
        return frob + sparse


def score_matching_loss(
    model,
    schedule: DiffusionSchedule,
    h0: torch.Tensor,
    adjacency: torch.Tensor,
    category_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = h0.shape[0]
    t = torch.randint(0, schedule.timesteps, (bsz,), device=h0.device)
    ht, noise = schedule.q_sample(h0, t)
    pred_score = model.score(ht, adjacency, t, category_id)
    target = schedule.score_target(noise, t)
    loss = (pred_score - target).pow(2).mean()
    residual = pred_score - target
    return loss, pred_score, residual


def laplacian_loss(h: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    lh = torch.bmm(laplacian, h)
    trace = (h * lh).sum(dim=(1, 2))
    return trace.sum()


def consistency_loss(energy_a: torch.Tensor, energy_b: torch.Tensor) -> torch.Tensor:
    return (energy_a - energy_b).abs().sum()
