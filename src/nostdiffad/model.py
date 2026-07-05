from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .diffusion import SinusoidalTimeEmbedding
from .features import make_feature_extractor, patch_coordinates, patch_mask_ids
from .graph import build_patch_graph, graph_laplacian, normalize_adjacency
from .wavelet import WaveletHighFrequency


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    blocks: list[nn.Module] = []
    current = in_dim
    for _ in range(max(layers - 1, 1)):
        blocks.extend([nn.Linear(current, hidden_dim), nn.GELU()])
        current = hidden_dim
    blocks.append(nn.Linear(current, out_dim))
    return nn.Sequential(*blocks)


class AttributeBranch(nn.Module):
    def __init__(self, feature_dim: int, wave_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.wave_dim = wave_dim
        self.net = mlp(feature_dim + wave_dim + 2, hidden_dim, hidden_dim, layers=3)

    def forward(self, z: torch.Tensor, wave: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz = z.shape[0]
        coords_b = coords.unsqueeze(0).expand(bsz, -1, -1)
        if self.wave_dim == 0:
            wave = z.new_zeros(z.shape[0], z.shape[1], 0)
        return self.net(torch.cat([z, wave, coords_b], dim=-1))


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        a_norm = normalize_adjacency(adjacency, add_self_loops=False)
        out = torch.bmm(a_norm, h)
        out = self.linear(out)
        return self.norm(torch.nn.functional.gelu(out))


class StructureBranch(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = feature_dim
        for _ in range(layers):
            blocks.append(GraphConv(current, hidden_dim))
            current = hidden_dim
        self.layers = nn.ModuleList(blocks)

    def forward(self, z: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.layers:
            h = layer(h, adjacency)
        return h


class FusionBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_attr: torch.Tensor, h_struct: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([h_attr, h_struct], dim=-1))
        return self.norm(h_attr + h_struct + gate * (h_attr - h_struct))


class ScoreGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.message = GraphConv(hidden_dim, hidden_dim)
        self.cond = nn.Linear(cond_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        msg = self.message(h, adjacency)
        cond_h = self.cond(cond).unsqueeze(1)
        h = self.norm(h + msg + cond_h)
        return h + self.ffn(h)


class ScoreNetwork(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int, num_classes: int, layers: int) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.class_embedding = nn.Embedding(num_classes, time_dim)
        self.blocks = nn.ModuleList([ScoreGraphBlock(hidden_dim, time_dim) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, ht: torch.Tensor, adjacency: torch.Tensor, t: torch.Tensor, category_id: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(self.time_embedding(t)) + self.class_embedding(category_id)
        h = ht
        for block in self.blocks:
            h = block(h, adjacency, cond)
        return self.out(h)


@dataclass
class ModelOutput:
    z: torch.Tensor
    wave: torch.Tensor
    coords: torch.Tensor
    adjacency: torch.Tensor
    laplacian: torch.Tensor
    h_attr: torch.Tensor
    h_struct: torch.Tensor
    h: torch.Tensor
    x_hat_patches: torch.Tensor
    wave_hat: torch.Tensor
    grid_hw: tuple[int, int]


class NOSTDiffAD(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        graph_cfg = config["graph"]
        self.model_cfg = model_cfg
        self.graph_cfg = graph_cfg
        self.feature_extractor = make_feature_extractor(model_cfg)
        feature_dim = int(model_cfg["feature_dim"])
        wave_dim = int(model_cfg.get("wave_dim", 9))
        hidden_dim = int(model_cfg["hidden_dim"])
        self.wave_dim = wave_dim
        self.wavelet = WaveletHighFrequency(int(model_cfg["patch_size"]))
        self.attribute_branch = AttributeBranch(feature_dim, wave_dim, hidden_dim)
        self.structure_branch = StructureBranch(feature_dim, hidden_dim, int(model_cfg["graph_layers"]))
        self.fusion = FusionBlock(hidden_dim)
        self.score_network = ScoreNetwork(
            hidden_dim=hidden_dim,
            time_dim=int(model_cfg["time_dim"]),
            num_classes=int(model_cfg["num_classes"]),
            layers=int(model_cfg["score_layers"]),
        )
        self.patch_decoder = nn.Linear(hidden_dim, 3 * int(model_cfg["patch_size"]) * int(model_cfg["patch_size"]))
        self.require_part_mask = bool(graph_cfg.get("use_mask_topology", True)) and float(graph_cfg.get("beta_m", 0.0)) != 0.0

    def encode(self, images: torch.Tensor, part_masks: torch.Tensor | None = None) -> ModelOutput:
        z, grid_hw = self.feature_extractor(images)
        coords = patch_coordinates(grid_hw, device=z.device, dtype=z.dtype)
        if self.require_part_mask and part_masks is None:
            raise ValueError("SAM/SAM2 part mask ids are required for the beta_m I[m_i=m_j] graph term.")
        mask_ids = patch_mask_ids(part_masks, grid_hw).to(z.device) if part_masks is not None else None
        wave = self.wavelet(images, grid_hw).to(dtype=z.dtype)
        if self.wave_dim == 0:
            wave = z.new_zeros(z.shape[0], z.shape[1], 0)
        adjacency = build_patch_graph(
            z=z,
            coords=coords,
            mask_ids=mask_ids,
            beta_s=float(self.graph_cfg["beta_s"]),
            beta_p=float(self.graph_cfg["beta_p"]),
            beta_m=float(self.graph_cfg["beta_m"]),
            sigma_p=float(self.graph_cfg["sigma_p"]),
            use_patch_graph=bool(self.graph_cfg.get("use_patch_graph", True)),
            use_mask_topology=bool(self.graph_cfg.get("use_mask_topology", True)),
        )
        laplacian = graph_laplacian(adjacency)
        h_attr = self.attribute_branch(z, wave, coords)
        h_struct = self.structure_branch(z, adjacency)
        h = self.fusion(h_attr, h_struct)
        patch_size = int(self.model_cfg["patch_size"])
        x_hat_patches = self.patch_decoder(h).view(h.shape[0], h.shape[1], 3, patch_size, patch_size)
        wave_hat = self.wavelet.patch_features(x_hat_patches).to(dtype=z.dtype) if self.wave_dim > 0 else wave
        return ModelOutput(
            z=z,
            wave=wave,
            coords=coords,
            adjacency=adjacency,
            laplacian=laplacian,
            h_attr=h_attr,
            h_struct=h_struct,
            h=h,
            x_hat_patches=x_hat_patches,
            wave_hat=wave_hat,
            grid_hw=grid_hw,
        )

    def score(self, ht: torch.Tensor, adjacency: torch.Tensor, t: torch.Tensor, category_id: torch.Tensor) -> torch.Tensor:
        return self.score_network(ht, adjacency, t, category_id)

    def forward(self, images: torch.Tensor, part_masks: torch.Tensor | None = None) -> ModelOutput:
        return self.encode(images, part_masks)
