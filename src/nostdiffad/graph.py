from __future__ import annotations

import torch
import torch.nn.functional as F


def build_patch_graph(
    z: torch.Tensor,
    coords: torch.Tensor,
    mask_ids: torch.Tensor | None,
    beta_s: float,
    beta_p: float,
    beta_m: float,
    sigma_p: float,
    use_patch_graph: bool = True,
    use_mask_topology: bool = True,
) -> torch.Tensor:
    """Compute A_ij from semantic, spatial, and mask topology terms."""

    bsz, num_nodes, _ = z.shape
    device = z.device
    dtype = z.dtype
    if not use_patch_graph:
        eye = torch.eye(num_nodes, device=device, dtype=dtype)
        return eye.unsqueeze(0).expand(bsz, -1, -1)

    z_norm = F.normalize(z, dim=-1)
    semantic = torch.bmm(z_norm, z_norm.transpose(1, 2))

    coord_diff = coords[:, None, :] - coords[None, :, :]
    spatial = torch.exp(-(coord_diff.pow(2).sum(dim=-1)) / max(sigma_p**2, 1e-12))
    spatial = spatial.to(device=device, dtype=dtype).unsqueeze(0)

    logits = beta_s * semantic + beta_p * spatial
    if mask_ids is not None and use_mask_topology and beta_m != 0.0:
        same_mask = (mask_ids[:, :, None] == mask_ids[:, None, :]).to(dtype=dtype)
        logits = logits + beta_m * same_mask

    adjacency = torch.sigmoid(logits)
    return adjacency


def graph_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    degree = torch.diag_embed(adjacency.sum(dim=-1))
    return degree - adjacency


def normalize_adjacency(adjacency: torch.Tensor, add_self_loops: bool = False) -> torch.Tensor:
    bsz, num_nodes, _ = adjacency.shape
    if add_self_loops:
        eye = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        adjacency = adjacency + eye
    degree = adjacency.sum(dim=-1).clamp_min(1e-6)
    inv_sqrt = degree.rsqrt()
    return adjacency * inv_sqrt[:, :, None] * inv_sqrt[:, None, :]
