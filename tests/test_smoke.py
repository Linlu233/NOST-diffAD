from __future__ import annotations

import torch

from nostdiffad.config import load_yaml
from nostdiffad.diffusion import DiffusionSchedule
from nostdiffad.energy import EnergyComputer
from nostdiffad.losses import NMFConstraint, PrototypeBank, laplacian_loss, score_matching_loss
from nostdiffad.metrics import au_pro
from nostdiffad.model import NOSTDiffAD


def test_formula_shapes_cpu() -> None:
    config = load_yaml("configs/default.yaml")
    config["device"] = "cpu"
    config["model"]["feature_extractor"] = "conv"
    config["model"]["feature_dim"] = 64
    config["model"]["hidden_dim"] = 32
    config["model"]["num_classes"] = 1
    config["model"]["num_prototypes"] = 4
    config["model"]["nmf_rank"] = 2
    config["model"]["time_dim"] = 16
    config["model"]["score_layers"] = 1
    config["model"]["graph_layers"] = 1
    config["data"]["image_size"] = 56
    config["model"]["patch_size"] = 14
    config["graph"]["use_mask_topology"] = False
    config["graph"]["beta_m"] = 0.0
    config["diffusion"]["timesteps"] = 20
    config["diffusion"]["t_min"] = 1
    config["diffusion"]["t_max"] = 18
    config["diffusion"]["energy_steps"] = 2

    model = NOSTDiffAD(config)
    images = torch.randn(2, 3, 56, 56)
    category = torch.zeros(2, dtype=torch.long)
    output = model(images)
    assert output.h.shape == (2, 16, 32)
    assert output.adjacency.shape == (2, 16, 16)
    assert output.laplacian.shape == (2, 16, 16)
    assert torch.all(output.adjacency.diagonal(dim1=1, dim2=2) > 0)

    schedule = DiffusionSchedule(20, 1e-4, 0.02)
    proto = PrototypeBank(1, 4, 32)
    nmf = NMFConstraint(1, 16, 32, 2)
    score_loss, _, _ = score_matching_loss(model, schedule, output.h, output.adjacency, category)
    assert torch.isfinite(score_loss)
    assert torch.isfinite(laplacian_loss(output.h, output.laplacian))
    assert torch.isfinite(nmf(NMFConstraint.normal_structure_matrix(output.h_struct), category, 1e-4, 1e-4))

    energy = EnergyComputer(schedule, proto, 1.0, 1.0, 0.5, 0.5, 1, 18, 2, 0.1)
    out = energy.total_energy(model, output, category)
    assert out["total"].shape == (2, 16)
    assert out["image_score"].shape == (2,)


def test_au_pro_prefers_region_aligned_scores() -> None:
    mask = torch.zeros(1, 16, 16).numpy()
    mask[:, 4:8, 4:8] = 1

    aligned = torch.zeros(1, 16, 16).numpy()
    aligned[:, 4:8, 4:8] = 10.0

    background_high = aligned.copy()
    background_high[:, :4, :4] = 20.0

    assert au_pro(mask, aligned, steps=50) > 0.99
    assert au_pro(mask, background_high, steps=50) < au_pro(mask, aligned, steps=50)
