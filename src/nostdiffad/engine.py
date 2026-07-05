from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .conformal import ConformalThreshold
from .data import collate_batch
from .diffusion import DiffusionSchedule
from .energy import EnergyComputer, patch_energy_to_pixel_map, score_energy_integral, topk_mean_score
from .losses import NMFConstraint, PrototypeBank, consistency_loss, laplacian_loss, score_matching_loss
from .metrics import compute_metrics
from .utils import AverageMeter, Timer, ensure_dir


def _normalize_score_map(score_map: torch.Tensor, norm: str, eval_cfg: dict[str, Any]) -> torch.Tensor:
    norm = norm.lower()
    eps = float(eval_cfg.get("score_map_normalization_eps", 1e-6))
    if norm in {"none", "raw"}:
        return score_map
    if norm in {"image_minmax", "per_image_minmax"}:
        min_value = score_map.amin(dim=(2, 3), keepdim=True)
        max_value = score_map.amax(dim=(2, 3), keepdim=True)
        return (score_map - min_value) / (max_value - min_value).clamp_min(eps)
    if norm in {"image_zscore", "per_image_zscore"}:
        mean = score_map.mean(dim=(2, 3), keepdim=True)
        std = score_map.std(dim=(2, 3), keepdim=True, unbiased=False)
        return (score_map - mean) / std.clamp_min(eps)
    if norm in {"image_center", "per_image_center"}:
        return score_map - score_map.mean(dim=(2, 3), keepdim=True)
    if norm in {"image_robust", "per_image_robust"}:
        flat = score_map.flatten(2)
        median = flat.quantile(0.5, dim=2, keepdim=True).view(score_map.shape[0], score_map.shape[1], 1, 1)
        q_hi = flat.quantile(float(eval_cfg.get("score_map_robust_q", 0.995)), dim=2, keepdim=True).view(
            score_map.shape[0], score_map.shape[1], 1, 1
        )
        return ((score_map - median) / (q_hi - median).clamp_min(eps)).clamp_min(0.0)
    raise ValueError(f"Unknown score map normalization: {norm}")


def _score_map_eval_postprocess(score_map: torch.Tensor, batch: dict[str, Any], config: dict) -> torch.Tensor:
    eval_cfg = config.get("eval", {}) or {}

    enhance = str(eval_cfg.get("score_map_enhancement", "none")).lower()
    if enhance in {"local_contrast", "local_contrast_abs"}:
        kernel = int(eval_cfg.get("local_contrast_kernel", 15) or 15)
        if kernel % 2 == 0:
            kernel += 1
        local_mean = torch.nn.functional.avg_pool2d(score_map, kernel_size=kernel, stride=1, padding=kernel // 2)
        score_map = score_map - local_mean if enhance == "local_contrast" else (score_map - local_mean).abs()
    elif enhance not in {"none", "raw"}:
        raise ValueError(f"Unknown eval.score_map_enhancement: {enhance}")

    smooth_kernel = int(eval_cfg.get("score_map_smoothing_kernel", 0) or 0)
    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        score_map = torch.nn.functional.avg_pool2d(score_map, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)

    if bool(eval_cfg.get("apply_foreground_mask", False)):
        available = batch.get("part_mask_available")
        part_mask = batch.get("part_mask")
        if torch.is_tensor(available) and torch.is_tensor(part_mask):
            foreground = part_mask[:, :1].to(device=score_map.device) > 0
            if foreground.shape[-2:] != score_map.shape[-2:]:
                foreground = torch.nn.functional.interpolate(foreground.float(), size=score_map.shape[-2:], mode="nearest") > 0
            fill_value = score_map.amin(dim=(2, 3), keepdim=True)
            foreground = torch.where(available.view(-1, 1, 1, 1).to(device=score_map.device), foreground, torch.ones_like(foreground))
            score_map = torch.where(foreground, score_map, fill_value)

    return _normalize_score_map(score_map, str(eval_cfg.get("score_map_normalization", "none")), eval_cfg)


def fit_score_map_calibration(score_maps: np.ndarray, config: dict) -> dict[str, torch.Tensor] | None:
    eval_cfg = config.get("eval", {}) or {}
    cal_cfg = eval_cfg.get("score_map_calibration", {}) or {}
    if not bool(cal_cfg.get("enabled", False)):
        return None
    maps = np.asarray(score_maps, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] == 0:
        return None
    mode = str(cal_cfg.get("mode", "spatial_quantile")).lower()
    min_scale = float(cal_cfg.get("min_scale", 1e-6))
    if mode in {"spatial_quantile", "quantile"}:
        offset = np.quantile(maps, float(cal_cfg.get("quantile", 0.9)), axis=0, keepdims=True).astype(np.float32)
        return {"offset": torch.from_numpy(offset[None]), "scale": torch.ones(1, 1, 1, 1), "min_scale": torch.tensor(min_scale)}
    if mode in {"spatial_mean_std", "mean_std"}:
        offset = maps.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        scale = maps.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        return {"offset": torch.from_numpy(offset[None]), "scale": torch.from_numpy(scale[None]), "min_scale": torch.tensor(min_scale)}
    if mode in {"spatial_median_mad", "median_mad"}:
        offset = np.median(maps, axis=0, keepdims=True).astype(np.float32)
        scale = np.median(np.abs(maps - offset), axis=0, keepdims=True).astype(np.float32) * 1.4826
        return {"offset": torch.from_numpy(offset[None]), "scale": torch.from_numpy(scale[None]), "min_scale": torch.tensor(min_scale)}
    if mode == "global_quantile":
        offset = np.asarray([[[[np.quantile(maps, float(cal_cfg.get("quantile", 0.9)))]]]], dtype=np.float32)
        return {"offset": torch.from_numpy(offset), "scale": torch.ones(1, 1, 1, 1), "min_scale": torch.tensor(min_scale)}
    raise ValueError(f"Unknown eval.score_map_calibration.mode: {mode}")


def _apply_score_map_calibration(
    score_map: torch.Tensor,
    calibration: dict[str, torch.Tensor] | None,
    config: dict,
) -> torch.Tensor:
    if calibration is None:
        return score_map
    eval_cfg = config.get("eval", {}) or {}
    cal_cfg = eval_cfg.get("score_map_calibration", {}) or {}
    offset = calibration["offset"].to(device=score_map.device, dtype=score_map.dtype)
    scale = calibration["scale"].to(device=score_map.device, dtype=score_map.dtype)
    min_scale = float(calibration.get("min_scale", torch.tensor(float(cal_cfg.get("min_scale", 1e-6)))).item())
    score_map = (score_map - offset) / scale.clamp_min(min_scale)
    if bool(cal_cfg.get("clamp_min", True)):
        score_map = score_map.clamp_min(0.0)
    power = float(cal_cfg.get("power", 1.0) or 1.0)
    if power != 1.0:
        score_map = score_map.clamp_min(0.0).pow(power)
    post_norm = str(cal_cfg.get("post_normalization", "none")).lower()
    if post_norm not in {"none", "raw"}:
        score_map = _normalize_score_map(score_map, post_norm, eval_cfg)
    return score_map


def _image_score_from_postprocessed_map(
    score_map: torch.Tensor,
    current_score: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    eval_cfg = config.get("eval", {}) or {}
    if not bool(eval_cfg.get("image_score_from_postprocessed_map", False)):
        return current_score
    mode = str(eval_cfg.get("postprocessed_image_score_mode", "map_topk")).lower()
    flat = score_map.flatten(1)
    ratio = float(config["energy"].get("topk_ratio", 0.05))
    if mode in {"map_topk", "pixel_topk"}:
        k = max(1, int(round(flat.shape[1] * ratio)))
        return torch.topk(flat, k=min(k, flat.shape[1]), dim=1).values.mean(dim=1)
    if mode in {"map_max", "pixel_max"}:
        return flat.amax(dim=1)
    if mode in {"map_mean", "pixel_mean"}:
        return flat.mean(dim=1)
    raise ValueError(f"Unknown eval.postprocessed_image_score_mode: {mode}")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def batch_part_masks(batch: dict[str, Any], require_part_mask: bool) -> torch.Tensor | None:
    available = batch.get("part_mask_available")
    if available is None:
        if require_part_mask:
            raise ValueError("Batch has no part_mask_available flag.")
        return None
    if torch.is_tensor(available) and bool(available.all().item()):
        return batch["part_mask"]
    if require_part_mask:
        raise ValueError("SAM/SAM2 part masks are required by graph.use_mask_topology and beta_m, but this batch lacks them.")
    return None


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=False,
    )


def _feature_memory_features(output, feature_name: str, memory_cfg: dict) -> torch.Tensor:
    features = output.h if feature_name == "h" else output.z
    normalize = bool(memory_cfg.get("normalize", True))
    if normalize:
        features = F.normalize(features.float(), dim=-1)
    else:
        features = features.float()

    local_kernel = int(memory_cfg.get("local_context_kernel", 0) or 0)
    if local_kernel > 1:
        if local_kernel % 2 == 0:
            local_kernel += 1
        height, width = output.grid_hw
        local = features.transpose(1, 2).reshape(features.shape[0], features.shape[-1], height, width)
        local = F.avg_pool2d(local, kernel_size=local_kernel, stride=1, padding=local_kernel // 2)
        local = local.flatten(2).transpose(1, 2)
        aggregation = str(memory_cfg.get("local_context_aggregation", "concat")).lower()
        if aggregation in {"concat", "cat"}:
            features = torch.cat([features, local], dim=-1)
        elif aggregation in {"mean", "avg"}:
            features = 0.5 * (features + local)
        elif aggregation in {"local", "replace"}:
            features = local
        else:
            raise ValueError(f"Unknown eval.feature_memory.local_context_aggregation: {aggregation}")

    spatial_weight = float(memory_cfg.get("spatial_weight", 0.0) or 0.0)
    if spatial_weight > 0.0:
        coords = output.coords.to(device=features.device, dtype=features.dtype)
        coords = coords.unsqueeze(0).expand(features.shape[0], -1, -1)
        features = torch.cat([features, coords * spatial_weight], dim=-1)
    return features


def _sample_feature_memory_bank(
    feature_parts: list[torch.Tensor],
    max_patches: int,
    memory_cfg: dict,
    config: dict,
) -> torch.Tensor:
    strategy = str(memory_cfg.get("sampling_strategy", "random")).lower()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(memory_cfg.get("seed", config.get("seed", 42))))
    if strategy in {"spatial_uniform", "spatial", "per_position"}:
        if not feature_parts:
            return torch.empty(0)
        nodes = {part.shape[1] for part in feature_parts if part.ndim == 3}
        if len(nodes) != 1:
            strategy = "random"
        else:
            stacked = torch.cat(feature_parts, dim=0)
            images, node_count, dim = stacked.shape
            if max_patches <= 0 or images * node_count <= max_patches:
                return stacked.reshape(-1, dim).contiguous()
            per_node = max(1, max_patches // node_count)
            remainder = max(0, max_patches - per_node * node_count)
            selected: list[torch.Tensor] = []
            for node_idx in range(node_count):
                quota = per_node + (1 if node_idx < remainder else 0)
                node_features = stacked[:, node_idx, :]
                if node_features.shape[0] > quota:
                    indices = torch.randperm(node_features.shape[0], generator=generator)[:quota]
                    node_features = node_features[indices]
                selected.append(node_features)
            return torch.cat(selected, dim=0).contiguous()

    if strategy not in {"random", "rand"}:
        raise ValueError(f"Unknown eval.feature_memory.sampling_strategy: {strategy}")
    bank = torch.cat([part.reshape(-1, part.shape[-1]) for part in feature_parts], dim=0)
    if max_patches > 0 and bank.shape[0] > max_patches:
        indices = torch.randperm(bank.shape[0], generator=generator)[:max_patches]
        bank = bank[indices]
    return bank.contiguous()


@torch.no_grad()
def build_feature_memory(
    model,
    loader: DataLoader,
    device: torch.device,
    config: dict,
) -> torch.Tensor | None:
    memory_cfg = ((config.get("eval", {}) or {}).get("feature_memory", {}) or {})
    if not bool(memory_cfg.get("enabled", False)):
        return None
    feature_name = str(memory_cfg.get("feature", "z")).lower()
    max_patches = int(memory_cfg.get("max_patches", 20000))
    bank_parts: list[torch.Tensor] = []
    model.eval()
    for batch in tqdm(loader, desc="feature-memory", leave=False):
        batch = move_batch(batch, device)
        part_masks = batch_part_masks(batch, getattr(model, "require_part_mask", False))
        output = model(batch["image"], part_masks)
        features = _feature_memory_features(output, feature_name, memory_cfg)
        bank_parts.append(features.detach().cpu())
    if not bank_parts:
        return None
    return _sample_feature_memory_bank(bank_parts, max_patches, memory_cfg, config)


def _feature_memory_patch_scores(features: torch.Tensor, bank: torch.Tensor, config: dict) -> torch.Tensor:
    memory_cfg = ((config.get("eval", {}) or {}).get("feature_memory", {}) or {})
    normalize = bool(memory_cfg.get("normalize", True)) and float(memory_cfg.get("spatial_weight", 0.0) or 0.0) <= 0.0
    k = max(1, int(memory_cfg.get("k", 1)))
    chunk_size = max(1, int(memory_cfg.get("chunk_size", 1024)))
    if normalize:
        features = F.normalize(features.float(), dim=-1)
        bank = F.normalize(bank.float(), dim=-1)
    else:
        features = features.float()
        bank = bank.float()
    bsz, nodes, dim = features.shape
    flat = features.reshape(-1, dim)
    scores: list[torch.Tensor] = []
    bank_t = bank.t().contiguous()
    for chunk in flat.split(chunk_size, dim=0):
        if normalize:
            dist = (2.0 - 2.0 * chunk @ bank_t).clamp_min_(0.0)
        else:
            dist = torch.cdist(chunk, bank, p=2.0).pow_(2.0)
        nearest = torch.topk(dist, k=min(k, dist.shape[1]), largest=False, dim=1).values.mean(dim=1)
        scores.append(nearest)
    return torch.cat(scores, dim=0).view(bsz, nodes)


def _fuse_with_feature_memory(
    score_map: torch.Tensor,
    image_score: torch.Tensor,
    output,
    config: dict,
    feature_memory: torch.Tensor | None,
    image_hw: tuple[int, int],
    upsample_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    memory_cfg = ((config.get("eval", {}) or {}).get("feature_memory", {}) or {})
    if feature_memory is None or not bool(memory_cfg.get("enabled", False)):
        return score_map, image_score
    feature_name = str(memory_cfg.get("feature", "z")).lower()
    features = _feature_memory_features(output, feature_name, memory_cfg)
    bank = feature_memory.to(device=features.device, non_blocking=True)
    memory_patch = _feature_memory_patch_scores(features, bank, config)
    memory_map = patch_energy_to_pixel_map(memory_patch, output.grid_hw, image_hw=image_hw, mode=upsample_mode)
    memory_image_score = topk_mean_score(memory_patch, float(config["energy"].get("topk_ratio", 0.05)))

    mode = str(memory_cfg.get("fusion", "replace")).lower()
    weight = float(memory_cfg.get("weight", 1.0))
    if bool(memory_cfg.get("normalize_maps", True)):
        score_map = _normalize_map_for_fusion(score_map)
        memory_map = _normalize_map_for_fusion(memory_map)
    power = float(memory_cfg.get("map_power", 1.0) or 1.0)
    if power != 1.0:
        memory_map = memory_map.clamp_min(0.0).pow(power)
    if mode == "replace":
        return memory_map, _image_score_from_map(memory_map, memory_patch, config, memory_image_score)
    if mode in {"add", "sum"}:
        fused_map = score_map + weight * memory_map
        fused_score = image_score + weight * _image_score_from_map(memory_map, memory_patch, config, memory_image_score)
        return fused_map, fused_score
    if mode in {"blend", "mix"}:
        weight = max(0.0, min(1.0, weight))
        fused_map = (1.0 - weight) * score_map + weight * memory_map
        fused_score = (1.0 - weight) * image_score + weight * _image_score_from_map(
            memory_map, memory_patch, config, memory_image_score
        )
        return fused_map, fused_score
    raise ValueError(f"Unknown eval.feature_memory.fusion: {mode}")


def _normalize_map_for_fusion(score_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    min_value = score_map.amin(dim=(2, 3), keepdim=True)
    max_value = score_map.amax(dim=(2, 3), keepdim=True)
    return (score_map - min_value) / (max_value - min_value).clamp_min(eps)


def _image_score_from_map(
    score_map: torch.Tensor,
    patch_scores: torch.Tensor,
    config: dict,
    default_score: torch.Tensor,
) -> torch.Tensor:
    memory_cfg = ((config.get("eval", {}) or {}).get("feature_memory", {}) or {})
    mode = str(memory_cfg.get("image_score_mode", "patch_topk")).lower()
    ratio = float(config["energy"].get("topk_ratio", 0.05))
    if mode in {"patch_topk", "patch"}:
        return topk_mean_score(patch_scores, ratio)
    if mode in {"map_topk", "pixel_topk"}:
        flat = score_map.flatten(1)
        k = max(1, int(round(flat.shape[1] * ratio)))
        return torch.topk(flat, k=min(k, flat.shape[1]), dim=1).values.mean(dim=1)
    if mode in {"map_max", "pixel_max"}:
        return score_map.flatten(1).amax(dim=1)
    if mode in {"default", "memory"}:
        return default_score
    raise ValueError(f"Unknown eval.feature_memory.image_score_mode: {mode}")


def tensor_intensity_augment(images: torch.Tensor) -> torch.Tensor:
    bsz = images.shape[0]
    scale = torch.empty(bsz, 1, 1, 1, device=images.device, dtype=images.dtype).uniform_(0.85, 1.15)
    noise = torch.randn_like(images) * 0.03
    return images * scale + noise


def differentiable_patch_energy(
    model,
    prototype_bank: PrototypeBank,
    schedule: DiffusionSchedule,
    output,
    category_id: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    energy_cfg = config["energy"]
    diffusion_cfg = config["diffusion"]
    e_score, residual = score_energy_integral(
        model=model,
        schedule=schedule,
        h=output.h,
        adjacency=output.adjacency,
        category_id=category_id,
        t_min=int(diffusion_cfg["t_min"]),
        t_max=int(diffusion_cfg["t_max"]),
        energy_steps=int(diffusion_cfg["energy_steps"]),
    )
    e_proto = prototype_bank.min_distance(output.h, category_id)
    diff = (residual[:, :, None, :] - residual[:, None, :, :]).abs().sum(dim=-1)
    e_topo = (output.adjacency * diff).sum(dim=-1)
    e_wave = (output.wave - output.wave_hat).abs().sum(dim=-1) if output.wave.shape[-1] > 0 else output.h.new_zeros(output.h.shape[:2])
    return (
        float(energy_cfg["alpha"]) * e_score
        + float(energy_cfg["beta"]) * e_proto
        + float(energy_cfg["gamma"]) * e_topo
        + float(energy_cfg["eta"]) * e_wave
    )


def make_components(config: dict, device: torch.device, max_nodes: int) -> tuple[DiffusionSchedule, PrototypeBank, NMFConstraint, EnergyComputer]:
    model_cfg = config["model"]
    diffusion_cfg = config["diffusion"]
    loss_cfg = config["loss"]
    energy_cfg = config["energy"]
    schedule = DiffusionSchedule(
        timesteps=int(diffusion_cfg["timesteps"]),
        beta_start=float(diffusion_cfg["beta_start"]),
        beta_end=float(diffusion_cfg["beta_end"]),
    ).to(device)
    prototype_bank = PrototypeBank(
        num_classes=int(model_cfg["num_classes"]),
        num_prototypes=int(model_cfg["num_prototypes"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
    ).to(device)
    nmf = NMFConstraint(
        num_classes=int(model_cfg["num_classes"]),
        max_nodes=max_nodes,
        hidden_dim=int(model_cfg["hidden_dim"]),
        rank=int(model_cfg["nmf_rank"]),
    ).to(device)
    energy = EnergyComputer(
        schedule=schedule,
        prototype_bank=prototype_bank,
        alpha=float(energy_cfg["alpha"]),
        beta=float(energy_cfg["beta"]),
        gamma=float(energy_cfg["gamma"]),
        eta=float(energy_cfg["eta"]),
        t_min=int(diffusion_cfg["t_min"]),
        t_max=int(diffusion_cfg["t_max"]),
        energy_steps=int(diffusion_cfg["energy_steps"]),
        topk_ratio=float(energy_cfg["topk_ratio"]),
    )
    _ = loss_cfg
    return schedule, prototype_bank, nmf, energy


def train_one_epoch(
    model,
    prototype_bank: PrototypeBank,
    nmf: NMFConstraint,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    prototype_bank.train()
    nmf.train()
    loss_cfg = config["loss"]
    train_cfg = config["train"]
    diffusion_disabled = bool(config.get("diffusion_disabled", False))
    scaler = GradScaler(device.type, enabled=bool(config.get("amp", True)) and device.type == "cuda")
    meters = {key: AverageMeter() for key in ["total", "score", "nmf", "lap", "proto", "wave", "cons"]}

    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for batch in pbar:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=bool(config.get("amp", True)) and device.type == "cuda"):
            part_masks = batch_part_masks(batch, getattr(model, "require_part_mask", False))
            output = model(batch["image"], part_masks)
            if diffusion_disabled:
                score_loss = output.h.new_tensor(0.0)
            else:
                score_loss, _, _ = score_matching_loss(model, schedule, output.h, output.adjacency, batch["category_id"])
            proto_loss = prototype_bank.compactness_loss(output.h, batch["category_id"])
            nmf_loss = nmf(
                NMFConstraint.normal_structure_matrix(output.h_struct),
                batch["category_id"],
                lambda_u=float(loss_cfg["lambda_u"]),
                lambda_v=float(loss_cfg["lambda_v"]),
            )
            lap_loss = laplacian_loss(output.h, output.laplacian)
            wave_loss = (output.wave - output.wave_hat).abs().sum() if output.wave.shape[-1] > 0 else output.h.new_tensor(0.0)
            cons_loss = output.h.new_tensor(0.0)
            if float(loss_cfg.get("lambda_cons", 0.0)) > 0.0:
                aug_images = tensor_intensity_augment(batch["image"])
                output_aug = model(aug_images, part_masks)
                energy_a = differentiable_patch_energy(model, prototype_bank, schedule, output, batch["category_id"], config)
                energy_b = differentiable_patch_energy(model, prototype_bank, schedule, output_aug, batch["category_id"], config)
                cons_loss = consistency_loss(energy_a, energy_b)
            total = (
                score_loss
                + float(loss_cfg["lambda_nmf"]) * nmf_loss
                + float(loss_cfg["lambda_lap"]) * lap_loss
                + float(loss_cfg["lambda_proto"]) * proto_loss
                + float(loss_cfg.get("lambda_cons", 0.0)) * cons_loss
            )
        scaler.scale(total).backward()
        if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(prototype_bank.parameters()) + list(nmf.parameters()),
                float(train_cfg["grad_clip_norm"]),
            )
        scaler.step(optimizer)
        scaler.update()

        bsz = batch["image"].shape[0]
        values = {
            "total": total.item(),
            "score": score_loss.item(),
            "nmf": nmf_loss.item(),
            "lap": lap_loss.item(),
            "proto": proto_loss.item(),
            "wave": wave_loss.item(),
            "cons": cons_loss.item(),
        }
        for key, value in values.items():
            meters[key].update(value, bsz)
        pbar.set_postfix({key: f"{meter.avg:.4f}" for key, meter in meters.items() if key in {"total", "score"}})
    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def collect_energy(
    model,
    energy: EnergyComputer,
    loader: DataLoader,
    device: torch.device,
    config: dict | None = None,
    upsample_mode: str = "bilinear",
    feature_memory: torch.Tensor | None = None,
    score_map_calibration: dict[str, torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    all_image_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_patch_energy: list[np.ndarray] = []
    all_maps: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    timer = Timer()
    for batch in tqdm(loader, desc="energy", leave=False):
        batch = move_batch(batch, device)
        part_masks = batch_part_masks(batch, getattr(model, "require_part_mask", False))
        output = model(batch["image"], part_masks)
        energy_out = energy.total_energy(model, output, batch["category_id"])
        score_map = patch_energy_to_pixel_map(
            energy_out["total"],
            output.grid_hw,
            image_hw=(batch["image"].shape[-2], batch["image"].shape[-1]),
            mode=upsample_mode,
        )
        image_score = energy_out["image_score"]
        if config is not None:
            score_map, image_score = _fuse_with_feature_memory(
                score_map=score_map,
                image_score=image_score,
                output=output,
                config=config,
                feature_memory=feature_memory,
                image_hw=(batch["image"].shape[-2], batch["image"].shape[-1]),
                upsample_mode=upsample_mode,
            )
        if config is not None:
            score_map = _score_map_eval_postprocess(score_map, batch, config)
            score_map = _apply_score_map_calibration(score_map, score_map_calibration, config)
            image_score = _image_score_from_postprocessed_map(score_map, image_score, config)
        all_image_scores.append(image_score.detach().cpu().numpy())
        all_labels.append(batch["label"].detach().cpu().numpy())
        all_patch_energy.append(energy_out["total"].detach().cpu().numpy())
        all_maps.append(score_map[:, 0].detach().cpu().numpy())
        all_masks.append(batch["mask"][:, 0].detach().cpu().numpy())
    elapsed = timer.elapsed()
    return {
        "image_scores": np.concatenate(all_image_scores, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "patch_energy": np.concatenate(all_patch_energy, axis=0),
        "score_maps": np.concatenate(all_maps, axis=0),
        "masks": np.concatenate(all_masks, axis=0),
        "elapsed": np.asarray([elapsed], dtype=np.float64),
    }


def evaluate_with_conformal(
    model,
    energy: EnergyComputer,
    cal_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    feature_memory: torch.Tensor | None = None,
) -> tuple[dict[str, float], float]:
    upsample_mode = str(config["energy"].get("upsample_mode", "bilinear"))
    cal = collect_energy(model, energy, cal_loader, device, config=config, upsample_mode=upsample_mode, feature_memory=feature_memory)
    score_map_calibration = fit_score_map_calibration(cal["score_maps"], config)
    if score_map_calibration is not None:
        cal = collect_energy(
            model,
            energy,
            cal_loader,
            device,
            config=config,
            upsample_mode=upsample_mode,
            feature_memory=feature_memory,
            score_map_calibration=score_map_calibration,
        )
    threshold: float | None = None
    if not bool(config.get("conformal_disabled", False)):
        conformal = ConformalThreshold(alpha=float(config["energy"]["conformal_alpha"]))
        threshold = conformal.fit(torch.from_numpy(cal["patch_energy"]))
    test = collect_energy(
        model,
        energy,
        test_loader,
        device,
        config=config,
        upsample_mode=upsample_mode,
        feature_memory=feature_memory,
        score_map_calibration=score_map_calibration,
    )
    metrics = compute_metrics(
        image_labels=test["labels"],
        image_scores=test["image_scores"],
        masks=test["masks"],
        score_maps=test["score_maps"],
        threshold=threshold,
        inference_seconds=float(test["elapsed"][0]),
        normal_calibration_scores=cal["patch_energy"].reshape(-1),
        target_coverage=1.0 - float(config["energy"]["conformal_alpha"]),
    )
    if threshold is not None:
        metrics["conformal_threshold"] = threshold
    return metrics, float("nan") if threshold is None else threshold


def save_checkpoint(
    path: str | Path,
    model,
    prototype_bank: PrototypeBank,
    nmf: NMFConstraint,
    optimizer: torch.optim.Optimizer | None,
    config: dict,
    epoch: int,
    metrics: dict[str, float] | None = None,
) -> None:
    ensure_dir(Path(path).parent)
    state = {
        "model": model.state_dict(),
        "prototype_bank": prototype_bank.state_dict(),
        "nmf": nmf.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "config": config,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    torch.save(state, path)


def load_checkpoint(path: str | Path, model, prototype_bank: PrototypeBank, nmf: NMFConstraint, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    prototype_bank.load_state_dict(state["prototype_bank"])
    nmf.load_state_dict(state["nmf"])
    return state
