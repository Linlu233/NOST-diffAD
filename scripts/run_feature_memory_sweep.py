#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_downloaded_official_experiments import Experiment, downloaded_experiments, quoted_override, shell_join


METRIC_KEYS = [
    "image_auroc",
    "image_ap",
    "pixel_auroc",
    "au_pro",
    "pixel_ap",
    "f1_max",
    "iou",
    "dice",
    "selection_score",
    "selection_score_pro",
    "inference_speed_fps",
]


@dataclass(frozen=True)
class Trial:
    name: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class Task:
    experiment: Experiment
    trial: Trial
    result_file: Path
    result_dir: Path
    log_file: Path

    @property
    def name(self) -> str:
        return f"{self.experiment.dataset}/{self.experiment.category}/{self.trial.name}"


@dataclass(frozen=True)
class RunningTask:
    task: Task
    process: subprocess.Popen
    handle: object
    attempts: int


BASE_OVERRIDES: dict[str, Any] = {
    "train.batch_size": 1,
    "train.epochs": 1,
    "graph.use_mask_topology": "false",
    "graph.beta_m": 0.0,
    "energy.alpha": 0.0,
    "energy.beta": 1.0,
    "energy.gamma": 0.0,
    "energy.eta": 0.0,
    "energy.upsample_mode": "bilinear",
    "data.mask_resize_mode": "nearest",
    "eval.apply_foreground_mask": "false",
    "eval.score_map_normalization": "none",
    "eval.score_map_smoothing_kernel": 0,
    "eval.feature_memory.enabled": "true",
    "eval.feature_memory.feature": "z",
    "eval.feature_memory.fusion": "replace",
    "eval.feature_memory.weight": 1.0,
    "eval.feature_memory.k": 1,
    "eval.feature_memory.chunk_size": 1024,
    "eval.feature_memory.normalize": "true",
    "eval.feature_memory.normalize_maps": "true",
}


def merged(name: str, **overrides: Any) -> Trial:
    payload = dict(BASE_OVERRIDES)
    payload.update(overrides)
    return Trial(name, payload)


def trial_sets() -> dict[str, list[Trial]]:
    return {
        "v8": [
            merged(
                "fm448_s_z_replace_tk002_mp040k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.feature_memory.max_patches": 40000,
                },
            )
        ],
        "v9": [
            merged(
                "fm448_s_z_replace_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_z_replace_tk010_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.010,
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_z_replace_fg_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.apply_foreground_mask": "true",
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_z_replace_s3_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_smoothing_kernel": 3,
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm672_s_z_replace_tk002_mp060k",
                **{
                    "data.image_size": 672,
                    "energy.topk_ratio": 0.002,
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_b_z_replace_tk002_mp080k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "energy.topk_ratio": 0.002,
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
        ],
        "v10": [
            merged(
                "fm448_s_sp02_robust_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_sp05_robust_p12_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.5,
                    "eval.feature_memory.map_power": 1.2,
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_lc31_robust_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_enhancement": "local_contrast",
                    "eval.local_contrast_kernel": 31,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_s5_robust_maptk010_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.010,
                    "eval.score_map_smoothing_kernel": 5,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.image_score_mode": "map_topk",
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_b_sp02_robust_tk002_mp080k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_b_lc31_robust_p12_tk002_mp080k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_enhancement": "local_contrast",
                    "eval.local_contrast_kernel": 31,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.map_power": 1.2,
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
        ],
        "v11": [
            merged(
                "fm448_s_l4cat_sp02_robust_tk002_mp060k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "concat",
                    "model.feature_dim": 1536,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_s_l4mean_sp02_robust_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_l4mean_lc31_robust_p12_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_enhancement": "local_contrast",
                    "eval.local_contrast_kernel": 31,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.map_power": 1.2,
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_b_l4mean_sp02_robust_tk002_mp080k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
        ],
        "v12": [
            merged(
                "fm448_s_calq90_p14_robust_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.4,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_calq95_p16_robust_tk001_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.001,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.95,
                    "eval.score_map_calibration.power": 1.6,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_l4mean_calq90_p14_sp02_tk002_mp080k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.4,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_b_calq90_p14_sp02_tk002_mp060k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.4,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
        ],
        "v13": [
            merged(
                "fm448_s_calmeanstd_p12_robust_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_mean_std",
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_calmedianmad_p12_robust_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_median_mad",
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_s_lc31_calq90_p14_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_enhancement": "local_contrast_abs",
                    "eval.local_contrast_kernel": 31,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.4,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.max_patches": 100000,
                },
            ),
            merged(
                "fm448_b_l4mean_calq90_p12_tk002_mp050k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.max_patches": 50000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
        ],
        "v14": [
            merged(
                "fm448_s_spatialctx3_k3_q90_p12_tk002_mp120k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 3,
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_s_spatialctx3_k5_q95_p14_tk001_mp120k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.001,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.95,
                    "eval.score_map_calibration.power": 1.4,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 5,
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_s_l4mean_spatialctx3_k3_sp02_q90_mp100k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 3,
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 100000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_b_spatialctx3_k3_q90_p12_mp060k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 3,
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
        ],
        "v15": [
            merged(
                "fm560_s_spatialctx3_k3_q90_p12_tk001_mp080k",
                **{
                    "data.image_size": 560,
                    "energy.topk_ratio": 0.001,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 3,
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 256,
                },
            ),
            merged(
                "fm448_s_spatialctx5_k3_q90_p12_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 3,
                    "eval.feature_memory.local_context_kernel": 5,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 100000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_s_ctx3_k3_globalq95_p12_tk002_mp120k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "global_quantile",
                    "eval.score_map_calibration.quantile": 0.95,
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 3,
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm448_s_l4cat_spatialctx3_k3_q90_mp060k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "concat",
                    "model.feature_dim": 1536,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 3,
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 256,
                },
            ),
        ],
        "v16": [
            merged(
                "fm448_s_fg_lc17_q85_p10_tk001_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.001,
                    "eval.apply_foreground_mask": "true",
                    "eval.score_map_enhancement": "local_contrast_abs",
                    "eval.local_contrast_kernel": 17,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.85,
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 1,
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 100000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_fg_lc9_q80_p10_tk0005_mp120k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.0005,
                    "eval.apply_foreground_mask": "true",
                    "eval.score_map_enhancement": "local_contrast_abs",
                    "eval.local_contrast_kernel": 9,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.80,
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 1,
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_fg_lc17_q85_p10_tk001_mp080k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.001,
                    "eval.apply_foreground_mask": "true",
                    "eval.score_map_enhancement": "local_contrast_abs",
                    "eval.local_contrast_kernel": 17,
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.85,
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 1,
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.spatial_weight": 0.1,
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm392_b_fg_q85_p10_tk001_mp050k",
                **{
                    "data.image_size": 392,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "energy.topk_ratio": 0.001,
                    "eval.apply_foreground_mask": "true",
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.85,
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 1,
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 50000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_fg_ctx3_k1_q85_p10_tk001_mp080k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.001,
                    "eval.apply_foreground_mask": "true",
                    "eval.score_map_normalization": "image_robust",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.85,
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "image_robust",
                    "eval.feature_memory.k": 1,
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
        ],
        "v17": [
            merged(
                "fm560_s_l4mean_sp02_robust_tk002_mp080k",
                **{
                    "data.image_size": 560,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
            merged(
                "fm560_s_l4cat_sp02_robust_tk002_mp050k",
                **{
                    "data.image_size": 560,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "concat",
                    "model.feature_dim": 1536,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 50000,
                    "eval.feature_memory.chunk_size": 256,
                },
            ),
            merged(
                "fm560_b_l4mean_sp02_robust_tk002_mp040k",
                **{
                    "data.image_size": 560,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 40000,
                    "eval.feature_memory.chunk_size": 256,
                },
            ),
            merged(
                "fm672_s_l4mean_sp02_robust_tk001_mp050k",
                **{
                    "data.image_size": 672,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.001,
                    "eval.score_map_normalization": "image_robust",
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 50000,
                    "eval.feature_memory.chunk_size": 256,
                },
            ),
            merged(
                "fm448_s_l4mean_sp02_robust_s3_tk002_mp150k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "image_robust",
                    "eval.score_map_smoothing_kernel": 3,
                    "eval.feature_memory.spatial_weight": 0.2,
                    "eval.feature_memory.max_patches": 150000,
                    "eval.feature_memory.chunk_size": 512,
                },
            ),
        ],
        "v18": [
            merged(
                "fm448_s_raw_normoff_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 100000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_raw_s3_normoff_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.score_map_smoothing_kernel": 3,
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 100000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_raw_normoff_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 100000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_calmeanstd_raw_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_mean_std",
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 100000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_b_raw_normoff_tk002_mp060k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
        ],
        "v19": [
            merged(
                "fm448_s_l4mean_q95raw_tk002_mp120k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.95,
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_madraw_p12_tk002_mp100k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_median_mad",
                    "eval.score_map_calibration.power": 1.2,
                    "eval.score_map_calibration.post_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 100000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_ctx3raw_tk002_mp080k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4cat_raw_tk002_mp050k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "concat",
                    "model.feature_dim": 1536,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 50000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm518_s_l4mean_raw_tk002_mp060k",
                **{
                    "data.image_size": 518,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
        ],
        "v20": [
            merged(
                "fm448_s_l4mean_ctx3raw_fg_tk002_mp080k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.apply_foreground_mask": "true",
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_ctx3raw_s3_tk002_mp080k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_smoothing_kernel": 3,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_ctx3raw_mapmax_tk0005_mp080k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0005,
                    "eval.score_map_normalization": "none",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_max",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_max",
                },
            ),
            merged(
                "fm518_s_l4mean_ctx3raw_tk001_mp060k",
                **{
                    "data.image_size": 518,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.001,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_b_l4mean_ctx3raw_tk002_mp040k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14",
                    "model.feature_dim": 768,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 40000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
        ],
        "v21": [
            merged(
                "fm448_breg_l4mean_raw_tk002_mp040k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14_reg",
                    "model.feature_dim": 768,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 40000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_breg_l4mean_ctx3raw_tk002_mp040k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14_reg",
                    "model.feature_dim": 768,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 40000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_l_l4mean_raw_tk002_mp020k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitl14",
                    "model.feature_dim": 1024,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.002,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 20000,
                    "eval.feature_memory.chunk_size": 128,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm784_s_l4mean_ctx3raw_s3_tk0005_mp030k",
                **{
                    "data.image_size": 784,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0005,
                    "eval.score_map_smoothing_kernel": 3,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 30000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_ctx3raw_pap_s5_tk0002_mp120k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0002,
                    "eval.score_map_smoothing_kernel": 5,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
        ],
        "v22": [
            merged(
                "fm784_s_l4mean_ctx3raw_s5_tk0002_mp050k",
                **{
                    "data.image_size": 784,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0002,
                    "eval.score_map_smoothing_kernel": 5,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 50000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm672_s_l4mean_ctx5raw_s5_tk0005_mp080k",
                **{
                    "data.image_size": 672,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0005,
                    "eval.score_map_smoothing_kernel": 5,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 5,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 80000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_breg_l4mean_ctx3raw_mapmax_tk0002_mp050k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitb14_reg",
                    "model.feature_dim": 768,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.0002,
                    "eval.score_map_normalization": "none",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_max",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 50000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_max",
                },
            ),
            merged(
                "fm448_l_l4mean_ctx3raw_tk0005_mp030k",
                **{
                    "data.image_size": 448,
                    "model.torchhub_model": "dinov2_vitl14",
                    "model.feature_dim": 1024,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "energy.topk_ratio": 0.0005,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 30000,
                    "eval.feature_memory.chunk_size": 128,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_hctx3raw_tk0005_mp120k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0005,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.feature": "h",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_ctx3_madraw_p10_tk0005_mp120k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0005,
                    "eval.score_map_normalization": "none",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_median_mad",
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm448_s_l4mean_ctx3_q90raw_p10_tk0005_mp120k",
                **{
                    "data.image_size": 448,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0005,
                    "eval.score_map_normalization": "none",
                    "eval.image_score_from_postprocessed_map": "true",
                    "eval.postprocessed_image_score_mode": "map_topk",
                    "eval.score_map_calibration.enabled": "true",
                    "eval.score_map_calibration.mode": "spatial_quantile",
                    "eval.score_map_calibration.quantile": 0.90,
                    "eval.score_map_calibration.power": 1.0,
                    "eval.score_map_calibration.post_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.local_context_kernel": 3,
                    "eval.feature_memory.local_context_aggregation": "concat",
                    "eval.feature_memory.sampling_strategy": "spatial_uniform",
                    "eval.feature_memory.max_patches": 120000,
                    "eval.feature_memory.chunk_size": 512,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
            merged(
                "fm672_s_l4mean_lc17raw_tk0005_mp060k",
                **{
                    "data.image_size": 672,
                    "model.intermediate_layers": 4,
                    "model.intermediate_aggregation": "mean",
                    "model.feature_dim": 384,
                    "energy.topk_ratio": 0.0005,
                    "eval.score_map_enhancement": "local_contrast_abs",
                    "eval.local_contrast_kernel": 17,
                    "eval.score_map_normalization": "none",
                    "eval.feature_memory.normalize_maps": "false",
                    "eval.feature_memory.max_patches": 60000,
                    "eval.feature_memory.chunk_size": 256,
                    "eval.feature_memory.image_score_mode": "map_topk",
                },
            ),
        ],
    }


def all_trials() -> dict[str, Trial]:
    out: dict[str, Trial] = {}
    for trials in trial_sets().values():
        for trial in trials:
            out[trial.name] = trial
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--categories", nargs="*", default=None, help="Optional dataset/category filters.")
    parser.add_argument("--trial-set", choices=sorted(trial_sets()), default="v9")
    parser.add_argument("--trials", nargs="*", default=None, help="Explicit trial names; overrides --trial-set.")
    parser.add_argument("--output-root", default="outputs/feature_memory_v9")
    parser.add_argument("--log-root", default="outputs/logs/feature_memory_v9")
    parser.add_argument("--test-split-role", choices=["tune", "final", "all"], default="tune")
    parser.add_argument("--test-split-fraction", type=float, default=0.5)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=int, default=30)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-running", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("completed"))


def running_command_text() -> str:
    try:
        result = subprocess.run(["ps", "-eo", "cmd"], check=False, capture_output=True, text=True)
    except Exception:
        return ""
    return result.stdout


def is_running(task: Task, command_text: str | None = None) -> bool:
    if command_text is None:
        command_text = running_command_text()
    return f"eval.result_dir={task.result_dir}" in command_text


def selected_trials(args: argparse.Namespace) -> list[Trial]:
    lookup = all_trials()
    if args.trials:
        missing = sorted(set(args.trials) - set(lookup))
        if missing:
            raise ValueError(f"Unknown feature-memory trials: {', '.join(missing)}")
        return [lookup[name] for name in args.trials]
    return trial_sets()[args.trial_set]


def command(args: argparse.Namespace, task: Task) -> list[str]:
    overrides = [
        "device=" + args.device,
        "data.few_shot=full",
        "data.robustness=none",
        "data.test_split_fraction=" + str(args.test_split_fraction),
        "data.test_split_role=" + args.test_split_role,
        "data.root=" + str(task.experiment.root),
        quoted_override("data.category", task.experiment.category),
        "data.part_mask_root=" + str(task.experiment.part_mask_root),
        "eval.result_dir=" + str(task.result_dir),
    ]
    overrides.extend(f"{key}={value}" for key, value in task.trial.overrides.items())
    return [sys.executable, "scripts/evaluate_feature_memory.py", "--config", args.config, "--set", *overrides]


def build_tasks(args: argparse.Namespace) -> list[Task]:
    dataset_filter = set(args.datasets or [])
    category_filter = set(args.categories or [])
    experiments = downloaded_experiments()
    if dataset_filter:
        experiments = [experiment for experiment in experiments if experiment.dataset in dataset_filter]
    if category_filter:
        experiments = [
            experiment
            for experiment in experiments
            if f"{experiment.dataset}/{experiment.category}" in category_filter or experiment.category in category_filter
        ]
    tasks: list[Task] = []
    for experiment in experiments:
        for trial in selected_trials(args):
            result_dir = Path(args.output_root) / "results" / experiment.dataset / trial.name / experiment.category
            tasks.append(
                Task(
                    experiment=experiment,
                    trial=trial,
                    result_file=result_dir / f"{experiment.category}_train_metrics.json",
                    result_dir=result_dir,
                    log_file=Path(args.log_root) / experiment.dataset / trial.name / f"{experiment.category}.log",
                )
            )
    return tasks


def launch(args: argparse.Namespace, task: Task, attempt: int = 1) -> tuple[subprocess.Popen, object]:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = task.log_file.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = command(args, task)
    print(f"[feature-memory-sweep] START {task.name} attempt={attempt} log={task.log_file}", flush=True)
    print(f"# attempt={attempt}", file=handle, flush=True)
    print("+ " + shell_join(cmd), file=handle, flush=True)
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env), handle


def metric_value(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def summarize(tasks: list[Task], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if not completed(task.result_file):
            continue
        payload = json.loads(task.result_file.read_text(encoding="utf-8"))
        metrics = payload.get("best_eval") or payload.get("latest_eval") or {}
        row = {
            "dataset": task.experiment.dataset,
            "category": task.experiment.category,
            "trial": task.trial.name,
            "path": str(task.result_file),
        }
        for key in METRIC_KEYS:
            row[key] = metric_value(metrics, key)
        rows.append(row)
    summary_path = output_root / "all_results.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "category", "trial", *METRIC_KEYS, "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[feature-memory-sweep] wrote {summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)
    command_text = running_command_text() if args.skip_running else ""
    pending = [
        task
        for task in tasks
        if not completed(task.result_file)
        and not (args.skip_running and is_running(task, command_text))
    ]
    print(
        f"[feature-memory-sweep] total={len(tasks)} skipped={len(tasks) - len(pending)} "
        f"pending={len(pending)} max_parallel={args.max_parallel} split={args.test_split_role}",
        flush=True,
    )
    if args.dry_run:
        for task in pending:
            print("+ " + shell_join(command(args, task)), flush=True)
        return
    failed: list[tuple[Task, int]] = []
    running: list[RunningTask] = []
    while pending or running:
        while pending and len(running) < max(1, int(args.max_parallel)):
            task = pending.pop(0)
            if completed(task.result_file) or (args.skip_running and is_running(task)):
                print(f"[feature-memory-sweep] SKIP {task.name}", flush=True)
                continue
            process, handle = launch(args, task, attempt=1)
            running.append(RunningTask(task=task, process=process, handle=handle, attempts=1))
        next_running: list[RunningTask] = []
        for item in running:
            code = item.process.poll()
            if code is None:
                next_running.append(item)
                continue
            item.handle.close()
            if code != 0:
                print(
                    f"[feature-memory-sweep] FAIL {item.task.name} exit={code} "
                    f"attempt={item.attempts}/{args.max_retries + 1} log={item.task.log_file}",
                    flush=True,
                )
                if item.attempts <= int(args.max_retries):
                    time.sleep(max(0, int(args.retry_backoff_seconds)))
                    process, handle = launch(args, item.task, attempt=item.attempts + 1)
                    next_running.append(RunningTask(task=item.task, process=process, handle=handle, attempts=item.attempts + 1))
                else:
                    failed.append((item.task, code))
                    if not bool(args.continue_on_fail):
                        raise SystemExit(code)
                continue
            if completed(item.task.result_file):
                print(f"[feature-memory-sweep] DONE {item.task.name}", flush=True)
            else:
                print(f"[feature-memory-sweep] FAIL {item.task.name} missing_result log={item.task.log_file}", flush=True)
                if item.attempts <= int(args.max_retries):
                    time.sleep(max(0, int(args.retry_backoff_seconds)))
                    process, handle = launch(args, item.task, attempt=item.attempts + 1)
                    next_running.append(RunningTask(task=item.task, process=process, handle=handle, attempts=item.attempts + 1))
                else:
                    failed.append((item.task, -1))
        running = next_running
        if pending or running:
            time.sleep(5)
    summarize(tasks, Path(args.output_root))
    if failed:
        failed_path = Path(args.output_root) / "failed_tasks.txt"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(
            "\n".join(f"{task.name}\texit={code}\tlog={task.log_file}" for task, code in failed) + "\n",
            encoding="utf-8",
        )
        print(f"[feature-memory-sweep] failed={len(failed)} wrote {failed_path}", flush=True)
        if not bool(args.continue_on_fail):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
