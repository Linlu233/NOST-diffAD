from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPatchFeatureExtractor(nn.Module):
    """Local fallback feature extractor for tests and offline runs."""

    def __init__(self, feature_dim: int = 384, patch_size: int = 14) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, feature_dim, kernel_size=patch_size, stride=patch_size),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feat = self.proj(images)
        bsz, channels, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, (height, width)


class HFTransformerFeatureExtractor(nn.Module):
    def __init__(self, kind: str, model_name: str, trainable: bool = False) -> None:
        super().__init__()
        self.kind = kind
        self.model_name = model_name
        self.trainable = trainable
        if kind == "dinov2":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name)
        elif kind == "clip":
            from transformers import CLIPVisionModel

            self.model = CLIPVisionModel.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported Hugging Face feature extractor: {kind}")
        if not trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        was_training = self.model.training
        if not self.trainable:
            self.model.eval()
        context = torch.enable_grad() if self.trainable else torch.no_grad()
        with context:
            outputs = self.model(pixel_values=images)
            hidden = outputs.last_hidden_state
        if not self.trainable and was_training:
            self.model.train()
        tokens = hidden[:, 1:, :]
        grid = int(tokens.shape[1] ** 0.5)
        if grid * grid != tokens.shape[1]:
            raise ValueError(f"Expected square patch token grid, got {tokens.shape[1]} tokens.")
        return tokens, (grid, grid)


class TorchHubDINOv2FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        repo_or_dir: str = "facebookresearch/dinov2",
        source: str = "github",
        trainable: bool = False,
        intermediate_layers: int = 1,
        intermediate_aggregation: str = "last",
    ) -> None:
        super().__init__()
        if source == "github":
            cached_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
            if repo_or_dir == "facebookresearch/dinov2" and (cached_repo / "hubconf.py").is_file():
                repo_or_dir = str(cached_repo)
                source = "local"
        self.model = torch.hub.load(repo_or_dir, model_name, source=source, trust_repo=True)
        self.trainable = trainable
        self.intermediate_layers = max(1, int(intermediate_layers))
        self.intermediate_aggregation = str(intermediate_aggregation).lower()
        if not trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        was_training = self.model.training
        if not self.trainable:
            self.model.eval()
        context = torch.enable_grad() if self.trainable else torch.no_grad()
        with context:
            if self.intermediate_layers > 1 and hasattr(self.model, "get_intermediate_layers"):
                outputs = self.model.get_intermediate_layers(
                    images,
                    n=self.intermediate_layers,
                    reshape=False,
                    return_class_token=False,
                    norm=True,
                )
                if self.intermediate_aggregation in {"concat", "cat"}:
                    tokens = torch.cat(list(outputs), dim=-1)
                elif self.intermediate_aggregation in {"mean", "avg"}:
                    tokens = torch.stack(list(outputs), dim=0).mean(dim=0)
                elif self.intermediate_aggregation in {"last", "final"}:
                    tokens = outputs[-1]
                else:
                    raise ValueError(f"Unknown DINOv2 intermediate aggregation: {self.intermediate_aggregation}")
            elif hasattr(self.model, "forward_features"):
                outputs = self.model.forward_features(images)
                if isinstance(outputs, dict):
                    tokens = outputs.get("x_norm_patchtokens")
                    if tokens is None:
                        raise ValueError("DINOv2 torch.hub output lacks x_norm_patchtokens.")
                else:
                    tokens = outputs[:, 1:, :]
            else:
                outputs = self.model(images)
                tokens = outputs[:, 1:, :]
        if not self.trainable and was_training:
            self.model.train()
        grid = int(tokens.shape[1] ** 0.5)
        if grid * grid != tokens.shape[1]:
            raise ValueError(f"Expected square patch token grid, got {tokens.shape[1]} tokens.")
        return tokens, (grid, grid)


class SAM2FeatureExtractor(nn.Module):
    def __init__(self, config_file: str, checkpoint: str, trainable: bool = False) -> None:
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
        except Exception as exc:
            raise ImportError("Install the official SAM2 package to use model.feature_extractor=sam2.") from exc
        if not config_file or not checkpoint:
            raise ValueError("SAM2 feature extraction requires model.sam2_config and model.sam2_checkpoint.")
        self.model = build_sam2(config_file, checkpoint)
        self.trainable = trainable
        if not trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        was_training = self.model.training
        if not self.trainable:
            self.model.eval()
        context = torch.enable_grad() if self.trainable else torch.no_grad()
        with context:
            if hasattr(self.model, "image_encoder"):
                features = self.model.image_encoder(images)
            else:
                features = self.model.forward_image(images)
        if not self.trainable and was_training:
            self.model.train()
        if isinstance(features, dict):
            for key in ("vision_features", "image_embed", "backbone_fpn"):
                if key in features:
                    features = features[key]
                    break
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim != 4:
            raise ValueError("SAM2 image encoder must return a BCHW feature map or a dict/list containing one.")
        bsz, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        return tokens, (height, width)


def make_feature_extractor(config: dict) -> nn.Module:
    kind = str(config.get("feature_extractor", "dinov2")).lower()
    feature_dim = int(config.get("feature_dim", 384))
    patch_size = int(config.get("patch_size", 14))
    if kind in {"conv", "fallback", "synthetic"}:
        return ConvPatchFeatureExtractor(feature_dim=feature_dim, patch_size=patch_size)
    if kind in {"dinov2", "clip"}:
        return HFTransformerFeatureExtractor(
            kind=kind,
            model_name=str(config["hf_model"]),
            trainable=bool(config.get("train_feature_extractor", False)),
        )
    if kind in {"dinov2_torchhub", "torchhub_dinov2"}:
        return TorchHubDINOv2FeatureExtractor(
            model_name=str(config.get("torchhub_model", "dinov2_vits14")),
            repo_or_dir=str(config.get("torchhub_repo_or_dir", "facebookresearch/dinov2")),
            source=str(config.get("torchhub_source", "github")),
            trainable=bool(config.get("train_feature_extractor", False)),
            intermediate_layers=int(config.get("intermediate_layers", 1)),
            intermediate_aggregation=str(config.get("intermediate_aggregation", "last")),
        )
    if kind == "sam2":
        return SAM2FeatureExtractor(
            config_file=str(config.get("sam2_config") or ""),
            checkpoint=str(config.get("sam2_checkpoint") or ""),
            trainable=bool(config.get("train_feature_extractor", False)),
        )
    raise ValueError(f"Unknown feature extractor: {kind}")


def patch_coordinates(grid_hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    height, width = grid_hw
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=-1).reshape(height * width, 2)


def patch_mask_ids(part_mask: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    pooled = F.interpolate(part_mask.float(), size=grid_hw, mode="nearest")
    return pooled[:, 0].flatten(1).long()
