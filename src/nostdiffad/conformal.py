from __future__ import annotations

import torch


class ConformalThreshold:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.threshold: float | None = None

    def fit(self, calibration_energy: torch.Tensor) -> float:
        flat = calibration_energy.detach().flatten().float().cpu()
        if flat.numel() == 0:
            raise ValueError("Cannot fit conformal threshold on empty calibration energy.")
        quantile = 1.0 - self.alpha
        self.threshold = float(torch.quantile(flat, quantile).item())
        return self.threshold

    def predict(self, energy: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("ConformalThreshold.fit must be called before predict.")
        return (energy > self.threshold).long()

    def coverage(self, normal_energy: torch.Tensor) -> float:
        if self.threshold is None:
            raise RuntimeError("ConformalThreshold.fit must be called before coverage.")
        return float((normal_energy <= self.threshold).float().mean().item())
