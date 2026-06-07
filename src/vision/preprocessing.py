from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .image_enhancement import enhance_rgb_to_gray


@dataclass(frozen=True)
class ModelInputSpec:
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    color_order: str = "RGB"


def preprocess_roi_for_model(
    roi_bgr: np.ndarray,
    input_spec: ModelInputSpec,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    size = int(input_spec.image_size)
    resized_bgr = cv2.resize(roi_bgr, (size, size), interpolation=cv2.INTER_AREA)
    if input_spec.color_order.upper() != "RGB":
        raise ValueError(f"Orden de color no soportado: {input_spec.color_order}")

    roi_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    enhanced_gray = enhance_rgb_to_gray(roi_rgb)
    img3 = np.stack([enhanced_gray, enhanced_gray, enhanced_gray], axis=2)
    img3 = img3.astype(np.float32) / 255.0

    chw = np.transpose(img3, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    mean = torch.tensor(input_spec.mean, device=device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(input_spec.std, device=device, dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor, enhanced_gray

