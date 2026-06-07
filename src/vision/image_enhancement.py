from __future__ import annotations

import cv2
import numpy as np


GAMMA = 0.75
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAUSSIAN_SIGMA = 1.0
GAUSSIAN_KERNEL_SIZE = 5
UNSHARP_LAMBDA = 1.5


def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def contrast_stretch(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    min_val = float(np.min(gray_f))
    max_val = float(np.max(gray_f))
    if max_val - min_val < 1e-5:
        return np.zeros_like(gray, dtype=np.uint8)
    stretched = (gray_f - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def gamma_correction(gray: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    corrected = np.power(gray_f, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray)


def gaussian_unsharp(gray: np.ndarray) -> np.ndarray:
    kernel = (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE)
    blurred = cv2.GaussianBlur(gray, kernel, sigmaX=GAUSSIAN_SIGMA, sigmaY=GAUSSIAN_SIGMA)
    blurred_again = cv2.GaussianBlur(
        blurred,
        kernel,
        sigmaX=GAUSSIAN_SIGMA,
        sigmaY=GAUSSIAN_SIGMA,
    )
    sharp = blurred.astype(np.float32) + UNSHARP_LAMBDA * (
        blurred.astype(np.float32) - blurred_again.astype(np.float32)
    )
    return np.clip(sharp, 0, 255).astype(np.uint8)


def enhance_rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img_rgb)
    stretched = contrast_stretch(gray)
    gamma_img = gamma_correction(stretched)
    clahe_img = apply_clahe(gamma_img)
    return gaussian_unsharp(clahe_img)

