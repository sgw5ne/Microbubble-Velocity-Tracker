from __future__ import annotations
from typing import Optional
import numpy as np


class Deconvolver:

    def __init__(self, psf: Optional[np.ndarray] = None, rl_iterations: int = 10):
        self.psf = None if psf is None else self._normalize_psf(psf)
        self.rl_iterations = int(max(1, rl_iterations))

    @staticmethod
    def _normalize_psf(psf: np.ndarray) -> np.ndarray:
        psf = psf.astype(np.float32)
        psf = np.maximum(psf, 0)
        return psf / (float(psf.sum()) + 1e-12)

    @staticmethod
    def _psf_flip(psf: np.ndarray) -> np.ndarray:
        return np.flipud(np.fliplr(psf))

    def enabled(self) -> bool:
        return self.psf is not None

    def apply(self, img_u8: np.ndarray) -> np.ndarray:
        """
        img_u8: uint8 grayscale
        returns: uint8 grayscale deconvolved
        """
        if self.psf is None:
            return img_u8

        img = img_u8.astype(np.float32) / 255.0
        est = self._richardson_lucy(img, self.psf, self.rl_iterations)
        return (np.clip(est, 0.0, 1.0) * 255).astype(np.uint8)

    def _richardson_lucy(self, img: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
        psf_m = self._psf_flip(psf)
        estimate = np.maximum(img, 1e-6)

        H, W = img.shape
        ph, pw = psf.shape

        psf_pad = np.zeros((H, W), dtype=np.float32)
        psf_pad[:ph, :pw] = psf
        psf_pad = np.roll(psf_pad, -ph // 2, axis=0)
        psf_pad = np.roll(psf_pad, -pw // 2, axis=1)

        psf_m_pad = np.zeros((H, W), dtype=np.float32)
        psf_m_pad[:ph, :pw] = psf_m
        psf_m_pad = np.roll(psf_m_pad, -ph // 2, axis=0)
        psf_m_pad = np.roll(psf_m_pad, -pw // 2, axis=1)

        H_psf = np.fft.rfft2(psf_pad)
        H_psf_m = np.fft.rfft2(psf_m_pad)

        eps = 1e-6
        for _ in range(iterations):
            conv_est = np.fft.irfft2(np.fft.rfft2(estimate) * H_psf, s=(H, W))
            relative_blur = img / (conv_est + eps)
            corr = np.fft.irfft2(np.fft.rfft2(relative_blur) * H_psf_m, s=(H, W))
            estimate *= corr
            estimate = np.maximum(estimate, eps)

        return estimate
