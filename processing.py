from __future__ import annotations
from typing import Optional, Tuple, List
import cv2
import numpy as np


class FramePreprocessor:
    """Convolution-based denoise + optional unsharp masking."""

    def __init__(self, gaussian_ksize: int = 5, gaussian_sigma: float = 1.2,
                 unsharp: bool = True, unsharp_amount: float = 1.0):
        self.gaussian_ksize = int(gaussian_ksize)
        self.gaussian_sigma = float(gaussian_sigma)
        self.unsharp = bool(unsharp)
        self.unsharp_amount = float(unsharp_amount)

    def apply(self, gray_u8: np.ndarray) -> np.ndarray:
        k = max(3, self.gaussian_ksize | 1)
        blurred = cv2.GaussianBlur(gray_u8, (k, k), self.gaussian_sigma)
        if not self.unsharp:
            return blurred
        return cv2.addWeighted(gray_u8, 1.0 + self.unsharp_amount, blurred, -self.unsharp_amount, 0)


class BackgroundSubtractorEMA:
    """EMA background model + abs(frame - background)."""

    def __init__(self, alpha: float = 0.02):
        self.alpha = float(alpha)
        self.bg_model: Optional[np.ndarray] = None

    def apply(self, gray_u8: np.ndarray) -> np.ndarray:
        fg = gray_u8.astype(np.float32)
        if self.bg_model is None:
            self.bg_model = fg.copy()
        else:
            self.bg_model = (1 - self.alpha) * self.bg_model + self.alpha * fg
        return cv2.absdiff(gray_u8, self.bg_model.astype(np.uint8))


class MicrobubbleDetector:
    """Threshold + morph + contour blob detection."""

    def __init__(self, thresh: int = 18, min_area: float = 6.0, max_area: float = 250.0, morph_ksize: int = 3):
        self.thresh = int(thresh)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.morph_ksize = int(morph_ksize)

    def detect(self, motion_img_u8: np.ndarray) -> List[Tuple[float, float, float]]:
        _, bw = cv2.threshold(motion_img_u8, self.thresh, 255, cv2.THRESH_BINARY)

        k = max(3, self.morph_ksize | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets: List[Tuple[float, float, float]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dets.append((float(cx), float(cy), float(area)))
        return dets
