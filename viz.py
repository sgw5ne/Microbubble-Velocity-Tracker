from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from tracking import Track


def speed_to_bgr(speed: float, min_speed: float, max_speed: float) -> Tuple[int, int, int]:
    """
    Map a speed value to a BGR color on a cool→warm gradient:
      0.0 → deep blue  (100, 0)   hue ~240°
      0.5 → cyan/green (90,  0)   hue ~180°  
      1.0 → red/orange (0,   255) hue ~0°
    Uses HSV so the gradient stays perceptually smooth.
    """
    span = max_speed - min_speed
    t = (speed - min_speed) / span if span > 1e-6 else 0.0
    t = float(np.clip(t, 0.0, 1.0))

    # Hue: 240° (blue) → 0° (red), travelling through cyan & green
    hue = int((1.0 - t) * 120)   # OpenCV hue is 0-179 (half of 360)
    sat = 220
    val = 230
    bgr = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


class Visualizer:
    def __init__(self, window_name: str = "microbubble_tracking"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    @staticmethod
    def playback_delay_ms(fps: float) -> int:
        return max(1, int(round(1000.0 / fps))) if fps > 0 else 1

    def draw(self,
             frame_bgr: np.ndarray,
             detections: List[Tuple[float, float, float]],
             tracks: Dict[int, Track],
             fps: float,
             frame_idx: int,
             units_per_px: Optional[float],
             unit_label: str,
             min_track_age: int = 2) -> np.ndarray:
        vis = frame_bgr.copy()

        # --- collect speed range across all active tracks for normalization ---
        speeds = [
            t.velocity_px_s
            for t in tracks.values()
            if t.age >= min_track_age and len(t.history) >= 2
        ]
        min_speed = min(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 1.0

        # --- detections ---
        for (cx, cy, area) in detections:
            r = int(max(2, min(12, math.sqrt(area))))
            cv2.circle(vis, (int(cx), int(cy)), r, (0, 255, 0), 1)

        # --- tracks ---
        for tid, t in tracks.items():
            x, y = t.last_pos
            track_color = speed_to_bgr(t.velocity_px_s, min_speed, max_speed)

            # draw trail with per-segment color based on local speed
            if len(t.history) >= 2:
                history = t.history[-15:]
                for i in range(1, len(history)):
                    p1 = (int(history[i - 1][0]), int(history[i - 1][1]))
                    p2 = (int(history[i][0]),     int(history[i][1]))
                    # interpolate color along the trail so older segments fade cool
                    seg_t = i / (len(history) - 1)
                    seg_speed = min_speed + seg_t * (t.velocity_px_s - min_speed)
                    seg_color = speed_to_bgr(seg_speed, min_speed, max_speed)
                    cv2.line(vis, p1, p2, seg_color, 2)

            if t.age >= min_track_age and t.missed == 0:
                if units_per_px is not None:
                    v_units_s = t.velocity_px_s * units_per_px
                    label = f"ID {tid}: {v_units_s:.2f} {unit_label}/s"
                else:
                    label = f"ID {tid}: {t.velocity_px_s:.1f} px/s"

                cv2.putText(vis, label, (int(x) + 8, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)

            cv2.circle(vis, (int(x), int(y)), 3, track_color, -1)

        # --- speed legend (color bar) ---
        self._draw_legend(vis, min_speed, max_speed, units_per_px, unit_label)

        cv2.putText(vis,
                    f"frame={frame_idx}  fps={fps:.1f}  tracks={len(tracks)}  dets={len(detections)}",
                    (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return vis

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    def _draw_legend(self,
                     vis: np.ndarray,
                     min_speed: float,
                     max_speed: float,
                     units_per_px: Optional[float],
                     unit_label: str) -> None:
        """Draw a vertical cool→warm color bar with speed labels in the top-right corner."""
        bar_h, bar_w = 120, 14
        margin = 10
        x0 = vis.shape[1] - bar_w - 50
        y0 = margin

        # draw gradient bar
        for i in range(bar_h):
            t = 1.0 - i / (bar_h - 1)          # top = fast (warm), bottom = slow (cool)
            speed = min_speed + t * (max_speed - min_speed)
            color = speed_to_bgr(speed, min_speed, max_speed)
            cv2.line(vis, (x0, y0 + i), (x0 + bar_w, y0 + i), color, 1)

        cv2.rectangle(vis, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

        # labels
        for frac, speed in [(0.0, max_speed), (0.5, (min_speed + max_speed) / 2), (1.0, min_speed)]:
            ly = y0 + int(frac * (bar_h - 1))
            display = speed * units_per_px if units_per_px else speed
            suffix = f" {unit_label}/s" if units_per_px else " px/s"
            cv2.putText(vis, f"{display:.1f}{suffix}",
                        (x0 + bar_w + 4, ly + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)