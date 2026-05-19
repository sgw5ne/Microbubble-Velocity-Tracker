from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from tracking import Track


class VelocityLogger:
    """
    Records per-track velocity samples and provides saving + plotting utilities.

    Usage:
        logger = VelocityLogger(units_per_px=0.5, unit_label="µm")
        # inside your frame loop:
        logger.record(tracks, frame_idx)
        # when done:
        logger.save("velocities.json")
        logger.plot("velocity_distributions.png")
    """

    def __init__(
        self,
        units_per_px: Optional[float] = None,
        unit_label: str = "px",
        min_track_age: int = 2,
    ):
        self.units_per_px = units_per_px
        self.unit_label = unit_label
        self.min_track_age = min_track_age

        # {track_id: [speed_sample, ...]}
        self._samples: Dict[int, List[float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, tracks: Dict[int, Track], frame_idx: int) -> None:
        """Call once per frame with the current track dict."""
        for tid, t in tracks.items():
            if t.age < self.min_track_age or t.missed > 0:
                continue
            speed = t.velocity_px_s
            if self.units_per_px is not None:
                speed *= self.units_per_px
            self._samples.setdefault(tid, []).append(speed)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save raw per-track samples plus summary statistics to JSON.

        Schema:
        {
          "unit": "µm/s",
          "tracks": {
            "1": {"samples": [...], "mean": ..., "std": ..., "median": ..., "n": ...},
            ...
          },
          "global": {"mean": ..., "std": ..., "median": ..., "n": ...}
        }
        """
        path = Path(path)
        unit_str = f"{self.unit_label}/s"

        tracks_out = {}
        all_speeds: List[float] = []

        for tid, samples in self._samples.items():
            arr = np.array(samples, dtype=float)
            all_speeds.extend(samples)
            tracks_out[str(tid)] = {
                "samples": arr.tolist(),
                "mean":   float(arr.mean()),
                "std":    float(arr.std()),
                "median": float(np.median(arr)),
                "n":      len(arr),
            }

        global_arr = np.array(all_speeds, dtype=float)
        global_stats = (
            {
                "mean":   float(global_arr.mean()),
                "std":    float(global_arr.std()),
                "median": float(np.median(global_arr)),
                "n":      len(global_arr),
            }
            if len(global_arr) > 0
            else {"mean": None, "std": None, "median": None, "n": 0}
        )

        payload = {"unit": unit_str, "tracks": tracks_out, "global": global_stats}
        path.write_text(json.dumps(payload, indent=2))
        print(f"[VelocityLogger] saved → {path}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        path: str | Path,
        bins: int = 30,
        max_per_track_plots: int = 8,
    ) -> None:
        """
        Save a figure with two panels:
          1. Global velocity distribution (all tracks pooled).
          2. Per-track mean ± std bar chart (up to `max_per_track_plots` tracks).
        """
        if not self._samples:
            print("[VelocityLogger] no data to plot.")
            return

        path = Path(path)
        unit_str = f"{self.unit_label}/s"
        all_speeds = [s for samples in self._samples.values() for s in samples]
        arr_all = np.array(all_speeds, dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Velocity Distributions", fontsize=14, fontweight="bold")

        # --- Panel 1: global histogram ---
        ax = axes[0]
        ax.hist(arr_all, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(arr_all.mean(),   color="tomato",  linestyle="--", linewidth=1.5,
                   label=f"mean = {arr_all.mean():.2f}")
        ax.axvline(np.median(arr_all), color="gold", linestyle=":",  linewidth=1.5,
                   label=f"median = {np.median(arr_all):.2f}")
        ax.set_xlabel(f"Speed ({unit_str})")
        ax.set_ylabel("Count")
        ax.set_title("All Tracks — Pooled")
        ax.legend(fontsize=9)

        # --- Panel 2: per-track mean ± std bar chart ---
        ax = axes[1]
        # sort by mean speed descending, cap number shown
        sorted_tracks = sorted(
            self._samples.items(),
            key=lambda kv: np.mean(kv[1]),
            reverse=True,
        )[:max_per_track_plots]

        tids   = [f"ID {tid}" for tid, _ in sorted_tracks]
        means  = [float(np.mean(s))  for _, s in sorted_tracks]
        stds   = [float(np.std(s))   for _, s in sorted_tracks]

        x = np.arange(len(tids))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color="steelblue", edgecolor="white",
                      error_kw={"ecolor": "tomato", "linewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(tids, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(f"Mean Speed ({unit_str})")
        ax.set_title(f"Per-Track Mean ± Std  (top {len(tids)} by speed)")

        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VelocityLogger] plot saved → {path}")