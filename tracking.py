from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Track:
    track_id: int
    last_pos: Tuple[float, float]
    last_frame_idx: int
    age: int = 0
    missed: int = 0
    velocity_px_s: float = 0.0
    history: List[Tuple[float, float]] = field(default_factory=list)

    def update(self, pos: Tuple[float, float], frame_idx: int, fps: float):
        x0, y0 = self.last_pos
        x1, y1 = pos
        dt = (frame_idx - self.last_frame_idx) / fps if fps > 0 else 0.0
        if dt > 0:
            self.velocity_px_s = math.hypot(x1 - x0, y1 - y0) / dt
        self.last_pos = pos
        self.last_frame_idx = frame_idx
        self.age += 1
        self.missed = 0
        self.history.append(pos)

    def mark_missed(self):
        self.missed += 1
        self.age += 1


class NearestNeighborTracker:
    def __init__(self, max_dist_px: float = 25.0, max_missed: int = 6):
        self.max_dist_px = float(max_dist_px)
        self.max_missed = int(max_missed)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def _associate(self, detections: List[Tuple[float, float, float]]):
        if(len(detections) == 0 or len(self.tracks) == 0):
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(self.tracks.keys())
            return {}, unmatched_tracks, unmatched_dets
        if not self.tracks:
            return {}, [], list(range(len(detections)))

        track_ids = list(self.tracks.keys())
        det_ids = list(range(len(detections)))

        dist = np.full((len(track_ids), len(det_ids)), np.inf, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            tx, ty = self.tracks[tid].last_pos
            for j in det_ids:
                dx = detections[j][0] - tx
                dy = detections[j][1] - ty
                dist[i, j] = math.hypot(dx, dy)

        matches = {}
        used_dets = set()

        while True:
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            dmin = dist[i, j]
            if not np.isfinite(dmin) or dmin > self.max_dist_px:
                break

            tid = track_ids[i]
            if int(j) in used_dets:
                dist[i, j] = np.inf
                continue

            matches[tid] = int(j)
            used_dets.add(int(j))
            dist[i, :] = np.inf
            dist[:, j] = np.inf

        unmatched_tracks = [tid for tid in track_ids if tid not in matches]
        unmatched_dets = [j for j in det_ids if j not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Tuple[float, float, float]], frame_idx: int, fps: float):
        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for tid, det_idx in matches.items():
            cx, cy, _ = detections[det_idx]
            self.tracks[tid].update((cx, cy), frame_idx, fps)

        for tid in unmatched_tracks:
            self.tracks[tid].mark_missed()

        for det_idx in unmatched_dets:
            cx, cy, _ = detections[det_idx]
            self.tracks[self.next_id] = Track(
                track_id=self.next_id,
                last_pos=(cx, cy),
                last_frame_idx=frame_idx,
                history=[(cx, cy)]
            )
            self.next_id += 1

        for tid in [tid for tid, t in self.tracks.items() if t.missed > self.max_missed]:
            del self.tracks[tid]
