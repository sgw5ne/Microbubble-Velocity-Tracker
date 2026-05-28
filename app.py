from __future__ import annotations
from typing import Optional, Tuple, List
from velocity_logger import VelocityLogger

import cv2
import numpy as np

from processing import FramePreprocessor, BackgroundSubtractorEMA, MicrobubbleDetector
from tracking import NearestNeighborTracker
from viz import Visualizer
from deconvolution import Deconvolver


class ROISelector:
    # allow user to select region of interest for video

    def __init__(self):
        self.roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self._drawing = False
        self._start: Optional[Tuple[int, int]] = None
        self._end: Optional[Tuple[int, int]] = None
        self._confirmed = False
        self._window_name = "Select ROI — drag to draw, Enter to confirm, R to reset, S to skip"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def select(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # open selection window for users to draw ROI 
        # note: implement ability to choose different shapes in the future (freehand, circles, etc)
        self._frame = frame.copy()
        self._overlay = frame.copy()
        self._confirmed = False
        self._drawing = False
        self._start = self._end = None

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._window_name, self._on_mouse)

        print("\n--- ROI Selection ---")
        print("  Drag  : draw region")
        print("  Enter : confirm")
        print("  R     : reset / redraw")
        print("  S     : skip (use full frame)")
        print("---------------------\n")

        while True:
            display = self._render()
            cv2.imshow(self._window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                if self._start and self._end:
                    self.roi = self._rect_from_points(self._start, self._end)
                    print(f"ROI confirmed: x={self.roi[0]}, y={self.roi[1]}, "
                          f"w={self.roi[2]}, h={self.roi[3]}")
                    break
                else:
                    print("No region drawn yet — drag on the image first.")
            elif key == ord("r"):
                self._start = self._end = None
                print("ROI reset.")
            elif key == ord("s"):
                self.roi = None
                print("ROI skipped — full frame will be used.")
                break

        cv2.destroyWindow(self._window_name)
        return self.roi

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (x, y)
            self._end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._end = (x, y)

    def _render(self) -> np.ndarray:
        display = self._frame.copy()
        if self._start and self._end:
            x, y, w, h = self._rect_from_points(self._start, self._end)
            # semi-transparent fill
            overlay = display.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            # solid border + corner handles
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
            for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
                cv2.circle(display, (cx, cy), 5, (0, 200, 255), -1)
            label = f"{w} x {h} px"
            cv2.putText(display, label, (x + 4, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        # instructions overlay
        cv2.putText(display, "Enter=confirm  R=reset  S=skip", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return display

    @staticmethod
    def _rect_from_points(p1: Tuple[int, int],
                           p2: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x = min(p1[0], p2[0])
        y = min(p1[1], p2[1])
        w = abs(p2[0] - p1[0])
        h = abs(p2[1] - p1[1])
        return x, y, w, h


class MicrobubbleTrackingApp:
    def __init__(self,
                 video_path: str,
                 units_per_px: Optional[float],
                 unit_label: str,
                 # pipeline params
                 gaussian_ksize: int = 5,
                 gaussian_sigma: float = 1.2,
                 unsharp: bool = True,
                 unsharp_amount: float = 1.0,
                 bg_alpha: float = 0.02,
                 # detection params
                 thresh: int = 18,
                 min_area: float = 6.0,
                 max_area: float = 250.0,
                 morph_ksize: int = 3,
                 # tracking params
                 max_dist: float = 25.0,
                 max_missed: int = 6,
                 min_track_age: int = 2,
                 # optional deconvolution
                 psf: Optional[np.ndarray] = None,
                 rl_iterations: int = 10,
                 # ROI
                 roi: Optional[Tuple[int, int, int, int]] = None,
                 interactive_roi: bool = True):
        self.video_path = video_path
        self.units_per_px = units_per_px
        self.unit_label = unit_label
        self.min_track_age = int(min_track_age)
        self.roi = roi                        # (x, y, w, h) or None = full frame
        self.interactive_roi = interactive_roi

        self.pre = FramePreprocessor(gaussian_ksize, gaussian_sigma, unsharp, unsharp_amount)
        self.bg = BackgroundSubtractorEMA(bg_alpha)
        self.detector = MicrobubbleDetector(thresh, min_area, max_area, morph_ksize)
        self.tracker = NearestNeighborTracker(max_dist, max_missed)
        self.deconv = Deconvolver(psf=psf, rl_iterations=rl_iterations)
        self.viz = Visualizer()
        self.logger = VelocityLogger(
            units_per_px=self.units_per_px,
            unit_label=self.unit_label,
            min_track_age=self.min_track_age
        )

    # Helper methods ...

    def _crop(self, img: np.ndarray) -> np.ndarray:
        """Return the ROI sub-image, or the full image if no ROI is set."""
        if self.roi is None:
            return img
        x, y, w, h = self.roi
        return img[y:y + h, x:x + w]

    def _offset_detections(self, detections: list) -> list:
        """
        Shift detection centroids from ROI-local coords back to full-frame
        coords so the tracker and visualiser always work in frame space.
        """
        if self.roi is None or not detections:
            return detections
        ox, oy = self.roi[0], self.roi[1]
        shifted = []
        for d in detections:
            if isinstance(d, dict):
                shifted.append({**d,
                                 "cx": d["cx"] + ox,
                                 "cy": d["cy"] + oy})
            elif hasattr(d, "_replace"):
                shifted.append(d._replace(cx=d.cx + ox, cy=d.cy + oy))
            else:
                d = list(d)
                d[0] += ox
                d[1] += oy
                shifted.append(tuple(d))
        return shifted

    def _draw_roi_overlay(self, vis: np.ndarray) -> np.ndarray:
        """Draw a subtle ROI border on the visualisation frame."""
        if self.roi is None:
            return vis
        x, y, w, h = self.roi
        # dim the area outside the ROI
        mask = np.zeros(vis.shape[:2], dtype=np.uint8)
        mask[y:y + h, x:x + w] = 255
        dimmed = (vis * 0.4).astype(np.uint8)
        vis = np.where(mask[:, :, None] == 255, vis, dimmed)
        # draw border
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(vis, "ROI", (x + 4, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return vis

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-6:
            fps = 30.0

        # ---- ROI selection on the first frame -------------------------
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read first frame.")

        if self.interactive_roi and self.roi is None:
            selector = ROISelector()
            self.roi = selector.select(first_frame)

        # replay first frame through the pipeline
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        print(f"\nLoaded: {self.video_path}")
        print(f"ROI: {self.roi if self.roi else 'full frame'}")
        print("Press 'q' to quit.\n")
        if self.deconv.enabled():
            print(f"Deconvolution: Richardson–Lucy enabled ({self.deconv.rl_iterations} iterations)\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- work inside the ROI (or full frame if roi is None) ---
            roi_gray = self._crop(gray)

            filt = self.pre.apply(roi_gray)
            proc = self.deconv.apply(filt)
            motion_img = self.bg.apply(proc)
            detections = self.detector.detect(motion_img)

            # shift centroids to full-frame coordinates
            detections = self._offset_detections(detections)

            self.tracker.update(detections, frame_idx, fps)

            self.logger.record(self.tracker.tracks, frame_idx)

            vis = self.viz.draw(
                frame_bgr=frame,
                detections=detections,
                tracks=self.tracker.tracks,
                fps=fps,
                frame_idx=frame_idx,
                units_per_px=self.units_per_px,
                unit_label=self.unit_label,
                min_track_age=self.min_track_age
            )

            # draw ROI border / dim outside region
            vis = self._draw_roi_overlay(vis)

            delay = Visualizer.playback_delay_ms(fps)
            cv2.imshow(self.viz.window_name, vis)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1



        cap.release()
        cv2.destroyAllWindows()
        self.logger.save("velocities.csv")
        self.logger.plot("velocity_distributions.png")