from __future__ import annotations
from typing import Optional, Tuple


class FilePicker:
    @staticmethod
    def pick_video_file() -> Optional[str]:
        """Open a native file dialog to select a video. Returns path or None if cancelled."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            print("Tkinter not available. On Linux try: sudo apt-get install python3-tk")
            return None

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select ultrasound video", filetypes=filetypes)
        root.destroy()
        return path if path else None


class ScaleCalibrator:
    @staticmethod
    def prompt() -> Tuple[Optional[float], str]:
        """
        Manual calibration: user enters scale bar pixels and real-world distance.
        Returns (units_per_px, unit_label). If skipped, returns (None, "px").
        """
        print("\n=== Scale Calibration (optional) ===")
        print("Measure the scale bar length in PIXELS using any viewer.")
        print("Then enter the real-world distance it represents.")
        print("Press Enter at the first prompt to skip and show velocity in px/s.\n")

        px_str = input("Scale bar length in pixels (e.g., 120) [Enter to skip]: ").strip()
        if px_str == "":
            return None, "px"

        try:
            bar_px = float(px_str)
            if bar_px <= 0:
                raise ValueError
        except ValueError:
            print("Invalid pixel length. Skipping calibration.")
            return None, "px"

        dist_str = input("Real-world distance represented (number only, e.g., 10): ").strip()
        unit_str = input("Unit (mm / cm / um / m) [default mm]: ").strip().lower() or "mm"

        try:
            dist_val = float(dist_str)
            if dist_val <= 0:
                raise ValueError
        except ValueError:
            print("Invalid distance. Skipping calibration.")
            return None, "px"

        valid_units = {"mm", "cm", "um", "µm", "m"}
        if unit_str not in valid_units:
            print(f"Unknown unit '{unit_str}'. Using mm.")
            unit_str = "mm"

        units_per_px = dist_val / bar_px
        print(f"Calibration set: {units_per_px:.6f} {unit_str}/px\n")
        return units_per_px, unit_str
