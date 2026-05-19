from __future__ import annotations

from ui import FilePicker, ScaleCalibrator
from app import MicrobubbleTrackingApp


def main():
    video_path = FilePicker.pick_video_file()
    if not video_path:
        print("No video selected. Exiting.")
        return

    units_per_px, unit_label = ScaleCalibrator.prompt()

    # If/when you have a PSF, load/construct it and pass as `psf=...` (numpy 2D array).
    psf = None
    rl_iterations = 10

    app = MicrobubbleTrackingApp(
        video_path=video_path,
        units_per_px=units_per_px,
        unit_label=unit_label,
        psf=psf,
        rl_iterations=rl_iterations,
    )
    app.run()


if __name__ == "__main__":
    main()
