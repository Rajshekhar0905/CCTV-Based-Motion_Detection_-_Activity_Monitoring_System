"""
main.py

Entry point for the CCTV Motion Detection System.

Run from a terminal with:
    python main.py

This opens the Tkinter GUI. No arguments are required; the video file is
selected from within the GUI via the "Select Video" button.
"""

import sys

import config
from gui import launch_app


def main():
    config.ensure_directories()
    try:
        launch_app()
    except Exception as exc:  # noqa: BLE001
        print(f"Fatal error while starting the application: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
