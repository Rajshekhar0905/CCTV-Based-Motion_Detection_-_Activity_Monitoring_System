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
