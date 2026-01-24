"""Enhanced MFC Platform Launcher.

Quick launcher script for the enhanced Streamlit MFC application.
Run this script to start the new multi-page navigation interface.

Usage:
    python launch_enhanced_gui.py

Or with pixi:
    pixi run -e amd-gpu streamlit run launch_enhanced_gui.py

Created: 2025-08-02
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int | None:
    """Launch the enhanced MFC platform."""
    # Add src to path
    gui_app_path = os.path.join("q-learning-mfcs", "src", "gui", "enhanced_main_app.py")

    if not os.path.exists(gui_app_path):
        return 1

    try:
        # Run streamlit with the enhanced app

        # Use streamlit run to launch the app
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            gui_app_path,
            "--server.port",
            "8501",
            "--server.address",
            "localhost",
        ]

        subprocess.run(cmd, check=False)

    except KeyboardInterrupt:
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    # Set working directory to script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Add src to Python path
    src_path = os.path.join(os.getcwd(), "q-learning-mfcs", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    exit_code = main()
    sys.exit(exit_code)
