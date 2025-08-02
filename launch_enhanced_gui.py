"""
Enhanced MFC Platform Launcher

Quick launcher script for the enhanced Streamlit MFC application.
Run this script to start the new multi-page navigation interface.

Usage:
    python launch_enhanced_gui.py

Or with pixi:
    pixi run -e amd-gpu streamlit run launch_enhanced_gui.py

Created: 2025-08-02
"""
import os
import subprocess
import sys


def main():
    """Launch the enhanced MFC platform."""

    print("🚀 Starting Enhanced MFC Scientific Platform...")
    print("=" * 50)

    # Add src to path
    gui_app_path = os.path.join("q-learning-mfcs", "src", "gui", "enhanced_main_app.py")

    if not os.path.exists(gui_app_path):
        print("❌ Error: Enhanced MFC app not found!")
        print(f"   Expected: {gui_app_path}")
        return 1

    try:
        # Run streamlit with the enhanced app
        print("📱 Launching Streamlit application...")
        print(f"   App: {gui_app_path}")
        print("   URL: http://localhost:8501")
        print("=" * 50)

        # Use streamlit run to launch the app
        cmd = [sys.executable, "-m", "streamlit", "run", gui_app_path,
               "--server.port", "8501",
               "--server.address", "localhost"]

        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching application: {e}")
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
