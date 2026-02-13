"""Constants for AppTest integration tests."""

from pathlib import Path

# Page labels as displayed in the sidebar radio widget
PAGE_LABELS: list[str] = [
    "\U0001f3e0 Dashboard",
    "\U0001f50b Electrode System",
    "\U0001f3d7\ufe0f Cell Configuration",
    "\u2697\ufe0f Physics Simulation",
    "\U0001f9e0 ML Optimization",
    "\U0001f9ec GSM Integration",
    "\U0001f4da Literature Validation",
    "\U0001f4ca Performance Monitor",
    "\u2699\ufe0f Configuration",
]

# Mapping from page label to internal page key
PAGE_KEYS: dict[str, str] = {
    "\U0001f3e0 Dashboard": "dashboard",
    "\U0001f50b Electrode System": "electrode_system",
    "\U0001f3d7\ufe0f Cell Configuration": "cell_configuration",
    "\u2697\ufe0f Physics Simulation": "advanced_physics",
    "\U0001f9e0 ML Optimization": "ml_optimization",
    "\U0001f9ec GSM Integration": "gsm_integration",
    "\U0001f4da Literature Validation": "literature_validation",
    "\U0001f4ca Performance Monitor": "performance_monitor",
    "\u2699\ufe0f Configuration": "system_configuration",
}

# Path to the main app file
APP_FILE = str(
    Path(__file__).parent / ".." / ".." / "src" / "gui" / "enhanced_main_app.py",
)
