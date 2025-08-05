"""Energy management and power optimization test suite."""

# Import existing test modules
try:
    from .test_energy_analysis import *
except ImportError:
    pass

__all__ = [
    'test_analyze_energy_sustainability_basic',
    'test_power_calculations',
    'test_visualization_functions',
    'test_summary_function',
    'test_main_function'
]
