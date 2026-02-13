# MFC Project Overview

## Purpose
Advanced Microbial Fuel Cell (MFC) biological configuration system with Q-learning control, parameter optimization, uncertainty quantification, and real-time analytics.

## Tech Stack
- Python 3.12+ with NumPy, SciPy, Pandas
- Mojo for high-performance calculations
- Pytest for testing
- Streamlit for GUI
- FastAPI for API
- Matplotlib/Plotly/Seaborn for visualization
- PyYAML for configuration
- Optional GPU acceleration (NVIDIA/AMD)

## Key Architecture
- Core MFC models in Mojo for performance
- Q-Learning controller for adaptive control
- Comprehensive biological configuration system
- Real-time monitoring and analytics
- Multi-cell stack simulation
- Advanced sensor integration (EIS, QCM)

## Core Modules (Current Testing Focus)
1. `src/mfc_model.py` - Basic MFC simulation script with Mojo integration
2. `src/mfc_stack_simulation.py` - Complete 5-cell stack simulation with classes:
   - MFCSensor, MFCActuator, MFCCell, MFCStack, MFCStackQLearningController

## Project Structure
- `/q-learning-mfcs/src/` - Main source code
- `/q-learning-mfcs/tests/` - Test suites
- `/q-learning-mfcs/configs/` - Configuration files
- `/configs/` - Additional configurations at root level