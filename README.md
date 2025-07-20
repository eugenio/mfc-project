# MFC Project

A comprehensive microbial fuel cell (MFC) research and development project featuring Q-learning control systems and advanced simulation capabilities.

## Project Status

This project contains a complete Q-learning control system for microbial fuel cell stacks, implemented with Mojo acceleration for high-performance real-time control.

### Q-Learning MFC Subproject

**Status**: ✅ Complete and functional

The `q-learning-mfcs/` directory contains a fully implemented Q-learning control system for a 5-cell MFC stack featuring:

- **Advanced Control**: Q-learning algorithm optimized for accelerator hardware (GPU/NPU/ASIC)
- **Complete Simulation**: 5-cell MFC stack with realistic electrochemical dynamics
- **Sensor Integration**: Voltage, current, pH, and substrate concentration monitoring
- **Actuator Control**: PWM duty cycle, pH buffer, and acetate addition systems
- **Performance**: 97.1% power stability, 100% cell reversal prevention
- **Documentation**: Comprehensive LaTeX reports with analysis and figures

**Key Achievements**:

- Training time: 0.65 seconds (1000 steps)
- Power output: 0.037-0.245 W
- Zero cell reversals during operation
- Complete sensor/actuator simulation with realistic noise and dynamics

For detailed technical information, implementation details, and usage instructions, see the [complete documentation](q-learning-mfcs/README.md).

## Development Environment

This project uses [Pixi](https://pixi.sh) for dependency management and development environment setup:

```bash
# Install dependencies
pixi install

# Install development tools
pixi install -e dev

# Run security checks
pixi run -e dev bandit -r . --exclude ./.pixi
pixi run -e dev detect-secrets scan --exclude-files '.pixi/.*'

# Format markdown files
pixi run -e dev mdformat .
```

## Project Structure

```text
├── q-learning-mfcs/          # Q-learning MFC control system
│   ├── README.md            # Detailed project documentation
│   ├── figures/             # Analysis graphs and diagrams
│   ├── reports/             # LaTeX research reports
│   └── *.py, *.mojo         # Implementation files
├── pixi.toml                # Dependency configuration
└── README.md                # This file
```

## License

This project is for educational and research purposes.
