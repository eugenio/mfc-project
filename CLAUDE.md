## Pixi Environment Usage

- For running individual pixi commands, use:
  - `pixi run <command>` - for tasks defined in pixi.toml
  - `pixi run -e <env> <command>` - to run in a specific environment

- `eval "$(pixi shell-hook -e <environment-name>)"` is only needed when you want to enter/activate a pixi shell environment interactively, not for running individual commands

- Example correct usages:
  - `pixi run -e default ruff check q-learning-mfcs/src/config/electrode_config.py`
  - `pixi run ruff-check-mfc`