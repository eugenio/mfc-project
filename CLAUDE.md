## Pixi Environment Usage

- For running individual pixi commands, use:
  - `pixi run <command>` - for tasks defined in pixi.toml
  - `pixi run -e <env> <command>` - to run in a specific environment

- `eval "$(pixi shell-hook -e <environment-name>)"` is only needed when you want to enter/activate a pixi shell environment interactively, not for running individual commands

- Example correct usages:
  - `pixi run -e default ruff check q-learning-mfcs/src/config/electrode_config.py`
  - `pixi run ruff-check-mfc`

- Add python from the mfc pixi environment to an environment variable if you need

- Stop your work and fix any pixi environment issue that arises before continuing

## Geometry Configuration Approach

- Instead of modifying the large file in chunks, let me create the density and mass functions and then add them to the existing geometry configuration section.