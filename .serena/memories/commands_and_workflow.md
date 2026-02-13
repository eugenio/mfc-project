# Commands and Development Workflow

## Pixi Environment Usage
- Use `pixi run <command>` for individual commands
- Use `pixi run -e <env> <command>` for specific environments
- Only use `eval "$(pixi shell-hook -e <env>)"` for interactive shell

## Testing Commands (from project root)
- `pixi run test` - Run all tests
- `pixi run test-coverage` - Run tests with coverage
- `pixi run test-fast` - Unit tests only, skip slow tests
- `pixi run test-ci` - CI-friendly test with XML output

## Code Quality Commands
- `pixi run ruff-check-mfc` - Check linting for MFC code
- `pixi run ruff-fix-mfc` - Fix linting issues
- `pixi run mypy-check-mfc` - Type checking
- `pixi run validate-mfc` - Full validation (linting + typing + tests)

## TDD Commands (Mandatory)
- `pixi run tdd-red` - Run failing test (pytest -x --tb=short -v)
- `pixi run tdd-green` - Make tests pass (pytest --tb=short)
- `pixi run tdd-refactor` - Refactor (pytest && pixi run validate-all)

## Coverage Reporting
- Tests with coverage: `pixi run test-coverage`
- Coverage threshold: 95% (configured in pyproject.toml)

## Environment Variables
- `DISABLE_AUDIO=true` - Disable audio notifications in CI
- `MOCK_EMAIL=true` - Mock email notifications in CI

## Current Working Directory
/home/uge/mfc-project/q-learning-mfcs