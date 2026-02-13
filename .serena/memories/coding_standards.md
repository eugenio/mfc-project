# Coding Standards and Conventions

## Python Standards
- Follow PEP 8 style guidelines
- Use type hints throughout the codebase
- Add comprehensive docstrings to all public functions
- Maximum line length: 88 characters (Black formatter)

## Testing Standards
- Write tests in pytest format, not simple Python scripts
- Use proper mocking for external dependencies
- Target 100% test coverage for assigned modules
- Test naming: `test_<functionality>.py`
- Use descriptive test method names: `test_should_<expected_behavior>_when_<condition>`

## Error Handling
- Use specific exception types
- Include meaningful error messages
- Log errors appropriately

## Documentation
- Use Google-style docstrings
- Include type annotations for all function parameters and return values
- Document complex algorithms and business logic

## Git Commit Guidelines
- No email addresses in commits
- Use conventional commit messages
- Each commit should be focused and atomic

## File Organization
- Keep related functionality together
- Use appropriate module structure
- Separate concerns properly

## Performance Considerations
- Use NumPy for numerical operations
- Consider GPU acceleration where appropriate
- Profile performance-critical code