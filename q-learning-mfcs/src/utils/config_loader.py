#!/usr/bin/env python3
"""
Configuration loader for GitLab integration

Loads configuration from .env file and environment variables.
"""

import os
from pathlib import Path


def load_env_file(env_path: str | None = None) -> dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, searches for .env in project root.

    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        # Look for .env file in project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent  # Go up to project root
        env_path = project_root / '.env'

    env_vars = {}

    if not Path(env_path).exists():
        print(f"⚠️  .env file not found at: {env_path}")
        return env_vars

    try:
        with open(env_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    env_vars[key] = value
                else:
                    print(f"⚠️  Invalid line {line_num} in .env: {line}")

    except Exception as e:
        print(f"❌ Error reading .env file: {e}")

    return env_vars

def setup_gitlab_config() -> bool:
    """
    Setup GitLab configuration from .env file.

    Returns:
        True if configuration was loaded successfully
    """
    # Load from .env file (but don't expose sensitive values)
    env_vars = load_env_file()

    # Set environment variables for GitLab integration
    gitlab_vars = ['GITLAB_TOKEN', 'GITLAB_PROJECT_ID', 'GITLAB_URL']

    config_loaded = False
    for var in gitlab_vars:
        if var in env_vars:
            os.environ[var] = env_vars[var]
            config_loaded = True
            # Don't print sensitive tokens
            if 'TOKEN' in var:
                print(f"✅ Loaded {var}: [HIDDEN]")
            else:
                print(f"✅ Loaded {var}: {env_vars[var]}")
        else:
            print(f"⚠️  {var} not found in .env file")

    return config_loaded

def get_gitlab_config() -> dict[str, str | None]:
    """
    Get current GitLab configuration.

    Returns:
        Dictionary with GitLab configuration
    """
    return {
        'token': os.getenv('GITLAB_TOKEN'),
        'project_id': os.getenv('GITLAB_PROJECT_ID'),
        'url': os.getenv('GITLAB_URL', 'https://gitlab.com')
    }

def validate_gitlab_config() -> bool:
    """
    Validate GitLab configuration.

    Returns:
        True if configuration is valid
    """
    config = get_gitlab_config()

    required_fields = ['token', 'project_id']
    missing_fields = [field for field in required_fields if not config[field]]

    if missing_fields:
        print(f"❌ Missing required GitLab configuration: {', '.join(missing_fields)}")
        return False

    print("✅ GitLab configuration is valid")
    return True

if __name__ == "__main__":
    print("🔧 Testing Configuration Loader")
    print("-" * 40)

    # Test loading .env file
    print("Loading .env file...")
    setup_gitlab_config()

    print("\nValidating configuration...")
    valid = validate_gitlab_config()

    print(f"\nConfiguration {'✅ Valid' if valid else '❌ Invalid'}")
