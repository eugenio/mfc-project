"""Example module following TDD methodology
GREEN PHASE: Minimal implementation to make tests pass.
"""


def hello_world() -> str:
    """Return a greeting message.

    Returns:
        str: Hello world message

    """
    return "Hello, World!"


def greet(name: str) -> str:
    """Return a personalized greeting.

    Args:
        name: Name to greet

    Returns:
        str: Personalized greeting message

    """
    if not name:
        return hello_world()
    return f"Hello, {name}!"
