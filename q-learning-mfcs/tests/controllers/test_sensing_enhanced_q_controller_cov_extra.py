"""Extra coverage tests for sensing_enhanced_q_controller.py.

Covers remaining uncovered line:
- Line 441: _choose_best_available_action returns 0 when q_values is empty
  (all available_actions >= len(self.actions))
"""
import sys
import os
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sensing_enhanced_q_controller import SensingEnhancedQLearningController


@pytest.fixture
def ctrl():
    c = SensingEnhancedQLearningController()
    if not hasattr(c, "actions"):
        c.actions = [
            (-10, -5), (-5, 0), (0, 0), (0, 5), (5, 0),
            (5, 5), (10, -5), (10, 0), (10, 5),
        ]
    return c


@pytest.mark.coverage_extra
class TestChooseBestAvailableActionEmpty:
    """Cover line 441: return 0 when no valid q_values."""

    def test_returns_zero_when_all_actions_out_of_range(self, ctrl):
        """When all available_actions are >= len(self.actions), q_values list
        is empty, so the method returns 0."""
        state = (0, 0, 0)
        # All action indices exceed the length of ctrl.actions
        available_actions = [100, 200, 300]
        result = ctrl._choose_best_available_action(state, available_actions)
        assert result == 0

    def test_returns_zero_when_available_actions_empty(self, ctrl):
        """When available_actions is empty, returns 0."""
        state = (0, 0, 0)
        result = ctrl._choose_best_available_action(state, [])
        assert result == 0
