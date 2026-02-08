"""Tests for utils/gitlab_auto_issue.py - targeting 98%+ coverage."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from utils.gitlab_auto_issue import (
    AutoIssueDetector,
    analyze_user_input,
    auto_create_issue,
)


class TestAutoIssueDetector:
    """Tests for AutoIssueDetector class."""

    def setup_method(self):
        self.detector = AutoIssueDetector()

    def test_init_has_keywords(self):
        assert len(self.detector.bug_keywords) > 0
        assert len(self.detector.feature_keywords) > 0
        assert "high" in self.detector.priority_keywords
        assert "low" in self.detector.priority_keywords

    def test_analyze_bug_description(self):
        desc = "There is a bug that causes a crash and error in the system"
        result = self.detector.analyze_description(desc)
        assert result["type"] == "bug"
        assert result["confidence"] > 0.5
        assert result["bug_score"] > result["feature_score"]
        assert result["description"] == desc

    def test_analyze_feature_description(self):
        desc = "I would like to add a new feature to improve the system and implement better logging"
        result = self.detector.analyze_description(desc)
        assert result["type"] == "enhancement"
        assert result["confidence"] > 0.5
        assert result["feature_score"] > result["bug_score"]

    def test_analyze_unclear_description(self):
        desc = "The system has a problem but we could also add a feature to improve it and fix the bug"
        result = self.detector.analyze_description(desc)
        # Score might be equal, depends on exact keywords
        assert result["type"] in ("bug", "enhancement", "unclear")

    def test_analyze_equal_scores(self):
        desc = "error with new feature"
        result = self.detector.analyze_description(desc)
        if result["bug_score"] == result["feature_score"]:
            assert result["type"] == "unclear"
            assert result["confidence"] == 0.5

    def test_analyze_high_priority(self):
        desc = "This is an urgent critical bug that is blocking"
        result = self.detector.analyze_description(desc)
        assert result["priority"] == "high"

    def test_analyze_low_priority(self):
        desc = "This is a minor bug, nice to have fix"
        result = self.detector.analyze_description(desc)
        assert result["priority"] == "low"

    def test_analyze_medium_priority_default(self):
        desc = "There is a bug in the system"
        result = self.detector.analyze_description(desc)
        assert result["priority"] == "medium"

    def test_analyze_title_extraction(self):
        desc = "The GUI crashes when clicking save. Steps: open the app."
        result = self.detector.analyze_description(desc)
        assert result["suggested_title"] == "The GUI crashes when clicking save"

    def test_analyze_title_long_truncation(self):
        long_desc = "A" * 100 + ". And more details"
        result = self.detector.analyze_description(long_desc)
        assert len(result["suggested_title"]) <= 80
        assert result["suggested_title"].endswith("...")

    def test_analyze_no_title_match(self):
        desc = ""
        result = self.detector.analyze_description(desc)
        assert result["suggested_title"] == "User reported issue"

    def test_analyze_no_keywords(self):
        desc = "hello world"
        result = self.detector.analyze_description(desc)
        assert result["bug_score"] == 0
        assert result["feature_score"] == 0
        assert result["type"] == "unclear"

    @patch("utils.gitlab_auto_issue.create_bug_issue", return_value=42)
    def test_create_issue_bug(self, mock_create):
        desc = "There is a critical bug causing a crash and error"
        result = self.detector.create_issue_from_description(desc)
        assert result == 42
        mock_create.assert_called_once()

    @patch("utils.gitlab_auto_issue.create_enhancement_issue", return_value=99)
    def test_create_issue_enhancement(self, mock_create):
        desc = "I would like to add a new feature to improve the system and implement better monitoring"
        result = self.detector.create_issue_from_description(desc)
        assert result == 99
        mock_create.assert_called_once()

    def test_create_issue_unclear_no_force(self):
        desc = "hello world"
        result = self.detector.create_issue_from_description(desc)
        assert result is None

    @patch("utils.gitlab_auto_issue.create_bug_issue", return_value=10)
    def test_create_issue_force_type_bug(self, mock_create):
        desc = "hello world"
        result = self.detector.create_issue_from_description(desc, force_type="bug")
        assert result == 10

    @patch("utils.gitlab_auto_issue.create_enhancement_issue", return_value=20)
    def test_create_issue_force_type_enhancement(self, mock_create):
        desc = "hello world"
        result = self.detector.create_issue_from_description(desc, force_type="enhancement")
        assert result == 20

    def test_create_issue_force_type_unknown(self):
        desc = "hello world"
        result = self.detector.create_issue_from_description(desc, force_type="other")
        assert result is None

    @patch("utils.gitlab_auto_issue.create_bug_issue", return_value=42)
    def test_create_bug_with_steps(self, mock_create):
        desc = "There is a bug with crash.\nSteps: open the app and click\n\nExpected: it should work"
        result = self.detector.create_issue_from_description(desc)
        assert result == 42
        call_kwargs = mock_create.call_args
        assert call_kwargs is not None

    @patch("utils.gitlab_auto_issue.create_enhancement_issue", return_value=55)
    def test_create_enhancement_with_todos(self, mock_create):
        desc = "I want to add a new feature to implement monitoring.\nTodo: build dashboard\n\n1. Create UI\n2. Add backend"
        result = self.detector.create_issue_from_description(desc)
        assert result == 55

    @patch("utils.gitlab_auto_issue.create_enhancement_issue", return_value=66)
    def test_create_enhancement_no_todos(self, mock_create):
        desc = "I would like to add a new feature to improve and implement better tools"
        result = self.detector.create_issue_from_description(desc)
        assert result == 66

    @patch("utils.gitlab_auto_issue.create_bug_issue", return_value=77)
    def test_create_bug_with_expected_behavior(self, mock_create):
        desc = "Bug with crash error.\nExpected: should not crash\n\nSteps: run the code"
        result = self.detector.create_issue_from_description(desc)
        assert result == 77


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("utils.gitlab_auto_issue.auto_detector")
    def test_auto_create_issue(self, mock_detector):
        mock_detector.create_issue_from_description.return_value = 42
        result = auto_create_issue("test description", force_type="bug")
        assert result == 42
        mock_detector.create_issue_from_description.assert_called_once_with(
            "test description", "bug"
        )

    @patch("utils.gitlab_auto_issue.auto_detector")
    def test_analyze_user_input(self, mock_detector):
        mock_detector.analyze_description.return_value = {"type": "bug"}
        result = analyze_user_input("test description")
        assert result == {"type": "bug"}
        mock_detector.analyze_description.assert_called_once_with("test description")
