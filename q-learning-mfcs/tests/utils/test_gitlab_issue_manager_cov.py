"""Tests for utils/gitlab_issue_manager.py - targeting 98%+ coverage."""
import importlib.util
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest

# ---------------------------------------------------------------------------
# Mock the 'gitlab' package BEFORE importing the module under test.
# We need proper exception classes that behave like real exceptions.
# ---------------------------------------------------------------------------
_mock_gitlab = MagicMock()


class _FakeGitlabGetError(Exception):
    pass


class _FakeGitlabCreateError(Exception):
    pass


class _FakeGitlabUpdateError(Exception):
    pass


class _FakeGitlabListError(Exception):
    pass


_mock_gitlab.exceptions.GitlabGetError = _FakeGitlabGetError
_mock_gitlab.exceptions.GitlabCreateError = _FakeGitlabCreateError
_mock_gitlab.exceptions.GitlabUpdateError = _FakeGitlabUpdateError
_mock_gitlab.exceptions.GitlabListError = _FakeGitlabListError

# Install mock before import
sys.modules.setdefault("gitlab", _mock_gitlab)
sys.modules.setdefault("gitlab.exceptions", _mock_gitlab.exceptions)

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")

_spec = importlib.util.spec_from_file_location(
    "utils.gitlab_issue_manager",
    os.path.join(_src, "utils", "gitlab_issue_manager.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["utils.gitlab_issue_manager"] = _mod
_spec.loader.exec_module(_mod)

IssueSeverity = _mod.IssueSeverity
IssueUrgency = _mod.IssueUrgency
IssueType = _mod.IssueType
IssueData = _mod.IssueData
GitLabIssueManager = _mod.GitLabIssueManager
TestResultProcessor = _mod.TestResultProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_project():
    """Create a mock gitlab project with issues manager."""
    project = MagicMock()
    return project


def _make_mock_issue(**kwargs):
    """Create a mock issue object with common attributes."""
    issue = MagicMock()
    issue.id = kwargs.get("id", 100)
    issue.iid = kwargs.get("iid", 42)
    issue.title = kwargs.get("title", "Test Issue")
    issue.description = kwargs.get("description", "Test description")
    issue.web_url = kwargs.get("web_url", "https://gitlab.com/test/42")
    issue.state = kwargs.get("state", "opened")
    issue.labels = kwargs.get("labels", ["bug"])
    issue.created_at = kwargs.get("created_at", "2025-01-01T00:00:00Z")
    issue.updated_at = kwargs.get("updated_at", "2025-01-01T00:00:00Z")
    issue.user_notes_count = kwargs.get("user_notes_count", 0)
    issue.author = kwargs.get("author", {"id": 1, "name": "Test", "username": "test"})
    issue.milestone = kwargs.get("milestone", None)
    issue.notes = MagicMock()
    return issue


def _make_manager(project=None):
    """Create a GitLabIssueManager with mocked gitlab connection."""
    if project is None:
        project = _make_mock_project()

    old_val = _mod.GITLAB_AVAILABLE
    _mod.GITLAB_AVAILABLE = True

    mock_gl = MagicMock()
    mock_gl.projects.get.return_value = project

    with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
        with patch.dict(os.environ, {
            "GITLAB_TOKEN": "fake-token",
            "GITLAB_PROJECT_ID": "123",
        }):
            mgr = GitLabIssueManager()

    _mod.GITLAB_AVAILABLE = old_val
    return mgr


# ===========================================================================
# Enum tests
# ===========================================================================
class TestIssueSeverity:
    def test_values(self):
        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.HIGH.value == "high"
        assert IssueSeverity.MEDIUM.value == "medium"
        assert IssueSeverity.LOW.value == "low"


class TestIssueUrgency:
    def test_values(self):
        assert IssueUrgency.URGENT.value == "urgent"
        assert IssueUrgency.HIGH.value == "high"
        assert IssueUrgency.MEDIUM.value == "medium"
        assert IssueUrgency.LOW.value == "low"


class TestIssueType:
    def test_values(self):
        assert IssueType.BUG.value == "bug"
        assert IssueType.ENHANCEMENT.value == "enhancement"
        assert IssueType.PERFORMANCE.value == "performance"
        assert IssueType.SECURITY.value == "security"
        assert IssueType.DOCUMENTATION.value == "documentation"
        assert IssueType.TEST.value == "test"


# ===========================================================================
# IssueData tests
# ===========================================================================
class TestIssueData:
    def test_defaults(self):
        d = IssueData(
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
        )
        assert d.labels == []
        assert d.component is None
        assert d.test_case is None
        assert d.error_message is None
        assert d.stack_trace is None

    def test_post_init_none_labels(self):
        d = IssueData(
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
            labels=None,
        )
        assert d.labels == []

    def test_custom_labels_preserved(self):
        d = IssueData(
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
            labels=["custom"],
        )
        assert d.labels == ["custom"]

    def test_all_optional_fields(self):
        d = IssueData(
            title="t",
            description="d",
            severity=IssueSeverity.HIGH,
            urgency=IssueUrgency.HIGH,
            issue_type=IssueType.ENHANCEMENT,
            labels=["a", "b"],
            component="comp",
            test_case="test_foo",
            error_message="oops",
            stack_trace="traceback here",
        )
        assert d.component == "comp"
        assert d.test_case == "test_foo"
        assert d.error_message == "oops"
        assert d.stack_trace == "traceback here"


# ===========================================================================
# GitLabIssueManager.__init__ tests
# ===========================================================================
class TestGitLabIssueManagerInit:
    def test_gitlab_not_available(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="python-gitlab is required"):
                GitLabIssueManager(project_id="1", token="t")
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_no_token(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        try:
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("GITLAB_TOKEN", None)
                os.environ.pop("GITLAB_PROJECT_ID", None)
                os.environ.pop("GITLAB_URL", None)
                with pytest.raises(ValueError, match="GitLab token is required"):
                    GitLabIssueManager()
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_no_project_id(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        try:
            with patch.dict(os.environ, {"GITLAB_TOKEN": "tok"}, clear=True):
                with pytest.raises(ValueError, match="GitLab project ID is required"):
                    GitLabIssueManager()
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_project_get_error(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.side_effect = _FakeGitlabGetError("not found")
        try:
            with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
                with patch.dict(os.environ, {
                    "GITLAB_TOKEN": "tok",
                    "GITLAB_PROJECT_ID": "999",
                }):
                    with pytest.raises(ValueError, match="Cannot access GitLab project"):
                        GitLabIssueManager()
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_success_with_env(self):
        mgr = _make_manager()
        assert mgr.token == "fake-token"
        assert mgr.project_id == "123"

    def test_success_with_args(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.return_value = _make_mock_project()
        try:
            with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
                mgr = GitLabIssueManager(project_id="456", token="my-token")
                assert mgr.project_id == "456"
                assert mgr.token == "my-token"
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_gitlab_url_from_env(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.return_value = _make_mock_project()
        try:
            with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
                with patch.dict(os.environ, {
                    "GITLAB_TOKEN": "tok",
                    "GITLAB_PROJECT_ID": "1",
                    "GITLAB_URL": "https://mygitlab.example.com",
                }):
                    mgr = GitLabIssueManager()
                    assert mgr.gitlab_url == "https://mygitlab.example.com"
        finally:
            _mod.GITLAB_AVAILABLE = old


# ===========================================================================
# GitLabIssueManager method tests
# ===========================================================================
class TestGitLabIssueManagerCreateIssue:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_create_issue_success(self):
        mock_issue = _make_mock_issue()
        self.project.issues.create.return_value = mock_issue

        data = IssueData(
            title="Test",
            description="desc",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
        )
        result = self.mgr.create_issue(data)
        assert result["id"] == 100
        assert result["iid"] == 42
        assert result["title"] == "Test Issue"
        assert result["web_url"] == "https://gitlab.com/test/42"
        assert result["state"] == "opened"

    def test_create_issue_gitlab_create_error(self):
        self.project.issues.create.side_effect = _FakeGitlabCreateError("fail")

        data = IssueData(
            title="Test",
            description="desc",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
        )
        with pytest.raises(_FakeGitlabCreateError):
            self.mgr.create_issue(data)


class TestGitLabIssueManagerCreateBugIssue:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)
        self.mock_issue = _make_mock_issue(title="Bug: test bug")
        self.project.issues.create.return_value = self.mock_issue

    def test_basic(self):
        result = self.mgr.create_bug_issue("test bug", "A bug")
        assert result["title"] == "Bug: test bug"

    def test_with_steps_to_reproduce(self):
        result = self.mgr.create_bug_issue(
            "bug", "desc",
            steps_to_reproduce="1. Do X\n2. Do Y",
        )
        call_args = self.project.issues.create.call_args[0][0]
        assert "Steps to Reproduce" in call_args["description"]

    def test_with_expected_behavior(self):
        result = self.mgr.create_bug_issue(
            "bug", "desc",
            expected_behavior="Should work",
        )
        call_args = self.project.issues.create.call_args[0][0]
        assert "Expected Behavior" in call_args["description"]

    def test_with_environment(self):
        result = self.mgr.create_bug_issue(
            "bug", "desc",
            environment="Linux Ubuntu 22.04",
        )
        call_args = self.project.issues.create.call_args[0][0]
        assert "Environment" in call_args["description"]

    def test_all_optional_params(self):
        result = self.mgr.create_bug_issue(
            "bug", "desc",
            steps_to_reproduce="steps",
            expected_behavior="expected",
            environment="env",
            severity=IssueSeverity.CRITICAL,
            urgency=IssueUrgency.URGENT,
        )
        assert result is not None


class TestGitLabIssueManagerCreateEnhancementIssue:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)
        self.mock_issue = _make_mock_issue(title="Enhancement: new feat")
        self.project.issues.create.return_value = self.mock_issue

    def test_basic(self):
        result = self.mgr.create_enhancement_issue("new feat", "desc")
        assert result["title"] == "Enhancement: new feat"

    def test_with_todo_list(self):
        result = self.mgr.create_enhancement_issue(
            "feat", "desc",
            todo_list=["Task 1", "Task 2"],
        )
        call_args = self.project.issues.create.call_args[0][0]
        assert "Implementation Tasks" in call_args["description"]
        assert "- [ ] Task 1" in call_args["description"]
        assert "- [ ] Task 2" in call_args["description"]

    def test_priority_high(self):
        result = self.mgr.create_enhancement_issue("feat", "desc", priority="high")
        call_args = self.project.issues.create.call_args[0][0]
        assert "priority::high" in call_args["labels"]

    def test_priority_low(self):
        result = self.mgr.create_enhancement_issue("feat", "desc", priority="low")
        call_args = self.project.issues.create.call_args[0][0]
        assert "priority::low" in call_args["labels"]

    def test_priority_medium(self):
        result = self.mgr.create_enhancement_issue("feat", "desc", priority="medium")
        call_args = self.project.issues.create.call_args[0][0]
        assert "priority::high" not in call_args["labels"]
        assert "priority::low" not in call_args["labels"]

    def test_priority_unknown_defaults_medium(self):
        result = self.mgr.create_enhancement_issue("feat", "desc", priority="unknown")
        assert result is not None


class TestGitLabIssueManagerUpdateIssue:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_update_success(self):
        mock_issue = _make_mock_issue()
        self.project.issues.get.return_value = mock_issue

        result = self.mgr.update_issue(42, {"title": "Updated"})
        assert result["iid"] == 42
        mock_issue.save.assert_called_once()

    def test_update_get_error(self):
        self.project.issues.get.side_effect = _FakeGitlabGetError("not found")
        with pytest.raises(_FakeGitlabGetError):
            self.mgr.update_issue(999, {"title": "x"})

    def test_update_save_error(self):
        mock_issue = _make_mock_issue()
        mock_issue.save.side_effect = _FakeGitlabUpdateError("update fail")
        self.project.issues.get.return_value = mock_issue
        with pytest.raises(_FakeGitlabUpdateError):
            self.mgr.update_issue(42, {"title": "x"})


class TestGitLabIssueManagerAddComment:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_add_comment_success(self):
        mock_issue = _make_mock_issue()
        self.project.issues.get.return_value = mock_issue
        result = self.mgr.add_comment(42, "A comment")
        assert result is True
        mock_issue.notes.create.assert_called_once_with({"body": "A comment"})

    def test_add_comment_failure(self):
        self.project.issues.get.side_effect = Exception("fail")
        result = self.mgr.add_comment(42, "comment")
        assert result is False


class TestGitLabIssueManagerCloseIssue:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_close_without_comment(self):
        mock_issue = _make_mock_issue()
        self.project.issues.get.return_value = mock_issue
        result = self.mgr.close_issue(42)
        assert result["iid"] == 42
        mock_issue.save.assert_called_once()
        mock_issue.notes.create.assert_not_called()

    def test_close_with_comment(self):
        mock_issue = _make_mock_issue()
        self.project.issues.get.return_value = mock_issue
        result = self.mgr.close_issue(42, comment="Closing this issue")
        mock_issue.notes.create.assert_called_once_with(
            {"body": "Closing this issue"}
        )
        mock_issue.save.assert_called_once()

    def test_close_get_error(self):
        self.project.issues.get.side_effect = _FakeGitlabGetError("not found")
        with pytest.raises(_FakeGitlabGetError):
            self.mgr.close_issue(999)


class TestGitLabIssueManagerSearchIssues:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_search_returns_results(self):
        i1 = _make_mock_issue(iid=1, title="First")
        i2 = _make_mock_issue(iid=2, title="Second")
        self.project.issues.list.return_value = [i1, i2]

        results = self.mgr.search_issues(["test"])
        assert len(results) == 2
        assert results[0]["iid"] == 1
        assert results[1]["iid"] == 2

    def test_search_empty(self):
        self.project.issues.list.return_value = []
        results = self.mgr.search_issues(["nothing"])
        assert results == []

    def test_search_list_error(self):
        self.project.issues.list.side_effect = _FakeGitlabListError("fail")
        results = self.mgr.search_issues(["test"])
        assert results == []


class TestGitLabIssueManagerGetIssueByTitle:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_found(self):
        i1 = _make_mock_issue(iid=1, title="Match Title")
        self.project.issues.list.return_value = [i1]
        result = self.mgr.get_issue_by_title("Match Title")
        assert result is not None
        assert result["title"] == "Match Title"

    def test_not_found(self):
        i1 = _make_mock_issue(iid=1, title="Other")
        self.project.issues.list.return_value = [i1]
        result = self.mgr.get_issue_by_title("Match Title")
        assert result is None

    def test_empty_list(self):
        self.project.issues.list.return_value = []
        result = self.mgr.get_issue_by_title("Anything")
        assert result is None


class TestGitLabIssueManagerGetIssueDetails:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_full_details_with_assignees(self):
        mock_issue = _make_mock_issue()
        mock_issue.assignees = [
            {"id": 10, "name": "Alice", "username": "alice"},
        ]
        note = MagicMock()
        note.id = 1
        note.author = {"name": "Author"}
        note.created_at = "2025-01-01"
        note.updated_at = "2025-01-02"
        note.body = "A note"
        note.system = False
        mock_issue.notes.list.return_value = [note]
        mock_issue.milestone = {"id": 5, "title": "v1.0", "description": "milestone"}
        self.project.issues.get.return_value = mock_issue

        result = self.mgr.get_issue_details(42)
        assert result is not None
        assert result["iid"] == 42
        assert len(result["comments"]) == 1
        assert result["comments"][0]["body"] == "A note"
        assert len(result["assignees"]) == 1
        assert result["assignees"][0]["name"] == "Alice"
        assert result["milestone"]["title"] == "v1.0"

    def test_details_with_single_assignee(self):
        mock_issue = _make_mock_issue()
        # No assignees attribute but has assignee
        del mock_issue.assignees
        mock_issue.assignee = {"id": 20, "name": "Bob", "username": "bob"}
        mock_issue.notes.list.return_value = []
        mock_issue.milestone = None
        self.project.issues.get.return_value = mock_issue

        result = self.mgr.get_issue_details(42)
        assert result is not None
        assert len(result["assignees"]) == 1
        assert result["assignees"][0]["name"] == "Bob"

    def test_details_no_assignees(self):
        mock_issue = _make_mock_issue()
        mock_issue.assignees = []
        mock_issue.assignee = None
        mock_issue.notes.list.return_value = []
        mock_issue.milestone = None
        self.project.issues.get.return_value = mock_issue

        result = self.mgr.get_issue_details(42)
        assert result is not None
        assert result["assignees"] == []

    def test_details_note_no_author(self):
        """Cover line 457: note.author is None -> 'System'."""
        mock_issue = _make_mock_issue()
        mock_issue.assignees = []
        mock_issue.milestone = None
        note = MagicMock()
        note.id = 2
        note.author = None
        note.created_at = "2025-01-01"
        note.updated_at = "2025-01-02"
        note.body = "System note"
        note.system = True
        mock_issue.notes.list.return_value = [note]
        self.project.issues.get.return_value = mock_issue

        result = self.mgr.get_issue_details(42)
        assert result["comments"][0]["author"] == "System"

    def test_details_author_none(self):
        """Cover lines 495-498: issue.author is None."""
        mock_issue = _make_mock_issue()
        mock_issue.author = None
        mock_issue.assignees = []
        mock_issue.milestone = None
        mock_issue.notes.list.return_value = []
        self.project.issues.get.return_value = mock_issue

        result = self.mgr.get_issue_details(42)
        assert result["author"]["id"] is None
        assert result["author"]["name"] == "Unknown"
        assert result["author"]["username"] == "unknown"

    def test_details_gitlab_get_error(self):
        self.project.issues.get.side_effect = _FakeGitlabGetError("not found")
        result = self.mgr.get_issue_details(999)
        assert result is None

    def test_details_generic_exception(self):
        self.project.issues.get.side_effect = RuntimeError("unexpected")
        result = self.mgr.get_issue_details(42)
        assert result is None

    def test_details_no_milestone(self):
        mock_issue = _make_mock_issue()
        mock_issue.assignees = []
        mock_issue.milestone = None
        mock_issue.notes.list.return_value = []
        self.project.issues.get.return_value = mock_issue

        result = self.mgr.get_issue_details(42)
        assert result["milestone"] is None


class TestGitLabIssueManagerPrepareLabels:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_basic_labels(self):
        data = IssueData(
            title="t",
            description="d",
            severity=IssueSeverity.HIGH,
            urgency=IssueUrgency.HIGH,
            issue_type=IssueType.BUG,
        )
        labels = self.mgr._prepare_labels(data)
        assert "bug" in labels
        assert "severity::high" in labels
        assert "urgency::high" in labels

    def test_with_component(self):
        data = IssueData(
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.ENHANCEMENT,
            component="sensor",
        )
        labels = self.mgr._prepare_labels(data)
        assert "component::sensor" in labels

    def test_with_test_case(self):
        data = IssueData(
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
            test_case="test_foo",
        )
        labels = self.mgr._prepare_labels(data)
        assert "test-failure" in labels

    def test_with_custom_labels(self):
        data = IssueData(
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
            labels=["custom1", "custom2"],
        )
        labels = self.mgr._prepare_labels(data)
        assert "custom1" in labels
        assert "custom2" in labels


class TestGitLabIssueManagerFormatDescription:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)

    def test_basic_format(self):
        data = IssueData(
            title="t",
            description="Main desc",
            severity=IssueSeverity.MEDIUM,
            urgency=IssueUrgency.MEDIUM,
            issue_type=IssueType.BUG,
        )
        desc = self.mgr._format_description(data)
        assert "Main desc" in desc
        assert "Issue Metadata" in desc
        assert "Type" in desc
        assert "Severity" in desc
        assert "Urgency" in desc
        assert "automatically created" in desc

    def test_with_component(self):
        data = IssueData(
            title="t",
            description="desc",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
            component="sensor",
        )
        desc = self.mgr._format_description(data)
        assert "Component" in desc
        assert "sensor" in desc

    def test_with_test_case(self):
        data = IssueData(
            title="t",
            description="desc",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
            test_case="test_something",
        )
        desc = self.mgr._format_description(data)
        assert "Test Case" in desc
        assert "test_something" in desc

    def test_with_error_message(self):
        data = IssueData(
            title="t",
            description="desc",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
            error_message="RuntimeError: boom",
        )
        desc = self.mgr._format_description(data)
        assert "Error Details" in desc
        assert "RuntimeError: boom" in desc

    def test_with_stack_trace(self):
        data = IssueData(
            title="t",
            description="desc",
            severity=IssueSeverity.LOW,
            urgency=IssueUrgency.LOW,
            issue_type=IssueType.BUG,
            stack_trace="File \"test.py\", line 1\n  raise ValueError",
        )
        desc = self.mgr._format_description(data)
        assert "Stack Trace" in desc
        assert "raise ValueError" in desc


# ===========================================================================
# TestResultProcessor tests
# ===========================================================================
class TestTestResultProcessorInit:
    def test_init(self):
        mgr = _make_manager()
        proc = TestResultProcessor(mgr)
        assert proc.issue_manager is mgr


class TestTestResultProcessorProcessFailures:
    def setup_method(self):
        self.project = _make_mock_project()
        self.mgr = _make_manager(self.project)
        self.proc = TestResultProcessor(self.mgr)

    def test_empty_results(self):
        result = self.proc.process_test_failures({})
        assert result == []

    def test_failures_creates_issues(self):
        mock_issue = _make_mock_issue()
        self.project.issues.create.return_value = mock_issue
        self.project.issues.list.return_value = []

        results = self.proc.process_test_failures({
            "failures": [("test_foo", "AssertionError: 1 != 2")],
            "errors": [],
        })
        assert len(results) == 1

    def test_failures_existing_issue_skipped(self):
        existing = _make_mock_issue(title="Test Failure: test_foo")
        self.project.issues.list.return_value = [existing]

        results = self.proc.process_test_failures({
            "failures": [("test_foo", "AssertionError")],
            "errors": [],
        })
        assert len(results) == 0

    def test_errors_creates_issues(self):
        mock_issue = _make_mock_issue()
        self.project.issues.create.return_value = mock_issue
        self.project.issues.list.return_value = []

        results = self.proc.process_test_failures({
            "failures": [],
            "errors": [("test_bar", "RuntimeError: fail")],
        })
        assert len(results) == 1

    def test_errors_existing_issue_skipped(self):
        existing = _make_mock_issue(title="Test Error: test_bar")
        self.project.issues.list.return_value = [existing]

        results = self.proc.process_test_failures({
            "failures": [],
            "errors": [("test_bar", "RuntimeError")],
        })
        assert len(results) == 0

    def test_mixed_failures_and_errors(self):
        mock_issue = _make_mock_issue()
        self.project.issues.create.return_value = mock_issue
        self.project.issues.list.return_value = []

        results = self.proc.process_test_failures({
            "failures": [("test_a", "assertion")],
            "errors": [("test_b", "runtime")],
        })
        assert len(results) == 2


class TestTestResultProcessorCreateIssueFromFailure:
    def setup_method(self):
        self.mgr = _make_manager()
        self.proc = TestResultProcessor(self.mgr)

    def test_basic(self):
        data = self.proc._create_issue_from_failure(
            ("test_sensor_integration", "File line 1\nAssertionError: fail"),
        )
        assert data.title == "Test Failure: test_sensor_integration"
        assert data.issue_type == IssueType.BUG
        assert "automated" in data.labels
        assert "test-failure" in data.labels
        assert data.component == "sensor-fusion"
        assert data.test_case == "test_sensor_integration"
        assert data.error_message is not None
        assert data.stack_trace is not None


class TestTestResultProcessorCreateIssueFromError:
    def setup_method(self):
        self.mgr = _make_manager()
        self.proc = TestResultProcessor(self.mgr)

    def test_basic(self):
        data = self.proc._create_issue_from_error(
            ("test_gpu_runtime", "File line 1\nRuntimeError: boom"),
        )
        assert data.title == "Test Error: test_gpu_runtime"
        assert data.severity == IssueSeverity.HIGH
        assert data.urgency == IssueUrgency.HIGH
        assert data.component == "gpu-acceleration"
        assert "test-error" in data.labels
        assert "runtime-error" in data.labels


class TestTestResultProcessorExtractComponent:
    def setup_method(self):
        self.mgr = _make_manager()
        self.proc = TestResultProcessor(self.mgr)

    def test_gpu(self):
        assert self.proc._extract_component_from_test_name("test_gpu_accel") == "gpu-acceleration"

    def test_biofilm(self):
        assert self.proc._extract_component_from_test_name("test_biofilm_model") == "biofilm-model"

    def test_metabolic(self):
        assert self.proc._extract_component_from_test_name("test_metabolic_rate") == "metabolic-model"

    def test_sensor(self):
        assert self.proc._extract_component_from_test_name("test_sensor_data") == "sensor-fusion"

    def test_qlearning(self):
        assert self.proc._extract_component_from_test_name("test_qlearning_agent") == "q-learning"

    def test_q_learning(self):
        assert self.proc._extract_component_from_test_name("test_q_learning_config") == "q-learning"

    def test_mfc(self):
        assert self.proc._extract_component_from_test_name("test_mfc_stack") == "mfc-stack"

    def test_config(self):
        assert self.proc._extract_component_from_test_name("test_config_loader") == "configuration"

    def test_path(self):
        assert self.proc._extract_component_from_test_name("test_path_management") == "path-management"

    def test_no_match(self):
        assert self.proc._extract_component_from_test_name("test_something_else") is None


class TestTestResultProcessorDetermineSeverity:
    def setup_method(self):
        self.mgr = _make_manager()
        self.proc = TestResultProcessor(self.mgr)

    def test_critical(self):
        assert self.proc._determine_severity_from_test_name("test_critical_path") == IssueSeverity.CRITICAL

    def test_security(self):
        assert self.proc._determine_severity_from_test_name("test_security_check") == IssueSeverity.CRITICAL

    def test_safety(self):
        assert self.proc._determine_severity_from_test_name("test_safety_monitor") == IssueSeverity.CRITICAL

    def test_performance(self):
        assert self.proc._determine_severity_from_test_name("test_performance_bench") == IssueSeverity.HIGH

    def test_stress(self):
        assert self.proc._determine_severity_from_test_name("test_stress_load") == IssueSeverity.HIGH

    def test_memory(self):
        assert self.proc._determine_severity_from_test_name("test_memory_usage") == IssueSeverity.HIGH

    def test_integration(self):
        assert self.proc._determine_severity_from_test_name("test_integration_flow") == IssueSeverity.MEDIUM

    def test_core(self):
        assert self.proc._determine_severity_from_test_name("test_core_function") == IssueSeverity.MEDIUM

    def test_main(self):
        assert self.proc._determine_severity_from_test_name("test_main_loop") == IssueSeverity.MEDIUM

    def test_default_low(self):
        assert self.proc._determine_severity_from_test_name("test_utility_helper") == IssueSeverity.LOW


class TestTestResultProcessorDetermineUrgency:
    def setup_method(self):
        self.mgr = _make_manager()
        self.proc = TestResultProcessor(self.mgr)

    def test_critical_to_urgent(self):
        assert self.proc._determine_urgency_from_severity(IssueSeverity.CRITICAL) == IssueUrgency.URGENT

    def test_high_to_high(self):
        assert self.proc._determine_urgency_from_severity(IssueSeverity.HIGH) == IssueUrgency.HIGH

    def test_medium_to_medium(self):
        assert self.proc._determine_urgency_from_severity(IssueSeverity.MEDIUM) == IssueUrgency.MEDIUM

    def test_low_to_low(self):
        assert self.proc._determine_urgency_from_severity(IssueSeverity.LOW) == IssueUrgency.LOW


class TestTestResultProcessorExtractErrorMessage:
    def setup_method(self):
        self.mgr = _make_manager()
        self.proc = TestResultProcessor(self.mgr)

    def test_extract_last_meaningful_line(self):
        tb = "File \"test.py\", line 10\n  x = 1 / 0\nZeroDivisionError: division by zero"
        msg = self.proc._extract_error_message(tb)
        assert msg == "ZeroDivisionError: division by zero"

    def test_all_file_lines(self):
        # After stripping, "x = 1" is a valid non-File line, so it returns it
        tb = "File \"a.py\", line 1\n  x = 1"
        msg = self.proc._extract_error_message(tb)
        assert msg == "x = 1"

    def test_only_file_and_indented_lines(self):
        # All lines either start with "File " or "  " after strip -> None
        tb = "File \"a.py\", line 1"
        msg = self.proc._extract_error_message(tb)
        assert msg is None

    def test_empty_traceback(self):
        msg = self.proc._extract_error_message("")
        assert msg is None

    def test_multiline_with_indented(self):
        tb = "File \"a.py\", line 1\n  indented\nAssertionError: bad"
        msg = self.proc._extract_error_message(tb)
        assert msg == "AssertionError: bad"


# ===========================================================================
# Convenience function tests
# ===========================================================================
class TestConvenienceCreateBugIssue:
    def test_success(self):
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue(iid=55)
        mock_project.issues.create.return_value = mock_issue

        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.return_value = mock_project
        try:
            with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
                with patch.dict(os.environ, {
                    "GITLAB_TOKEN": "tok",
                    "GITLAB_PROJECT_ID": "1",
                }):
                    result = _mod.create_bug_issue("bug title", "desc")
                    assert result == 55
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_failure_returns_none(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = False
        try:
            result = _mod.create_bug_issue("bug", "desc")
            assert result is None
        finally:
            _mod.GITLAB_AVAILABLE = old


class TestConvenienceCreateEnhancementIssue:
    def test_success(self):
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue(iid=66)
        mock_project.issues.create.return_value = mock_issue

        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.return_value = mock_project
        try:
            with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
                with patch.dict(os.environ, {
                    "GITLAB_TOKEN": "tok",
                    "GITLAB_PROJECT_ID": "1",
                }):
                    result = _mod.create_enhancement_issue(
                        "feat", "desc",
                        todo_list=["a"],
                        priority="high",
                    )
                    assert result == 66
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_failure_returns_none(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = False
        try:
            result = _mod.create_enhancement_issue("feat", "desc")
            assert result is None
        finally:
            _mod.GITLAB_AVAILABLE = old


class TestConvenienceUpdateIssue:
    def test_success_no_close(self):
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue()
        mock_project.issues.get.return_value = mock_issue

        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.return_value = mock_project
        try:
            with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
                with patch.dict(os.environ, {
                    "GITLAB_TOKEN": "tok",
                    "GITLAB_PROJECT_ID": "1",
                }):
                    result = _mod.update_issue(42, "comment")
                    assert result is True
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_success_with_close(self):
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue()
        mock_project.issues.get.return_value = mock_issue

        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.return_value = mock_project
        try:
            with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
                with patch.dict(os.environ, {
                    "GITLAB_TOKEN": "tok",
                    "GITLAB_PROJECT_ID": "1",
                }):
                    result = _mod.update_issue(42, "closing", close=True)
                    assert result is True
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_failure_returns_false(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = False
        try:
            result = _mod.update_issue(42, "comment")
            assert result is False
        finally:
            _mod.GITLAB_AVAILABLE = old


# ===========================================================================
# main() tests
# ===========================================================================
class TestMain:
    def _make_manager_patch(self, mock_project):
        """Return a context manager that patches GitLabIssueManager init."""
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.return_value = mock_project
        return patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl)

    def test_gitlab_not_available(self):
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = False
        try:
            with patch("sys.argv", ["prog"]):
                with pytest.raises(SystemExit) as exc_info:
                    _mod.main()
                assert exc_info.value.code == 1
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_test_results(self, tmp_path):
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue(iid=10, title="Test Failure: test_x")
        mock_project.issues.create.return_value = mock_issue
        mock_project.issues.list.return_value = []

        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps({
            "failures": [["test_x", "AssertionError"]],
            "errors": [],
        }))

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog", "--test-results", str(results_file)]):
                    _mod.main()

    def test_list_issues(self, capsys):
        mock_project = _make_mock_project()
        i1 = _make_mock_issue(iid=1, title="Issue 1", labels=["bug", "high", "extra"])
        mock_project.issues.list.return_value = [i1]

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog", "--list-issues"]):
                    _mod.main()

        captured = capsys.readouterr()
        assert "1 open issues" in captured.out

    def test_close_issue(self, capsys):
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue(iid=5, title="Closed Issue")
        mock_project.issues.get.return_value = mock_issue

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog", "--close-issue", "5"]):
                    _mod.main()

        captured = capsys.readouterr()
        assert "Closed issue #5" in captured.out

    def test_get_issue_found(self, capsys):
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue(
            iid=7,
            title="Details Issue",
            description="Full description here",
            labels=["bug", "low"],
        )
        mock_issue.assignees = []
        mock_issue.milestone = None
        mock_issue.notes.list.return_value = []
        mock_project.issues.get.return_value = mock_issue

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog", "--get-issue", "7"]):
                    _mod.main()

        captured = capsys.readouterr()
        assert "Issue #7" in captured.out
        assert "Details Issue" in captured.out

    def test_get_issue_not_found(self, capsys):
        mock_project = _make_mock_project()
        mock_project.issues.get.side_effect = _FakeGitlabGetError("nope")

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog", "--get-issue", "999"]):
                    _mod.main()

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_create_issue_interactive(self, capsys):
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue(iid=88)
        mock_project.issues.create.return_value = mock_issue

        inputs = iter(["My Bug Title", "Line 1 of desc", EOFError, "bug", "medium"])

        def fake_input(prompt=""):
            val = next(inputs)
            if isinstance(val, type) and issubclass(val, BaseException):
                raise val()
            return val

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog", "--create-issue"]):
                    with patch("builtins.input", side_effect=fake_input):
                        _mod.main()

        captured = capsys.readouterr()
        assert "Created issue #88" in captured.out

    def test_create_issue_empty_title(self):
        mock_project = _make_mock_project()

        def fake_input(prompt=""):
            return ""

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog", "--create-issue"]):
                    with patch("builtins.input", side_effect=fake_input):
                        with pytest.raises(SystemExit) as exc_info:
                            _mod.main()
                        assert exc_info.value.code == 1

    def test_no_args_shows_help(self, capsys):
        mock_project = _make_mock_project()

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog"]):
                    _mod.main()

        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "GitLab" in captured.out

    def test_value_error_in_main(self, capsys):
        """Cover line 987-989: ValueError caught in main."""
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        try:
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("GITLAB_TOKEN", None)
                os.environ.pop("GITLAB_PROJECT_ID", None)
                with patch("sys.argv", ["prog", "--list-issues"]):
                    with pytest.raises(SystemExit) as exc_info:
                        _mod.main()
                    assert exc_info.value.code == 1
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_generic_error_in_main(self, capsys):
        """Cover lines 990-992: generic Exception caught in main."""
        old = _mod.GITLAB_AVAILABLE
        _mod.GITLAB_AVAILABLE = True
        mock_gl = MagicMock()
        mock_gl.projects.get.side_effect = RuntimeError("unexpected")
        try:
            with patch.object(_mock_gitlab, "Gitlab", return_value=mock_gl):
                with patch.dict(os.environ, {
                    "GITLAB_TOKEN": "tok",
                    "GITLAB_PROJECT_ID": "1",
                }):
                    with patch("sys.argv", ["prog", "--list-issues"]):
                        with pytest.raises(SystemExit) as exc_info:
                            _mod.main()
                        assert exc_info.value.code == 1
        finally:
            _mod.GITLAB_AVAILABLE = old

    def test_create_issue_enhancement_type(self, capsys):
        """Cover line 978: IssueType.ENHANCEMENT branch."""
        mock_project = _make_mock_project()
        mock_issue = _make_mock_issue(iid=99)
        mock_project.issues.create.return_value = mock_issue

        inputs = iter([
            "My Enhancement",
            "Some description",
            EOFError,
            "enhancement",
            "high",
        ])

        def fake_input(prompt=""):
            val = next(inputs)
            if isinstance(val, type) and issubclass(val, BaseException):
                raise val()
            return val

        with self._make_manager_patch(mock_project):
            with patch.dict(os.environ, {
                "GITLAB_TOKEN": "tok",
                "GITLAB_PROJECT_ID": "1",
            }):
                with patch("sys.argv", ["prog", "--create-issue"]):
                    with patch("builtins.input", side_effect=fake_input):
                        _mod.main()

        captured = capsys.readouterr()
        assert "Created issue #99" in captured.out
