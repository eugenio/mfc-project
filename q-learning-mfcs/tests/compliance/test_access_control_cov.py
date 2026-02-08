"""Tests for access_control.py module - comprehensive coverage."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from compliance.access_control import (
    AccessControl,
    AccessControlError,
    AuthenticationError,
    AuthorizationError,
    Permission,
    Role,
    Session,
    User,
    main,
)


class TestPermission:
    """Tests for the Permission dataclass."""

    def test_create_permission(self):
        perm = Permission("read_data", "mfc_system", "read", "Read MFC data")
        assert perm.name == "read_data"
        assert perm.resource == "mfc_system"
        assert perm.action == "read"
        assert perm.description == "Read MFC data"
        assert perm.permission_id is not None
        assert isinstance(perm.created_at, datetime)

    def test_permission_empty_name_raises(self):
        with pytest.raises(ValueError, match="Permission name cannot be empty"):
            Permission("", "resource", "read")

    def test_permission_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="Permission name cannot be empty"):
            Permission("   ", "resource", "read")

    def test_permission_empty_resource_raises(self):
        with pytest.raises(ValueError, match="Permission resource cannot be empty"):
            Permission("name", "", "read")

    def test_permission_whitespace_resource_raises(self):
        with pytest.raises(ValueError, match="Permission resource cannot be empty"):
            Permission("name", "   ", "read")

    def test_permission_empty_action_raises(self):
        with pytest.raises(ValueError, match="Permission action cannot be empty"):
            Permission("name", "resource", "")

    def test_permission_whitespace_action_raises(self):
        with pytest.raises(ValueError, match="Permission action cannot be empty"):
            Permission("name", "resource", "   ")

    def test_permission_to_dict(self):
        perm = Permission("read_data", "mfc_system", "read", "desc")
        d = perm.to_dict()
        assert d["name"] == "read_data"
        assert d["resource"] == "mfc_system"
        assert d["action"] == "read"
        assert d["description"] == "desc"
        assert "permission_id" in d
        assert "created_at" in d

    def test_permission_from_dict(self):
        data = {
            "name": "write_data",
            "resource": "db",
            "action": "write",
            "description": "Write data",
            "permission_id": "test-id-123",
            "created_at": "2025-01-01T00:00:00",
        }
        perm = Permission.from_dict(data)
        assert perm.name == "write_data"
        assert perm.resource == "db"
        assert perm.action == "write"
        assert perm.permission_id == "test-id-123"
        assert perm.created_at == datetime.fromisoformat("2025-01-01T00:00:00")

    def test_permission_from_dict_no_optional(self):
        data = {"name": "read", "resource": "sys", "action": "read"}
        perm = Permission.from_dict(data)
        assert perm.name == "read"
        assert perm.description == ""


class TestRole:
    """Tests for the Role dataclass."""

    def test_create_role(self):
        role = Role("admin", "Administrator role")
        assert role.name == "admin"
        assert role.description == "Administrator role"
        assert role.permissions == []
        assert role.parent_roles == []

    def test_role_empty_name_raises(self):
        with pytest.raises(ValueError, match="Role name cannot be empty"):
            Role("")

    def test_role_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="Role name cannot be empty"):
            Role("   ")

    def test_add_permission(self):
        role = Role("admin")
        perm = Permission("read", "sys", "read")
        role.add_permission(perm)
        assert perm in role.permissions

    def test_add_permission_duplicate(self):
        role = Role("admin")
        perm = Permission("read", "sys", "read")
        role.add_permission(perm)
        role.add_permission(perm)
        assert len(role.permissions) == 1

    def test_remove_permission(self):
        role = Role("admin")
        perm = Permission("read", "sys", "read")
        role.add_permission(perm)
        role.remove_permission(perm)
        assert perm not in role.permissions

    def test_remove_permission_not_present(self):
        role = Role("admin")
        perm = Permission("read", "sys", "read")
        role.remove_permission(perm)
        assert len(role.permissions) == 0

    def test_has_permission(self):
        perm = Permission("read_data", "sys", "read")
        role = Role("reader", permissions=[perm])
        assert role.has_permission("read_data") is True
        assert role.has_permission("write_data") is False

    def test_has_permission_from_parent(self):
        perm = Permission("admin_access", "sys", "admin")
        parent_role = Role("super_admin", permissions=[perm])
        child_role = Role("admin", parent_roles=[parent_role])
        assert child_role.has_permission("admin_access") is True

    def test_has_permission_for_resource(self):
        perm = Permission("read_data", "mfc", "read")
        role = Role("reader", permissions=[perm])
        assert role.has_permission_for_resource("mfc", "read") is True
        assert role.has_permission_for_resource("mfc", "write") is False
        assert role.has_permission_for_resource("db", "read") is False

    def test_has_permission_for_resource_from_parent(self):
        perm = Permission("admin_perm", "mfc", "admin")
        parent = Role("super", permissions=[perm])
        child = Role("admin", parent_roles=[parent])
        assert child.has_permission_for_resource("mfc", "admin") is True
        assert child.has_permission_for_resource("mfc", "read") is False


class TestUser:
    """Tests for the User dataclass."""

    def test_create_user(self):
        user = User("testuser", "test@example.com", "password123")
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.password == ""
        assert user.password_hash != ""
        assert user.is_active is True
        assert user.last_login is None

    def test_user_empty_username_raises(self):
        with pytest.raises(ValueError, match="Username cannot be empty"):
            User("", "test@example.com", "pass")

    def test_user_whitespace_username_raises(self):
        with pytest.raises(ValueError, match="Username cannot be empty"):
            User("   ", "test@example.com", "pass")

    def test_user_empty_email_raises(self):
        with pytest.raises(ValueError, match="Email cannot be empty"):
            User("user", "", "pass")

    def test_user_whitespace_email_raises(self):
        with pytest.raises(ValueError, match="Email cannot be empty"):
            User("user", "   ", "pass")

    def test_verify_password_correct(self):
        user = User("testuser", "test@example.com", "SecurePass123!")
        assert user.verify_password("SecurePass123!") is True

    def test_verify_password_incorrect(self):
        user = User("testuser", "test@example.com", "SecurePass123!")
        assert user.verify_password("WrongPassword") is False

    def test_verify_password_invalid_hash(self):
        user = User("testuser", "test@example.com", "pass")
        user.password_hash = "invalid_hash_no_colon"
        assert user.verify_password("pass") is False

    def test_has_role(self):
        role = Role("admin")
        user = User("admin", "admin@example.com", "pass", roles=[role])
        assert user.has_role("admin") is True
        assert user.has_role("viewer") is False

    def test_has_permission(self):
        perm = Permission("read_data", "sys", "read")
        role = Role("reader", permissions=[perm])
        user = User("reader", "r@example.com", "pass", roles=[role])
        assert user.has_permission("read_data") is True
        assert user.has_permission("write_data") is False

    def test_can_access_resource(self):
        perm = Permission("r", "mfc", "read")
        role = Role("reader", permissions=[perm])
        user = User("u", "u@ex.com", "p", roles=[role])
        assert user.can_access_resource("mfc", "read") is True
        assert user.can_access_resource("mfc", "write") is False

    def test_get_attribute(self):
        user = User("u", "u@ex.com", "p", attributes={"dept": "research"})
        assert user.get_attribute("dept") == "research"
        assert user.get_attribute("nonexistent") is None

    def test_update_last_login(self):
        user = User("u", "u@ex.com", "p")
        assert user.last_login is None
        user.update_last_login()
        assert user.last_login is not None
        assert isinstance(user.last_login, datetime)


class TestSession:
    """Tests for the Session dataclass."""

    def test_create_session(self):
        user = User("u", "u@ex.com", "p")
        session = Session(user=user)
        assert session.user == user
        assert session.session_id is not None
        assert session.timeout_minutes == 60
        assert isinstance(session.expires_at, datetime)
        assert session.activity_log == []

    def test_session_is_valid(self):
        user = User("u", "u@ex.com", "p")
        session = Session(user=user, timeout_minutes=60)
        assert session.is_valid() is True

    def test_session_expired(self):
        user = User("u", "u@ex.com", "p")
        session = Session(user=user, timeout_minutes=0)
        session.expires_at = datetime.now() - timedelta(minutes=1)
        assert session.is_valid() is False

    def test_refresh_session_default(self):
        user = User("u", "u@ex.com", "p")
        session = Session(user=user, timeout_minutes=30)
        old_expires = session.expires_at
        session.refresh()
        assert session.expires_at >= old_expires

    def test_refresh_session_custom_minutes(self):
        user = User("u", "u@ex.com", "p")
        session = Session(user=user, timeout_minutes=30)
        session.refresh(additional_minutes=120)
        assert session.is_valid() is True

    def test_refresh_session_zero_minutes(self):
        user = User("u", "u@ex.com", "p")
        session = Session(user=user, timeout_minutes=30)
        session.refresh(additional_minutes=0)
        assert session.is_valid() is True

    def test_record_activity(self):
        user = User("u", "u@ex.com", "p")
        session = Session(user=user)
        session.record_activity("login")
        assert len(session.activity_log) == 1
        assert session.activity_log[0]["action"] == "login"

    def test_record_activity_with_metadata(self):
        user = User("u", "u@ex.com", "p")
        session = Session(user=user)
        session.record_activity("read", {"resource": "mfc"})
        assert session.activity_log[0]["metadata"] == {"resource": "mfc"}


class TestAccessControl:
    """Tests for the AccessControl class."""

    def test_register_user(self):
        ac = AccessControl()
        user = ac.register_user("testuser", "test@ex.com", "pass123")
        assert user.username == "testuser"
        assert "testuser" in ac.users

    def test_register_user_duplicate(self):
        ac = AccessControl()
        ac.register_user("testuser", "test@ex.com", "pass123")
        with pytest.raises(ValueError, match="User testuser already exists"):
            ac.register_user("testuser", "test2@ex.com", "pass456")

    def test_register_user_with_roles(self):
        ac = AccessControl()
        role = Role("admin")
        user = ac.register_user("admin", "a@ex.com", "pass", roles=[role])
        assert user.has_role("admin")

    def test_register_user_with_attributes(self):
        ac = AccessControl()
        user = ac.register_user("u", "u@ex.com", "p", attributes={"dept": "R&D"})
        assert user.get_attribute("dept") == "R&D"

    def test_authenticate_success(self):
        ac = AccessControl()
        ac.register_user("testuser", "t@ex.com", "SecurePass!")
        user = ac.authenticate("testuser", "SecurePass!")
        assert user.username == "testuser"
        assert user.last_login is not None

    def test_authenticate_unknown_user(self):
        ac = AccessControl()
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            ac.authenticate("unknown", "pass")

    def test_authenticate_wrong_password(self):
        ac = AccessControl()
        ac.register_user("testuser", "t@ex.com", "correct_pass")
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            ac.authenticate("testuser", "wrong_pass")
        assert "testuser" in ac.failed_logins

    def test_authenticate_inactive_user(self):
        ac = AccessControl()
        user = ac.register_user("testuser", "t@ex.com", "pass")
        user.is_active = False
        with pytest.raises(AuthenticationError, match="User account is inactive"):
            ac.authenticate("testuser", "pass")

    def test_login_creates_session(self):
        ac = AccessControl()
        ac.register_user("testuser", "t@ex.com", "pass")
        session = ac.login("testuser", "pass")
        assert session.session_id in ac.active_sessions
        assert len(session.activity_log) == 1
        assert session.activity_log[0]["action"] == "login"

    def test_logout_success(self):
        ac = AccessControl()
        ac.register_user("testuser", "t@ex.com", "pass")
        session = ac.login("testuser", "pass")
        ac.logout(session.session_id)
        assert session.session_id not in ac.active_sessions

    def test_logout_invalid_session(self):
        ac = AccessControl()
        with pytest.raises(AuthenticationError, match="Invalid session"):
            ac.logout("nonexistent-session-id")

    def test_validate_session_valid(self):
        ac = AccessControl()
        ac.register_user("testuser", "t@ex.com", "pass")
        session = ac.login("testuser", "pass")
        validated = ac.validate_session(session.session_id)
        assert validated.session_id == session.session_id

    def test_validate_session_invalid(self):
        ac = AccessControl()
        with pytest.raises(AuthenticationError, match="Invalid session"):
            ac.validate_session("nonexistent")

    def test_validate_session_expired(self):
        ac = AccessControl()
        ac.register_user("testuser", "t@ex.com", "pass")
        session = ac.login("testuser", "pass")
        session.expires_at = datetime.now() - timedelta(minutes=1)
        with pytest.raises(AuthenticationError, match="Session expired"):
            ac.validate_session(session.session_id)
        assert session.session_id not in ac.active_sessions

    def test_authorize_success(self):
        ac = AccessControl()
        perm = Permission("read", "mfc", "read")
        role = Role("reader", permissions=[perm])
        ac.register_user("u", "u@ex.com", "p", roles=[role])
        session = ac.login("u", "p")
        assert ac.authorize(session, "mfc", "read") is True

    def test_authorize_denied(self):
        ac = AccessControl()
        role = Role("empty_role")
        ac.register_user("u", "u@ex.com", "p", roles=[role])
        session = ac.login("u", "p")
        assert ac.authorize(session, "mfc", "admin") is False

    def test_authorize_expired_session(self):
        ac = AccessControl()
        ac.register_user("u", "u@ex.com", "p")
        session = ac.login("u", "p")
        session.expires_at = datetime.now() - timedelta(minutes=1)
        assert ac.authorize(session, "mfc", "read") is False

    def test_record_failed_login_cleans_old(self):
        ac = AccessControl()
        ac.register_user("u", "u@ex.com", "correct")
        for _ in range(5):
            try:
                ac.authenticate("u", "wrong")
            except AuthenticationError:
                pass
        assert len(ac.failed_logins["u"]) == 5


class TestExceptionHierarchy:
    """Test exception classes."""

    def test_access_control_error(self):
        with pytest.raises(AccessControlError):
            raise AccessControlError("test")

    def test_authentication_error_inherits(self):
        assert issubclass(AuthenticationError, AccessControlError)

    def test_authorization_error_inherits(self):
        assert issubclass(AuthorizationError, AccessControlError)


class TestMainFunction:
    """Test the main() demonstration function."""

    def test_main_runs_successfully(self):
        result = main()
        assert result is True
