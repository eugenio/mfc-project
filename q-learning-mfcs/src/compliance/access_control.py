"""
Access Control Implementation - Testing TDD Approach

This is a standalone implementation to test the access control functionality
before integrating into the main compliance module.
"""
import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


class AccessControlError(Exception):
    """Base exception for access control errors"""
    pass


class AuthenticationError(AccessControlError):
    """Raised when authentication fails"""
    pass


class AuthorizationError(AccessControlError):
    """Raised when authorization fails"""
    pass

@dataclass
class Permission:
    """Represents a system permission"""
    name: str
    resource: str
    action: str
    description: str = ""
    permission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate permission data"""
        if not self.name or not self.name.strip():
            raise ValueError("Permission name cannot be empty")
        if not self.resource or not self.resource.strip():
            raise ValueError("Permission resource cannot be empty")
        if not self.action or not self.action.strip():
            raise ValueError("Permission action cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert permission to dictionary"""
        return {
            "permission_id": self.permission_id,
            "name": self.name,
            "resource": self.resource,
            "action": self.action,
            "description": self.description,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Permission':
        """Create permission from dictionary"""
        permission = cls(
            name=data["name"],
            resource=data["resource"],
            action=data["action"],
            description=data.get("description", ""),
            permission_id=data.get("permission_id", str(uuid.uuid4()))
        )
        if "created_at" in data:
            permission.created_at = datetime.fromisoformat(data["created_at"])
        return permission

@dataclass
class Role:
    """Represents a system role with permissions"""
    name: str
    description: str = ""
    permissions: list[Permission] = field(default_factory=list)
    parent_roles: list['Role'] = field(default_factory=list)
    role_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate role data"""
        if not self.name or not self.name.strip():
            raise ValueError("Role name cannot be empty")

    def add_permission(self, permission: Permission) -> None:
        """Add permission to role"""
        if permission not in self.permissions:
            self.permissions.append(permission)

    def remove_permission(self, permission: Permission) -> None:
        """Remove permission from role"""
        if permission in self.permissions:
            self.permissions.remove(permission)

    def has_permission(self, permission_name: str) -> bool:
        """Check if role has specific permission"""
        for perm in self.permissions:
            if perm.name == permission_name:
                return True

        for parent_role in self.parent_roles:
            if parent_role.has_permission(permission_name):
                return True

        return False

    def has_permission_for_resource(self, resource: str, action: str) -> bool:
        """Check if role has permission for specific resource and action"""
        for perm in self.permissions:
            if perm.resource == resource and perm.action == action:
                return True

        for parent_role in self.parent_roles:
            if parent_role.has_permission_for_resource(resource, action):
                return True

        return False

@dataclass
class User:
    """Represents a system user"""
    username: str
    email: str
    password: str  # Will be hashed in __post_init__
    roles: list[Role] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    last_login: datetime | None = None
    password_hash: str = field(init=False)

    def __post_init__(self):
        """Initialize user with hashed password"""
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")
        if not self.email or not self.email.strip():
            raise ValueError("Email cannot be empty")

        # Hash the password
        self.password_hash = self._hash_password(self.password)
        # Clear the plain text password for security
        self.password = ""

    def _hash_password(self, password: str) -> str:
        """Hash password using PBKDF2"""
        salt = secrets.token_bytes(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return f"{salt.hex()}:{pwdhash.hex()}"

    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt_hex, pwdhash_hex = self.password_hash.split(':')
            salt = bytes.fromhex(salt_hex)
            pwdhash = bytes.fromhex(pwdhash_hex)

            test_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return hmac.compare_digest(pwdhash, test_hash)
        except ValueError:
            return False

    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        return any(role.name == role_name for role in self.roles)

    def has_permission(self, permission_name: str) -> bool:
        """Check if user has specific permission through roles"""
        return any(role.has_permission(permission_name) for role in self.roles)

    def can_access_resource(self, resource: str, action: str) -> bool:
        """Check if user can access resource with specific action"""
        return any(role.has_permission_for_resource(resource, action) for role in self.roles)

    def get_attribute(self, attribute_name: str) -> Any:
        """Get user attribute for ABAC"""
        return self.attributes.get(attribute_name)

    def update_last_login(self) -> None:
        """Update last login timestamp"""
        self.last_login = datetime.now()

@dataclass
class Session:
    """Represents a user session"""
    user: User
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(init=False)
    timeout_minutes: int = 60
    activity_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize session expiration"""
        self.expires_at = self.created_at + timedelta(minutes=self.timeout_minutes)

    def is_valid(self) -> bool:
        """Check if session is still valid"""
        return datetime.now() < self.expires_at

    def refresh(self, additional_minutes: int = 0) -> None:
        """Refresh session expiration"""
        if additional_minutes > 0:
            self.expires_at = datetime.now() + timedelta(minutes=additional_minutes)
        else:
            self.expires_at = datetime.now() + timedelta(minutes=self.timeout_minutes)

    def record_activity(self, action: str, metadata: dict[str, Any] | None = None) -> None:
        """Record session activity"""
        self.activity_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "metadata": metadata or {}
        })

class AccessControl:
    """Main access control system"""

    def __init__(self):
        self.users: dict[str, User] = {}
        self.roles: dict[str, Role] = {}
        self.active_sessions: dict[str, Session] = {}
        self.failed_logins: dict[str, list[datetime]] = {}

    def register_user(self, username: str, email: str, password: str,
                     roles: list[Role] | None = None,
                     attributes: dict[str, Any] | None = None) -> User:
        """Register a new user"""
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        user = User(
            username=username,
            email=email,
            password=password,
            roles=roles or [],
            attributes=attributes or {}
        )

        self.users[username] = user
        return user

    def authenticate(self, username: str, password: str) -> User:
        """Authenticate user with username and password"""
        user = self.users.get(username)
        if not user:
            raise AuthenticationError("Invalid username or password")

        if not user.verify_password(password):
            self._record_failed_login(username)
            raise AuthenticationError("Invalid username or password")

        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        user.update_last_login()
        return user

    def login(self, username: str, password: str) -> Session:
        """Login user and create session"""
        user = self.authenticate(username, password)
        session = Session(user=user)
        self.active_sessions[session.session_id] = session
        session.record_activity("login")
        return session

    def logout(self, session_id: str) -> None:
        """Logout user and invalidate session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.record_activity("logout")
            del self.active_sessions[session_id]
        else:
            raise AuthenticationError("Invalid session")

    def validate_session(self, session_id: str) -> Session:
        """Validate session token"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise AuthenticationError("Invalid session")

        if not session.is_valid():
            del self.active_sessions[session_id]
            raise AuthenticationError("Session expired")

        return session

    def authorize(self, session: Session, resource: str, action: str) -> bool:
        """Authorize user access to resource"""
        if not session.is_valid():
            return False

        session.record_activity("authorization_check", {
            "resource": resource,
            "action": action
        })

        return session.user.can_access_resource(resource, action)

    def _record_failed_login(self, username: str) -> None:
        """Record failed login attempt"""
        now = datetime.now()
        if username not in self.failed_logins:
            self.failed_logins[username] = []

        self.failed_logins[username].append(now)

        # Keep only attempts from the last hour
        one_hour_ago = now - timedelta(hours=1)
        self.failed_logins[username] = [
            attempt for attempt in self.failed_logins[username]
            if attempt > one_hour_ago
        ]

def main():
    """Demonstration of access control functionality"""
    print("🛡️  Access Control System - TDD Implementation Demo")
    print("=" * 50)

    # Initialize access control system
    ac = AccessControl()

    # Create permissions
    read_permission = Permission("mfc_read", "mfc_system", "read", "Read MFC data")
    write_permission = Permission("mfc_write", "mfc_system", "write", "Write MFC data")
    admin_permission = Permission("mfc_admin", "mfc_system", "admin", "Administer MFC system")

    print(f"✓ Created permissions: {read_permission.name}, {write_permission.name}, {admin_permission.name}")

    # Create roles
    operator_role = Role("operator", "MFC Operator", [read_permission, write_permission])
    admin_role = Role("admin", "MFC Administrator", [read_permission, write_permission, admin_permission])

    print(f"✓ Created roles: {operator_role.name}, {admin_role.name}")

    # Register users
    operator_user = ac.register_user("operator01", "operator@mfc.com", "SecurePass123!", [operator_role])
    admin_user = ac.register_user("admin01", "admin@mfc.com", "AdminPass123!", [admin_role])

    print(f"✓ Registered users: {operator_user.username}, {admin_user.username}")

    # Test authentication and authorization
    try:
        # Login as operator
        op_session = ac.login("operator01", "SecurePass123!")
        print(f"✓ Operator logged in: {op_session.session_id}")

        # Test authorization
        can_read = ac.authorize(op_session, "mfc_system", "read")
        can_admin = ac.authorize(op_session, "mfc_system", "admin")

        print(f"✓ Operator can read: {can_read}")
        print(f"✓ Operator can admin: {can_admin}")

        # Login as admin
        admin_session = ac.login("admin01", "AdminPass123!")
        print(f"✓ Admin logged in: {admin_session.session_id}")

        # Test admin authorization
        admin_can_admin = ac.authorize(admin_session, "mfc_system", "admin")
        print(f"✓ Admin can admin: {admin_can_admin}")

        # Logout
        ac.logout(op_session.session_id)
        ac.logout(admin_session.session_id)
        print("✓ Users logged out successfully")

        print("\n🎉 All tests passed! Access Control system working correctly.")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
