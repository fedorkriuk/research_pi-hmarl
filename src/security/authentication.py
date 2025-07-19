"""Authentication and Authorization System

This module provides secure authentication and role-based access control
for the PI-HMARL system.
"""

import jwt
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path
import bcrypt
import hmac

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions"""
    # System permissions
    SYSTEM_ADMIN = "system.admin"
    SYSTEM_MONITOR = "system.monitor"
    SYSTEM_CONFIG = "system.config"
    
    # Agent permissions
    AGENT_CONTROL = "agent.control"
    AGENT_MONITOR = "agent.monitor"
    AGENT_DEPLOY = "agent.deploy"
    
    # Task permissions
    TASK_CREATE = "task.create"
    TASK_ASSIGN = "task.assign"
    TASK_MONITOR = "task.monitor"
    
    # Data permissions
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    
    # Communication permissions
    COMM_SEND = "comm.send"
    COMM_RECEIVE = "comm.receive"
    COMM_BROADCAST = "comm.broadcast"


@dataclass
class Role:
    """User role with permissions"""
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission"""
        return permission in self.permissions
    
    def add_permission(self, permission: Permission):
        """Add permission to role"""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role"""
        self.permissions.discard(permission)


@dataclass
class User:
    """System user"""
    username: str
    user_id: str
    roles: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has role"""
        return role_name in self.roles
    
    def add_role(self, role_name: str):
        """Add role to user"""
        self.roles.add(role_name)
    
    def remove_role(self, role_name: str):
        """Remove role from user"""
        self.roles.discard(role_name)


class AuthenticationManager:
    """Manages authentication and authorization"""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_expiry: int = 3600,
        refresh_expiry: int = 86400
    ):
        """Initialize authentication manager
        
        Args:
            secret_key: Secret key for JWT
            token_expiry: Access token expiry in seconds
            refresh_expiry: Refresh token expiry in seconds
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.token_expiry = token_expiry
        self.refresh_expiry = refresh_expiry
        
        # Storage
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.passwords: Dict[str, bytes] = {}  # Hashed passwords
        self.tokens: Dict[str, Dict[str, Any]] = {}  # Active tokens
        
        # Initialize default roles
        self._init_default_roles()
        
        logger.info("Initialized AuthenticationManager")
    
    def _init_default_roles(self):
        """Initialize default system roles"""
        # Admin role
        admin_role = Role(
            name="admin",
            description="System administrator",
            permissions={
                Permission.SYSTEM_ADMIN,
                Permission.SYSTEM_MONITOR,
                Permission.SYSTEM_CONFIG,
                Permission.AGENT_CONTROL,
                Permission.AGENT_DEPLOY,
                Permission.TASK_CREATE,
                Permission.TASK_ASSIGN,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_DELETE,
                Permission.COMM_BROADCAST
            }
        )
        
        # Operator role
        operator_role = Role(
            name="operator",
            description="System operator",
            permissions={
                Permission.SYSTEM_MONITOR,
                Permission.AGENT_CONTROL,
                Permission.AGENT_MONITOR,
                Permission.TASK_CREATE,
                Permission.TASK_ASSIGN,
                Permission.TASK_MONITOR,
                Permission.DATA_READ,
                Permission.COMM_SEND,
                Permission.COMM_RECEIVE
            }
        )
        
        # Monitor role
        monitor_role = Role(
            name="monitor",
            description="Read-only monitoring",
            permissions={
                Permission.SYSTEM_MONITOR,
                Permission.AGENT_MONITOR,
                Permission.TASK_MONITOR,
                Permission.DATA_READ,
                Permission.COMM_RECEIVE
            }
        )
        
        # Agent role
        agent_role = Role(
            name="agent",
            description="Autonomous agent",
            permissions={
                Permission.AGENT_MONITOR,
                Permission.TASK_MONITOR,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.COMM_SEND,
                Permission.COMM_RECEIVE
            }
        )
        
        self.roles["admin"] = admin_role
        self.roles["operator"] = operator_role
        self.roles["monitor"] = monitor_role
        self.roles["agent"] = agent_role
    
    def create_user(
        self,
        username: str,
        password: str,
        roles: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> User:
        """Create new user
        
        Args:
            username: Username
            password: Password
            roles: Initial roles
            metadata: User metadata
            
        Returns:
            Created user
        """
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        # Generate user ID
        user_id = secrets.token_urlsafe(16)
        
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create user
        user = User(
            username=username,
            user_id=user_id,
            roles=set(roles or []),
            metadata=metadata or {}
        )
        
        # Store
        self.users[username] = user
        self.passwords[username] = hashed_password
        
        logger.info(f"Created user: {username}")
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User if authenticated, None otherwise
        """
        if username not in self.users:
            return None
        
        user = self.users[username]
        
        if not user.is_active:
            logger.warning(f"Inactive user login attempt: {username}")
            return None
        
        # Check password
        stored_password = self.passwords.get(username)
        if not stored_password:
            return None
        
        if bcrypt.checkpw(password.encode('utf-8'), stored_password):
            user.last_login = datetime.now()
            logger.info(f"User authenticated: {username}")
            return user
        
        logger.warning(f"Failed authentication attempt: {username}")
        return None
    
    def generate_token(self, user: User) -> Dict[str, str]:
        """Generate access and refresh tokens
        
        Args:
            user: Authenticated user
            
        Returns:
            Token dictionary
        """
        # Access token payload
        access_payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': list(user.roles),
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        
        # Refresh token payload
        refresh_payload = {
            'user_id': user.user_id,
            'exp': datetime.utcnow() + timedelta(seconds=self.refresh_expiry),
            'iat': datetime.utcnow(),
            'type': 'refresh'
        }
        
        # Generate tokens
        access_token = jwt.encode(access_payload, self.secret_key, algorithm='HS256')
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm='HS256')
        
        # Store active tokens
        self.tokens[access_token] = {
            'user_id': user.user_id,
            'type': 'access',
            'expires': access_payload['exp']
        }
        
        self.tokens[refresh_token] = {
            'user_id': user.user_id,
            'type': 'refresh',
            'expires': refresh_payload['exp']
        }
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': self.token_expiry
        }
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode token
        
        Args:
            token: JWT token
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is in active tokens
            if token not in self.tokens:
                logger.warning("Token not in active tokens")
                return None
            
            # Check expiry
            if datetime.fromtimestamp(payload['exp']) < datetime.utcnow():
                logger.warning("Token expired")
                self.tokens.pop(token, None)
                return None
            
            return payload
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Refresh access token
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New token dictionary if valid
        """
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get('type') != 'refresh':
            return None
        
        # Get user
        user = self.get_user_by_id(payload['user_id'])
        if not user:
            return None
        
        # Revoke old refresh token
        self.tokens.pop(refresh_token, None)
        
        # Generate new tokens
        return self.generate_token(user)
    
    def revoke_token(self, token: str):
        """Revoke token
        
        Args:
            token: Token to revoke
        """
        self.tokens.pop(token, None)
        logger.info("Token revoked")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User if found
        """
        for user in self.users.values():
            if user.user_id == user_id:
                return user
        return None
    
    def check_permission(
        self,
        user: User,
        permission: Permission
    ) -> bool:
        """Check if user has permission
        
        Args:
            user: User
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and role.has_permission(permission):
                return True
        return False
    
    def require_permission(self, permission: Permission):
        """Decorator to require permission
        
        Args:
            permission: Required permission
            
        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract token from context (simplified)
                token = kwargs.get('auth_token')
                if not token:
                    raise PermissionError("No authentication token provided")
                
                # Verify token
                payload = self.verify_token(token)
                if not payload:
                    raise PermissionError("Invalid or expired token")
                
                # Get user
                user = self.get_user_by_id(payload['user_id'])
                if not user:
                    raise PermissionError("User not found")
                
                # Check permission
                if not self.check_permission(user, permission):
                    raise PermissionError(f"Missing permission: {permission.value}")
                
                # Add user to kwargs
                kwargs['auth_user'] = user
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


class TokenManager:
    """Manages secure token generation and validation"""
    
    def __init__(self, secret_key: str):
        """Initialize token manager
        
        Args:
            secret_key: Secret key for signing
        """
        self.secret_key = secret_key
    
    def generate_api_key(self, identifier: str) -> str:
        """Generate API key
        
        Args:
            identifier: Unique identifier
            
        Returns:
            API key
        """
        # Generate random component
        random_part = secrets.token_urlsafe(32)
        
        # Create signature
        message = f"{identifier}:{random_part}".encode()
        signature = hmac.new(
            self.secret_key.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        
        # Combine parts
        api_key = f"{identifier}.{random_part}.{signature}"
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key
        
        Args:
            api_key: API key to verify
            
        Returns:
            Identifier if valid, None otherwise
        """
        try:
            parts = api_key.split('.')
            if len(parts) != 3:
                return None
            
            identifier, random_part, signature = parts
            
            # Recreate signature
            message = f"{identifier}:{random_part}".encode()
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            if hmac.compare_digest(signature, expected_signature):
                return identifier
            
        except Exception as e:
            logger.error(f"API key verification error: {e}")
        
        return None
    
    def generate_session_token(
        self,
        user_id: str,
        duration: int = 3600
    ) -> str:
        """Generate session token
        
        Args:
            user_id: User ID
            duration: Token duration in seconds
            
        Returns:
            Session token
        """
        payload = {
            'user_id': user_id,
            'created': time.time(),
            'expires': time.time() + duration,
            'session_id': secrets.token_urlsafe(16)
        }
        
        # Sign payload
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        return token
    
    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify session token
        
        Args:
            token: Session token
            
        Returns:
            Payload if valid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check expiry
            if time.time() > payload.get('expires', 0):
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None


# Example usage
def create_secure_system():
    """Create secure system with authentication"""
    
    # Initialize authentication
    auth_manager = AuthenticationManager()
    
    # Create users
    admin = auth_manager.create_user(
        username="admin",
        password="secure_password_123",
        roles=["admin"],
        metadata={"email": "admin@pi-hmarl.org"}
    )
    
    operator = auth_manager.create_user(
        username="operator1",
        password="operator_pass_456",
        roles=["operator"],
        metadata={"department": "operations"}
    )
    
    # Create agent users
    for i in range(5):
        auth_manager.create_user(
            username=f"agent_{i}",
            password=f"agent_pass_{i}",
            roles=["agent"],
            metadata={"agent_id": i}
        )
    
    return auth_manager