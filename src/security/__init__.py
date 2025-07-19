"""Security and Robustness Module

This module provides security enhancements and robustness features
for the PI-HMARL system.
"""

from .authentication import (
    AuthenticationManager, TokenManager,
    User, Role, Permission
)
from .encryption import (
    EncryptionManager, SecureChannel,
    MessageEncryption, KeyManager
)
from .robustness import (
    FaultDetector, AnomalyDetector,
    SystemValidator, RecoveryManager
)
from .attack_defense import (
    AttackDetector, DefenseStrategy,
    AdversarialDefense, IntrusionPrevention
)
from .secure_communication import (
    SecureCommunicationProtocol, MessageVerifier,
    BlockchainLogger, TrustManager
)

__all__ = [
    # Authentication
    'AuthenticationManager', 'TokenManager',
    'User', 'Role', 'Permission',
    
    # Encryption
    'EncryptionManager', 'SecureChannel',
    'MessageEncryption', 'KeyManager',
    
    # Robustness
    'FaultDetector', 'AnomalyDetector',
    'SystemValidator', 'RecoveryManager',
    
    # Attack Defense
    'AttackDetector', 'DefenseStrategy',
    'AdversarialDefense', 'IntrusionPrevention',
    
    # Secure Communication
    'SecureCommunicationProtocol', 'MessageVerifier',
    'BlockchainLogger', 'TrustManager'
]