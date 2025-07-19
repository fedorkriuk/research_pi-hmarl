"""Encryption and Secure Communication

This module provides encryption capabilities for secure data transmission
and storage in the PI-HMARL system.
"""

import os
import base64
import json
from typing import Dict, Any, Optional, Tuple, List
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import nacl.secret
import nacl.utils
from nacl.public import PrivateKey, PublicKey, Box
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class EncryptedMessage:
    """Encrypted message container"""
    ciphertext: bytes
    nonce: Optional[bytes] = None
    timestamp: float = None
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None
    signature: Optional[bytes] = None


class EncryptionManager:
    """Manages encryption operations"""
    
    def __init__(self):
        """Initialize encryption manager"""
        self.backend = default_backend()
        self._keys: Dict[str, Any] = {}
        
        logger.info("Initialized EncryptionManager")
    
    def generate_symmetric_key(self) -> bytes:
        """Generate symmetric encryption key
        
        Returns:
            Symmetric key
        """
        return Fernet.generate_key()
    
    def generate_asymmetric_keypair(self) -> Tuple[bytes, bytes]:
        """Generate RSA keypair
        
        Returns:
            Private key and public key as PEM bytes
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=self.backend
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_symmetric(self, data: bytes, key: bytes) -> EncryptedMessage:
        """Encrypt data using symmetric encryption
        
        Args:
            data: Data to encrypt
            key: Symmetric key
            
        Returns:
            Encrypted message
        """
        f = Fernet(key)
        ciphertext = f.encrypt(data)
        
        return EncryptedMessage(
            ciphertext=ciphertext,
            timestamp=datetime.now().timestamp()
        )
    
    def decrypt_symmetric(self, encrypted: EncryptedMessage, key: bytes) -> bytes:
        """Decrypt symmetric encrypted data
        
        Args:
            encrypted: Encrypted message
            key: Symmetric key
            
        Returns:
            Decrypted data
        """
        f = Fernet(key)
        return f.decrypt(encrypted.ciphertext)
    
    def encrypt_asymmetric(
        self,
        data: bytes,
        public_key_pem: bytes
    ) -> EncryptedMessage:
        """Encrypt data using RSA public key
        
        Args:
            data: Data to encrypt
            public_key_pem: Public key in PEM format
            
        Returns:
            Encrypted message
        """
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=self.backend
        )
        
        # Encrypt data
        ciphertext = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return EncryptedMessage(
            ciphertext=ciphertext,
            timestamp=datetime.now().timestamp()
        )
    
    def decrypt_asymmetric(
        self,
        encrypted: EncryptedMessage,
        private_key_pem: bytes
    ) -> bytes:
        """Decrypt RSA encrypted data
        
        Args:
            encrypted: Encrypted message
            private_key_pem: Private key in PEM format
            
        Returns:
            Decrypted data
        """
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=self.backend
        )
        
        # Decrypt data
        plaintext = private_key.decrypt(
            encrypted.ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext
    
    def create_hybrid_encryption(
        self,
        data: bytes,
        public_key_pem: bytes
    ) -> Tuple[EncryptedMessage, EncryptedMessage]:
        """Hybrid encryption (RSA + AES)
        
        Args:
            data: Data to encrypt
            public_key_pem: Recipient's public key
            
        Returns:
            Encrypted symmetric key and encrypted data
        """
        # Generate symmetric key
        symmetric_key = self.generate_symmetric_key()
        
        # Encrypt data with symmetric key
        encrypted_data = self.encrypt_symmetric(data, symmetric_key)
        
        # Encrypt symmetric key with public key
        encrypted_key = self.encrypt_asymmetric(symmetric_key, public_key_pem)
        
        return encrypted_key, encrypted_data
    
    def decrypt_hybrid_encryption(
        self,
        encrypted_key: EncryptedMessage,
        encrypted_data: EncryptedMessage,
        private_key_pem: bytes
    ) -> bytes:
        """Decrypt hybrid encrypted data
        
        Args:
            encrypted_key: Encrypted symmetric key
            encrypted_data: Encrypted data
            private_key_pem: Private key
            
        Returns:
            Decrypted data
        """
        # Decrypt symmetric key
        symmetric_key = self.decrypt_asymmetric(encrypted_key, private_key_pem)
        
        # Decrypt data
        return self.decrypt_symmetric(encrypted_data, symmetric_key)


class SecureChannel:
    """Secure communication channel between two parties"""
    
    def __init__(self, identity: str):
        """Initialize secure channel
        
        Args:
            identity: Channel identity
        """
        self.identity = identity
        self.private_key = PrivateKey.generate()
        self.public_key = self.private_key.public_key
        self.peer_keys: Dict[str, PublicKey] = {}
        self.boxes: Dict[str, Box] = {}
        
        logger.info(f"Initialized SecureChannel for {identity}")
    
    def get_public_key(self) -> bytes:
        """Get public key for sharing
        
        Returns:
            Public key bytes
        """
        return bytes(self.public_key)
    
    def add_peer(self, peer_id: str, public_key_bytes: bytes):
        """Add peer's public key
        
        Args:
            peer_id: Peer identity
            public_key_bytes: Peer's public key
        """
        peer_public_key = PublicKey(public_key_bytes)
        self.peer_keys[peer_id] = peer_public_key
        
        # Create box for encryption
        self.boxes[peer_id] = Box(self.private_key, peer_public_key)
        
        logger.info(f"Added peer {peer_id} to secure channel")
    
    def encrypt_for_peer(
        self,
        peer_id: str,
        data: bytes
    ) -> EncryptedMessage:
        """Encrypt data for specific peer
        
        Args:
            peer_id: Peer identity
            data: Data to encrypt
            
        Returns:
            Encrypted message
        """
        if peer_id not in self.boxes:
            raise ValueError(f"Unknown peer: {peer_id}")
        
        box = self.boxes[peer_id]
        
        # Encrypt
        ciphertext = box.encrypt(data)
        
        return EncryptedMessage(
            ciphertext=ciphertext,
            sender_id=self.identity,
            recipient_id=peer_id,
            timestamp=datetime.now().timestamp()
        )
    
    def decrypt_from_peer(
        self,
        peer_id: str,
        encrypted: EncryptedMessage
    ) -> bytes:
        """Decrypt data from peer
        
        Args:
            peer_id: Peer identity
            encrypted: Encrypted message
            
        Returns:
            Decrypted data
        """
        if peer_id not in self.boxes:
            raise ValueError(f"Unknown peer: {peer_id}")
        
        box = self.boxes[peer_id]
        
        # Decrypt
        plaintext = box.decrypt(encrypted.ciphertext)
        
        return plaintext


class MessageEncryption:
    """End-to-end message encryption"""
    
    def __init__(self, node_id: str):
        """Initialize message encryption
        
        Args:
            node_id: Node identifier
        """
        self.node_id = node_id
        self.channel = SecureChannel(node_id)
        self.message_counter = 0
        
    def encrypt_message(
        self,
        recipient_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encrypt message for transmission
        
        Args:
            recipient_id: Recipient node ID
            message: Message dictionary
            
        Returns:
            Encrypted message envelope
        """
        # Add metadata
        message['sender'] = self.node_id
        message['recipient'] = recipient_id
        message['counter'] = self.message_counter
        message['timestamp'] = datetime.now().isoformat()
        
        self.message_counter += 1
        
        # Serialize message
        message_bytes = json.dumps(message).encode()
        
        # Encrypt
        encrypted = self.channel.encrypt_for_peer(recipient_id, message_bytes)
        
        # Create envelope
        envelope = {
            'type': 'encrypted_message',
            'sender': self.node_id,
            'recipient': recipient_id,
            'ciphertext': base64.b64encode(encrypted.ciphertext).decode(),
            'timestamp': encrypted.timestamp
        }
        
        return envelope
    
    def decrypt_message(
        self,
        envelope: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decrypt received message
        
        Args:
            envelope: Encrypted message envelope
            
        Returns:
            Decrypted message
        """
        sender_id = envelope['sender']
        
        # Decode ciphertext
        ciphertext = base64.b64decode(envelope['ciphertext'])
        
        encrypted = EncryptedMessage(
            ciphertext=ciphertext,
            sender_id=sender_id,
            recipient_id=envelope['recipient'],
            timestamp=envelope['timestamp']
        )
        
        # Decrypt
        plaintext = self.channel.decrypt_from_peer(sender_id, encrypted)
        
        # Parse message
        message = json.loads(plaintext.decode())
        
        return message


class KeyManager:
    """Manages encryption keys"""
    
    def __init__(self, storage_path: str = "keys"):
        """Initialize key manager
        
        Args:
            storage_path: Path for key storage
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.master_key = None
        
        # Initialize master key
        self._init_master_key()
        
        logger.info("Initialized KeyManager")
    
    def _init_master_key(self):
        """Initialize or load master key"""
        master_key_path = os.path.join(self.storage_path, "master.key")
        
        if os.path.exists(master_key_path):
            # Load existing master key
            with open(master_key_path, 'rb') as f:
                self.master_key = f.read()
        else:
            # Generate new master key
            self.master_key = Fernet.generate_key()
            
            # Save master key (in production, use HSM or secure storage)
            with open(master_key_path, 'wb') as f:
                f.write(self.master_key)
            
            os.chmod(master_key_path, 0o600)  # Restrict permissions
    
    def generate_node_keys(self, node_id: str) -> Dict[str, bytes]:
        """Generate keys for a node
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary of keys
        """
        # Generate keys
        private_key, public_key = EncryptionManager().generate_asymmetric_keypair()
        symmetric_key = EncryptionManager().generate_symmetric_key()
        
        # Store keys (encrypted with master key)
        f = Fernet(self.master_key)
        
        keys = {
            'node_id': node_id,
            'private_key': private_key,
            'public_key': public_key,
            'symmetric_key': symmetric_key,
            'created': datetime.now().isoformat()
        }
        
        # Encrypt sensitive keys
        encrypted_keys = {
            'node_id': node_id,
            'private_key': f.encrypt(private_key),
            'public_key': public_key,  # Public key doesn't need encryption
            'symmetric_key': f.encrypt(symmetric_key),
            'created': keys['created']
        }
        
        self.keys[node_id] = encrypted_keys
        
        # Save to disk
        self._save_keys(node_id, encrypted_keys)
        
        return keys
    
    def get_node_keys(self, node_id: str) -> Optional[Dict[str, bytes]]:
        """Get keys for a node
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary of keys
        """
        # Check memory cache
        if node_id not in self.keys:
            # Try loading from disk
            self._load_keys(node_id)
        
        if node_id not in self.keys:
            return None
        
        encrypted_keys = self.keys[node_id]
        f = Fernet(self.master_key)
        
        # Decrypt keys
        keys = {
            'node_id': node_id,
            'private_key': f.decrypt(encrypted_keys['private_key']),
            'public_key': encrypted_keys['public_key'],
            'symmetric_key': f.decrypt(encrypted_keys['symmetric_key']),
            'created': encrypted_keys['created']
        }
        
        return keys
    
    def _save_keys(self, node_id: str, keys: Dict[str, Any]):
        """Save keys to disk
        
        Args:
            node_id: Node identifier
            keys: Encrypted keys
        """
        key_file = os.path.join(self.storage_path, f"{node_id}.keys")
        
        # Convert bytes to base64 for JSON serialization
        serializable_keys = {
            'node_id': keys['node_id'],
            'private_key': base64.b64encode(keys['private_key']).decode(),
            'public_key': base64.b64encode(keys['public_key']).decode(),
            'symmetric_key': base64.b64encode(keys['symmetric_key']).decode(),
            'created': keys['created']
        }
        
        with open(key_file, 'w') as f:
            json.dump(serializable_keys, f)
        
        os.chmod(key_file, 0o600)  # Restrict permissions
    
    def _load_keys(self, node_id: str):
        """Load keys from disk
        
        Args:
            node_id: Node identifier
        """
        key_file = os.path.join(self.storage_path, f"{node_id}.keys")
        
        if not os.path.exists(key_file):
            return
        
        with open(key_file, 'r') as f:
            serializable_keys = json.load(f)
        
        # Convert base64 back to bytes
        keys = {
            'node_id': serializable_keys['node_id'],
            'private_key': base64.b64decode(serializable_keys['private_key']),
            'public_key': base64.b64decode(serializable_keys['public_key']),
            'symmetric_key': base64.b64decode(serializable_keys['symmetric_key']),
            'created': serializable_keys['created']
        }
        
        self.keys[node_id] = keys
    
    def rotate_keys(self, node_id: str) -> Dict[str, bytes]:
        """Rotate keys for a node
        
        Args:
            node_id: Node identifier
            
        Returns:
            New keys
        """
        # Generate new keys
        new_keys = self.generate_node_keys(node_id)
        
        logger.info(f"Rotated keys for node {node_id}")
        
        return new_keys


# Example usage
def setup_encrypted_communication():
    """Setup encrypted communication between nodes"""
    
    # Initialize encryption manager
    enc_manager = EncryptionManager()
    
    # Initialize key manager
    key_manager = KeyManager()
    
    # Generate keys for nodes
    node1_keys = key_manager.generate_node_keys("node1")
    node2_keys = key_manager.generate_node_keys("node2")
    
    # Setup message encryption
    node1_encryption = MessageEncryption("node1")
    node2_encryption = MessageEncryption("node2")
    
    # Exchange public keys
    node1_encryption.channel.add_peer("node2", node2_keys['public_key'])
    node2_encryption.channel.add_peer("node1", node1_keys['public_key'])
    
    return node1_encryption, node2_encryption