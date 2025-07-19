"""Secure Communication Protocol

This module provides secure communication protocols, message verification,
blockchain logging, and trust management for the PI-HMARL system.
"""

import hashlib
import hmac
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from enum import Enum
import threading
import asyncio
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import nacl.secret
import nacl.utils
from nacl.signing import SigningKey, VerifyKey

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages"""
    CONTROL = "control"
    DATA = "data"
    STATUS = "status"
    COORDINATION = "coordination"
    CONSENSUS = "consensus"
    HEARTBEAT = "heartbeat"


class TrustLevel(Enum):
    """Trust levels for entities"""
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    FULL = 4


@dataclass
class SecureMessage:
    """Secure message container"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float
    signature: bytes
    encrypted: bool = False
    nonce: Optional[bytes] = None
    sequence_number: int = 0


@dataclass
class BlockchainEntry:
    """Blockchain log entry"""
    index: int
    timestamp: float
    data: Dict[str, Any]
    previous_hash: str
    hash: str
    nonce: int = 0


class SecureCommunicationProtocol:
    """Secure communication protocol implementation"""
    
    def __init__(
        self,
        node_id: str,
        signing_key: Optional[SigningKey] = None
    ):
        """Initialize secure communication protocol
        
        Args:
            node_id: Node identifier
            signing_key: Signing key for messages
        """
        self.node_id = node_id
        self.signing_key = signing_key or SigningKey.generate()
        self.verify_key = self.signing_key.verify_key
        
        # Key management
        self.peer_keys: Dict[str, VerifyKey] = {}
        self.session_keys: Dict[str, bytes] = {}
        
        # Message tracking
        self.message_counter = 0
        self.received_messages: deque = deque(maxlen=10000)
        self.sequence_numbers: Dict[str, int] = defaultdict(int)
        
        # Protocol parameters
        self.max_message_age = 300  # 5 minutes
        self.replay_window = 100
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized SecureCommunicationProtocol for {node_id}")
    
    def create_message(
        self,
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        encrypt: bool = True
    ) -> SecureMessage:
        """Create secure message
        
        Args:
            recipient_id: Recipient identifier
            message_type: Type of message
            payload: Message payload
            encrypt: Whether to encrypt
            
        Returns:
            Secure message
        """
        with self._lock:
            # Create message
            message = SecureMessage(
                message_id=f"{self.node_id}_{self.message_counter}",
                sender_id=self.node_id,
                recipient_id=recipient_id,
                message_type=message_type,
                payload=payload,
                timestamp=time.time(),
                signature=b'',  # Will be set later
                encrypted=encrypt,
                sequence_number=self.message_counter
            )
            
            self.message_counter += 1
            
            # Sign message
            message.signature = self._sign_message(message)
            
            # Encrypt if requested
            if encrypt and recipient_id in self.session_keys:
                message = self._encrypt_message(message, recipient_id)
            
            return message
    
    def verify_message(
        self,
        message: SecureMessage
    ) -> Tuple[bool, Optional[str]]:
        """Verify message integrity and authenticity
        
        Args:
            message: Message to verify
            
        Returns:
            Verification status and error message
        """
        with self._lock:
            # Check message age
            age = time.time() - message.timestamp
            if age > self.max_message_age:
                return False, "Message too old"
            
            # Check sender key
            if message.sender_id not in self.peer_keys:
                return False, "Unknown sender"
            
            # Decrypt if needed
            if message.encrypted:
                success, decrypted = self._decrypt_message(message)
                if not success:
                    return False, "Decryption failed"
                message = decrypted
            
            # Verify signature
            if not self._verify_signature(message):
                return False, "Invalid signature"
            
            # Check replay attack
            if self._is_replay(message):
                return False, "Replay attack detected"
            
            # Update sequence number
            self.sequence_numbers[message.sender_id] = max(
                self.sequence_numbers[message.sender_id],
                message.sequence_number
            )
            
            # Record message
            self.received_messages.append({
                'message_id': message.message_id,
                'timestamp': message.timestamp,
                'hash': self._hash_message(message)
            })
            
            return True, None
    
    def _sign_message(self, message: SecureMessage) -> bytes:
        """Sign message
        
        Args:
            message: Message to sign
            
        Returns:
            Signature
        """
        # Create message digest
        digest = self._create_message_digest(message)
        
        # Sign digest
        signature = self.signing_key.sign(digest.encode())
        
        return signature.signature
    
    def _verify_signature(self, message: SecureMessage) -> bool:
        """Verify message signature
        
        Args:
            message: Message to verify
            
        Returns:
            Verification status
        """
        if message.sender_id not in self.peer_keys:
            return False
        
        verify_key = self.peer_keys[message.sender_id]
        digest = self._create_message_digest(message)
        
        try:
            verify_key.verify(digest.encode(), message.signature)
            return True
        except Exception:
            return False
    
    def _create_message_digest(self, message: SecureMessage) -> str:
        """Create message digest for signing
        
        Args:
            message: Message
            
        Returns:
            Message digest
        """
        # Create deterministic representation
        content = {
            'message_id': message.message_id,
            'sender_id': message.sender_id,
            'recipient_id': message.recipient_id,
            'message_type': message.message_type.value,
            'payload': message.payload,
            'timestamp': message.timestamp,
            'sequence_number': message.sequence_number
        }
        
        # Create hash
        content_str = json.dumps(content, sort_keys=True)
        digest = hashlib.sha256(content_str.encode()).hexdigest()
        
        return digest
    
    def _encrypt_message(
        self,
        message: SecureMessage,
        recipient_id: str
    ) -> SecureMessage:
        """Encrypt message payload
        
        Args:
            message: Message to encrypt
            recipient_id: Recipient
            
        Returns:
            Encrypted message
        """
        if recipient_id not in self.session_keys:
            return message
        
        # Get session key
        session_key = self.session_keys[recipient_id]
        
        # Encrypt payload
        box = nacl.secret.SecretBox(session_key)
        plaintext = json.dumps(message.payload).encode()
        encrypted = box.encrypt(plaintext)
        
        # Update message
        message.payload = {'encrypted_data': encrypted.hex()}
        message.nonce = encrypted.nonce
        message.encrypted = True
        
        return message
    
    def _decrypt_message(
        self,
        message: SecureMessage
    ) -> Tuple[bool, Optional[SecureMessage]]:
        """Decrypt message payload
        
        Args:
            message: Encrypted message
            
        Returns:
            Success status and decrypted message
        """
        if message.sender_id not in self.session_keys:
            return False, None
        
        try:
            # Get session key
            session_key = self.session_keys[message.sender_id]
            
            # Decrypt payload
            box = nacl.secret.SecretBox(session_key)
            encrypted_data = bytes.fromhex(message.payload['encrypted_data'])
            
            plaintext = box.decrypt(encrypted_data)
            
            # Restore payload
            message.payload = json.loads(plaintext.decode())
            message.encrypted = False
            
            return True, message
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return False, None
    
    def _is_replay(self, message: SecureMessage) -> bool:
        """Check for replay attack
        
        Args:
            message: Message to check
            
        Returns:
            True if replay detected
        """
        # Check sequence number
        if message.sender_id in self.sequence_numbers:
            expected_seq = self.sequence_numbers[message.sender_id]
            if message.sequence_number <= expected_seq - self.replay_window:
                return True
        
        # Check message hash
        message_hash = self._hash_message(message)
        for recorded in self.received_messages:
            if recorded['hash'] == message_hash:
                return True
        
        return False
    
    def _hash_message(self, message: SecureMessage) -> str:
        """Create message hash
        
        Args:
            message: Message to hash
            
        Returns:
            Message hash
        """
        content = f"{message.message_id}_{message.timestamp}_{message.sequence_number}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def add_peer(
        self,
        peer_id: str,
        verify_key_bytes: bytes
    ):
        """Add peer's verification key
        
        Args:
            peer_id: Peer identifier
            verify_key_bytes: Peer's verification key
        """
        self.peer_keys[peer_id] = VerifyKey(verify_key_bytes)
        logger.info(f"Added peer {peer_id} to secure communication")
    
    def establish_session(
        self,
        peer_id: str,
        session_key: bytes
    ):
        """Establish session with peer
        
        Args:
            peer_id: Peer identifier
            session_key: Shared session key
        """
        self.session_keys[peer_id] = session_key
        logger.info(f"Established secure session with {peer_id}")
    
    def get_public_key(self) -> bytes:
        """Get public verification key
        
        Returns:
            Public key bytes
        """
        return bytes(self.verify_key)


class MessageVerifier:
    """Message verification service"""
    
    def __init__(self):
        """Initialize message verifier"""
        self.verification_rules: List[Callable] = []
        self.verified_messages: deque = deque(maxlen=10000)
        
        # Initialize default rules
        self._init_default_rules()
        
        logger.info("Initialized MessageVerifier")
    
    def _init_default_rules(self):
        """Initialize default verification rules"""
        self.verification_rules.extend([
            self._verify_format,
            self._verify_timestamp,
            self._verify_sender,
            self._verify_content
        ])
    
    def verify(
        self,
        message: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Verify message
        
        Args:
            message: Message to verify
            context: Verification context
            
        Returns:
            Verification status and issues
        """
        issues = []
        
        for rule in self.verification_rules:
            passed, issue = rule(message, context or {})
            if not passed:
                issues.append(issue)
        
        if not issues:
            self.verified_messages.append({
                'message_id': message.get('message_id', 'unknown'),
                'timestamp': datetime.now(),
                'verified': True
            })
        
        return len(issues) == 0, issues
    
    def _verify_format(
        self,
        message: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Verify message format
        
        Args:
            message: Message to verify
            context: Context
            
        Returns:
            Verification result
        """
        required_fields = ['sender_id', 'message_type', 'timestamp']
        
        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"
        
        return True, None
    
    def _verify_timestamp(
        self,
        message: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Verify message timestamp
        
        Args:
            message: Message to verify
            context: Context
            
        Returns:
            Verification result
        """
        timestamp = message.get('timestamp', 0)
        current_time = time.time()
        
        # Check if timestamp is reasonable
        if abs(current_time - timestamp) > 300:  # 5 minutes
            return False, "Timestamp out of acceptable range"
        
        return True, None
    
    def _verify_sender(
        self,
        message: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Verify sender
        
        Args:
            message: Message to verify
            context: Context
            
        Returns:
            Verification result
        """
        sender_id = message.get('sender_id')
        known_senders = context.get('known_senders', set())
        
        if known_senders and sender_id not in known_senders:
            return False, f"Unknown sender: {sender_id}"
        
        return True, None
    
    def _verify_content(
        self,
        message: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Verify message content
        
        Args:
            message: Message to verify
            context: Context
            
        Returns:
            Verification result
        """
        # Check for suspicious content
        payload = message.get('payload', {})
        
        # Size check
        payload_size = len(json.dumps(payload))
        if payload_size > 1024 * 1024:  # 1MB limit
            return False, "Payload too large"
        
        return True, None
    
    def add_rule(self, rule: Callable):
        """Add verification rule
        
        Args:
            rule: Verification rule function
        """
        self.verification_rules.append(rule)


class BlockchainLogger:
    """Blockchain-based audit logging"""
    
    def __init__(self):
        """Initialize blockchain logger"""
        self.chain: List[BlockchainEntry] = []
        self.pending_entries: List[Dict[str, Any]] = []
        self.mining_difficulty = 4  # Number of leading zeros
        
        # Create genesis block
        self._create_genesis_block()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Initialized BlockchainLogger")
    
    def _create_genesis_block(self):
        """Create genesis block"""
        genesis = BlockchainEntry(
            index=0,
            timestamp=time.time(),
            data={'type': 'genesis', 'version': '1.0'},
            previous_hash='0',
            hash=''
        )
        
        genesis.hash = self._calculate_hash(genesis)
        self.chain.append(genesis)
    
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> str:
        """Log event to blockchain
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            Block hash
        """
        with self._lock:
            # Create entry
            entry_data = {
                'type': event_type,
                'data': data,
                'timestamp': time.time()
            }
            
            # Add to pending
            self.pending_entries.append(entry_data)
            
            # Mine new block if enough entries
            if len(self.pending_entries) >= 10:
                return self._mine_block()
            
            return ""
    
    def _mine_block(self) -> str:
        """Mine new block
        
        Returns:
            Block hash
        """
        if not self.pending_entries:
            return ""
        
        # Get previous block
        previous_block = self.chain[-1]
        
        # Create new block
        block = BlockchainEntry(
            index=len(self.chain),
            timestamp=time.time(),
            data={
                'entries': self.pending_entries,
                'count': len(self.pending_entries)
            },
            previous_hash=previous_block.hash,
            hash=''
        )
        
        # Mine block
        block.hash, block.nonce = self._proof_of_work(block)
        
        # Add to chain
        self.chain.append(block)
        
        # Clear pending
        self.pending_entries.clear()
        
        logger.info(f"Mined block {block.index} with hash {block.hash[:8]}...")
        
        return block.hash
    
    def _proof_of_work(
        self,
        block: BlockchainEntry
    ) -> Tuple[str, int]:
        """Perform proof of work
        
        Args:
            block: Block to mine
            
        Returns:
            Hash and nonce
        """
        nonce = 0
        
        while True:
            block.nonce = nonce
            hash_value = self._calculate_hash(block)
            
            if hash_value.startswith('0' * self.mining_difficulty):
                return hash_value, nonce
            
            nonce += 1
    
    def _calculate_hash(self, block: BlockchainEntry) -> str:
        """Calculate block hash
        
        Args:
            block: Block to hash
            
        Returns:
            Block hash
        """
        block_data = {
            'index': block.index,
            'timestamp': block.timestamp,
            'data': block.data,
            'previous_hash': block.previous_hash,
            'nonce': block.nonce
        }
        
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def verify_chain(self) -> bool:
        """Verify blockchain integrity
        
        Returns:
            Verification status
        """
        with self._lock:
            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i - 1]
                
                # Check hash
                if current.hash != self._calculate_hash(current):
                    logger.error(f"Invalid hash at block {i}")
                    return False
                
                # Check previous hash
                if current.previous_hash != previous.hash:
                    logger.error(f"Invalid chain at block {i}")
                    return False
                
                # Check proof of work
                if not current.hash.startswith('0' * self.mining_difficulty):
                    logger.error(f"Invalid proof of work at block {i}")
                    return False
            
            return True
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get logged events
        
        Args:
            event_type: Filter by event type
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of events
        """
        events = []
        
        with self._lock:
            for block in self.chain[1:]:  # Skip genesis
                for entry in block.data.get('entries', []):
                    # Apply filters
                    if event_type and entry['type'] != event_type:
                        continue
                    
                    if start_time and entry['timestamp'] < start_time:
                        continue
                    
                    if end_time and entry['timestamp'] > end_time:
                        continue
                    
                    events.append({
                        'block_index': block.index,
                        'block_hash': block.hash,
                        **entry
                    })
        
        return events


class TrustManager:
    """Trust management system"""
    
    def __init__(self):
        """Initialize trust manager"""
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        self.trust_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.trust_relationships: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Trust parameters
        self.decay_factor = 0.95
        self.update_weight = 0.1
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Initialized TrustManager")
    
    def update_trust(
        self,
        entity_id: str,
        interaction_result: float,
        interaction_type: str = "default"
    ):
        """Update trust score based on interaction
        
        Args:
            entity_id: Entity identifier
            interaction_result: Result score (0-1)
            interaction_type: Type of interaction
        """
        with self._lock:
            # Get current trust
            current_trust = self.trust_scores[entity_id]
            
            # Update trust score
            new_trust = (
                current_trust * (1 - self.update_weight) +
                interaction_result * self.update_weight
            )
            
            # Apply bounds
            new_trust = max(0.0, min(1.0, new_trust))
            
            # Update score
            self.trust_scores[entity_id] = new_trust
            
            # Record history
            self.trust_history[entity_id].append({
                'timestamp': datetime.now(),
                'interaction_type': interaction_type,
                'result': interaction_result,
                'new_trust': new_trust
            })
    
    def get_trust_level(self, entity_id: str) -> TrustLevel:
        """Get trust level for entity
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Trust level
        """
        score = self.trust_scores.get(entity_id, 0.5)
        
        if score >= 0.9:
            return TrustLevel.FULL
        elif score >= 0.7:
            return TrustLevel.HIGH
        elif score >= 0.5:
            return TrustLevel.MEDIUM
        elif score >= 0.3:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED
    
    def establish_trust_relationship(
        self,
        entity1: str,
        entity2: str,
        initial_trust: float = 0.5
    ):
        """Establish trust relationship between entities
        
        Args:
            entity1: First entity
            entity2: Second entity
            initial_trust: Initial trust score
        """
        with self._lock:
            self.trust_relationships[entity1][entity2] = initial_trust
            self.trust_relationships[entity2][entity1] = initial_trust
    
    def propagate_trust(
        self,
        source: str,
        target: str,
        recommender: str
    ):
        """Propagate trust through recommendation
        
        Args:
            source: Source entity
            target: Target entity
            recommender: Recommending entity
        """
        with self._lock:
            # Get trust in recommender
            recommender_trust = self.trust_scores.get(recommender, 0.5)
            
            # Get recommender's trust in target
            recommended_trust = self.trust_relationships.get(
                recommender, {}
            ).get(target, 0.5)
            
            # Calculate propagated trust
            propagated_trust = recommender_trust * recommended_trust
            
            # Update relationship
            current = self.trust_relationships[source].get(target, 0.5)
            new_trust = current * 0.7 + propagated_trust * 0.3
            
            self.trust_relationships[source][target] = new_trust
    
    def decay_trust(self):
        """Apply trust decay over time"""
        with self._lock:
            for entity_id in list(self.trust_scores.keys()):
                # Apply decay
                self.trust_scores[entity_id] *= self.decay_factor
                
                # Remove if too low
                if self.trust_scores[entity_id] < 0.1:
                    del self.trust_scores[entity_id]
    
    def get_trusted_entities(
        self,
        min_trust_level: TrustLevel = TrustLevel.MEDIUM
    ) -> List[str]:
        """Get list of trusted entities
        
        Args:
            min_trust_level: Minimum trust level
            
        Returns:
            List of entity IDs
        """
        trusted = []
        
        with self._lock:
            for entity_id, score in self.trust_scores.items():
                level = self.get_trust_level(entity_id)
                if level.value >= min_trust_level.value:
                    trusted.append(entity_id)
        
        return trusted
    
    def get_trust_report(self) -> Dict[str, Any]:
        """Get trust system report
        
        Returns:
            Trust report
        """
        with self._lock:
            trust_distribution = defaultdict(int)
            
            for entity_id in self.trust_scores:
                level = self.get_trust_level(entity_id)
                trust_distribution[level.name] += 1
            
            return {
                'total_entities': len(self.trust_scores),
                'average_trust': sum(self.trust_scores.values()) / len(self.trust_scores) if self.trust_scores else 0.5,
                'distribution': dict(trust_distribution),
                'trusted_count': len(self.get_trusted_entities()),
                'relationships': sum(len(rels) for rels in self.trust_relationships.values())
            }


# Example integrated secure communication system
def create_secure_communication_system(node_id: str) -> Dict[str, Any]:
    """Create integrated secure communication system
    
    Args:
        node_id: Node identifier
        
    Returns:
        Communication system components
    """
    protocol = SecureCommunicationProtocol(node_id)
    verifier = MessageVerifier()
    blockchain = BlockchainLogger()
    trust_manager = TrustManager()
    
    return {
        'protocol': protocol,
        'verifier': verifier,
        'blockchain': blockchain,
        'trust_manager': trust_manager
    }