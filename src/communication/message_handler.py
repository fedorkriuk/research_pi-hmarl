"""Message Handling and Processing

This module implements message handling, queueing, filtering,
compression, and encryption for multi-agent communication.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import heapq
import threading
import queue
import time
import json
import zlib
import hashlib
from cryptography.fernet import Fernet
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueuedMessage:
    """Message with priority queue support"""
    priority: int
    timestamp: float
    message: Any
    retry_count: int = 0
    
    def __lt__(self, other):
        """Compare by priority (lower value = higher priority)"""
        return self.priority < other.priority


class MessageQueue:
    """Priority-based message queue with QoS"""
    
    def __init__(
        self,
        max_size: int = 10000,
        enable_qos: bool = True
    ):
        """Initialize message queue
        
        Args:
            max_size: Maximum queue size
            enable_qos: Enable quality of service
        """
        self.max_size = max_size
        self.enable_qos = enable_qos
        
        # Priority queues for different message types
        self.queues = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'routine': []
        }
        
        # Queue statistics
        self.stats = {
            'enqueued': 0,
            'dequeued': 0,
            'dropped': 0,
            'expired': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # QoS parameters
        self.qos_params = {
            'critical': {'ttl': 60.0, 'max_retries': 5},
            'high': {'ttl': 30.0, 'max_retries': 3},
            'medium': {'ttl': 15.0, 'max_retries': 2},
            'low': {'ttl': 10.0, 'max_retries': 1},
            'routine': {'ttl': 5.0, 'max_retries': 0}
        }
    
    def enqueue(
        self,
        message: Any,
        priority: str = 'medium'
    ) -> bool:
        """Add message to queue
        
        Args:
            message: Message to enqueue
            priority: Priority level
            
        Returns:
            Success status
        """
        with self.lock:
            # Check queue size
            total_size = sum(len(q) for q in self.queues.values())
            if total_size >= self.max_size:
                # Drop lowest priority messages if needed
                if not self._make_space(priority):
                    self.stats['dropped'] += 1
                    return False
            
            # Create queued message
            queued_msg = QueuedMessage(
                priority=self._priority_to_int(priority),
                timestamp=time.time(),
                message=message
            )
            
            # Add to appropriate queue
            heapq.heappush(self.queues[priority], queued_msg)
            self.stats['enqueued'] += 1
            
            return True
    
    def dequeue(self) -> Optional[Any]:
        """Get highest priority message
        
        Returns:
            Message or None
        """
        with self.lock:
            # Check queues in priority order
            for priority in ['critical', 'high', 'medium', 'low', 'routine']:
                if self.queues[priority]:
                    # Clean expired messages
                    self._clean_expired(priority)
                    
                    if self.queues[priority]:
                        queued_msg = heapq.heappop(self.queues[priority])
                        self.stats['dequeued'] += 1
                        return queued_msg.message
            
            return None
    
    def peek(self, priority: Optional[str] = None) -> Optional[Any]:
        """Peek at next message without removing
        
        Args:
            priority: Specific priority to check
            
        Returns:
            Message or None
        """
        with self.lock:
            if priority:
                if self.queues[priority]:
                    return self.queues[priority][0].message
            else:
                # Check all queues
                for p in ['critical', 'high', 'medium', 'low', 'routine']:
                    if self.queues[p]:
                        return self.queues[p][0].message
            
            return None
    
    def size(self, priority: Optional[str] = None) -> int:
        """Get queue size
        
        Args:
            priority: Specific priority
            
        Returns:
            Queue size
        """
        with self.lock:
            if priority:
                return len(self.queues.get(priority, []))
            else:
                return sum(len(q) for q in self.queues.values())
    
    def _priority_to_int(self, priority: str) -> int:
        """Convert priority string to integer
        
        Args:
            priority: Priority level
            
        Returns:
            Priority integer
        """
        priority_map = {
            'critical': 0,
            'high': 1,
            'medium': 2,
            'low': 3,
            'routine': 4
        }
        return priority_map.get(priority, 2)
    
    def _make_space(self, incoming_priority: str) -> bool:
        """Make space for incoming message
        
        Args:
            incoming_priority: Priority of incoming message
            
        Returns:
            Whether space was made
        """
        incoming_int = self._priority_to_int(incoming_priority)
        
        # Try to drop lower priority messages
        for priority in ['routine', 'low', 'medium', 'high', 'critical']:
            if self._priority_to_int(priority) > incoming_int:
                if self.queues[priority]:
                    # Drop oldest message
                    heapq.heappop(self.queues[priority])
                    self.stats['dropped'] += 1
                    return True
        
        return False
    
    def _clean_expired(self, priority: str):
        """Remove expired messages from queue
        
        Args:
            priority: Priority level
        """
        if not self.enable_qos:
            return
        
        current_time = time.time()
        ttl = self.qos_params[priority]['ttl']
        
        # Remove expired messages
        cleaned = []
        for msg in self.queues[priority]:
            if current_time - msg.timestamp < ttl:
                cleaned.append(msg)
            else:
                self.stats['expired'] += 1
        
        self.queues[priority] = cleaned
        heapq.heapify(self.queues[priority])


class MessageFilter:
    """Filters messages based on rules"""
    
    def __init__(self):
        """Initialize message filter"""
        self.filters = []
        self.blacklist = set()
        self.whitelist = set()
        self.rate_limiters = {}
    
    def add_filter(
        self,
        name: str,
        filter_func: Callable[[Any], bool],
        priority: int = 0
    ):
        """Add filter rule
        
        Args:
            name: Filter name
            filter_func: Filter function
            priority: Filter priority
        """
        self.filters.append({
            'name': name,
            'func': filter_func,
            'priority': priority
        })
        
        # Sort by priority
        self.filters.sort(key=lambda x: x['priority'])
    
    def add_sender_filter(
        self,
        sender_id: int,
        action: str = 'blacklist'
    ):
        """Add sender-based filter
        
        Args:
            sender_id: Sender ID
            action: 'blacklist' or 'whitelist'
        """
        if action == 'blacklist':
            self.blacklist.add(sender_id)
        elif action == 'whitelist':
            self.whitelist.add(sender_id)
    
    def add_rate_limit(
        self,
        sender_id: int,
        max_rate: float,
        window: float = 60.0
    ):
        """Add rate limiting for sender
        
        Args:
            sender_id: Sender ID
            max_rate: Maximum messages per window
            window: Time window in seconds
        """
        self.rate_limiters[sender_id] = {
            'max_rate': max_rate,
            'window': window,
            'timestamps': deque()
        }
    
    def should_accept(self, message: Any) -> bool:
        """Check if message should be accepted
        
        Args:
            message: Message to check
            
        Returns:
            Whether to accept message
        """
        # Check blacklist/whitelist
        sender_id = getattr(message, 'sender_id', None)
        if sender_id:
            if sender_id in self.blacklist:
                return False
            if self.whitelist and sender_id not in self.whitelist:
                return False
        
        # Check rate limits
        if sender_id and sender_id in self.rate_limiters:
            if not self._check_rate_limit(sender_id):
                return False
        
        # Apply custom filters
        for filter_rule in self.filters:
            if not filter_rule['func'](message):
                logger.debug(f"Message rejected by filter: {filter_rule['name']}")
                return False
        
        return True
    
    def _check_rate_limit(self, sender_id: int) -> bool:
        """Check rate limit for sender
        
        Args:
            sender_id: Sender ID
            
        Returns:
            Whether within rate limit
        """
        limiter = self.rate_limiters[sender_id]
        current_time = time.time()
        
        # Clean old timestamps
        cutoff = current_time - limiter['window']
        while limiter['timestamps'] and limiter['timestamps'][0] < cutoff:
            limiter['timestamps'].popleft()
        
        # Check rate
        if len(limiter['timestamps']) >= limiter['max_rate']:
            return False
        
        # Add timestamp
        limiter['timestamps'].append(current_time)
        return True


class MessageCompressor:
    """Handles message compression"""
    
    def __init__(
        self,
        compression_threshold: int = 1024,
        compression_level: int = 6
    ):
        """Initialize message compressor
        
        Args:
            compression_threshold: Minimum size for compression
            compression_level: zlib compression level (1-9)
        """
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level
        
        # Compression statistics
        self.stats = {
            'messages_compressed': 0,
            'bytes_saved': 0,
            'compression_ratio': 0.0
        }
    
    def compress(self, data: bytes) -> Tuple[bytes, bool]:
        """Compress data if beneficial
        
        Args:
            data: Data to compress
            
        Returns:
            (compressed_data, was_compressed)
        """
        if len(data) < self.compression_threshold:
            return data, False
        
        try:
            compressed = zlib.compress(data, self.compression_level)
            
            # Only use compression if it saves space
            if len(compressed) < len(data) * 0.9:  # 10% savings threshold
                self.stats['messages_compressed'] += 1
                self.stats['bytes_saved'] += len(data) - len(compressed)
                
                # Update compression ratio
                if self.stats['messages_compressed'] > 0:
                    self.stats['compression_ratio'] = (
                        self.stats['bytes_saved'] / 
                        (self.stats['bytes_saved'] + 
                         self.stats['messages_compressed'] * len(compressed))
                    )
                
                return compressed, True
            else:
                return data, False
                
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return data, False
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress data
        
        Args:
            data: Compressed data
            
        Returns:
            Decompressed data
        """
        try:
            return zlib.decompress(data)
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            return data


class MessageEncryption:
    """Handles message encryption and authentication"""
    
    def __init__(self):
        """Initialize encryption handler"""
        # Generate keys
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        
        # Key management
        self.peer_keys = {}  # agent_id -> key
        self.session_keys = {}  # session_id -> key
        
        # Authentication
        self.auth_tokens = {}
        self.nonce_cache = deque(maxlen=1000)
    
    def encrypt(
        self,
        data: bytes,
        recipient_id: Optional[int] = None
    ) -> bytes:
        """Encrypt data
        
        Args:
            data: Data to encrypt
            recipient_id: Recipient agent ID
            
        Returns:
            Encrypted data
        """
        try:
            # Use recipient-specific key if available
            if recipient_id and recipient_id in self.peer_keys:
                cipher = Fernet(self.peer_keys[recipient_id])
                return cipher.encrypt(data)
            else:
                return self.cipher_suite.encrypt(data)
                
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt(
        self,
        data: bytes,
        sender_id: Optional[int] = None
    ) -> bytes:
        """Decrypt data
        
        Args:
            data: Encrypted data
            sender_id: Sender agent ID
            
        Returns:
            Decrypted data
        """
        try:
            # Use sender-specific key if available
            if sender_id and sender_id in self.peer_keys:
                cipher = Fernet(self.peer_keys[sender_id])
                return cipher.decrypt(data)
            else:
                return self.cipher_suite.decrypt(data)
                
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    def generate_auth_token(
        self,
        agent_id: int,
        validity: float = 3600.0
    ) -> str:
        """Generate authentication token
        
        Args:
            agent_id: Agent ID
            validity: Token validity in seconds
            
        Returns:
            Authentication token
        """
        # Create token data
        token_data = {
            'agent_id': agent_id,
            'timestamp': time.time(),
            'validity': validity,
            'nonce': hashlib.sha256(
                f"{agent_id}{time.time()}{np.random.random()}".encode()
            ).hexdigest()[:16]
        }
        
        # Encrypt token
        token_bytes = json.dumps(token_data).encode()
        encrypted_token = self.cipher_suite.encrypt(token_bytes)
        
        # Store token
        self.auth_tokens[agent_id] = token_data
        
        return encrypted_token.decode()
    
    def verify_auth_token(self, token: str) -> Optional[int]:
        """Verify authentication token
        
        Args:
            token: Authentication token
            
        Returns:
            Agent ID if valid, None otherwise
        """
        try:
            # Decrypt token
            decrypted = self.cipher_suite.decrypt(token.encode())
            token_data = json.loads(decrypted.decode())
            
            # Check validity
            current_time = time.time()
            if current_time - token_data['timestamp'] > token_data['validity']:
                return None
            
            # Check nonce
            nonce = token_data['nonce']
            if nonce in self.nonce_cache:
                return None  # Replay attack
            
            self.nonce_cache.append(nonce)
            
            return token_data['agent_id']
            
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def exchange_keys(
        self,
        peer_id: int,
        peer_public_key: bytes
    ) -> bytes:
        """Exchange keys with peer
        
        Args:
            peer_id: Peer agent ID
            peer_public_key: Peer's public key
            
        Returns:
            Own public key
        """
        # Simplified key exchange
        # In practice would use Diffie-Hellman or similar
        shared_key = Fernet.generate_key()
        self.peer_keys[peer_id] = shared_key
        
        return self.key  # Return own key


class MessageHandler:
    """Main message handling system"""
    
    def __init__(
        self,
        agent_id: int,
        enable_encryption: bool = True,
        enable_compression: bool = True
    ):
        """Initialize message handler
        
        Args:
            agent_id: Agent identifier
            enable_encryption: Enable encryption
            enable_compression: Enable compression
        """
        self.agent_id = agent_id
        self.enable_encryption = enable_encryption
        self.enable_compression = enable_compression
        
        # Components
        self.queue = MessageQueue()
        self.filter = MessageFilter()
        self.compressor = MessageCompressor()
        self.encryption = MessageEncryption() if enable_encryption else None
        
        # Message processing
        self.handlers = {}  # message_type -> handler
        self.preprocessors = []
        self.postprocessors = []
        
        # Threading
        self.processing_thread = None
        self.running = False
        
        logger.info(f"Initialized message handler for agent {agent_id}")
    
    def register_handler(
        self,
        message_type: str,
        handler: Callable[[Any], Any]
    ):
        """Register message type handler
        
        Args:
            message_type: Message type
            handler: Handler function
        """
        self.handlers[message_type] = handler
    
    def add_preprocessor(self, processor: Callable[[Any], Any]):
        """Add message preprocessor
        
        Args:
            processor: Preprocessor function
        """
        self.preprocessors.append(processor)
    
    def add_postprocessor(self, processor: Callable[[Any], Any]):
        """Add message postprocessor
        
        Args:
            processor: Postprocessor function
        """
        self.postprocessors.append(processor)
    
    def process_incoming(self, raw_data: bytes) -> Optional[Any]:
        """Process incoming message
        
        Args:
            raw_data: Raw message data
            
        Returns:
            Processed message or None
        """
        try:
            # Decrypt if needed
            if self.enable_encryption and self.encryption:
                data = self.encryption.decrypt(raw_data)
            else:
                data = raw_data
            
            # Decompress if needed
            if self.enable_compression:
                data = self.compressor.decompress(data)
            
            # Deserialize
            message = json.loads(data.decode())
            
            # Apply filters
            if not self.filter.should_accept(message):
                return None
            
            # Apply preprocessors
            for processor in self.preprocessors:
                message = processor(message)
                if message is None:
                    return None
            
            # Queue for processing
            priority = message.get('priority', 'medium')
            self.queue.enqueue(message, priority)
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing incoming message: {e}")
            return None
    
    def prepare_outgoing(
        self,
        message: Dict[str, Any],
        recipient_id: Optional[int] = None
    ) -> bytes:
        """Prepare outgoing message
        
        Args:
            message: Message to send
            recipient_id: Recipient ID
            
        Returns:
            Prepared message data
        """
        try:
            # Apply postprocessors
            for processor in self.postprocessors:
                message = processor(message)
            
            # Serialize
            data = json.dumps(message).encode()
            
            # Compress if beneficial
            if self.enable_compression:
                data, compressed = self.compressor.compress(data)
                message['compressed'] = compressed
            
            # Encrypt
            if self.enable_encryption and self.encryption:
                data = self.encryption.encrypt(data, recipient_id)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing outgoing message: {e}")
            raise
    
    def start_processing(self):
        """Start message processing thread"""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop_processing(self):
        """Stop message processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get next message
                message = self.queue.dequeue()
                
                if message:
                    # Get handler
                    msg_type = message.get('type', 'unknown')
                    handler = self.handlers.get(msg_type)
                    
                    if handler:
                        # Process message
                        handler(message)
                    else:
                        logger.warning(f"No handler for message type: {msg_type}")
                else:
                    # No messages, sleep briefly
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'queue_stats': self.queue.stats,
            'queue_sizes': {
                priority: self.queue.size(priority)
                for priority in ['critical', 'high', 'medium', 'low', 'routine']
            },
            'compression_stats': self.compressor.stats if self.enable_compression else {},
            'filter_count': len(self.filter.filters),
            'handler_count': len(self.handlers)
        }