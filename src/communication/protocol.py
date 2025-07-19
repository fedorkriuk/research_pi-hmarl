"""Core Communication Protocol

This module implements the fundamental communication protocol for
multi-agent coordination with real-world constraints.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import json
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the protocol"""
    # Core messages
    HEARTBEAT = "heartbeat"
    STATUS = "status"
    COMMAND = "command"
    ACK = "acknowledgment"
    
    # Coordination messages
    TASK_ASSIGNMENT = "task_assignment"
    TASK_UPDATE = "task_update"
    COORDINATION = "coordination"
    FORMATION = "formation"
    
    # Data messages
    SENSOR_DATA = "sensor_data"
    TELEMETRY = "telemetry"
    MAP_UPDATE = "map_update"
    DISCOVERY = "discovery"
    
    # Emergency messages
    EMERGENCY = "emergency"
    COLLISION_WARNING = "collision_warning"
    SYSTEM_FAILURE = "system_failure"
    LOW_BATTERY = "low_battery"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    ROUTINE = 4  # Lowest priority


@dataclass
class Message:
    """Communication message structure"""
    msg_id: str
    msg_type: MessageType
    sender_id: int
    receiver_id: Optional[int]  # None for broadcast
    timestamp: float
    priority: MessagePriority
    payload: Dict[str, Any]
    ttl: int = 10  # Time to live (hops)
    requires_ack: bool = False
    encrypted: bool = False
    compressed: bool = False
    checksum: Optional[str] = None
    route: List[int] = field(default_factory=list)


@dataclass
class ProtocolConfig:
    """Communication protocol configuration"""
    # Network parameters (based on real-world constraints)
    max_range: float = 5000.0  # meters (typical drone communication range)
    frequency: float = 2.4e9  # Hz (2.4 GHz band)
    bandwidth: float = 20e6  # Hz (20 MHz channel)
    data_rate: float = 10e6  # bps (10 Mbps)
    
    # Protocol parameters
    heartbeat_interval: float = 1.0  # seconds
    message_timeout: float = 5.0  # seconds
    max_retries: int = 3
    buffer_size: int = 1000
    
    # QoS parameters
    latency_target: float = 0.1  # seconds (100ms)
    packet_loss_threshold: float = 0.05  # 5%
    jitter_threshold: float = 0.02  # 20ms
    
    # Security
    encryption_enabled: bool = True
    authentication_required: bool = True
    
    # Power management
    tx_power: float = 20.0  # dBm (100mW)
    rx_sensitivity: float = -90.0  # dBm


class CommunicationProtocol:
    """Main communication protocol implementation"""
    
    def __init__(
        self,
        agent_id: int,
        config: Optional[ProtocolConfig] = None
    ):
        """Initialize communication protocol
        
        Args:
            agent_id: Agent identifier
            config: Protocol configuration
        """
        self.agent_id = agent_id
        self.config = config or ProtocolConfig()
        
        # Message handling
        self.message_counter = 0
        self.pending_acks = {}
        self.received_messages = set()
        self.message_buffer = deque(maxlen=self.config.buffer_size)
        
        # Network state
        self.neighbors = {}  # agent_id -> last_seen
        self.link_quality = {}  # agent_id -> quality metrics
        self.routing_table = {}  # destination -> next_hop
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'avg_latency': 0.0,
            'packet_loss_rate': 0.0
        }
        
        # Protocol state
        self.running = True
        self.last_heartbeat = 0
        
        logger.info(f"Initialized communication protocol for agent {agent_id}")
    
    def create_message(
        self,
        msg_type: MessageType,
        receiver_id: Optional[int],
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.MEDIUM,
        requires_ack: bool = False
    ) -> Message:
        """Create a new message
        
        Args:
            msg_type: Type of message
            receiver_id: Receiver ID (None for broadcast)
            payload: Message payload
            priority: Message priority
            requires_ack: Whether acknowledgment is required
            
        Returns:
            Created message
        """
        self.message_counter += 1
        msg_id = f"{self.agent_id}_{self.message_counter}_{int(time.time() * 1000)}"
        
        message = Message(
            msg_id=msg_id,
            msg_type=msg_type,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            timestamp=time.time(),
            priority=priority,
            payload=payload,
            requires_ack=requires_ack,
            encrypted=self.config.encryption_enabled
        )
        
        # Calculate checksum
        message.checksum = self._calculate_checksum(message)
        
        return message
    
    def send_message(
        self,
        message: Message,
        channel: Any = None
    ) -> bool:
        """Send a message
        
        Args:
            message: Message to send
            channel: Communication channel
            
        Returns:
            Success status
        """
        try:
            # Check if receiver is reachable
            if message.receiver_id is not None:
                if not self._is_reachable(message.receiver_id):
                    # Try multi-hop routing
                    return self._route_message(message)
            
            # Apply QoS
            if not self._apply_qos(message):
                self.stats['messages_dropped'] += 1
                return False
            
            # Compress if needed
            if self._should_compress(message):
                message = self._compress_message(message)
            
            # Encrypt if enabled
            if message.encrypted:
                message = self._encrypt_message(message)
            
            # Simulate transmission
            transmission_success = self._transmit(message, channel)
            
            if transmission_success:
                self.stats['messages_sent'] += 1
                self.stats['bytes_sent'] += self._estimate_message_size(message)
                
                # Track acknowledgments
                if message.requires_ack:
                    self.pending_acks[message.msg_id] = {
                        'message': message,
                        'timestamp': time.time(),
                        'retries': 0
                    }
                
                return True
            else:
                self.stats['messages_dropped'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def receive_message(
        self,
        data: bytes,
        channel: Any = None
    ) -> Optional[Message]:
        """Receive and process a message
        
        Args:
            data: Raw message data
            channel: Communication channel
            
        Returns:
            Received message or None
        """
        try:
            # Deserialize message
            message = self._deserialize_message(data)
            
            # Check if already received (duplicate detection)
            if message.msg_id in self.received_messages:
                return None
            
            # Verify checksum
            if not self._verify_checksum(message):
                logger.warning(f"Checksum verification failed for message {message.msg_id}")
                return None
            
            # Decrypt if needed
            if message.encrypted:
                message = self._decrypt_message(message)
            
            # Decompress if needed
            if message.compressed:
                message = self._decompress_message(message)
            
            # Update statistics
            self.stats['messages_received'] += 1
            self.stats['bytes_received'] += len(data)
            self.received_messages.add(message.msg_id)
            
            # Update neighbor information
            self._update_neighbor_info(message.sender_id, channel)
            
            # Handle acknowledgments
            if message.msg_type == MessageType.ACK:
                self._handle_acknowledgment(message)
                return None
            
            # Send acknowledgment if required
            if message.requires_ack:
                self._send_acknowledgment(message)
            
            # Buffer message
            self.message_buffer.append(message)
            
            # Check if message needs forwarding
            if message.receiver_id != self.agent_id and message.receiver_id is not None:
                self._forward_message(message)
                return None
            
            return message
            
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    def process_heartbeat(self):
        """Send periodic heartbeat messages"""
        current_time = time.time()
        
        if current_time - self.last_heartbeat >= self.config.heartbeat_interval:
            # Create heartbeat message
            heartbeat = self.create_message(
                msg_type=MessageType.HEARTBEAT,
                receiver_id=None,  # Broadcast
                payload={
                    'position': [0, 0, 0],  # Would be actual position
                    'status': 'active',
                    'battery': 0.8,
                    'neighbors': list(self.neighbors.keys())
                },
                priority=MessagePriority.LOW
            )
            
            # Send heartbeat
            self.send_message(heartbeat)
            self.last_heartbeat = current_time
            
            # Clean up old neighbors
            self._cleanup_neighbors()
    
    def _is_reachable(self, agent_id: int) -> bool:
        """Check if agent is directly reachable
        
        Args:
            agent_id: Target agent ID
            
        Returns:
            Reachability status
        """
        if agent_id not in self.neighbors:
            return False
        
        # Check if neighbor is still active
        last_seen = self.neighbors[agent_id]
        if time.time() - last_seen > self.config.heartbeat_interval * 3:
            return False
        
        # Check link quality
        if agent_id in self.link_quality:
            quality = self.link_quality[agent_id]
            if quality.get('packet_loss', 0) > self.config.packet_loss_threshold:
                return False
        
        return True
    
    def _route_message(self, message: Message) -> bool:
        """Route message through multi-hop path
        
        Args:
            message: Message to route
            
        Returns:
            Success status
        """
        if message.receiver_id not in self.routing_table:
            # No route available
            return False
        
        # Get next hop
        next_hop = self.routing_table[message.receiver_id]
        
        # Update route in message
        message.route.append(self.agent_id)
        
        # Decrease TTL
        message.ttl -= 1
        if message.ttl <= 0:
            return False
        
        # Forward to next hop
        forwarded_msg = Message(
            msg_id=message.msg_id,
            msg_type=message.msg_type,
            sender_id=message.sender_id,
            receiver_id=next_hop,
            timestamp=message.timestamp,
            priority=message.priority,
            payload=message.payload,
            ttl=message.ttl,
            requires_ack=message.requires_ack,
            encrypted=message.encrypted,
            compressed=message.compressed,
            checksum=message.checksum,
            route=message.route
        )
        
        return self.send_message(forwarded_msg)
    
    def _apply_qos(self, message: Message) -> bool:
        """Apply Quality of Service rules
        
        Args:
            message: Message to send
            
        Returns:
            Whether message should be sent
        """
        # Priority-based queueing would be implemented here
        # For now, always allow critical messages
        if message.priority == MessagePriority.CRITICAL:
            return True
        
        # Check bandwidth constraints
        # Simplified - would implement token bucket or similar
        return True
    
    def _should_compress(self, message: Message) -> bool:
        """Determine if message should be compressed
        
        Args:
            message: Message to check
            
        Returns:
            Whether to compress
        """
        # Compress large payloads
        payload_size = len(json.dumps(message.payload))
        return payload_size > 1024  # 1KB threshold
    
    def _compress_message(self, message: Message) -> Message:
        """Compress message payload
        
        Args:
            message: Message to compress
            
        Returns:
            Compressed message
        """
        # Simplified compression - would use zlib or similar
        message.compressed = True
        return message
    
    def _encrypt_message(self, message: Message) -> Message:
        """Encrypt message
        
        Args:
            message: Message to encrypt
            
        Returns:
            Encrypted message
        """
        # Simplified encryption - would use AES or similar
        message.encrypted = True
        return message
    
    def _decrypt_message(self, message: Message) -> Message:
        """Decrypt message
        
        Args:
            message: Message to decrypt
            
        Returns:
            Decrypted message
        """
        # Simplified decryption
        return message
    
    def _decompress_message(self, message: Message) -> Message:
        """Decompress message
        
        Args:
            message: Message to decompress
            
        Returns:
            Decompressed message
        """
        # Simplified decompression
        return message
    
    def _calculate_checksum(self, message: Message) -> str:
        """Calculate message checksum
        
        Args:
            message: Message
            
        Returns:
            Checksum string
        """
        # Create checksum from key fields
        checksum_data = f"{message.msg_id}{message.msg_type.value}{message.sender_id}"
        checksum_data += f"{message.receiver_id}{message.timestamp}"
        checksum_data += json.dumps(message.payload, sort_keys=True)
        
        return hashlib.sha256(checksum_data.encode()).hexdigest()[:16]
    
    def _verify_checksum(self, message: Message) -> bool:
        """Verify message checksum
        
        Args:
            message: Message to verify
            
        Returns:
            Verification status
        """
        expected = self._calculate_checksum(message)
        return message.checksum == expected
    
    def _estimate_message_size(self, message: Message) -> int:
        """Estimate message size in bytes
        
        Args:
            message: Message
            
        Returns:
            Estimated size
        """
        # Simplified estimation
        base_size = 100  # Header overhead
        payload_size = len(json.dumps(message.payload))
        return base_size + payload_size
    
    def _transmit(self, message: Message, channel: Any) -> bool:
        """Simulate message transmission
        
        Args:
            message: Message to transmit
            channel: Communication channel
            
        Returns:
            Success status
        """
        # Simulate transmission with success probability based on link quality
        if message.receiver_id and message.receiver_id in self.link_quality:
            quality = self.link_quality[message.receiver_id]
            success_prob = 1.0 - quality.get('packet_loss', 0)
            return np.random.random() < success_prob
        
        return True  # Broadcast always succeeds locally
    
    def _deserialize_message(self, data: bytes) -> Message:
        """Deserialize message from bytes
        
        Args:
            data: Raw message data
            
        Returns:
            Deserialized message
        """
        # Simplified deserialization
        # In practice would use protobuf or similar
        msg_dict = json.loads(data.decode())
        
        return Message(
            msg_id=msg_dict['msg_id'],
            msg_type=MessageType(msg_dict['msg_type']),
            sender_id=msg_dict['sender_id'],
            receiver_id=msg_dict['receiver_id'],
            timestamp=msg_dict['timestamp'],
            priority=MessagePriority(msg_dict['priority']),
            payload=msg_dict['payload'],
            ttl=msg_dict.get('ttl', 10),
            requires_ack=msg_dict.get('requires_ack', False),
            encrypted=msg_dict.get('encrypted', False),
            compressed=msg_dict.get('compressed', False),
            checksum=msg_dict.get('checksum'),
            route=msg_dict.get('route', [])
        )
    
    def _update_neighbor_info(self, sender_id: int, channel: Any):
        """Update neighbor information
        
        Args:
            sender_id: Sender agent ID
            channel: Communication channel
        """
        self.neighbors[sender_id] = time.time()
        
        # Update link quality metrics
        if sender_id not in self.link_quality:
            self.link_quality[sender_id] = {
                'rssi': -70.0,  # dBm
                'packet_loss': 0.01,
                'latency': 0.05,
                'jitter': 0.01
            }
    
    def _handle_acknowledgment(self, ack_message: Message):
        """Handle received acknowledgment
        
        Args:
            ack_message: Acknowledgment message
        """
        original_msg_id = ack_message.payload.get('original_msg_id')
        
        if original_msg_id in self.pending_acks:
            # Calculate round-trip time
            send_time = self.pending_acks[original_msg_id]['timestamp']
            rtt = time.time() - send_time
            
            # Update latency statistics
            self.stats['avg_latency'] = (
                0.9 * self.stats['avg_latency'] + 0.1 * rtt
            )
            
            # Remove from pending
            del self.pending_acks[original_msg_id]
    
    def _send_acknowledgment(self, message: Message):
        """Send acknowledgment for received message
        
        Args:
            message: Message to acknowledge
        """
        ack = self.create_message(
            msg_type=MessageType.ACK,
            receiver_id=message.sender_id,
            payload={'original_msg_id': message.msg_id},
            priority=MessagePriority.HIGH
        )
        self.send_message(ack)
    
    def _forward_message(self, message: Message):
        """Forward message to next hop
        
        Args:
            message: Message to forward
        """
        if message.receiver_id in self.routing_table:
            next_hop = self.routing_table[message.receiver_id]
            message.route.append(self.agent_id)
            message.ttl -= 1
            
            if message.ttl > 0:
                # Create forwarded message
                forward_msg = Message(
                    msg_id=message.msg_id,
                    msg_type=message.msg_type,
                    sender_id=message.sender_id,
                    receiver_id=next_hop,
                    timestamp=message.timestamp,
                    priority=message.priority,
                    payload=message.payload,
                    ttl=message.ttl,
                    requires_ack=message.requires_ack,
                    encrypted=message.encrypted,
                    compressed=message.compressed,
                    checksum=message.checksum,
                    route=message.route
                )
                self.send_message(forward_msg)
    
    def _cleanup_neighbors(self):
        """Remove inactive neighbors"""
        current_time = time.time()
        timeout = self.config.heartbeat_interval * 3
        
        inactive = [
            agent_id for agent_id, last_seen in self.neighbors.items()
            if current_time - last_seen > timeout
        ]
        
        for agent_id in inactive:
            del self.neighbors[agent_id]
            if agent_id in self.link_quality:
                del self.link_quality[agent_id]
            
            # Remove routes through inactive neighbor
            routes_to_remove = [
                dest for dest, next_hop in self.routing_table.items()
                if next_hop == agent_id
            ]
            for dest in routes_to_remove:
                del self.routing_table[dest]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            'active_neighbors': len(self.neighbors),
            'pending_acks': len(self.pending_acks),
            'buffer_usage': len(self.message_buffer) / self.config.buffer_size,
            'avg_link_quality': np.mean([
                1.0 - q.get('packet_loss', 0)
                for q in self.link_quality.values()
            ]) if self.link_quality else 1.0
        }