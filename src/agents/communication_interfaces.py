"""Communication Interfaces for Hierarchical Agents

This module implements communication protocols between hierarchy levels
with real network latency constraints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


class MessagePassing(nn.Module):
    """Message passing between agents with real network constraints"""
    
    def __init__(
        self,
        agent_id: int,
        hidden_dim: int,
        message_dim: int = 64,
        max_agents: int = 20,
        latency_ms: float = 50.0,  # Real WiFi latency
        bandwidth_kbps: float = 1000.0,  # Real bandwidth limit
        dropout_rate: float = 0.05  # Packet loss rate
    ):
        """Initialize message passing
        
        Args:
            agent_id: Agent identifier
            hidden_dim: Hidden state dimension
            message_dim: Message dimension
            max_agents: Maximum number of agents
            latency_ms: Network latency in milliseconds
            bandwidth_kbps: Bandwidth in kilobits per second
            dropout_rate: Message dropout rate (packet loss)
        """
        super().__init__()
        
        self.agent_id = agent_id
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.max_agents = max_agents
        self.latency_ms = latency_ms
        self.bandwidth_kbps = bandwidth_kbps
        self.dropout_rate = dropout_rate
        
        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(hidden_dim, message_dim),
            nn.ReLU(),
            nn.LayerNorm(message_dim),
            nn.Linear(message_dim, message_dim)
        )
        
        # Message aggregator
        self.message_aggregator = nn.Sequential(
            nn.Linear(message_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for message selection
        self.attention = nn.MultiheadAttention(
            message_dim, num_heads=4, batch_first=True
        )
        
        # Message buffer for latency simulation
        self.message_buffer = MessageBuffer(latency_ms)
        
        # Bandwidth tracker
        self.bandwidth_tracker = BandwidthTracker(bandwidth_kbps)
        
        logger.info(f"Initialized MessagePassing for agent {agent_id}")
    
    def create_message(
        self,
        state: torch.Tensor,
        priority: int = 1
    ) -> torch.Tensor:
        """Create message from current state
        
        Args:
            state: Current agent state
            priority: Message priority (1-5)
            
        Returns:
            Encoded message
        """
        # Encode state into message
        message = self.message_encoder(state)
        
        # Add metadata (agent ID, priority, timestamp)
        metadata = torch.tensor([
            self.agent_id / self.max_agents,  # Normalized agent ID
            priority / 5.0,  # Normalized priority
            time.time() % 1000  # Timestamp
        ], device=message.device)
        
        # Check bandwidth constraints
        message_size_bits = message.numel() * 32  # 32 bits per float
        if not self.bandwidth_tracker.can_send(message_size_bits):
            # Return zero message if bandwidth exceeded
            return torch.zeros_like(message)
        
        return message
    
    def process_messages(
        self,
        own_state: torch.Tensor,
        messages: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Process received messages
        
        Args:
            own_state: Agent's own state
            messages: Dictionary of messages from other agents
            
        Returns:
            Updated state
        """
        if not messages:
            return own_state
        
        # Apply latency and packet loss
        available_messages = {}
        for agent_id, message in messages.items():
            # Check if message has arrived (latency)
            if self.message_buffer.is_available(agent_id):
                # Simulate packet loss
                if torch.rand(1).item() > self.dropout_rate:
                    available_messages[agent_id] = message
        
        if not available_messages:
            return own_state
        
        # Stack messages for attention
        message_stack = torch.stack(list(available_messages.values()))
        if message_stack.dim() == 2:
            message_stack = message_stack.unsqueeze(0)
        
        # Self-attention over messages
        own_message = self.message_encoder(own_state).unsqueeze(0).unsqueeze(0)
        attended_messages, _ = self.attention(
            own_message, message_stack, message_stack
        )
        
        # Aggregate with own state
        aggregated = self.message_aggregator(
            torch.cat([own_state, attended_messages.squeeze()], dim=-1)
        )
        
        # Residual connection
        updated_state = own_state + 0.5 * aggregated
        
        return updated_state


class CommandInterface(nn.Module):
    """Interface for meta-controller to execution policy commands"""
    
    def __init__(
        self,
        command_dim: int = 32,
        hidden_dim: int = 128
    ):
        """Initialize command interface
        
        Args:
            command_dim: Command vector dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.command_dim = command_dim
        
        # Command encoder
        self.command_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, command_dim),
            nn.Tanh()
        )
        
        # Command decoder
        self.command_decoder = nn.Sequential(
            nn.Linear(command_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Command types based on real operations
        self.command_types = {
            0: "maintain_position",
            1: "move_to_waypoint",
            2: "follow_trajectory",
            3: "join_formation",
            4: "search_pattern",
            5: "return_to_base",
            6: "emergency_stop",
            7: "change_altitude"
        }
    
    def encode_command(
        self,
        strategic_state: torch.Tensor,
        command_type: int,
        parameters: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode high-level command
        
        Args:
            strategic_state: Strategic level state
            command_type: Type of command
            parameters: Optional command parameters
            
        Returns:
            Encoded command
        """
        # Base command from strategic state
        base_command = self.command_encoder(strategic_state)
        
        # Add command type encoding
        type_encoding = F.one_hot(
            torch.tensor(command_type), 
            num_classes=len(self.command_types)
        ).float()
        
        # Combine with parameters if provided
        if parameters is not None:
            combined = torch.cat([
                base_command[:self.command_dim-len(type_encoding)-parameters.shape[0]],
                type_encoding,
                parameters
            ])
            return combined[:self.command_dim]
        else:
            combined = torch.cat([
                base_command[:self.command_dim-len(type_encoding)],
                type_encoding
            ])
            return combined
    
    def decode_command(
        self,
        command: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Decode command for execution
        
        Args:
            command: Encoded command
            
        Returns:
            Decoded features and command type
        """
        # Extract command type
        type_logits = command[-len(self.command_types):]
        command_type = torch.argmax(type_logits).item()
        
        # Decode features
        features = self.command_decoder(command)
        
        return features, command_type


class FeedbackLoop(nn.Module):
    """Feedback from execution to meta-controller"""
    
    def __init__(
        self,
        hidden_dim: int,
        feedback_dim: int = 32,
        history_len: int = 10
    ):
        """Initialize feedback loop
        
        Args:
            hidden_dim: Hidden state dimension
            feedback_dim: Feedback vector dimension
            history_len: Length of feedback history
        """
        super().__init__()
        
        self.feedback_dim = feedback_dim
        self.history_len = history_len
        
        # Feedback encoder
        self.feedback_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, feedback_dim),  # +1 for value
            nn.ReLU(),
            nn.Linear(feedback_dim, feedback_dim)
        )
        
        # Feedback aggregator (LSTM for temporal modeling)
        self.feedback_lstm = nn.LSTM(
            feedback_dim, hidden_dim,
            batch_first=True
        )
        
        # Feedback history
        self.feedback_history = deque(maxlen=history_len)
    
    def update(
        self,
        execution_state: torch.Tensor,
        action: torch.Tensor,
        value: float
    ):
        """Update feedback with execution results
        
        Args:
            execution_state: Current execution state
            action: Action taken
            value: Value estimate
        """
        # Encode feedback
        value_tensor = torch.tensor([value], device=execution_state.device)
        feedback_input = torch.cat([execution_state, value_tensor])
        feedback = self.feedback_encoder(feedback_input)
        
        # Add to history
        self.feedback_history.append(feedback)
    
    def get_feedback_summary(self) -> torch.Tensor:
        """Get aggregated feedback for meta-controller
        
        Returns:
            Feedback summary
        """
        if not self.feedback_history:
            return torch.zeros(self.feedback_dim)
        
        # Stack feedback history
        feedback_stack = torch.stack(list(self.feedback_history))
        if feedback_stack.dim() == 2:
            feedback_stack = feedback_stack.unsqueeze(0)
        
        # Process through LSTM
        _, (hidden, _) = self.feedback_lstm(feedback_stack)
        
        return hidden.squeeze(0)


class StateSharing(nn.Module):
    """State sharing mechanism for multi-agent coordination"""
    
    def __init__(
        self,
        state_dim: int,
        shared_dim: int = 64,
        compression_ratio: float = 0.5
    ):
        """Initialize state sharing
        
        Args:
            state_dim: Full state dimension
            shared_dim: Shared state dimension
            compression_ratio: State compression ratio
        """
        super().__init__()
        
        self.shared_dim = shared_dim
        self.compression_ratio = compression_ratio
        
        # State compressor
        compressed_dim = int(state_dim * compression_ratio)
        self.state_compressor = nn.Sequential(
            nn.Linear(state_dim, compressed_dim),
            nn.ReLU(),
            nn.Linear(compressed_dim, shared_dim)
        )
        
        # State decompressor
        self.state_decompressor = nn.Sequential(
            nn.Linear(shared_dim, compressed_dim),
            nn.ReLU(),
            nn.Linear(compressed_dim, state_dim)
        )
        
        # Importance weighting
        self.importance_net = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, state_dim),
            nn.Sigmoid()
        )
    
    def compress_state(
        self,
        state: torch.Tensor,
        importance_threshold: float = 0.5
    ) -> torch.Tensor:
        """Compress state for sharing
        
        Args:
            state: Full state
            importance_threshold: Threshold for importance
            
        Returns:
            Compressed state
        """
        # Get importance weights
        importance = self.importance_net(state)
        
        # Mask unimportant features
        masked_state = state * (importance > importance_threshold).float()
        
        # Compress
        compressed = self.state_compressor(masked_state)
        
        return compressed
    
    def decompress_state(
        self,
        compressed: torch.Tensor
    ) -> torch.Tensor:
        """Decompress shared state
        
        Args:
            compressed: Compressed state
            
        Returns:
            Reconstructed state
        """
        return self.state_decompressor(compressed)


class MessageBuffer:
    """Buffer for simulating network latency"""
    
    def __init__(self, latency_ms: float):
        """Initialize message buffer
        
        Args:
            latency_ms: Network latency in milliseconds
        """
        self.latency_ms = latency_ms
        self.buffer = {}
        self.timestamps = {}
    
    def add_message(self, agent_id: int, message: Any):
        """Add message to buffer
        
        Args:
            agent_id: Sender agent ID
            message: Message content
        """
        self.buffer[agent_id] = message
        self.timestamps[agent_id] = time.time()
    
    def is_available(self, agent_id: int) -> bool:
        """Check if message is available (latency passed)
        
        Args:
            agent_id: Sender agent ID
            
        Returns:
            Whether message is available
        """
        if agent_id not in self.timestamps:
            return False
        
        elapsed_ms = (time.time() - self.timestamps[agent_id]) * 1000
        return elapsed_ms >= self.latency_ms
    
    def get_message(self, agent_id: int) -> Optional[Any]:
        """Get message if available
        
        Args:
            agent_id: Sender agent ID
            
        Returns:
            Message or None
        """
        if self.is_available(agent_id):
            return self.buffer.get(agent_id)
        return None


class BandwidthTracker:
    """Tracks bandwidth usage with real constraints"""
    
    def __init__(self, bandwidth_kbps: float):
        """Initialize bandwidth tracker
        
        Args:
            bandwidth_kbps: Bandwidth limit in kilobits per second
        """
        self.bandwidth_kbps = bandwidth_kbps
        self.bandwidth_bps = bandwidth_kbps * 1000
        self.sent_bits = 0
        self.last_reset = time.time()
        self.reset_interval = 1.0  # Reset every second
    
    def can_send(self, message_bits: int) -> bool:
        """Check if message can be sent within bandwidth
        
        Args:
            message_bits: Size of message in bits
            
        Returns:
            Whether message can be sent
        """
        # Reset counter if interval passed
        current_time = time.time()
        if current_time - self.last_reset >= self.reset_interval:
            self.sent_bits = 0
            self.last_reset = current_time
        
        # Check if sending would exceed bandwidth
        if self.sent_bits + message_bits > self.bandwidth_bps:
            return False
        
        # Update counter
        self.sent_bits += message_bits
        return True
    
    def get_usage_ratio(self) -> float:
        """Get current bandwidth usage ratio
        
        Returns:
            Usage ratio (0-1)
        """
        return min(1.0, self.sent_bits / self.bandwidth_bps)