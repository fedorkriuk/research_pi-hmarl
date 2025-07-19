"""Communication Protocol Implementation

This module implements robust communication protocols for multi-agent
coordination with real-world constraints.
"""

from .protocol import (
    CommunicationProtocol, MessageType, Message,
    ProtocolConfig, MessagePriority
)
from .network_manager import (
    NetworkManager, NetworkTopology, LinkQuality,
    NetworkMetrics, RoutingTable
)
from .message_handler import (
    MessageHandler, MessageQueue, MessageFilter,
    MessageCompressor, MessageEncryption
)
from .consensus import (
    ConsensusProtocol, VotingMechanism, ConsensusState,
    ByzantineFaultTolerance, LeaderElection
)
from .bandwidth_manager import (
    BandwidthManager, QoSManager, TrafficShaper,
    CongestionControl, AdaptiveBitrate
)

__all__ = [
    # Protocol
    'CommunicationProtocol', 'MessageType', 'Message',
    'ProtocolConfig', 'MessagePriority',
    
    # Network Manager
    'NetworkManager', 'NetworkTopology', 'LinkQuality',
    'NetworkMetrics', 'RoutingTable',
    
    # Message Handler
    'MessageHandler', 'MessageQueue', 'MessageFilter',
    'MessageCompressor', 'MessageEncryption',
    
    # Consensus
    'ConsensusProtocol', 'VotingMechanism', 'ConsensusState',
    'ByzantineFaultTolerance', 'LeaderElection',
    
    # Bandwidth Manager
    'BandwidthManager', 'QoSManager', 'TrafficShaper',
    'CongestionControl', 'AdaptiveBitrate'
]