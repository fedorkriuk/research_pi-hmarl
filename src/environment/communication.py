"""Communication Protocol for Multi-Agent Coordination

This module implements communication protocols between agents with
real-world constraints like latency, bandwidth, and range limitations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import logging

from ..data.real_parameter_extractor import RealParameterExtractor, CommunicationSpecifications

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Container for agent messages"""
    sender_id: int
    receiver_id: int  # -1 for broadcast
    content: np.ndarray
    timestamp: float
    priority: int = 0  # 0=low, 1=normal, 2=high
    msg_type: str = "data"  # data, control, emergency
    
    # Network properties
    latency: float = 0.0
    packet_loss: bool = False
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority > other.priority


@dataclass 
class CommunicationLink:
    """Represents a communication link between agents"""
    agent1_id: int
    agent2_id: int
    distance: float
    link_quality: float  # 0-1
    latency: float  # seconds
    bandwidth: float  # bits/second
    packet_loss_rate: float  # 0-1
    active: bool = True
    
    # Statistics
    messages_sent: int = 0
    messages_lost: int = 0
    total_data_sent: float = 0.0  # bits


class CommunicationProtocol:
    """Manages inter-agent communication with realistic constraints"""
    
    def __init__(
        self,
        communication_range: float = 50.0,
        latency: float = 0.001,
        bandwidth: float = 1e6,  # 1 Mbps default
        packet_loss_rate: float = 0.001,
        protocol_type: str = "wifi_ac",
        parameter_extractor: Optional[RealParameterExtractor] = None
    ):
        """Initialize communication protocol
        
        Args:
            communication_range: Maximum communication range
            latency: Base latency in seconds
            bandwidth: Available bandwidth in bits/second
            packet_loss_rate: Base packet loss rate
            protocol_type: Communication protocol type
            parameter_extractor: Real parameter extractor
        """
        self.communication_range = communication_range
        self.base_latency = latency
        self.bandwidth = bandwidth
        self.base_packet_loss = packet_loss_rate
        self.protocol_type = protocol_type
        self.parameter_extractor = parameter_extractor or RealParameterExtractor()
        
        # Get real communication specifications
        self.comm_specs = self.parameter_extractor.get_communication_specs(protocol_type)
        if self.comm_specs:
            self.communication_range = self.comm_specs.max_range
            self.base_latency = self.comm_specs.latency_mean / 1000.0  # ms to s
            self.bandwidth = self.comm_specs.max_data_rate * 1e6  # Mbps to bps
            self.base_packet_loss = self.comm_specs.packet_loss_rate
        
        # Communication state
        self.active_agents: Set[int] = set()
        self.links: Dict[Tuple[int, int], CommunicationLink] = {}
        self.message_queues: Dict[int, deque] = {}  # Per-agent message queues
        self.sent_messages: Dict[int, List[Message]] = {}  # Sent but not delivered
        
        # Network topology
        self.topology_type = "mesh"  # mesh, star, hierarchical
        self.central_node: Optional[int] = None  # For star topology
        
        # Statistics
        self.total_messages_sent = 0
        self.total_messages_delivered = 0
        self.total_messages_lost = 0
        self.total_bandwidth_used = 0.0
        
        logger.info(
            f"Initialized CommunicationProtocol: "
            f"range={self.communication_range}m, "
            f"latency={self.base_latency*1000:.1f}ms, "
            f"bandwidth={self.bandwidth/1e6:.1f}Mbps"
        )
    
    def reset(self, agent_ids: Set[int]):
        """Reset communication state
        
        Args:
            agent_ids: Set of active agent IDs
        """
        self.active_agents = set(agent_ids)
        self.links.clear()
        self.message_queues.clear()
        self.sent_messages.clear()
        
        # Initialize message queues
        for agent_id in agent_ids:
            self.message_queues[agent_id] = deque(maxlen=1000)
            self.sent_messages[agent_id] = []
        
        # Reset statistics
        self.total_messages_sent = 0
        self.total_messages_delivered = 0
        self.total_messages_lost = 0
        self.total_bandwidth_used = 0.0
        
        logger.debug(f"Reset communication for {len(agent_ids)} agents")
    
    def add_agent(self, agent_id: int):
        """Add an agent to the communication network
        
        Args:
            agent_id: Agent ID to add
        """
        if agent_id not in self.active_agents:
            self.active_agents.add(agent_id)
            self.message_queues[agent_id] = deque(maxlen=1000)
            self.sent_messages[agent_id] = []
            logger.debug(f"Added agent {agent_id} to communication network")
    
    def remove_agent(self, agent_id: int):
        """Remove an agent from the communication network
        
        Args:
            agent_id: Agent ID to remove
        """
        if agent_id in self.active_agents:
            self.active_agents.remove(agent_id)
            
            # Remove associated links
            links_to_remove = [
                (a1, a2) for (a1, a2) in self.links
                if a1 == agent_id or a2 == agent_id
            ]
            for link in links_to_remove:
                del self.links[link]
            
            # Clear queues
            if agent_id in self.message_queues:
                del self.message_queues[agent_id]
            if agent_id in self.sent_messages:
                del self.sent_messages[agent_id]
            
            logger.debug(f"Removed agent {agent_id} from communication network")
    
    def update(
        self,
        agent_positions: Dict[int, np.ndarray],
        messages: Dict[int, Any],
        dt: float
    ):
        """Update communication state
        
        Args:
            agent_positions: Current positions of agents
            messages: Messages to send (agent_id -> message content)
            dt: Time step
        """
        # Update communication links
        self._update_links(agent_positions)
        
        # Process new messages
        for sender_id, content in messages.items():
            if sender_id in self.active_agents:
                self._send_message(sender_id, content)
        
        # Update message delivery
        self._deliver_messages(dt)
    
    def _update_links(self, agent_positions: Dict[int, np.ndarray]):
        """Update communication links based on positions
        
        Args:
            agent_positions: Current agent positions
        """
        # Update existing links and create new ones
        agent_list = list(self.active_agents)
        
        for i, agent1_id in enumerate(agent_list):
            if agent1_id not in agent_positions:
                continue
                
            for agent2_id in agent_list[i+1:]:
                if agent2_id not in agent_positions:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(
                    agent_positions[agent1_id] - agent_positions[agent2_id]
                )
                
                link_key = (min(agent1_id, agent2_id), max(agent1_id, agent2_id))
                
                if distance <= self.communication_range:
                    # Calculate link properties
                    link_quality = self._calculate_link_quality(distance)
                    latency = self._calculate_latency(distance)
                    packet_loss = self._calculate_packet_loss(distance, link_quality)
                    
                    if link_key in self.links:
                        # Update existing link
                        link = self.links[link_key]
                        link.distance = distance
                        link.link_quality = link_quality
                        link.latency = latency
                        link.packet_loss_rate = packet_loss
                        link.active = True
                    else:
                        # Create new link
                        self.links[link_key] = CommunicationLink(
                            agent1_id=agent1_id,
                            agent2_id=agent2_id,
                            distance=distance,
                            link_quality=link_quality,
                            latency=latency,
                            bandwidth=self.bandwidth * link_quality,
                            packet_loss_rate=packet_loss
                        )
                else:
                    # Deactivate link if out of range
                    if link_key in self.links:
                        self.links[link_key].active = False
    
    def _calculate_link_quality(self, distance: float) -> float:
        """Calculate link quality based on distance
        
        Args:
            distance: Distance between agents
            
        Returns:
            Link quality (0-1)
        """
        if distance > self.communication_range:
            return 0.0
        
        # Path loss model
        if self.comm_specs:
            # Use real propagation model
            path_loss = 20 * np.log10(distance + 1) + 20 * np.log10(
                self.comm_specs.frequency_band * 1e9
            ) - 147.55
            
            # Convert to link quality
            rx_power = self.comm_specs.transmit_power - path_loss
            quality = np.clip(
                (rx_power - self.comm_specs.receiver_sensitivity) / 20.0,
                0.0, 1.0
            )
        else:
            # Simple model
            quality = 1.0 - (distance / self.communication_range) ** 2
        
        return quality
    
    def _calculate_latency(self, distance: float) -> float:
        """Calculate communication latency
        
        Args:
            distance: Distance between agents
            
        Returns:
            Latency in seconds
        """
        # Base latency + propagation delay + processing
        propagation_delay = distance / 3e8  # Speed of light
        
        if self.comm_specs:
            # Add variance
            latency = np.random.normal(
                self.comm_specs.latency_mean / 1000.0,
                self.comm_specs.latency_std / 1000.0
            )
            latency = np.clip(
                latency,
                self.comm_specs.latency_min / 1000.0,
                self.comm_specs.latency_max / 1000.0
            )
        else:
            latency = self.base_latency
        
        return latency + propagation_delay
    
    def _calculate_packet_loss(self, distance: float, link_quality: float) -> float:
        """Calculate packet loss rate
        
        Args:
            distance: Distance between agents
            link_quality: Link quality
            
        Returns:
            Packet loss rate (0-1)
        """
        if self.comm_specs:
            # Base rate + distance/quality effects
            base_loss = self.comm_specs.packet_loss_rate
            quality_factor = (1.0 - link_quality) ** 2
            
            # Burst loss probability
            if np.random.random() < self.comm_specs.packet_loss_burst_prob:
                return min(0.5, base_loss * 10)  # Burst loss
            
            return min(1.0, base_loss + quality_factor * 0.1)
        else:
            return self.base_packet_loss * (2.0 - link_quality)
    
    def _send_message(
        self,
        sender_id: int,
        content: Any,
        receiver_id: int = -1,
        priority: int = 1,
        msg_type: str = "data"
    ):
        """Send a message from an agent
        
        Args:
            sender_id: Sender agent ID
            content: Message content
            receiver_id: Receiver ID (-1 for broadcast)
            priority: Message priority
            msg_type: Type of message
        """
        # Convert content to numpy array if needed
        if not isinstance(content, np.ndarray):
            content = np.array(content, dtype=np.float32)
        
        # Create message
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            timestamp=0.0,  # Will be set on delivery
            priority=priority,
            msg_type=msg_type
        )
        
        # Add to sent queue
        self.sent_messages[sender_id].append(message)
        self.total_messages_sent += 1
        
        # Calculate message size (bits)
        message_size = content.size * 32  # 32 bits per float
        self.total_bandwidth_used += message_size
    
    def _deliver_messages(self, dt: float):
        """Process message delivery with latency
        
        Args:
            dt: Time step
        """
        for sender_id in list(self.sent_messages.keys()):
            delivered = []
            
            for i, message in enumerate(self.sent_messages[sender_id]):
                # Update message timestamp
                message.timestamp += dt
                
                if message.receiver_id == -1:
                    # Broadcast message
                    for receiver_id in self.active_agents:
                        if receiver_id != sender_id:
                            self._attempt_delivery(
                                message, sender_id, receiver_id
                            )
                    delivered.append(i)
                else:
                    # Unicast message
                    if self._attempt_delivery(
                        message, sender_id, message.receiver_id
                    ):
                        delivered.append(i)
            
            # Remove delivered messages
            for i in reversed(delivered):
                del self.sent_messages[sender_id][i]
    
    def _attempt_delivery(
        self,
        message: Message,
        sender_id: int,
        receiver_id: int
    ) -> bool:
        """Attempt to deliver a message
        
        Args:
            message: Message to deliver
            sender_id: Sender ID
            receiver_id: Receiver ID
            
        Returns:
            Whether delivery was successful
        """
        # Check if link exists
        link_key = (min(sender_id, receiver_id), max(sender_id, receiver_id))
        
        if link_key not in self.links or not self.links[link_key].active:
            return False
        
        link = self.links[link_key]
        
        # Check if enough time has passed for latency
        if message.timestamp < link.latency:
            return False
        
        # Check packet loss
        if np.random.random() < link.packet_loss_rate:
            message.packet_loss = True
            self.total_messages_lost += 1
            link.messages_lost += 1
            return True  # Remove from queue even if lost
        
        # Deliver message
        delivered_message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=message.content.copy(),
            timestamp=message.timestamp,
            priority=message.priority,
            msg_type=message.msg_type,
            latency=link.latency,
            packet_loss=False
        )
        
        self.message_queues[receiver_id].append(delivered_message)
        self.total_messages_delivered += 1
        link.messages_sent += 1
        
        return True
    
    def get_received_messages(self, agent_id: int) -> List[Message]:
        """Get messages received by an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of received messages
        """
        if agent_id not in self.message_queues:
            return []
        
        # Get all messages and clear queue
        messages = list(self.message_queues[agent_id])
        self.message_queues[agent_id].clear()
        
        return messages
    
    def get_connected_agents(self, agent_id: int) -> List[int]:
        """Get list of agents connected to given agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of connected agent IDs
        """
        connected = []
        
        for (a1, a2), link in self.links.items():
            if link.active:
                if a1 == agent_id:
                    connected.append(a2)
                elif a2 == agent_id:
                    connected.append(a1)
        
        return connected
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get current network state
        
        Returns:
            Network state information
        """
        active_links = sum(1 for link in self.links.values() if link.active)
        total_bandwidth = sum(
            link.bandwidth for link in self.links.values() if link.active
        )
        
        avg_latency = np.mean([
            link.latency for link in self.links.values() if link.active
        ]) if active_links > 0 else 0.0
        
        avg_quality = np.mean([
            link.link_quality for link in self.links.values() if link.active
        ]) if active_links > 0 else 0.0
        
        return {
            "num_agents": len(self.active_agents),
            "active_links": active_links,
            "total_bandwidth": total_bandwidth,
            "avg_latency": avg_latency * 1000,  # ms
            "avg_link_quality": avg_quality,
            "messages_sent": self.total_messages_sent,
            "messages_delivered": self.total_messages_delivered,
            "messages_lost": self.total_messages_lost,
            "delivery_rate": self.total_messages_delivered / max(1, self.total_messages_sent),
            "bandwidth_utilization": self.total_bandwidth_used / max(1, total_bandwidth)
        }
    
    def broadcast(
        self,
        sender_id: int,
        content: Any,
        priority: int = 1,
        msg_type: str = "data"
    ):
        """Broadcast a message to all connected agents
        
        Args:
            sender_id: Sender agent ID
            content: Message content
            priority: Message priority
            msg_type: Type of message
        """
        self._send_message(sender_id, content, -1, priority, msg_type)
    
    def unicast(
        self,
        sender_id: int,
        receiver_id: int,
        content: Any,
        priority: int = 1,
        msg_type: str = "data"
    ):
        """Send a message to a specific agent
        
        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            content: Message content
            priority: Message priority
            msg_type: Type of message
        """
        self._send_message(sender_id, content, receiver_id, priority, msg_type)
    
    def get_communication_graph(self) -> Dict[int, List[int]]:
        """Get communication graph as adjacency list
        
        Returns:
            Adjacency list representation
        """
        graph = {agent_id: [] for agent_id in self.active_agents}
        
        for (a1, a2), link in self.links.items():
            if link.active:
                graph[a1].append(a2)
                graph[a2].append(a1)
        
        return graph