"""Network Management and Topology

This module manages network topology, routing, and link quality
for multi-agent communication.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import time
import logging

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Network topology types"""
    MESH = "mesh"
    STAR = "star"
    HIERARCHICAL = "hierarchical"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"


@dataclass
class LinkQuality:
    """Link quality metrics"""
    rssi: float  # Received Signal Strength Indicator (dBm)
    snr: float  # Signal-to-Noise Ratio (dB)
    packet_loss: float  # Packet loss rate (0-1)
    latency: float  # Round-trip time (seconds)
    jitter: float  # Latency variation (seconds)
    bandwidth: float  # Available bandwidth (bps)
    last_update: float  # Timestamp


@dataclass
class NetworkMetrics:
    """Network-wide metrics"""
    total_nodes: int
    active_links: int
    avg_hop_count: float
    network_diameter: int
    connectivity: float  # 0-1
    partition_count: int
    avg_link_quality: float
    total_bandwidth: float
    congestion_level: float


@dataclass
class RoutingEntry:
    """Routing table entry"""
    destination: int
    next_hop: int
    hop_count: int
    metric: float  # Combined routing metric
    last_update: float
    alternate_routes: List[Tuple[int, int, float]]  # (next_hop, hops, metric)


class NetworkManager:
    """Manages network topology and routing"""
    
    def __init__(
        self,
        agent_id: int,
        topology_type: TopologyType = TopologyType.DYNAMIC,
        max_range: float = 5000.0  # meters
    ):
        """Initialize network manager
        
        Args:
            agent_id: Agent identifier
            topology_type: Network topology type
            max_range: Maximum communication range
        """
        self.agent_id = agent_id
        self.topology_type = topology_type
        self.max_range = max_range
        
        # Network graph
        self.network = nx.Graph()
        self.network.add_node(agent_id)
        
        # Routing
        self.routing_table: Dict[int, RoutingEntry] = {}
        self.route_cache = {}
        
        # Link management
        self.links: Dict[int, LinkQuality] = {}
        self.link_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Topology management
        self.topology_manager = TopologyManager(topology_type)
        self.last_topology_update = 0
        
        # Metrics
        self.metrics_collector = MetricsCollector()
        
        logger.info(f"Initialized network manager for agent {agent_id}")
    
    def update_agent_positions(
        self,
        positions: Dict[int, torch.Tensor]
    ):
        """Update agent positions for topology calculation
        
        Args:
            positions: Agent positions {agent_id: position}
        """
        # Update network based on positions
        for agent_id, pos in positions.items():
            if agent_id not in self.network:
                self.network.add_node(agent_id, position=pos.numpy())
            else:
                self.network.nodes[agent_id]['position'] = pos.numpy()
        
        # Update links based on range
        self._update_links_by_range(positions)
        
        # Update topology if needed
        current_time = time.time()
        if current_time - self.last_topology_update > 5.0:  # 5 second interval
            self._update_topology()
            self.last_topology_update = current_time
    
    def update_link_quality(
        self,
        neighbor_id: int,
        quality_data: Dict[str, float]
    ):
        """Update link quality metrics
        
        Args:
            neighbor_id: Neighbor agent ID
            quality_data: Quality metrics
        """
        # Create or update link quality
        link_quality = LinkQuality(
            rssi=quality_data.get('rssi', -80.0),
            snr=quality_data.get('snr', 10.0),
            packet_loss=quality_data.get('packet_loss', 0.01),
            latency=quality_data.get('latency', 0.05),
            jitter=quality_data.get('jitter', 0.01),
            bandwidth=quality_data.get('bandwidth', 10e6),
            last_update=time.time()
        )
        
        self.links[neighbor_id] = link_quality
        
        # Update history
        self.link_history[neighbor_id].append({
            'quality': self._calculate_link_score(link_quality),
            'timestamp': time.time()
        })
        
        # Update network edge
        if self.network.has_edge(self.agent_id, neighbor_id):
            self.network[self.agent_id][neighbor_id]['quality'] = link_quality
        else:
            self.network.add_edge(
                self.agent_id,
                neighbor_id,
                quality=link_quality
            )
    
    def calculate_routes(self) -> Dict[int, RoutingEntry]:
        """Calculate optimal routes to all reachable agents
        
        Returns:
            Routing table
        """
        routing_table = {}
        
        # Get all reachable nodes
        if self.agent_id in self.network:
            reachable = nx.node_connected_component(self.network, self.agent_id)
        else:
            reachable = {self.agent_id}
        
        for target in reachable:
            if target == self.agent_id:
                continue
            
            try:
                # Calculate shortest path with custom weight
                path = nx.shortest_path(
                    self.network,
                    self.agent_id,
                    target,
                    weight=self._routing_weight
                )
                
                if len(path) > 1:
                    next_hop = path[1]
                    hop_count = len(path) - 1
                    
                    # Calculate path metric
                    metric = self._calculate_path_metric(path)
                    
                    # Find alternate routes
                    alternates = self._find_alternate_routes(target, path)
                    
                    routing_table[target] = RoutingEntry(
                        destination=target,
                        next_hop=next_hop,
                        hop_count=hop_count,
                        metric=metric,
                        last_update=time.time(),
                        alternate_routes=alternates
                    )
                    
            except nx.NetworkXNoPath:
                continue
        
        self.routing_table = routing_table
        return routing_table
    
    def get_next_hop(self, destination: int) -> Optional[int]:
        """Get next hop for destination
        
        Args:
            destination: Target agent ID
            
        Returns:
            Next hop agent ID or None
        """
        if destination in self.routing_table:
            return self.routing_table[destination].next_hop
        return None
    
    def get_network_metrics(self) -> NetworkMetrics:
        """Calculate network-wide metrics
        
        Returns:
            Network metrics
        """
        return self.metrics_collector.calculate_metrics(
            self.network,
            self.links
        )
    
    def _update_links_by_range(self, positions: Dict[int, torch.Tensor]):
        """Update links based on communication range
        
        Args:
            positions: Agent positions
        """
        if self.agent_id not in positions:
            return
        
        my_pos = positions[self.agent_id]
        
        for agent_id, pos in positions.items():
            if agent_id == self.agent_id:
                continue
            
            # Calculate distance
            distance = torch.norm(pos - my_pos).item()
            
            if distance <= self.max_range:
                # Link is possible
                if agent_id not in self.links:
                    # Estimate initial link quality based on distance
                    rssi = self._estimate_rssi(distance)
                    self.update_link_quality(agent_id, {
                        'rssi': rssi,
                        'packet_loss': self._estimate_packet_loss(rssi),
                        'latency': self._estimate_latency(distance),
                        'bandwidth': self._estimate_bandwidth(rssi)
                    })
            else:
                # Link is out of range
                if agent_id in self.links:
                    del self.links[agent_id]
                    if self.network.has_edge(self.agent_id, agent_id):
                        self.network.remove_edge(self.agent_id, agent_id)
    
    def _update_topology(self):
        """Update network topology based on current state"""
        # Apply topology-specific rules
        if self.topology_type == TopologyType.HIERARCHICAL:
            self._enforce_hierarchical_topology()
        elif self.topology_type == TopologyType.STAR:
            self._enforce_star_topology()
        elif self.topology_type == TopologyType.HYBRID:
            self._optimize_hybrid_topology()
        
        # Recalculate routes after topology change
        self.calculate_routes()
    
    def _routing_weight(self, u: int, v: int, d: Dict) -> float:
        """Calculate routing weight for edge
        
        Args:
            u: Source node
            v: Target node
            d: Edge data
            
        Returns:
            Routing weight
        """
        if 'quality' not in d:
            return 1.0
        
        quality = d['quality']
        
        # Multi-metric routing
        # Lower weight is better
        weight = (
            0.3 * quality.packet_loss +
            0.3 * (quality.latency / 0.1) +  # Normalized to 100ms
            0.2 * (1.0 - quality.bandwidth / 10e6) +  # Normalized to 10Mbps
            0.2 * ((-quality.rssi + 40) / 50)  # Normalized RSSI
        )
        
        return max(weight, 0.01)  # Avoid zero weight
    
    def _calculate_link_score(self, quality: LinkQuality) -> float:
        """Calculate overall link quality score
        
        Args:
            quality: Link quality metrics
            
        Returns:
            Quality score (0-1)
        """
        # Combine metrics into single score
        score = (
            0.3 * (1.0 - quality.packet_loss) +
            0.2 * min(1.0, 100 / (quality.latency * 1000)) +  # ms
            0.2 * min(1.0, quality.bandwidth / 10e6) +
            0.2 * min(1.0, (quality.rssi + 90) / 50) +
            0.1 * min(1.0, quality.snr / 30)
        )
        
        return np.clip(score, 0, 1)
    
    def _calculate_path_metric(self, path: List[int]) -> float:
        """Calculate path quality metric
        
        Args:
            path: Node path
            
        Returns:
            Path metric (lower is better)
        """
        if len(path) < 2:
            return 0.0
        
        total_metric = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.network.has_edge(u, v):
                total_metric += self._routing_weight(u, v, self.network[u][v])
            else:
                total_metric += 10.0  # High penalty for missing edge
        
        return total_metric
    
    def _find_alternate_routes(
        self,
        target: int,
        primary_path: List[int],
        max_alternates: int = 3
    ) -> List[Tuple[int, int, float]]:
        """Find alternate routes to target
        
        Args:
            target: Target node
            primary_path: Primary path
            max_alternates: Maximum alternate routes
            
        Returns:
            List of (next_hop, hop_count, metric)
        """
        alternates = []
        
        try:
            # Find k-shortest paths
            paths = list(nx.shortest_simple_paths(
                self.network,
                self.agent_id,
                target,
                weight=self._routing_weight
            ))[:max_alternates + 1]
            
            for path in paths:
                if path != primary_path and len(path) > 1:
                    next_hop = path[1]
                    hop_count = len(path) - 1
                    metric = self._calculate_path_metric(path)
                    alternates.append((next_hop, hop_count, metric))
                    
                    if len(alternates) >= max_alternates:
                        break
                        
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        return alternates
    
    def _estimate_rssi(self, distance: float) -> float:
        """Estimate RSSI based on distance
        
        Args:
            distance: Distance in meters
            
        Returns:
            RSSI in dBm
        """
        # Simplified path loss model
        # RSSI = Tx_power - Path_loss
        tx_power = 20.0  # dBm
        
        # Free space path loss
        frequency = 2.4e9  # Hz
        path_loss = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
        
        rssi = tx_power - path_loss
        
        # Add some randomness
        rssi += np.random.normal(0, 2)
        
        return np.clip(rssi, -100, 0)
    
    def _estimate_packet_loss(self, rssi: float) -> float:
        """Estimate packet loss based on RSSI
        
        Args:
            rssi: RSSI in dBm
            
        Returns:
            Packet loss rate (0-1)
        """
        # Simplified model
        if rssi > -60:
            return 0.001
        elif rssi > -70:
            return 0.01
        elif rssi > -80:
            return 0.05
        elif rssi > -90:
            return 0.1
        else:
            return 0.5
    
    def _estimate_latency(self, distance: float) -> float:
        """Estimate latency based on distance
        
        Args:
            distance: Distance in meters
            
        Returns:
            Latency in seconds
        """
        # Speed of light propagation
        propagation_delay = distance / 3e8
        
        # Processing delay
        processing_delay = 0.001  # 1ms
        
        # Queuing delay (simplified)
        queuing_delay = np.random.exponential(0.005)  # 5ms average
        
        return propagation_delay + processing_delay + queuing_delay
    
    def _estimate_bandwidth(self, rssi: float) -> float:
        """Estimate available bandwidth based on RSSI
        
        Args:
            rssi: RSSI in dBm
            
        Returns:
            Bandwidth in bps
        """
        # Simplified adaptive rate selection
        if rssi > -55:
            return 54e6  # 54 Mbps
        elif rssi > -65:
            return 24e6  # 24 Mbps
        elif rssi > -75:
            return 12e6  # 12 Mbps
        elif rssi > -85:
            return 6e6   # 6 Mbps
        else:
            return 1e6   # 1 Mbps
    
    def _enforce_hierarchical_topology(self):
        """Enforce hierarchical network topology"""
        # Implement cluster-based hierarchy
        # This is simplified - would use proper clustering
        pass
    
    def _enforce_star_topology(self):
        """Enforce star network topology"""
        # Select central node based on connectivity
        # This is simplified
        pass
    
    def _optimize_hybrid_topology(self):
        """Optimize hybrid network topology"""
        # Combine multiple topology types
        # This is simplified
        pass


class TopologyManager:
    """Manages network topology formation and optimization"""
    
    def __init__(self, topology_type: TopologyType):
        """Initialize topology manager
        
        Args:
            topology_type: Type of topology to maintain
        """
        self.topology_type = topology_type
        self.topology_params = self._get_default_params(topology_type)
    
    def _get_default_params(self, topology_type: TopologyType) -> Dict[str, Any]:
        """Get default parameters for topology type
        
        Args:
            topology_type: Topology type
            
        Returns:
            Default parameters
        """
        params = {
            TopologyType.MESH: {
                'max_degree': 6,
                'redundancy': 2
            },
            TopologyType.STAR: {
                'hub_selection': 'centrality',
                'backup_hubs': 2
            },
            TopologyType.HIERARCHICAL: {
                'levels': 3,
                'cluster_size': 5,
                'inter_cluster_links': 2
            },
            TopologyType.DYNAMIC: {
                'adaptation_rate': 0.1,
                'optimization_interval': 10.0
            },
            TopologyType.HYBRID: {
                'mesh_ratio': 0.3,
                'star_ratio': 0.3,
                'hierarchical_ratio': 0.4
            }
        }
        
        return params.get(topology_type, {})
    
    def optimize_topology(
        self,
        network: nx.Graph,
        objectives: Dict[str, float]
    ) -> List[Tuple[int, int]]:
        """Optimize network topology
        
        Args:
            network: Current network graph
            objectives: Optimization objectives
            
        Returns:
            List of edges to add/remove
        """
        # Simplified topology optimization
        # Would implement proper multi-objective optimization
        return []


class MetricsCollector:
    """Collects and calculates network metrics"""
    
    def calculate_metrics(
        self,
        network: nx.Graph,
        links: Dict[int, LinkQuality]
    ) -> NetworkMetrics:
        """Calculate network metrics
        
        Args:
            network: Network graph
            links: Link quality data
            
        Returns:
            Network metrics
        """
        # Basic metrics
        total_nodes = network.number_of_nodes()
        active_links = network.number_of_edges()
        
        # Connectivity metrics
        if total_nodes > 1:
            components = list(nx.connected_components(network))
            partition_count = len(components)
            largest_component = max(components, key=len)
            connectivity = len(largest_component) / total_nodes
        else:
            partition_count = 1
            connectivity = 1.0
        
        # Path metrics
        if total_nodes > 1 and nx.is_connected(network):
            avg_shortest_path = nx.average_shortest_path_length(network)
            diameter = nx.diameter(network)
        else:
            avg_shortest_path = 0.0
            diameter = 0
        
        # Link quality metrics
        if links:
            avg_link_quality = np.mean([
                self._calculate_link_score(lq) for lq in links.values()
            ])
            total_bandwidth = sum(lq.bandwidth for lq in links.values())
        else:
            avg_link_quality = 0.0
            total_bandwidth = 0.0
        
        # Congestion (simplified)
        if active_links > 0:
            congestion_level = min(1.0, total_nodes / (active_links * 2))
        else:
            congestion_level = 1.0
        
        return NetworkMetrics(
            total_nodes=total_nodes,
            active_links=active_links,
            avg_hop_count=avg_shortest_path,
            network_diameter=diameter,
            connectivity=connectivity,
            partition_count=partition_count,
            avg_link_quality=avg_link_quality,
            total_bandwidth=total_bandwidth,
            congestion_level=congestion_level
        )
    
    def _calculate_link_score(self, quality: LinkQuality) -> float:
        """Calculate link quality score
        
        Args:
            quality: Link quality
            
        Returns:
            Score (0-1)
        """
        score = (
            0.3 * (1.0 - quality.packet_loss) +
            0.2 * min(1.0, 100 / (quality.latency * 1000)) +
            0.2 * min(1.0, quality.bandwidth / 10e6) +
            0.2 * min(1.0, (quality.rssi + 90) / 50) +
            0.1 * min(1.0, quality.snr / 30)
        )
        
        return np.clip(score, 0, 1)


class RoutingTable:
    """Advanced routing table with multiple protocols"""
    
    def __init__(self):
        """Initialize routing table"""
        self.entries: Dict[int, RoutingEntry] = {}
        self.protocol_handlers = {
            'ospf': self._ospf_update,
            'aodv': self._aodv_update,
            'dsr': self._dsr_update,
            'custom': self._custom_update
        }
    
    def update_route(
        self,
        destination: int,
        route_info: Dict[str, Any],
        protocol: str = 'custom'
    ):
        """Update routing entry
        
        Args:
            destination: Destination node
            route_info: Route information
            protocol: Routing protocol
        """
        if protocol in self.protocol_handlers:
            self.protocol_handlers[protocol](destination, route_info)
    
    def _ospf_update(self, destination: int, route_info: Dict[str, Any]):
        """OSPF-style route update"""
        # Simplified OSPF update
        pass
    
    def _aodv_update(self, destination: int, route_info: Dict[str, Any]):
        """AODV-style route update"""
        # Simplified AODV update
        pass
    
    def _dsr_update(self, destination: int, route_info: Dict[str, Any]):
        """DSR-style route update"""
        # Simplified DSR update
        pass
    
    def _custom_update(self, destination: int, route_info: Dict[str, Any]):
        """Custom protocol route update"""
        # Update with provided information
        self.entries[destination] = RoutingEntry(
            destination=destination,
            next_hop=route_info.get('next_hop'),
            hop_count=route_info.get('hop_count', 1),
            metric=route_info.get('metric', 1.0),
            last_update=time.time(),
            alternate_routes=route_info.get('alternates', [])
        )