"""Bandwidth Management and QoS

This module implements bandwidth management, quality of service,
and congestion control for multi-agent communication.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque, defaultdict
import threading
import logging

logger = logging.getLogger(__name__)


class TrafficClass(Enum):
    """Traffic classification for QoS"""
    CONTROL = 0      # Highest priority - control messages
    REALTIME = 1     # Real-time sensor data
    PRIORITY = 2     # High priority data
    STANDARD = 3     # Standard data
    BULK = 4         # Bulk transfers - lowest priority


@dataclass
class BandwidthAllocation:
    """Bandwidth allocation for traffic class"""
    traffic_class: TrafficClass
    min_bandwidth: float  # bps
    max_bandwidth: float  # bps
    current_allocation: float  # bps
    priority_weight: float
    burst_allowance: float  # bytes


@dataclass
class FlowStats:
    """Statistics for a communication flow"""
    source: int
    destination: int
    traffic_class: TrafficClass
    bytes_sent: int
    packets_sent: int
    start_time: float
    last_update: float
    avg_rate: float  # bps
    peak_rate: float  # bps
    jitter: float
    packet_loss: float


class BandwidthManager:
    """Manages bandwidth allocation and scheduling"""
    
    def __init__(
        self,
        total_bandwidth: float = 10e6,  # 10 Mbps default
        enable_qos: bool = True
    ):
        """Initialize bandwidth manager
        
        Args:
            total_bandwidth: Total available bandwidth in bps
            enable_qos: Enable quality of service
        """
        self.total_bandwidth = total_bandwidth
        self.enable_qos = enable_qos
        
        # Bandwidth allocations
        self.allocations = self._initialize_allocations()
        
        # Traffic tracking
        self.active_flows: Dict[Tuple[int, int], FlowStats] = {}
        self.flow_history = deque(maxlen=1000)
        
        # Token buckets for rate limiting
        self.token_buckets = {}
        
        # Statistics
        self.stats = {
            'bytes_transmitted': 0,
            'packets_transmitted': 0,
            'bytes_dropped': 0,
            'packets_dropped': 0,
            'avg_utilization': 0.0,
            'peak_utilization': 0.0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Initialized bandwidth manager with {total_bandwidth/1e6:.1f} Mbps")
    
    def _initialize_allocations(self) -> Dict[TrafficClass, BandwidthAllocation]:
        """Initialize default bandwidth allocations
        
        Returns:
            Bandwidth allocations by traffic class
        """
        # Default QoS allocations (percentages of total)
        allocations = {
            TrafficClass.CONTROL: BandwidthAllocation(
                traffic_class=TrafficClass.CONTROL,
                min_bandwidth=self.total_bandwidth * 0.2,  # 20% guaranteed
                max_bandwidth=self.total_bandwidth * 0.4,  # 40% max
                current_allocation=self.total_bandwidth * 0.2,
                priority_weight=5.0,
                burst_allowance=1024 * 1024  # 1MB burst
            ),
            TrafficClass.REALTIME: BandwidthAllocation(
                traffic_class=TrafficClass.REALTIME,
                min_bandwidth=self.total_bandwidth * 0.3,  # 30% guaranteed
                max_bandwidth=self.total_bandwidth * 0.5,  # 50% max
                current_allocation=self.total_bandwidth * 0.3,
                priority_weight=4.0,
                burst_allowance=512 * 1024  # 512KB burst
            ),
            TrafficClass.PRIORITY: BandwidthAllocation(
                traffic_class=TrafficClass.PRIORITY,
                min_bandwidth=self.total_bandwidth * 0.2,  # 20% guaranteed
                max_bandwidth=self.total_bandwidth * 0.4,  # 40% max
                current_allocation=self.total_bandwidth * 0.2,
                priority_weight=3.0,
                burst_allowance=256 * 1024  # 256KB burst
            ),
            TrafficClass.STANDARD: BandwidthAllocation(
                traffic_class=TrafficClass.STANDARD,
                min_bandwidth=self.total_bandwidth * 0.2,  # 20% guaranteed
                max_bandwidth=self.total_bandwidth * 0.6,  # 60% max
                current_allocation=self.total_bandwidth * 0.2,
                priority_weight=2.0,
                burst_allowance=128 * 1024  # 128KB burst
            ),
            TrafficClass.BULK: BandwidthAllocation(
                traffic_class=TrafficClass.BULK,
                min_bandwidth=self.total_bandwidth * 0.1,  # 10% guaranteed
                max_bandwidth=self.total_bandwidth * 0.8,  # 80% max
                current_allocation=self.total_bandwidth * 0.1,
                priority_weight=1.0,
                burst_allowance=64 * 1024  # 64KB burst
            )
        }
        
        return allocations
    
    def allocate_bandwidth(
        self,
        source: int,
        destination: int,
        traffic_class: TrafficClass,
        requested_bandwidth: float
    ) -> float:
        """Allocate bandwidth for a flow
        
        Args:
            source: Source agent ID
            destination: Destination agent ID
            traffic_class: Traffic class
            requested_bandwidth: Requested bandwidth in bps
            
        Returns:
            Allocated bandwidth in bps
        """
        with self.lock:
            flow_key = (source, destination)
            
            # Get allocation for traffic class
            allocation = self.allocations[traffic_class]
            
            # Check available bandwidth
            used_bandwidth = self._calculate_used_bandwidth()
            available = self.total_bandwidth - used_bandwidth
            
            # Apply QoS rules
            if self.enable_qos:
                allocated = min(
                    requested_bandwidth,
                    allocation.max_bandwidth,
                    available + allocation.min_bandwidth
                )
                
                # Ensure minimum bandwidth
                allocated = max(allocated, allocation.min_bandwidth)
            else:
                # Simple fair share without QoS
                allocated = min(requested_bandwidth, available)
            
            # Update flow statistics
            if flow_key not in self.active_flows:
                self.active_flows[flow_key] = FlowStats(
                    source=source,
                    destination=destination,
                    traffic_class=traffic_class,
                    bytes_sent=0,
                    packets_sent=0,
                    start_time=time.time(),
                    last_update=time.time(),
                    avg_rate=allocated,
                    peak_rate=allocated,
                    jitter=0.0,
                    packet_loss=0.0
                )
            
            return allocated
    
    def transmit(
        self,
        source: int,
        destination: int,
        data_size: int,
        traffic_class: TrafficClass
    ) -> Tuple[bool, float]:
        """Attempt to transmit data
        
        Args:
            source: Source agent ID
            destination: Destination agent ID
            data_size: Data size in bytes
            traffic_class: Traffic class
            
        Returns:
            (success, estimated_time)
        """
        with self.lock:
            flow_key = (source, destination)
            
            # Check token bucket
            if not self._check_token_bucket(flow_key, data_size, traffic_class):
                self.stats['bytes_dropped'] += data_size
                self.stats['packets_dropped'] += 1
                return False, 0.0
            
            # Get allocated bandwidth
            if flow_key in self.active_flows:
                flow = self.active_flows[flow_key]
                bandwidth = flow.avg_rate
            else:
                bandwidth = self.allocate_bandwidth(
                    source, destination, traffic_class, self.total_bandwidth * 0.1
                )
            
            # Calculate transmission time
            transmission_time = data_size * 8 / bandwidth  # Convert bytes to bits
            
            # Update statistics
            self._update_flow_stats(flow_key, data_size, bandwidth)
            self.stats['bytes_transmitted'] += data_size
            self.stats['packets_transmitted'] += 1
            
            return True, transmission_time
    
    def _check_token_bucket(
        self,
        flow_key: Tuple[int, int],
        data_size: int,
        traffic_class: TrafficClass
    ) -> bool:
        """Check token bucket for rate limiting
        
        Args:
            flow_key: Flow identifier
            data_size: Data size in bytes
            traffic_class: Traffic class
            
        Returns:
            Whether transmission is allowed
        """
        current_time = time.time()
        
        if flow_key not in self.token_buckets:
            allocation = self.allocations[traffic_class]
            self.token_buckets[flow_key] = {
                'tokens': allocation.burst_allowance,
                'last_update': current_time,
                'rate': allocation.current_allocation / 8,  # bytes per second
                'burst_size': allocation.burst_allowance
            }
        
        bucket = self.token_buckets[flow_key]
        
        # Refill tokens
        elapsed = current_time - bucket['last_update']
        new_tokens = elapsed * bucket['rate']
        bucket['tokens'] = min(
            bucket['tokens'] + new_tokens,
            bucket['burst_size']
        )
        bucket['last_update'] = current_time
        
        # Check if enough tokens
        if bucket['tokens'] >= data_size:
            bucket['tokens'] -= data_size
            return True
        
        return False
    
    def _calculate_used_bandwidth(self) -> float:
        """Calculate currently used bandwidth
        
        Returns:
            Used bandwidth in bps
        """
        current_time = time.time()
        used_bandwidth = 0.0
        
        for flow in self.active_flows.values():
            # Only count recent flows
            if current_time - flow.last_update < 5.0:
                used_bandwidth += flow.avg_rate
        
        return used_bandwidth
    
    def _update_flow_stats(
        self,
        flow_key: Tuple[int, int],
        data_size: int,
        bandwidth: float
    ):
        """Update flow statistics
        
        Args:
            flow_key: Flow identifier
            data_size: Data size in bytes
            bandwidth: Allocated bandwidth
        """
        current_time = time.time()
        
        if flow_key in self.active_flows:
            flow = self.active_flows[flow_key]
            
            # Update counters
            flow.bytes_sent += data_size
            flow.packets_sent += 1
            
            # Update rates
            time_delta = current_time - flow.last_update
            if time_delta > 0:
                instant_rate = (data_size * 8) / time_delta
                flow.avg_rate = 0.9 * flow.avg_rate + 0.1 * instant_rate
                flow.peak_rate = max(flow.peak_rate, instant_rate)
            
            flow.last_update = current_time
    
    def update_qos_parameters(
        self,
        traffic_class: TrafficClass,
        min_bandwidth: Optional[float] = None,
        max_bandwidth: Optional[float] = None,
        priority_weight: Optional[float] = None
    ):
        """Update QoS parameters for traffic class
        
        Args:
            traffic_class: Traffic class to update
            min_bandwidth: Minimum guaranteed bandwidth
            max_bandwidth: Maximum allowed bandwidth
            priority_weight: Priority weight
        """
        with self.lock:
            allocation = self.allocations[traffic_class]
            
            if min_bandwidth is not None:
                allocation.min_bandwidth = min_bandwidth
            
            if max_bandwidth is not None:
                allocation.max_bandwidth = max_bandwidth
            
            if priority_weight is not None:
                allocation.priority_weight = priority_weight
            
            # Rebalance allocations
            self._rebalance_allocations()
    
    def _rebalance_allocations(self):
        """Rebalance bandwidth allocations"""
        # Ensure minimum bandwidths don't exceed total
        total_min = sum(a.min_bandwidth for a in self.allocations.values())
        
        if total_min > self.total_bandwidth:
            # Scale down proportionally
            scale = self.total_bandwidth / total_min
            for allocation in self.allocations.values():
                allocation.min_bandwidth *= scale
                allocation.current_allocation = allocation.min_bandwidth
    
    def get_flow_statistics(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Get flow statistics
        
        Returns:
            Flow statistics by flow key
        """
        with self.lock:
            return {
                flow_key: {
                    'source': flow.source,
                    'destination': flow.destination,
                    'traffic_class': flow.traffic_class.value,
                    'bytes_sent': flow.bytes_sent,
                    'packets_sent': flow.packets_sent,
                    'avg_rate': flow.avg_rate,
                    'peak_rate': flow.peak_rate,
                    'duration': time.time() - flow.start_time
                }
                for flow_key, flow in self.active_flows.items()
            }
    
    def get_utilization(self) -> Dict[str, float]:
        """Get bandwidth utilization
        
        Returns:
            Utilization metrics
        """
        with self.lock:
            used_bandwidth = self._calculate_used_bandwidth()
            utilization = used_bandwidth / self.total_bandwidth
            
            # Update statistics
            self.stats['avg_utilization'] = (
                0.95 * self.stats['avg_utilization'] + 0.05 * utilization
            )
            self.stats['peak_utilization'] = max(
                self.stats['peak_utilization'],
                utilization
            )
            
            return {
                'current_utilization': utilization,
                'avg_utilization': self.stats['avg_utilization'],
                'peak_utilization': self.stats['peak_utilization'],
                'used_bandwidth': used_bandwidth,
                'available_bandwidth': self.total_bandwidth - used_bandwidth
            }


class QoSManager:
    """Quality of Service manager"""
    
    def __init__(self):
        """Initialize QoS manager"""
        self.policies = {}
        self.classifiers = []
        self.shapers = {}
    
    def add_policy(
        self,
        name: str,
        conditions: Dict[str, Any],
        actions: Dict[str, Any]
    ):
        """Add QoS policy
        
        Args:
            name: Policy name
            conditions: Policy conditions
            actions: Policy actions
        """
        self.policies[name] = {
            'conditions': conditions,
            'actions': actions,
            'stats': {
                'matches': 0,
                'bytes_affected': 0
            }
        }
    
    def classify_traffic(
        self,
        source: int,
        destination: int,
        data: Dict[str, Any]
    ) -> TrafficClass:
        """Classify traffic based on rules
        
        Args:
            source: Source agent
            destination: Destination agent
            data: Traffic data
            
        Returns:
            Traffic class
        """
        # Check message type
        msg_type = data.get('type', '')
        
        if msg_type in ['emergency', 'collision_warning', 'system_failure']:
            return TrafficClass.CONTROL
        elif msg_type in ['telemetry', 'sensor_data']:
            return TrafficClass.REALTIME
        elif data.get('priority', 'medium') == 'high':
            return TrafficClass.PRIORITY
        elif data.get('bulk', False):
            return TrafficClass.BULK
        else:
            return TrafficClass.STANDARD
    
    def apply_policy(
        self,
        flow_key: Tuple[int, int],
        traffic_class: TrafficClass,
        bandwidth_manager: BandwidthManager
    ) -> Dict[str, Any]:
        """Apply QoS policy to flow
        
        Args:
            flow_key: Flow identifier
            traffic_class: Traffic class
            bandwidth_manager: Bandwidth manager
            
        Returns:
            Applied actions
        """
        applied_actions = {}
        
        for name, policy in self.policies.items():
            # Check conditions
            conditions = policy['conditions']
            
            if self._check_conditions(flow_key, traffic_class, conditions):
                # Apply actions
                actions = policy['actions']
                
                if 'min_bandwidth' in actions:
                    bandwidth_manager.update_qos_parameters(
                        traffic_class,
                        min_bandwidth=actions['min_bandwidth']
                    )
                    applied_actions['min_bandwidth'] = actions['min_bandwidth']
                
                if 'max_bandwidth' in actions:
                    bandwidth_manager.update_qos_parameters(
                        traffic_class,
                        max_bandwidth=actions['max_bandwidth']
                    )
                    applied_actions['max_bandwidth'] = actions['max_bandwidth']
                
                if 'priority_boost' in actions:
                    current_weight = bandwidth_manager.allocations[traffic_class].priority_weight
                    new_weight = current_weight * actions['priority_boost']
                    bandwidth_manager.update_qos_parameters(
                        traffic_class,
                        priority_weight=new_weight
                    )
                    applied_actions['priority_weight'] = new_weight
                
                # Update statistics
                policy['stats']['matches'] += 1
        
        return applied_actions
    
    def _check_conditions(
        self,
        flow_key: Tuple[int, int],
        traffic_class: TrafficClass,
        conditions: Dict[str, Any]
    ) -> bool:
        """Check if conditions are met
        
        Args:
            flow_key: Flow identifier
            traffic_class: Traffic class
            conditions: Conditions to check
            
        Returns:
            Whether conditions are met
        """
        # Check traffic class condition
        if 'traffic_class' in conditions:
            if traffic_class.value != conditions['traffic_class']:
                return False
        
        # Check source/destination conditions
        if 'source' in conditions:
            if flow_key[0] != conditions['source']:
                return False
        
        if 'destination' in conditions:
            if flow_key[1] != conditions['destination']:
                return False
        
        # All conditions met
        return True


class TrafficShaper:
    """Shapes traffic to meet bandwidth constraints"""
    
    def __init__(self, rate: float, burst_size: float):
        """Initialize traffic shaper
        
        Args:
            rate: Shaping rate in bps
            burst_size: Burst size in bytes
        """
        self.rate = rate
        self.burst_size = burst_size
        self.queue = deque()
        self.tokens = burst_size
        self.last_update = time.time()
        
        # Statistics
        self.stats = {
            'packets_shaped': 0,
            'packets_dropped': 0,
            'avg_queue_length': 0.0,
            'max_queue_length': 0
        }
    
    def shape(
        self,
        packet: bytes,
        max_queue_size: int = 100
    ) -> Optional[bytes]:
        """Shape packet
        
        Args:
            packet: Packet to shape
            max_queue_size: Maximum queue size
            
        Returns:
            Shaped packet or None if dropped
        """
        current_time = time.time()
        
        # Update tokens
        elapsed = current_time - self.last_update
        self.tokens = min(
            self.tokens + elapsed * self.rate / 8,
            self.burst_size
        )
        self.last_update = current_time
        
        packet_size = len(packet)
        
        # Check if can transmit immediately
        if self.tokens >= packet_size:
            self.tokens -= packet_size
            self.stats['packets_shaped'] += 1
            return packet
        
        # Queue packet
        if len(self.queue) < max_queue_size:
            self.queue.append(packet)
            
            # Update statistics
            queue_length = len(self.queue)
            self.stats['avg_queue_length'] = (
                0.95 * self.stats['avg_queue_length'] + 0.05 * queue_length
            )
            self.stats['max_queue_length'] = max(
                self.stats['max_queue_length'],
                queue_length
            )
            
            return None
        else:
            # Drop packet
            self.stats['packets_dropped'] += 1
            return None
    
    def get_queued_packet(self) -> Optional[bytes]:
        """Get next packet from queue if tokens available
        
        Returns:
            Packet or None
        """
        if not self.queue:
            return None
        
        # Update tokens
        current_time = time.time()
        elapsed = current_time - self.last_update
        self.tokens = min(
            self.tokens + elapsed * self.rate / 8,
            self.burst_size
        )
        self.last_update = current_time
        
        # Check if can transmit
        packet = self.queue[0]
        if self.tokens >= len(packet):
            self.tokens -= len(packet)
            return self.queue.popleft()
        
        return None


class CongestionControl:
    """Implements congestion control algorithms"""
    
    def __init__(
        self,
        algorithm: str = 'cubic',
        initial_window: int = 10
    ):
        """Initialize congestion control
        
        Args:
            algorithm: Congestion control algorithm
            initial_window: Initial congestion window
        """
        self.algorithm = algorithm
        self.cwnd = initial_window  # Congestion window
        self.ssthresh = 65535  # Slow start threshold
        self.rtt = 0.1  # Round trip time estimate
        self.rtt_var = 0.01  # RTT variance
        
        # Algorithm-specific state
        if algorithm == 'cubic':
            self.cubic_state = {
                'w_max': initial_window,
                'k': 0,
                'c': 0.4,
                'beta': 0.7
            }
        
        # Statistics
        self.stats = {
            'packets_sent': 0,
            'packets_lost': 0,
            'retransmissions': 0,
            'avg_throughput': 0.0
        }
    
    def on_ack(self, ack_time: float):
        """Handle acknowledgment
        
        Args:
            ack_time: Time to receive acknowledgment
        """
        # Update RTT
        self._update_rtt(ack_time)
        
        # Increase congestion window
        if self.algorithm == 'cubic':
            self._cubic_increase()
        else:
            self._reno_increase()
        
        self.stats['packets_sent'] += 1
    
    def on_loss(self):
        """Handle packet loss"""
        self.stats['packets_lost'] += 1
        
        # Decrease congestion window
        if self.algorithm == 'cubic':
            self._cubic_decrease()
        else:
            self._reno_decrease()
    
    def _update_rtt(self, sample_rtt: float):
        """Update RTT estimate
        
        Args:
            sample_rtt: Sampled RTT
        """
        alpha = 0.125
        beta = 0.25
        
        self.rtt_var = (1 - beta) * self.rtt_var + beta * abs(sample_rtt - self.rtt)
        self.rtt = (1 - alpha) * self.rtt + alpha * sample_rtt
    
    def _reno_increase(self):
        """TCP Reno congestion window increase"""
        if self.cwnd < self.ssthresh:
            # Slow start
            self.cwnd += 1
        else:
            # Congestion avoidance
            self.cwnd += 1 / self.cwnd
    
    def _reno_decrease(self):
        """TCP Reno congestion window decrease"""
        self.ssthresh = max(self.cwnd / 2, 2)
        self.cwnd = 1  # Fast retransmit
    
    def _cubic_increase(self):
        """CUBIC congestion window increase"""
        state = self.cubic_state
        
        # Time since last decrease
        t = time.time() - state.get('last_decrease', 0)
        
        # CUBIC window function
        k = state['k']
        c = state['c']
        w_cubic = c * (t - k) ** 3 + state['w_max']
        
        if w_cubic > self.cwnd:
            self.cwnd = w_cubic
        else:
            # TCP-friendly region
            self.cwnd += 1 / self.cwnd
    
    def _cubic_decrease(self):
        """CUBIC congestion window decrease"""
        state = self.cubic_state
        
        state['w_max'] = self.cwnd
        self.cwnd = self.cwnd * state['beta']
        state['k'] = (state['w_max'] * (1 - state['beta']) / state['c']) ** (1/3)
        state['last_decrease'] = time.time()
    
    def get_send_rate(self) -> float:
        """Get current sending rate
        
        Returns:
            Sending rate in packets per second
        """
        if self.rtt > 0:
            return self.cwnd / self.rtt
        else:
            return self.cwnd * 10  # Default assumption


class AdaptiveBitrate:
    """Adaptive bitrate control for video/sensor streams"""
    
    def __init__(
        self,
        bitrates: List[float] = None,
        initial_bitrate: Optional[float] = None
    ):
        """Initialize adaptive bitrate
        
        Args:
            bitrates: Available bitrates in bps
            initial_bitrate: Initial bitrate
        """
        if bitrates is None:
            # Default bitrates for drone video streaming
            bitrates = [
                500e3,   # 500 kbps - Low quality
                1e6,     # 1 Mbps - Standard quality
                2e6,     # 2 Mbps - High quality
                5e6,     # 5 Mbps - Full HD
                10e6     # 10 Mbps - 4K
            ]
        
        self.bitrates = sorted(bitrates)
        self.current_index = len(bitrates) // 2
        
        if initial_bitrate:
            # Find closest bitrate
            self.current_index = min(
                range(len(bitrates)),
                key=lambda i: abs(bitrates[i] - initial_bitrate)
            )
        
        # Adaptation state
        self.bandwidth_estimates = deque(maxlen=10)
        self.buffer_level = 0.0  # seconds
        self.target_buffer = 5.0  # seconds
        
        # Statistics
        self.stats = {
            'switches': 0,
            'avg_bitrate': self.bitrates[self.current_index],
            'time_at_bitrate': defaultdict(float)
        }
        
        self.last_update = time.time()
    
    def update(
        self,
        bandwidth_estimate: float,
        buffer_level: float,
        packet_loss: float = 0.0
    ) -> float:
        """Update bitrate based on conditions
        
        Args:
            bandwidth_estimate: Estimated bandwidth in bps
            buffer_level: Current buffer level in seconds
            packet_loss: Packet loss rate
            
        Returns:
            Selected bitrate
        """
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # Update statistics
        self.stats['time_at_bitrate'][self.bitrates[self.current_index]] += elapsed
        
        # Store estimates
        self.bandwidth_estimates.append(bandwidth_estimate)
        self.buffer_level = buffer_level
        
        # Adaptation logic
        if self._should_switch_up():
            self._switch_up()
        elif self._should_switch_down():
            self._switch_down()
        
        # Update statistics
        current_bitrate = self.bitrates[self.current_index]
        self.stats['avg_bitrate'] = (
            0.95 * self.stats['avg_bitrate'] + 0.05 * current_bitrate
        )
        
        self.last_update = current_time
        return current_bitrate
    
    def _should_switch_up(self) -> bool:
        """Check if should switch to higher bitrate
        
        Returns:
            Whether to switch up
        """
        if self.current_index >= len(self.bitrates) - 1:
            return False
        
        # Need stable bandwidth estimates
        if len(self.bandwidth_estimates) < 5:
            return False
        
        # Check bandwidth headroom
        avg_bandwidth = np.mean(list(self.bandwidth_estimates))
        next_bitrate = self.bitrates[self.current_index + 1]
        
        # Need 20% headroom
        if avg_bandwidth < next_bitrate * 1.2:
            return False
        
        # Check buffer health
        if self.buffer_level < self.target_buffer:
            return False
        
        return True
    
    def _should_switch_down(self) -> bool:
        """Check if should switch to lower bitrate
        
        Returns:
            Whether to switch down
        """
        if self.current_index <= 0:
            return False
        
        # Emergency switch on low buffer
        if self.buffer_level < 1.0:
            return True
        
        # Check bandwidth
        if self.bandwidth_estimates:
            recent_bandwidth = list(self.bandwidth_estimates)[-3:]
            avg_bandwidth = np.mean(recent_bandwidth)
            current_bitrate = self.bitrates[self.current_index]
            
            # Switch if bandwidth insufficient
            if avg_bandwidth < current_bitrate * 0.9:
                return True
        
        return False
    
    def _switch_up(self):
        """Switch to higher bitrate"""
        if self.current_index < len(self.bitrates) - 1:
            self.current_index += 1
            self.stats['switches'] += 1
            logger.info(f"Switched up to {self.bitrates[self.current_index]/1e6:.1f} Mbps")
    
    def _switch_down(self):
        """Switch to lower bitrate"""
        if self.current_index > 0:
            self.current_index -= 1
            self.stats['switches'] += 1
            logger.info(f"Switched down to {self.bitrates[self.current_index]/1e6:.1f} Mbps")
    
    def get_current_bitrate(self) -> float:
        """Get current bitrate
        
        Returns:
            Current bitrate in bps
        """
        return self.bitrates[self.current_index]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'current_bitrate': self.bitrates[self.current_index],
            'switches': self.stats['switches'],
            'avg_bitrate': self.stats['avg_bitrate'],
            'time_distribution': dict(self.stats['time_at_bitrate']),
            'buffer_level': self.buffer_level
        }