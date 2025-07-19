"""Return-to-Base Planning with Energy Constraints

This module implements energy-aware return-to-base planning that ensures
agents can safely return before depleting their energy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ReturnToBasePlanner:
    """Plans safe return to base considering energy constraints"""
    
    def __init__(
        self,
        base_locations: List[Tuple[float, float]],
        energy_model: Any,
        safety_margin: float = 0.2,  # 20% safety margin
        wind_model: Optional[Any] = None
    ):
        """Initialize return-to-base planner
        
        Args:
            base_locations: List of base station locations
            energy_model: Energy consumption model
            safety_margin: Safety margin for energy
            wind_model: Wind model for path planning
        """
        self.base_locations = torch.tensor(base_locations)
        self.energy_model = energy_model
        self.safety_margin = safety_margin
        self.wind_model = wind_model
        
        # Path planner
        self.path_planner = EnergyConstrainedPath(
            energy_model=energy_model,
            wind_model=wind_model
        )
        
        # Safe return policy
        self.safe_return = SafeReturnPolicy(
            safety_margin=safety_margin
        )
        
        # Charging station selector
        self.station_selector = ChargingStationSelector(
            station_locations=base_locations
        )
        
        # Emergency return handler
        self.emergency_handler = EmergencyReturn()
        
        logger.info(f"Initialized ReturnToBasePlanner with {len(base_locations)} bases")
    
    def should_return(
        self,
        position: torch.Tensor,
        energy_state: Dict[str, torch.Tensor],
        task_status: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Determine if agent should return to base
        
        Args:
            position: Current position
            energy_state: Current energy state
            task_status: Current task status
            
        Returns:
            Should return flag and decision info
        """
        # Get nearest base
        nearest_base, distance = self._find_nearest_base(position)
        
        # Estimate energy to return
        return_energy = self.path_planner.estimate_return_energy(
            position, self.base_locations[nearest_base], energy_state
        )
        
        # Current available energy
        soc = energy_state['soc']
        capacity = energy_state.get('capacity', 18.5)
        available_energy = soc * capacity
        
        # Required energy with safety margin
        required_energy = return_energy * (1 + self.safety_margin)
        
        # Decision logic
        decision_info = {
            'nearest_base': nearest_base,
            'distance': distance.item(),
            'return_energy': return_energy.item(),
            'available_energy': available_energy.item(),
            'required_energy': required_energy.item(),
            'energy_margin': (available_energy - required_energy).item()
        }
        
        # Check if we must return
        must_return = available_energy <= required_energy
        
        # Additional checks
        if not must_return and task_status:
            # Check if task can be completed
            if 'estimated_energy' in task_status:
                total_energy = return_energy + task_status['estimated_energy']
                must_return = available_energy <= total_energy * (1 + self.safety_margin)
        
        # Emergency check
        if soc < 0.15:  # Below 15%
            must_return = True
            decision_info['emergency'] = True
        
        decision_info['must_return'] = must_return
        
        return must_return, decision_info
    
    def plan_return_path(
        self,
        position: torch.Tensor,
        energy_state: Dict[str, torch.Tensor],
        target_base: Optional[int] = None
    ) -> Dict[str, Any]:
        """Plan optimal return path to base
        
        Args:
            position: Current position
            energy_state: Current energy state
            target_base: Target base index (None for nearest)
            
        Returns:
            Return path plan
        """
        # Select target base
        if target_base is None:
            target_base, _ = self._find_nearest_base(position)
        
        base_position = self.base_locations[target_base]
        
        # Plan energy-optimal path
        path = self.path_planner.plan_path(
            position, base_position, energy_state
        )
        
        # Verify path feasibility
        path_energy = self.path_planner.compute_path_energy(
            path['waypoints'], energy_state
        )
        
        available_energy = energy_state['soc'] * energy_state.get('capacity', 18.5)
        
        plan = {
            'target_base': target_base,
            'waypoints': path['waypoints'],
            'estimated_energy': path_energy.item(),
            'estimated_time': path['estimated_time'],
            'feasible': available_energy > path_energy * (1 + self.safety_margin),
            'optimization_metric': path.get('optimization_metric', 'energy')
        }
        
        # If not feasible, try emergency path
        if not plan['feasible']:
            emergency_path = self.emergency_handler.plan_emergency_return(
                position, base_position, energy_state
            )
            plan['emergency_path'] = emergency_path
            plan['emergency'] = True
        
        return plan
    
    def _find_nearest_base(
        self,
        position: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """Find nearest base station
        
        Args:
            position: Current position
            
        Returns:
            Base index and distance
        """
        distances = torch.norm(self.base_locations - position, dim=1)
        nearest_idx = torch.argmin(distances)
        
        return nearest_idx.item(), distances[nearest_idx]


class EnergyConstrainedPath(nn.Module):
    """Plans paths with energy constraints"""
    
    def __init__(
        self,
        energy_model: Any,
        wind_model: Optional[Any] = None,
        num_waypoints: int = 10
    ):
        """Initialize path planner
        
        Args:
            energy_model: Energy consumption model
            wind_model: Wind model
            num_waypoints: Number of waypoints
        """
        super().__init__()
        
        self.energy_model = energy_model
        self.wind_model = wind_model
        self.num_waypoints = num_waypoints
        
        # Path optimization network
        self.path_net = nn.Sequential(
            nn.Linear(6, 128),  # start(2) + end(2) + energy_state(2)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_waypoints * 2)  # waypoint positions
        )
        
        # Energy prediction network
        self.energy_net = nn.Sequential(
            nn.Linear(num_waypoints * 2 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def plan_path(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        energy_state: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Plan energy-optimal path
        
        Args:
            start: Start position
            end: End position
            energy_state: Energy state
            
        Returns:
            Path plan
        """
        # Direct path as baseline
        direct_distance = torch.norm(end - start)
        direct_energy = self.estimate_direct_energy(start, end, energy_state)
        
        # Neural network path
        features = torch.cat([
            start,
            end,
            torch.tensor([energy_state['soc'], energy_state.get('power', 50.0)])
        ])
        
        waypoints_flat = self.path_net(features)
        waypoints = waypoints_flat.view(self.num_waypoints, 2)
        
        # Add start and end
        full_path = torch.cat([
            start.unsqueeze(0),
            waypoints,
            end.unsqueeze(0)
        ])
        
        # Optimize path for energy
        optimized_path = self._optimize_path(
            full_path, energy_state
        )
        
        # Compute path metrics
        path_energy = self.compute_path_energy(optimized_path, energy_state)
        path_time = self._estimate_path_time(optimized_path)
        
        return {
            'waypoints': optimized_path,
            'direct_distance': direct_distance.item(),
            'direct_energy': direct_energy.item(),
            'path_energy': path_energy.item(),
            'energy_savings': (direct_energy - path_energy).item(),
            'estimated_time': path_time
        }
    
    def estimate_return_energy(
        self,
        position: torch.Tensor,
        base: torch.Tensor,
        energy_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Estimate energy needed to return to base
        
        Args:
            position: Current position
            base: Base position
            energy_state: Energy state
            
        Returns:
            Estimated energy consumption
        """
        # Quick estimate using direct path
        direct_energy = self.estimate_direct_energy(position, base, energy_state)
        
        # Add margin for wind and maneuvers
        if self.wind_model is not None:
            wind_factor = 1.2  # 20% extra for wind
        else:
            wind_factor = 1.1
        
        return direct_energy * wind_factor
    
    def estimate_direct_energy(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        energy_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Estimate energy for direct path
        
        Args:
            start: Start position
            end: End position
            energy_state: Energy state
            
        Returns:
            Energy estimate
        """
        distance = torch.norm(end - start)
        
        # Average speed (m/s)
        avg_speed = 10.0
        
        # Flight time
        flight_time = distance / avg_speed
        
        # Average power (W)
        avg_power = energy_state.get('power', 80.0)
        
        # Energy consumption (Wh)
        energy = avg_power * flight_time / 3600.0
        
        return energy
    
    def compute_path_energy(
        self,
        waypoints: torch.Tensor,
        energy_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute energy for a path
        
        Args:
            waypoints: Path waypoints
            energy_state: Energy state
            
        Returns:
            Total energy consumption
        """
        total_energy = torch.tensor(0.0)
        
        # Sum energy for each segment
        for i in range(len(waypoints) - 1):
            segment_energy = self._compute_segment_energy(
                waypoints[i], waypoints[i+1], energy_state
            )
            total_energy += segment_energy
        
        return total_energy
    
    def _compute_segment_energy(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        energy_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute energy for path segment
        
        Args:
            start: Segment start
            end: Segment end
            energy_state: Energy state
            
        Returns:
            Segment energy
        """
        # Basic physics model
        distance = torch.norm(end - start)
        
        # Height change
        if len(start) > 2 and len(end) > 2:
            height_change = end[2] - start[2]
        else:
            height_change = torch.tensor(0.0)
        
        # Base power for level flight
        base_power = 80.0  # W
        
        # Additional power for climbing (negative for descending)
        climb_power = height_change * 10.0  # W per m/s climb
        
        # Wind effect if available
        if self.wind_model is not None:
            wind_effect = self.wind_model.get_wind_effect(start, end)
            wind_power = wind_effect * 20.0
        else:
            wind_power = 0.0
        
        # Total power
        total_power = base_power + climb_power + wind_power
        
        # Time for segment
        avg_speed = 10.0  # m/s
        time = distance / avg_speed
        
        # Energy (Wh)
        energy = total_power * time / 3600.0
        
        return energy
    
    def _optimize_path(
        self,
        initial_path: torch.Tensor,
        energy_state: Dict[str, torch.Tensor],
        iterations: int = 20
    ) -> torch.Tensor:
        """Optimize path for minimum energy
        
        Args:
            initial_path: Initial waypoints
            energy_state: Energy state
            iterations: Optimization iterations
            
        Returns:
            Optimized path
        """
        path = initial_path.clone()
        path.requires_grad_(True)
        
        optimizer = torch.optim.Adam([path], lr=0.1)
        
        for _ in range(iterations):
            optimizer.zero_grad()
            
            # Compute energy
            energy = self.compute_path_energy(path, energy_state)
            
            # Add smoothness penalty
            smoothness = torch.tensor(0.0)
            for i in range(1, len(path) - 1):
                curvature = path[i-1] - 2*path[i] + path[i+1]
                smoothness += torch.norm(curvature)
            
            # Total loss
            loss = energy + 0.1 * smoothness
            
            loss.backward()
            optimizer.step()
            
            # Keep endpoints fixed
            with torch.no_grad():
                path[0] = initial_path[0]
                path[-1] = initial_path[-1]
        
        return path.detach()
    
    def _estimate_path_time(self, waypoints: torch.Tensor) -> float:
        """Estimate time to traverse path
        
        Args:
            waypoints: Path waypoints
            
        Returns:
            Time in seconds
        """
        total_distance = 0.0
        
        for i in range(len(waypoints) - 1):
            segment_distance = torch.norm(waypoints[i+1] - waypoints[i])
            total_distance += segment_distance
        
        avg_speed = 10.0  # m/s
        total_time = total_distance / avg_speed
        
        return total_time.item()


class SafeReturnPolicy(nn.Module):
    """Policy for safe return decisions"""
    
    def __init__(
        self,
        safety_margin: float = 0.2,
        hidden_dim: int = 64
    ):
        """Initialize safe return policy
        
        Args:
            safety_margin: Safety margin
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.safety_margin = safety_margin
        
        # Decision network
        self.decision_net = nn.Sequential(
            nn.Linear(8, hidden_dim),  # position(3) + energy(2) + task(3)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [continue, return, emergency]
        )
    
    def forward(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Make return decision
        
        Args:
            state: Current state
            
        Returns:
            Decision outputs
        """
        # Extract features
        features = torch.cat([
            state.get('position', torch.zeros(3)),
            torch.tensor([state['soc'], state.get('power', 50.0)]),
            state.get('task_features', torch.zeros(3))
        ])
        
        # Get decision logits
        logits = self.decision_net(features)
        probs = F.softmax(logits, dim=-1)
        
        # Decision
        decision_idx = torch.argmax(probs)
        decisions = ['continue', 'return', 'emergency']
        
        return {
            'decision': decisions[decision_idx],
            'confidence': probs[decision_idx],
            'continue_prob': probs[0],
            'return_prob': probs[1],
            'emergency_prob': probs[2]
        }


class ChargingStationSelector:
    """Selects optimal charging station"""
    
    def __init__(
        self,
        station_locations: List[Tuple[float, float]],
        station_capacity: Optional[List[int]] = None
    ):
        """Initialize station selector
        
        Args:
            station_locations: Charging station locations
            station_capacity: Capacity of each station
        """
        self.station_locations = torch.tensor(station_locations)
        self.num_stations = len(station_locations)
        
        if station_capacity is None:
            self.station_capacity = [2] * self.num_stations
        else:
            self.station_capacity = station_capacity
        
        # Station occupancy tracking
        self.station_occupancy = [0] * self.num_stations
    
    def select_station(
        self,
        position: torch.Tensor,
        energy_state: Dict[str, torch.Tensor],
        urgency: float = 0.5
    ) -> Dict[str, Any]:
        """Select optimal charging station
        
        Args:
            position: Current position
            energy_state: Energy state
            urgency: Urgency level (0-1)
            
        Returns:
            Station selection info
        """
        # Compute distances
        distances = torch.norm(self.station_locations - position, dim=1)
        
        # Compute scores
        scores = []
        
        for i in range(self.num_stations):
            # Distance score (closer is better)
            dist_score = 1.0 / (distances[i] + 1.0)
            
            # Availability score
            occupancy_rate = self.station_occupancy[i] / self.station_capacity[i]
            availability_score = 1.0 - occupancy_rate
            
            # Combined score with urgency weighting
            if urgency > 0.7:
                # High urgency: prioritize distance
                score = 0.8 * dist_score + 0.2 * availability_score
            else:
                # Low urgency: balance distance and availability
                score = 0.5 * dist_score + 0.5 * availability_score
            
            scores.append(score)
        
        scores = torch.tensor(scores)
        best_station = torch.argmax(scores)
        
        return {
            'station_id': best_station.item(),
            'distance': distances[best_station].item(),
            'score': scores[best_station].item(),
            'occupancy': self.station_occupancy[best_station.item()],
            'capacity': self.station_capacity[best_station.item()],
            'available_slots': (self.station_capacity[best_station.item()] -
                              self.station_occupancy[best_station.item()])
        }
    
    def update_occupancy(self, station_id: int, change: int):
        """Update station occupancy
        
        Args:
            station_id: Station ID
            change: Change in occupancy (+1 for arrival, -1 for departure)
        """
        self.station_occupancy[station_id] += change
        self.station_occupancy[station_id] = max(
            0, min(self.station_occupancy[station_id],
                  self.station_capacity[station_id])
        )


class EmergencyReturn:
    """Handles emergency return scenarios"""
    
    def __init__(self):
        """Initialize emergency return handler"""
        self.emergency_speed = 15.0  # m/s - faster than normal
        self.emergency_altitude = 30.0  # m - lower altitude
    
    def plan_emergency_return(
        self,
        position: torch.Tensor,
        base: torch.Tensor,
        energy_state: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Plan emergency return path
        
        Args:
            position: Current position
            base: Base position
            energy_state: Energy state
            
        Returns:
            Emergency path plan
        """
        # Direct path only
        direct_distance = torch.norm(base - position)
        
        # Time at emergency speed
        time = direct_distance / self.emergency_speed
        
        # Power in emergency mode (reduced)
        emergency_power = 60.0  # W - minimum power
        
        # Energy needed
        energy = emergency_power * time / 3600.0
        
        # Check feasibility
        available = energy_state['soc'] * energy_state.get('capacity', 18.5)
        feasible = available > energy * 1.1  # 10% margin
        
        plan = {
            'path_type': 'emergency_direct',
            'waypoints': torch.stack([position, base]),
            'distance': direct_distance.item(),
            'time': time.item(),
            'energy': energy.item(),
            'feasible': feasible,
            'speed': self.emergency_speed,
            'altitude': self.emergency_altitude
        }
        
        if not feasible:
            # Find nearest safe landing spot
            landing_spot = self._find_emergency_landing(position, base)
            plan['emergency_landing'] = landing_spot
            plan['landing_required'] = True
        
        return plan
    
    def _find_emergency_landing(
        self,
        position: torch.Tensor,
        base: torch.Tensor
    ) -> torch.Tensor:
        """Find emergency landing location
        
        Args:
            position: Current position
            base: Base position
            
        Returns:
            Landing position
        """
        # Simple strategy: land at fraction of distance to base
        direction = base - position
        direction = direction / torch.norm(direction)
        
        # Land 1km ahead or halfway, whichever is closer
        landing_distance = min(1000.0, torch.norm(base - position) * 0.5)
        
        landing_position = position + direction * landing_distance
        
        return landing_position