"""Episode Manager for Multi-Agent Environment

This module manages episode-specific logic, scenarios, and termination
conditions for different mission types.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """Target or waypoint in the environment"""
    position: np.ndarray
    radius: float = 2.0
    type: str = "waypoint"  # waypoint, search_target, delivery_point
    id: int = 0
    found: bool = False
    visited_by: List[int] = field(default_factory=list)
    
    def is_reached_by(self, position: np.ndarray) -> bool:
        """Check if position is within target radius"""
        return np.linalg.norm(position - self.position) <= self.radius


class ScenarioBase(ABC):
    """Base class for scenario implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize scenario
        
        Args:
            config: Scenario configuration
        """
        self.config = config
        self.targets: List[Target] = []
        self.start_time: float = 0.0
        self.max_time: float = config.get("max_time", 300.0)
        
    @abstractmethod
    def reset(self, num_agents: int) -> Dict[str, Any]:
        """Reset scenario for new episode"""
        pass
    
    @abstractmethod
    def step(self, agent_states: Dict[int, Dict[str, Any]], time: float) -> Dict[int, Any]:
        """Update scenario state"""
        pass
    
    @abstractmethod
    def get_agent_rewards(self, agent_id: int, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get scenario-specific rewards for agent"""
        pass
    
    @abstractmethod
    def is_mission_completed(self, agent_id: int) -> bool:
        """Check if mission is completed for agent"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get scenario information"""
        pass


class SearchRescueScenario(ScenarioBase):
    """Search and rescue scenario"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.search_area_size = config.get("search_area_size", (200, 200))
        self.num_targets = config.get("num_targets", 3)
        self.coverage_map = None
        self.coverage_resolution = 5.0  # meters per cell
        self.agent_distances = {}  # Track distances for progress
        
    def reset(self, num_agents: int) -> Dict[str, Any]:
        """Reset search and rescue scenario"""
        self.targets.clear()
        self.agent_distances.clear()
        
        # Generate random target positions
        for i in range(self.num_targets):
            position = np.array([
                np.random.uniform(10, self.search_area_size[0] - 10),
                np.random.uniform(10, self.search_area_size[1] - 10),
                0.0  # Ground level
            ])
            
            target = Target(
                position=position,
                radius=5.0,
                type="search_target",
                id=i
            )
            self.targets.append(target)
        
        # Initialize coverage map
        map_size = (
            int(self.search_area_size[0] / self.coverage_resolution),
            int(self.search_area_size[1] / self.coverage_resolution)
        )
        self.coverage_map = np.zeros(map_size, dtype=bool)
        
        # Initialize agent distances
        for agent_id in range(num_agents):
            self.agent_distances[agent_id] = {
                target.id: float('inf') for target in self.targets
            }
        
        return {
            "scenario_type": "search_rescue",
            "num_targets": self.num_targets,
            "targets_found": 0,
            "area_covered": 0.0
        }
    
    def step(self, agent_states: Dict[int, Dict[str, Any]], time: float) -> Dict[int, Any]:
        """Update search and rescue scenario"""
        targets_found = 0
        
        # Check target discovery
        for target in self.targets:
            if not target.found:
                for agent_id, state in agent_states.items():
                    if target.is_reached_by(state["position"]):
                        target.found = True
                        target.visited_by.append(agent_id)
                        logger.info(f"Target {target.id} found by agent {agent_id}")
                        break
            
            if target.found:
                targets_found += 1
        
        # Update coverage map
        for agent_id, state in agent_states.items():
            pos = state["position"]
            # Convert to grid coordinates
            grid_x = int(pos[0] / self.coverage_resolution)
            grid_y = int(pos[1] / self.coverage_resolution)
            
            # Mark covered area (with sensor radius)
            sensor_radius_cells = int(10.0 / self.coverage_resolution)  # 10m sensor
            for dx in range(-sensor_radius_cells, sensor_radius_cells + 1):
                for dy in range(-sensor_radius_cells, sensor_radius_cells + 1):
                    gx, gy = grid_x + dx, grid_y + dy
                    if (0 <= gx < self.coverage_map.shape[0] and 
                        0 <= gy < self.coverage_map.shape[1]):
                        self.coverage_map[gx, gy] = True
        
        # Calculate coverage percentage
        area_covered = np.sum(self.coverage_map) / self.coverage_map.size
        
        return {
            "targets_found": targets_found,
            "area_covered": area_covered,
            "time_elapsed": time - self.start_time
        }
    
    def get_agent_rewards(self, agent_id: int, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get search-specific rewards"""
        rewards = {}
        
        # Check if agent found a target
        for target in self.targets:
            if agent_id in target.visited_by:
                rewards["target_reached"] = True
                break
        
        # Distance to nearest unfound target
        min_distance = float('inf')
        for target in self.targets:
            if not target.found:
                distance = np.linalg.norm(agent_state["position"] - target.position)
                
                # Track progress
                prev_distance = self.agent_distances.get(agent_id, {}).get(target.id, distance)
                self.agent_distances[agent_id][target.id] = distance
                
                if distance < min_distance:
                    min_distance = distance
                    rewards["distance_to_target"] = distance
                    rewards["prev_distance_to_target"] = prev_distance
        
        # Mission completion
        if all(target.found for target in self.targets):
            rewards["mission_completed"] = True
        
        return rewards
    
    def is_mission_completed(self, agent_id: int) -> bool:
        """Check if all targets found"""
        return all(target.found for target in self.targets)
    
    def get_info(self) -> Dict[str, Any]:
        """Get scenario information"""
        return {
            "type": "search_rescue",
            "targets_total": len(self.targets),
            "targets_found": sum(1 for t in self.targets if t.found),
            "coverage": float(np.sum(self.coverage_map)) / self.coverage_map.size
        }


class FormationScenario(ScenarioBase):
    """Formation flying scenario"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.formation_type = config.get("formation_type", "line")
        self.formation_spacing = config.get("formation_spacing", 5.0)
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.formation_errors = {}
        
    def reset(self, num_agents: int) -> Dict[str, Any]:
        """Reset formation scenario"""
        self.targets.clear()
        self.formation_errors.clear()
        self.current_waypoint_idx = 0
        
        # Generate waypoints
        waypoint_positions = [
            np.array([50, 50, 15]),
            np.array([150, 50, 20]),
            np.array([150, 150, 15]),
            np.array([50, 150, 20]),
            np.array([100, 100, 15])
        ]
        
        for i, pos in enumerate(waypoint_positions):
            self.targets.append(Target(
                position=pos,
                radius=10.0,
                type="waypoint",
                id=i
            ))
        
        # Initialize formation errors
        for agent_id in range(num_agents):
            self.formation_errors[agent_id] = []
        
        return {
            "scenario_type": "formation",
            "formation_type": self.formation_type,
            "num_waypoints": len(self.targets),
            "current_waypoint": 0
        }
    
    def step(self, agent_states: Dict[int, Dict[str, Any]], time: float) -> Dict[int, Any]:
        """Update formation scenario"""
        if self.current_waypoint_idx >= len(self.targets):
            return {"mission_completed": True}
        
        current_target = self.targets[self.current_waypoint_idx]
        
        # Check if formation reached waypoint
        positions = [state["position"] for state in agent_states.values()]
        if positions:
            centroid = np.mean(positions, axis=0)
            
            if current_target.is_reached_by(centroid):
                current_target.found = True
                self.current_waypoint_idx += 1
                logger.info(f"Formation reached waypoint {current_target.id}")
        
        # Calculate formation error
        formation_error = self._calculate_formation_error(positions)
        
        # Store errors for each agent
        for agent_id in agent_states:
            self.formation_errors[agent_id].append(formation_error)
        
        return {
            "current_waypoint": self.current_waypoint_idx,
            "formation_error": formation_error,
            "waypoints_completed": self.current_waypoint_idx
        }
    
    def _calculate_formation_error(self, positions: List[np.ndarray]) -> float:
        """Calculate how well agents maintain formation"""
        if len(positions) < 2:
            return 0.0
        
        if self.formation_type == "line":
            # Check if agents are in a line
            # Simplified: calculate variance from best-fit line
            positions_2d = np.array([p[:2] for p in positions])
            
            if len(positions_2d) > 2:
                # Fit line using PCA
                centered = positions_2d - np.mean(positions_2d, axis=0)
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                
                # Project points onto principal axis
                main_axis = eigenvectors[:, np.argmax(eigenvalues)]
                projections = centered @ main_axis
                perpendicular = centered - np.outer(projections, main_axis)
                
                # Average perpendicular distance
                error = np.mean(np.linalg.norm(perpendicular, axis=1))
            else:
                error = 0.0
                
        elif self.formation_type == "v":
            # V-formation error
            # Simplified: check angle and spacing
            if len(positions) >= 3:
                # Assume first agent is leader
                leader = positions[0]
                followers = positions[1:]
                
                # Check V-shape
                errors = []
                for i, follower in enumerate(followers):
                    rel_pos = follower - leader
                    
                    # Expected position in V
                    side = 1 if i % 2 == 0 else -1
                    expected_lateral = side * self.formation_spacing * ((i // 2) + 1)
                    expected_back = -self.formation_spacing * ((i // 2) + 1)
                    
                    error_lateral = abs(rel_pos[1] - expected_lateral)
                    error_back = abs(rel_pos[0] - expected_back)
                    
                    errors.append(np.sqrt(error_lateral**2 + error_back**2))
                
                error = np.mean(errors)
            else:
                error = 0.0
                
        else:
            # Default: average pairwise distance variance
            distances = []
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    distances.append(np.linalg.norm(positions[i] - positions[j]))
            
            if distances:
                error = np.std(distances)
            else:
                error = 0.0
        
        return error
    
    def get_agent_rewards(self, agent_id: int, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get formation-specific rewards"""
        rewards = {}
        
        # Formation maintenance
        if agent_id in self.formation_errors and self.formation_errors[agent_id]:
            current_error = self.formation_errors[agent_id][-1]
            rewards["formation_error"] = current_error
            
            # Reward good formation
            if current_error < 2.0:
                rewards["formation_maintained"] = True
        
        # Waypoint progress
        rewards["waypoints_completed"] = self.current_waypoint_idx
        
        # Mission completion
        if self.current_waypoint_idx >= len(self.targets):
            rewards["mission_completed"] = True
        
        return rewards
    
    def is_mission_completed(self, agent_id: int) -> bool:
        """Check if all waypoints visited"""
        return self.current_waypoint_idx >= len(self.targets)
    
    def get_info(self) -> Dict[str, Any]:
        """Get scenario information"""
        avg_error = np.mean([
            errors[-1] if errors else 0.0 
            for errors in self.formation_errors.values()
        ])
        
        return {
            "type": "formation",
            "formation_type": self.formation_type,
            "waypoints_total": len(self.targets),
            "waypoints_completed": self.current_waypoint_idx,
            "avg_formation_error": avg_error
        }


class EpisodeManager:
    """Manages episode logic and scenarios"""
    
    def __init__(
        self,
        max_steps: int = 10000,
        timestep: float = 0.01
    ):
        """Initialize episode manager
        
        Args:
            max_steps: Maximum steps per episode
            timestep: Simulation timestep
        """
        self.max_steps = max_steps
        self.timestep = timestep
        self.current_step = 0
        self.episode_time = 0.0
        
        # Scenario
        self.scenario = None
        self.scenario_type = "default"
        
        # Episode statistics
        self.episode_stats = {
            "total_distance": 0.0,
            "total_energy": 0.0,
            "collisions": 0,
            "targets_found": 0
        }
        
        logger.info("Initialized EpisodeManager")
    
    def reset(self, scenario_type: str = "default", **kwargs):
        """Reset episode with specified scenario
        
        Args:
            scenario_type: Type of scenario
            **kwargs: Additional scenario configuration
        """
        self.current_step = 0
        self.episode_time = 0.0
        self.scenario_type = scenario_type
        
        # Reset statistics
        self.episode_stats = {
            "total_distance": 0.0,
            "total_energy": 0.0,
            "collisions": 0,
            "targets_found": 0
        }
        
        # Create scenario
        if scenario_type == "search_rescue":
            config = {
                "search_area_size": kwargs.get("world_size", (200, 200))[:2],
                "num_targets": kwargs.get("num_targets", 3),
                "max_time": kwargs.get("max_time", 300.0)
            }
            self.scenario = SearchRescueScenario(config)
        elif scenario_type == "formation":
            config = {
                "formation_type": kwargs.get("formation_type", "line"),
                "formation_spacing": kwargs.get("formation_spacing", 5.0),
                "max_time": kwargs.get("max_time", 240.0)
            }
            self.scenario = FormationScenario(config)
        else:
            # Default scenario (free flight)
            self.scenario = None
        
        # Initialize scenario
        if self.scenario:
            num_agents = kwargs.get("num_agents", 5)
            scenario_info = self.scenario.reset(num_agents)
            logger.info(f"Reset episode with {scenario_type} scenario: {scenario_info}")
    
    def step(self):
        """Update episode state"""
        self.current_step += 1
        self.episode_time = self.current_step * self.timestep
    
    def update_scenario(self, agent_states: Dict[int, Dict[str, Any]]):
        """Update scenario with agent states
        
        Args:
            agent_states: Current states of all agents
        """
        if self.scenario:
            scenario_state = self.scenario.step(agent_states, self.episode_time)
            
            # Update episode stats
            if "targets_found" in scenario_state:
                self.episode_stats["targets_found"] = scenario_state["targets_found"]
    
    def get_agent_rewards(self, agent_id: int) -> Dict[str, Any]:
        """Get scenario-specific rewards for agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Reward information
        """
        if self.scenario:
            # Get agent state from somewhere (would be passed in real implementation)
            # For now, return empty dict
            return self.scenario.get_agent_rewards(agent_id, {})
        
        return {}
    
    def is_mission_completed(self, agent_id: int) -> bool:
        """Check if mission is completed for agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Whether mission is completed
        """
        if self.scenario:
            return self.scenario.is_mission_completed(agent_id)
        
        return False
    
    def get_targets(self) -> List[Dict[str, Any]]:
        """Get current targets/waypoints
        
        Returns:
            List of target information
        """
        if self.scenario and hasattr(self.scenario, 'targets'):
            return [
                {
                    "position": target.position.tolist(),
                    "radius": target.radius,
                    "type": target.type,
                    "found": target.found
                }
                for target in self.scenario.targets
            ]
        
        return []
    
    def get_agent_info(self, agent_id: int) -> Dict[str, Any]:
        """Get scenario-specific info for agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent-specific scenario information
        """
        info = {
            "episode_time": self.episode_time,
            "episode_progress": self.current_step / self.max_steps
        }
        
        if self.scenario:
            scenario_info = self.scenario.get_info()
            info.update(scenario_info)
        
        return info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics
        
        Returns:
            Episode statistics
        """
        stats = self.episode_stats.copy()
        stats["duration"] = self.episode_time
        stats["steps"] = self.current_step
        
        if self.scenario:
            stats["scenario_info"] = self.scenario.get_info()
        
        return stats