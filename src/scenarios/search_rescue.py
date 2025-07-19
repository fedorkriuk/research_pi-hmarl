"""Search and Rescue Scenario

This module implements a multi-agent search and rescue scenario where
agents collaborate to find and rescue victims in a disaster area.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class VictimStatus(Enum):
    """Status of victims"""
    UNDISCOVERED = "undiscovered"
    DETECTED = "detected"
    BEING_RESCUED = "being_rescued"
    RESCUED = "rescued"
    LOST = "lost"


class AgentRole(Enum):
    """Agent roles in search and rescue"""
    SEARCHER = "searcher"
    RESCUER = "rescuer"
    COORDINATOR = "coordinator"
    SUPPORT = "support"


@dataclass
class VictimModel:
    """Model for victims in the scenario"""
    victim_id: str
    position: np.ndarray
    severity: float  # 0-1, higher is more critical
    discovery_time: Optional[float] = None
    rescue_time: Optional[float] = None
    status: VictimStatus = VictimStatus.UNDISCOVERED
    health: float = 1.0  # Decreases over time
    required_rescuers: int = 1
    
    def update_health(self, dt: float, decay_rate: float = 0.01):
        """Update victim health over time
        
        Args:
            dt: Time step
            decay_rate: Health decay rate
        """
        if self.status not in [VictimStatus.RESCUED, VictimStatus.LOST]:
            self.health = max(0, self.health - decay_rate * dt * self.severity)
            
            if self.health == 0:
                self.status = VictimStatus.LOST


@dataclass
class SearchPattern:
    """Search pattern for agents"""
    pattern_type: str  # 'spiral', 'grid', 'random', 'informed'
    start_position: np.ndarray
    parameters: Dict[str, Any]
    waypoints: List[np.ndarray]
    current_index: int = 0
    
    def get_next_waypoint(self) -> Optional[np.ndarray]:
        """Get next waypoint in pattern
        
        Returns:
            Next waypoint or None if complete
        """
        if self.current_index < len(self.waypoints):
            waypoint = self.waypoints[self.current_index]
            self.current_index += 1
            return waypoint
        return None
    
    def reset(self):
        """Reset pattern to beginning"""
        self.current_index = 0


class SearchAgent:
    """Individual search and rescue agent"""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        position: np.ndarray,
        sensor_range: float = 10.0,
        move_speed: float = 2.0,
        communication_range: float = 50.0
    ):
        """Initialize search agent
        
        Args:
            agent_id: Unique agent identifier
            role: Agent's role
            position: Initial position
            sensor_range: Detection range
            move_speed: Movement speed
            communication_range: Communication range
        """
        self.agent_id = agent_id
        self.role = role
        self.position = position.copy()
        self.sensor_range = sensor_range
        self.move_speed = move_speed
        self.communication_range = communication_range
        
        # State
        self.detected_victims: Set[str] = set()
        self.assigned_victim: Optional[str] = None
        self.search_pattern: Optional[SearchPattern] = None
        self.path: List[np.ndarray] = []
        self.energy = 1.0  # Energy level
        
        # Knowledge base
        self.victim_beliefs: Dict[str, Dict[str, Any]] = {}
        self.teammate_positions: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized SearchAgent {agent_id} with role {role.value}")
    
    def sense_environment(
        self,
        victims: List[VictimModel],
        obstacles: List[np.ndarray],
        time: float = 0.0
    ) -> List[str]:
        """Sense nearby victims
        
        Args:
            victims: List of all victims
            obstacles: List of obstacles
            
        Returns:
            List of detected victim IDs
        """
        newly_detected = []
        
        for victim in victims:
            if victim.status == VictimStatus.UNDISCOVERED:
                distance = np.linalg.norm(victim.position - self.position)
                
                # Check if within sensor range and line of sight
                if distance <= self.sensor_range:
                    if self._has_line_of_sight(victim.position, obstacles):
                        victim.status = VictimStatus.DETECTED
                        victim.discovery_time = time
                        
                        self.detected_victims.add(victim.victim_id)
                        newly_detected.append(victim.victim_id)
                        
                        # Update beliefs
                        self.victim_beliefs[victim.victim_id] = {
                            'position': victim.position.copy(),
                            'severity': victim.severity,
                            'last_seen': time
                        }
        
        return newly_detected
    
    def plan_action(
        self,
        victims: Dict[str, VictimModel],
        teammates: List['SearchAgent'],
        obstacles: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Plan next action
        
        Args:
            victims: Dictionary of victims
            teammates: List of teammate agents
            obstacles: List of obstacles
            
        Returns:
            Action dictionary
        """
        # Update teammate positions
        for teammate in teammates:
            if teammate.agent_id != self.agent_id:
                self.teammate_positions[teammate.agent_id] = teammate.position.copy()
        
        action = {'type': 'move', 'target': None}
        
        if self.role == AgentRole.SEARCHER:
            action = self._plan_search(victims, obstacles)
        elif self.role == AgentRole.RESCUER:
            action = self._plan_rescue(victims, obstacles)
        elif self.role == AgentRole.COORDINATOR:
            action = self._plan_coordination(victims, teammates)
        
        return action
    
    def _plan_search(
        self,
        victims: Dict[str, VictimModel],
        obstacles: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Plan search action
        
        Args:
            victims: Dictionary of victims
            obstacles: List of obstacles
            
        Returns:
            Search action
        """
        # If no search pattern, request one
        if not self.search_pattern:
            return {'type': 'request_pattern', 'area': 'unassigned'}
        
        # Follow search pattern
        next_waypoint = self.search_pattern.get_next_waypoint()
        
        if next_waypoint is not None:
            # Plan path to waypoint
            path = self._plan_path(next_waypoint, obstacles)
            
            if path:
                self.path = path
                return {'type': 'move', 'target': path[0]}
        
        # Pattern complete, request new one
        return {'type': 'request_pattern', 'area': 'next'}
    
    def _plan_rescue(
        self,
        victims: Dict[str, VictimModel],
        obstacles: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Plan rescue action
        
        Args:
            victims: Dictionary of victims
            obstacles: List of obstacles
            
        Returns:
            Rescue action
        """
        # If assigned to victim
        if self.assigned_victim and self.assigned_victim in victims:
            victim = victims[self.assigned_victim]
            
            # Move to victim
            distance = np.linalg.norm(victim.position - self.position)
            
            if distance > 2.0:  # Not close enough
                path = self._plan_path(victim.position, obstacles)
                if path:
                    self.path = path
                    return {'type': 'move', 'target': path[0]}
            else:
                # Start rescue
                return {
                    'type': 'rescue',
                    'victim_id': self.assigned_victim,
                    'position': self.position
                }
        
        # No assignment, request one
        return {'type': 'request_assignment'}
    
    def _plan_coordination(
        self,
        victims: Dict[str, VictimModel],
        teammates: List['SearchAgent']
    ) -> Dict[str, Any]:
        """Plan coordination action
        
        Args:
            victims: Dictionary of victims
            teammates: List of teammates
            
        Returns:
            Coordination action
        """
        # Analyze situation
        unrescued_victims = [
            v for v in victims.values()
            if v.status in [VictimStatus.DETECTED, VictimStatus.BEING_RESCUED]
        ]
        
        # Sort by priority (severity and health)
        unrescued_victims.sort(
            key=lambda v: v.severity * (1 - v.health),
            reverse=True
        )
        
        # Create assignments
        assignments = {}
        available_rescuers = [
            t for t in teammates
            if t.role == AgentRole.RESCUER and not t.assigned_victim
        ]
        
        for victim in unrescued_victims[:len(available_rescuers)]:
            # Find closest rescuer
            best_rescuer = min(
                available_rescuers,
                key=lambda r: np.linalg.norm(r.position - victim.position)
            )
            
            assignments[best_rescuer.agent_id] = victim.victim_id
            available_rescuers.remove(best_rescuer)
        
        return {
            'type': 'coordinate',
            'assignments': assignments,
            'priority_victims': [v.victim_id for v in unrescued_victims[:3]]
        }
    
    def _plan_path(
        self,
        target: np.ndarray,
        obstacles: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Plan path to target avoiding obstacles
        
        Args:
            target: Target position
            obstacles: List of obstacles
            
        Returns:
            Path as list of waypoints
        """
        # Improved pathfinding with better obstacle avoidance
        # Check if direct path is clear
        if self._has_line_of_sight(target, obstacles):
            return [target]
        
        # Find path around obstacles
        waypoints = []
        current = self.position.copy()
        
        # Simple but more robust obstacle avoidance
        direction_to_target = target[:2] - current[:2]
        distance_to_target = np.linalg.norm(direction_to_target)
        
        if distance_to_target > 0:
            direction_to_target = direction_to_target / distance_to_target
            
            # Try different avoidance directions
            for obstacle in obstacles:
                if self._line_intersects_circle(current[:2], target[:2], obstacle[:2], 6.0):
                    # Calculate perpendicular directions
                    perp_dir = np.array([-direction_to_target[1], direction_to_target[0]])
                    
                    # Try both sides of obstacle
                    for side in [1, -1]:
                        avoidance_point = obstacle[:2] + side * perp_dir * 8.0
                        # Add Z coordinate
                        avoidance_3d = np.array([avoidance_point[0], avoidance_point[1], target[2]])
                        waypoints.append(avoidance_3d)
                        break  # Take first viable path
        
        waypoints.append(target)
        return waypoints
    
    def _has_line_of_sight(
        self,
        target: np.ndarray,
        obstacles: List[np.ndarray]
    ) -> bool:
        """Check if there's line of sight to target
        
        Args:
            target: Target position
            obstacles: List of obstacles
            
        Returns:
            True if line of sight exists
        """
        for obstacle in obstacles:
            if self._line_intersects_circle(
                self.position[:2], target[:2], obstacle[:2], 5.0
            ):
                return False
        return True
    
    def _line_intersects_circle(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        center: np.ndarray,
        radius: float
    ) -> bool:
        """Check if line segment intersects circle
        
        Args:
            p1: Line start
            p2: Line end
            center: Circle center
            radius: Circle radius
            
        Returns:
            True if intersection exists
        """
        # Vector from p1 to p2
        d = p2 - p1
        # Vector from p1 to center
        f = p1 - center
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return False
        
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)
    
    def execute_action(self, action: Dict[str, Any], dt: float):
        """Execute planned action
        
        Args:
            action: Action to execute
            dt: Time step
        """
        if action['type'] == 'move' and action['target'] is not None:
            # Move towards target
            if len(action['target']) == 2:
                target = np.array([action['target'][0], action['target'][1], self.position[2]])
            else:
                target = np.array(action['target'])
            direction = target - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
                move_distance = min(self.move_speed * dt, distance)
                self.position += direction * move_distance
                
                # Consume energy
                self.energy = max(0, self.energy - 0.01 * dt)
        
        elif action['type'] == 'rescue':
            # Perform rescue (handled by scenario)
            pass
    
    def communicate(
        self,
        teammates: List['SearchAgent']
    ) -> List[Dict[str, Any]]:
        """Communicate with nearby teammates
        
        Args:
            teammates: List of teammates
            
        Returns:
            List of messages
        """
        messages = []
        
        # Share detected victims
        if self.detected_victims:
            message = {
                'type': 'victim_report',
                'sender': self.agent_id,
                'victims': list(self.detected_victims),
                'position': self.position.copy()
            }
            messages.append(message)
        
        # Request help if needed
        if self.energy < 0.2:
            message = {
                'type': 'help_request',
                'sender': self.agent_id,
                'position': self.position.copy(),
                'energy': self.energy
            }
            messages.append(message)
        
        return messages
    
    def receive_message(self, message: Dict[str, Any]):
        """Process received message
        
        Args:
            message: Received message
        """
        if message['type'] == 'victim_report':
            # Update victim beliefs
            for victim_id in message['victims']:
                if victim_id not in self.victim_beliefs:
                    self.detected_victims.add(victim_id)
        
        elif message['type'] == 'assignment':
            # Accept assignment
            if message['target'] == self.agent_id:
                self.assigned_victim = message['victim_id']
        
        elif message['type'] == 'pattern_assignment':
            # Accept search pattern
            if message['target'] == self.agent_id:
                self.search_pattern = message['pattern']


class RescueCoordinator:
    """Central coordinator for rescue operations"""
    
    def __init__(self):
        """Initialize rescue coordinator"""
        self.active_missions: Dict[str, 'RescueMission'] = {}
        self.completed_missions: List['RescueMission'] = []
        self.search_areas: Dict[str, Dict[str, Any]] = {}
        self.agent_assignments: Dict[str, str] = {}
        
        logger.info("Initialized RescueCoordinator")
    
    def create_search_patterns(
        self,
        area_bounds: Tuple[float, float, float, float],
        num_agents: int
    ) -> List[SearchPattern]:
        """Create search patterns for agents
        
        Args:
            area_bounds: (min_x, min_y, max_x, max_y)
            num_agents: Number of search agents
            
        Returns:
            List of search patterns
        """
        patterns = []
        min_x, min_y, max_x, max_y = area_bounds
        
        # Divide area into sectors
        sectors_per_side = int(np.sqrt(num_agents))
        sector_width = (max_x - min_x) / sectors_per_side
        sector_height = (max_y - min_y) / sectors_per_side
        
        for i in range(num_agents):
            sector_x = i % sectors_per_side
            sector_y = i // sectors_per_side
            
            # Create grid search pattern for sector
            start_x = min_x + sector_x * sector_width
            start_y = min_y + sector_y * sector_height
            
            pattern = self._create_grid_pattern(
                start_x, start_y,
                start_x + sector_width,
                start_y + sector_height,
                spacing=5.0
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _create_grid_pattern(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        spacing: float
    ) -> SearchPattern:
        """Create grid search pattern
        
        Args:
            min_x, min_y: Minimum coordinates
            max_x, max_y: Maximum coordinates
            spacing: Grid spacing
            
        Returns:
            Search pattern
        """
        waypoints = []
        
        # Serpentine pattern
        y = min_y
        direction = 1
        
        while y <= max_y:
            if direction == 1:
                x = min_x
                while x <= max_x:
                    waypoints.append(np.array([x, y, 0]))
                    x += spacing
            else:
                x = max_x
                while x >= min_x:
                    waypoints.append(np.array([x, y, 0]))
                    x -= spacing
            
            y += spacing
            direction *= -1
        
        return SearchPattern(
            pattern_type='grid',
            start_position=np.array([min_x, min_y, 0]),
            parameters={'spacing': spacing},
            waypoints=waypoints
        )
    
    def assign_agents(
        self,
        agents: List[SearchAgent],
        victims: Dict[str, VictimModel]
    ) -> Dict[str, str]:
        """Assign agents to victims
        
        Args:
            agents: List of agents
            victims: Dictionary of victims
            
        Returns:
            Agent-victim assignments
        """
        assignments = {}
        
        # Get available rescuers
        rescuers = [a for a in agents if a.role == AgentRole.RESCUER]
        
        # Get unassigned detected victims - include being_rescued to reassign if needed
        unassigned_victims = [
            v for v in victims.values()
            if v.status in [VictimStatus.DETECTED, VictimStatus.BEING_RESCUED]
        ]
        
        # Sort by priority
        unassigned_victims.sort(
            key=lambda v: v.severity * (1 - v.health),
            reverse=True
        )
        
        # Assign closest rescuer to each victim
        for victim in unassigned_victims:
            if not rescuers:
                break
            
            # Find closest available rescuer
            best_rescuer = min(
                rescuers,
                key=lambda r: np.linalg.norm(r.position - victim.position)
            )
            
            assignments[best_rescuer.agent_id] = victim.victim_id
            rescuers.remove(best_rescuer)
            
            # Update victim status
            victim.status = VictimStatus.BEING_RESCUED
        
        return assignments
    
    def create_mission(
        self,
        victim: VictimModel,
        assigned_agents: List[str]
    ) -> 'RescueMission':
        """Create rescue mission
        
        Args:
            victim: Victim to rescue
            assigned_agents: Assigned agent IDs
            
        Returns:
            Rescue mission
        """
        mission = RescueMission(
            mission_id=f"mission_{victim.victim_id}",
            victim_id=victim.victim_id,
            assigned_agents=assigned_agents,
            start_time=0.0,  # Use scenario time
            priority=victim.severity
        )
        
        self.active_missions[mission.mission_id] = mission
        
        return mission
    
    def update_missions(self, dt: float):
        """Update active missions
        
        Args:
            dt: Time step
        """
        completed = []
        
        for mission_id, mission in self.active_missions.items():
            mission.update(dt)
            
            if mission.status == 'completed':
                completed.append(mission_id)
        
        # Move completed missions
        for mission_id in completed:
            mission = self.active_missions.pop(mission_id)
            self.completed_missions.append(mission)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rescue statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'active_missions': len(self.active_missions),
            'completed_missions': len(self.completed_missions),
            'success_rate': self._calculate_success_rate(),
            'average_rescue_time': self._calculate_average_rescue_time()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate mission success rate"""
        if not self.completed_missions:
            return 0.0
        
        successful = sum(
            1 for m in self.completed_missions
            if m.successful
        )
        
        return successful / len(self.completed_missions)
    
    def _calculate_average_rescue_time(self) -> float:
        """Calculate average rescue time"""
        if not self.completed_missions:
            return 0.0
        
        total_time = sum(
            m.completion_time - m.start_time
            for m in self.completed_missions
            if m.successful
        )
        
        successful_count = sum(
            1 for m in self.completed_missions
            if m.successful
        )
        
        return total_time / successful_count if successful_count > 0 else 0.0


@dataclass
class RescueMission:
    """Individual rescue mission"""
    mission_id: str
    victim_id: str
    assigned_agents: List[str]
    start_time: float
    priority: float
    status: str = 'active'
    completion_time: Optional[float] = None
    successful: bool = False
    
    def update(self, dt: float):
        """Update mission status
        
        Args:
            dt: Time step
        """
        # Mission logic handled by scenario
        pass


class SearchRescueScenario:
    """Main search and rescue scenario"""
    
    def __init__(
        self,
        area_size: Tuple[float, float] = (100.0, 100.0),
        num_victims: int = 10,
        num_agents: int = 6,
        obstacle_density: float = 0.1
    ):
        """Initialize search and rescue scenario
        
        Args:
            area_size: Size of search area
            num_victims: Number of victims
            num_agents: Number of agents
            obstacle_density: Density of obstacles
        """
        self.area_size = area_size
        self.num_victims = num_victims
        self.num_agents = num_agents
        
        # Initialize components
        self.victims: Dict[str, VictimModel] = {}
        self.agents: List[SearchAgent] = []
        self.obstacles: List[np.ndarray] = []
        self.coordinator = RescueCoordinator()
        
        # Simulation state
        self.time = 0.0
        self.completed = False
        
        # Initialize scenario
        self._initialize_victims()
        self._initialize_agents()
        self._initialize_obstacles(obstacle_density)
        
        logger.info(
            f"Initialized SearchRescueScenario with {num_victims} victims "
            f"and {num_agents} agents"
        )
    
    def _initialize_victims(self):
        """Initialize victims in the area"""
        for i in range(self.num_victims):
            position = np.array([
                np.random.uniform(0, self.area_size[0]),
                np.random.uniform(0, self.area_size[1]),
                0.0
            ])
            
            victim = VictimModel(
                victim_id=f"victim_{i}",
                position=position,
                severity=np.random.uniform(0.3, 1.0),
                required_rescuers=np.random.choice([1, 2], p=[0.7, 0.3])
            )
            
            self.victims[victim.victim_id] = victim
    
    def _initialize_agents(self):
        """Initialize search and rescue agents"""
        # Determine agent roles - fix role distribution
        num_coordinators = 1
        remaining_agents = self.num_agents - num_coordinators
        num_searchers = max(1, remaining_agents // 2)  # At least 1 searcher
        num_rescuers = max(1, remaining_agents - num_searchers)  # Rest are rescuers
        num_support = 0  # Simplify by removing support role for now
        
        agent_id = 0
        
        # Create searchers
        for i in range(num_searchers):
            position = np.array([
                self.area_size[0] / 2 + np.random.uniform(-10, 10),
                self.area_size[1] / 2 + np.random.uniform(-10, 10),
                0.0
            ])
            
            agent = SearchAgent(
                agent_id=f"agent_{agent_id}",
                role=AgentRole.SEARCHER,
                position=position,
                sensor_range=15.0,
                move_speed=3.0
            )
            
            self.agents.append(agent)
            agent_id += 1
        
        # Create rescuers
        for i in range(num_rescuers):
            position = np.array([
                self.area_size[0] / 2 + np.random.uniform(-10, 10),
                self.area_size[1] / 2 + np.random.uniform(-10, 10),
                0.0
            ])
            
            agent = SearchAgent(
                agent_id=f"agent_{agent_id}",
                role=AgentRole.RESCUER,
                position=position,
                sensor_range=10.0,
                move_speed=2.0
            )
            
            self.agents.append(agent)
            agent_id += 1
        
        # Create coordinator
        position = np.array([self.area_size[0] / 2, self.area_size[1] / 2, 0.0])
        
        coordinator_agent = SearchAgent(
            agent_id=f"agent_{agent_id}",
            role=AgentRole.COORDINATOR,
            position=position,
            sensor_range=20.0,
            move_speed=1.0,
            communication_range=100.0
        )
        
        self.agents.append(coordinator_agent)
        
        # Assign search patterns
        search_patterns = self.coordinator.create_search_patterns(
            (0, 0, self.area_size[0], self.area_size[1]),
            num_searchers
        )
        
        searcher_idx = 0
        for agent in self.agents:
            if agent.role == AgentRole.SEARCHER:
                agent.search_pattern = search_patterns[searcher_idx]
                searcher_idx += 1
    
    def _initialize_obstacles(self, density: float):
        """Initialize obstacles in the area
        
        Args:
            density: Obstacle density
        """
        num_obstacles = int(self.area_size[0] * self.area_size[1] * density / 100)
        
        for _ in range(num_obstacles):
            position = np.array([
                np.random.uniform(10, self.area_size[0] - 10),
                np.random.uniform(10, self.area_size[1] - 10),
                0.0
            ])
            
            self.obstacles.append(position)
    
    def step(self, dt: float = 0.1):
        """Run one simulation step
        
        Args:
            dt: Time step
        """
        self.time += dt
        
        # Update victims
        for victim in self.victims.values():
            victim.update_health(dt)
        
        # Agent sensing phase
        for agent in self.agents:
            newly_detected = agent.sense_environment(
                list(self.victims.values()),
                self.obstacles,
                self.time
            )
            
            # Report detections
            if newly_detected:
                logger.info(
                    f"Agent {agent.agent_id} detected victims: {newly_detected}"
                )
        
        # Planning phase
        actions = {}
        for agent in self.agents:
            action = agent.plan_action(
                self.victims,
                self.agents,
                self.obstacles
            )
            actions[agent.agent_id] = action
        
        # Coordination phase
        coordinator = next(
            (a for a in self.agents if a.role == AgentRole.COORDINATOR),
            None
        )
        
        if coordinator:
            coord_action = actions[coordinator.agent_id]
            
            if coord_action['type'] == 'coordinate':
                # Send assignments
                for agent_id, victim_id in coord_action['assignments'].items():
                    for agent in self.agents:
                        if agent.agent_id == agent_id:
                            agent.assigned_victim = victim_id
                            
                            # Create mission
                            self.coordinator.create_mission(
                                self.victims[victim_id],
                                [agent_id]
                            )
        
        # Execution phase
        for agent in self.agents:
            action = actions[agent.agent_id]
            agent.execute_action(action, dt)
            
            # Handle rescue actions
            if action['type'] == 'rescue':
                victim_id = action['victim_id']
                if victim_id in self.victims:
                    victim = self.victims[victim_id]
                    
                    # Improved rescue logic with time requirement
                    distance = np.linalg.norm(victim.position - agent.position)
                    
                    if victim.status == VictimStatus.BEING_RESCUED:
                        # Rescue takes time - require agent to stay near victim
                        rescue_time_required = 2.0  # 2 seconds to complete rescue
                        if not hasattr(victim, 'rescue_start_time'):
                            victim.rescue_start_time = self.time
                        
                        if self.time - victim.rescue_start_time >= rescue_time_required:
                            victim.status = VictimStatus.RESCUED
                            victim.rescue_time = self.time
                            agent.assigned_victim = None
                            
                            logger.info(
                                f"Agent {agent.agent_id} rescued {victim_id} after {rescue_time_required}s"
                            )
                        # If agent moves away during rescue, reset rescue
                        elif distance > 3.0:
                            victim.status = VictimStatus.DETECTED
                            if hasattr(victim, 'rescue_start_time'):
                                delattr(victim, 'rescue_start_time')
                            agent.assigned_victim = None
                    # Start rescue if close enough
                    elif victim.status == VictimStatus.DETECTED and distance <= 2.0:
                        victim.status = VictimStatus.BEING_RESCUED
                        victim.rescue_start_time = self.time
        
        # Communication phase
        all_messages = []
        for agent in self.agents:
            messages = agent.communicate(self.agents)
            all_messages.extend(messages)
        
        # Deliver messages
        for message in all_messages:
            for agent in self.agents:
                if agent.agent_id != message['sender']:
                    distance = np.linalg.norm(
                        agent.position - message['position']
                    )
                    
                    # Check if within communication range
                    sender = next(
                        a for a in self.agents
                        if a.agent_id == message['sender']
                    )
                    
                    if distance <= sender.communication_range:
                        agent.receive_message(message)
        
        # Update missions
        self.coordinator.update_missions(dt)
        
        # Check completion
        rescued_count = sum(
            1 for v in self.victims.values()
            if v.status == VictimStatus.RESCUED
        )
        
        lost_count = sum(
            1 for v in self.victims.values()
            if v.status == VictimStatus.LOST
        )
        
        if rescued_count + lost_count == self.num_victims:
            self.completed = True
            logger.info(
                f"Scenario completed: {rescued_count} rescued, "
                f"{lost_count} lost"
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current scenario state
        
        Returns:
            State dictionary
        """
        return {
            'time': self.time,
            'victims': {
                'total': self.num_victims,
                'detected': sum(
                    1 for v in self.victims.values()
                    if v.status != VictimStatus.UNDISCOVERED
                ),
                'rescued': sum(
                    1 for v in self.victims.values()
                    if v.status == VictimStatus.RESCUED
                ),
                'lost': sum(
                    1 for v in self.victims.values()
                    if v.status == VictimStatus.LOST
                )
            },
            'agents': {
                agent.agent_id: {
                    'position': agent.position.tolist(),
                    'role': agent.role.value,
                    'energy': agent.energy,
                    'assigned_victim': agent.assigned_victim
                }
                for agent in self.agents
            },
            'missions': self.coordinator.get_statistics(),
            'completed': self.completed
        }
    
    def reset(self):
        """Reset scenario"""
        self.time = 0.0
        self.completed = False
        
        # Reset victims
        self.victims.clear()
        self._initialize_victims()
        
        # Reset agents
        self.agents.clear()
        self._initialize_agents()
        
        # Reset coordinator
        self.coordinator = RescueCoordinator()
        
        logger.info("Scenario reset")


# Example usage
def run_search_rescue_scenario():
    """Run a search and rescue scenario"""
    # Create scenario
    scenario = SearchRescueScenario(
        area_size=(200.0, 200.0),
        num_victims=15,
        num_agents=8,
        obstacle_density=0.05
    )
    
    # Run simulation
    max_time = 300.0  # 5 minutes
    dt = 0.1
    
    while scenario.time < max_time and not scenario.completed:
        scenario.step(dt)
        
        # Log state periodically
        if int(scenario.time) % 10 == 0:
            state = scenario.get_state()
            logger.info(f"Time: {state['time']:.1f}s, State: {state['victims']}")
    
    # Final statistics
    final_state = scenario.get_state()
    logger.info(f"Final state: {final_state}")