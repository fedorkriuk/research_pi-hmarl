"""Swarm Exploration Scenario

This module implements a swarm exploration scenario where multiple agents
collaboratively explore and map an unknown environment.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, deque
import heapq
import cv2

logger = logging.getLogger(__name__)


class ExplorationStrategy(Enum):
    """Exploration strategies"""
    FRONTIER = "frontier"
    COVERAGE = "coverage"
    INFORMATION_GAIN = "information_gain"
    COORDINATED = "coordinated"


class CellStatus(Enum):
    """Grid cell status"""
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    FRONTIER = 3


@dataclass
class MapCell:
    """Individual map cell"""
    x: int
    y: int
    status: CellStatus = CellStatus.UNKNOWN
    confidence: float = 0.0
    last_update: float = 0.0
    visits: int = 0
    
    def update(self, new_status: CellStatus, confidence: float, time: float):
        """Update cell status
        
        Args:
            new_status: New cell status
            confidence: Confidence in observation
            time: Update time
        """
        # Bayesian update of confidence
        self.confidence = min(
            1.0,
            self.confidence + confidence * (1 - self.confidence)
        )
        
        if self.confidence > 0.8:  # High confidence threshold
            self.status = new_status
        
        self.last_update = time
        self.visits += 1


class MapBuilder:
    """Builds and maintains exploration map"""
    
    def __init__(
        self,
        map_size: Tuple[int, int],
        resolution: float = 1.0
    ):
        """Initialize map builder
        
        Args:
            map_size: Map dimensions in cells
            resolution: Meters per cell
        """
        self.map_size = map_size
        self.resolution = resolution
        
        # Initialize map
        self.grid = {}
        for x in range(map_size[0]):
            for y in range(map_size[1]):
                self.grid[(x, y)] = MapCell(x, y)
        
        # Frontier tracking
        self.frontiers: Set[Tuple[int, int]] = set()
        
        # Exploration metrics
        self.explored_cells = 0
        self.total_cells = map_size[0] * map_size[1]
        
        logger.info(f"Initialized MapBuilder with size {map_size}")
    
    def update_from_sensor(
        self,
        sensor_data: Dict[str, Any],
        robot_pose: Tuple[float, float, float],
        time: float
    ):
        """Update map from sensor data
        
        Args:
            sensor_data: Sensor observations
            robot_pose: Robot position and orientation
            time: Current time
        """
        robot_x, robot_y, robot_theta = robot_pose
        
        # Convert to grid coordinates
        robot_cell_x = int(robot_x / self.resolution)
        robot_cell_y = int(robot_y / self.resolution)
        
        # Process laser scan
        if 'ranges' in sensor_data:
            ranges = sensor_data['ranges']
            angles = sensor_data['angles']
            
            for i, (range_val, angle) in enumerate(zip(ranges, angles)):
                if range_val > 0 and range_val < sensor_data.get('max_range', 30.0):
                    # Ray tracing
                    self._trace_ray(
                        robot_cell_x, robot_cell_y,
                        robot_theta + angle,
                        range_val / self.resolution,
                        time
                    )
        
        # Update frontiers
        self._update_frontiers()
    
    def _trace_ray(
        self,
        start_x: int,
        start_y: int,
        angle: float,
        max_dist: float,
        time: float
    ):
        """Trace a sensor ray through the map
        
        Args:
            start_x, start_y: Starting cell
            angle: Ray angle
            max_dist: Maximum distance
            time: Current time
        """
        # Bresenham's line algorithm
        end_x = start_x + int(max_dist * np.cos(angle))
        end_y = start_y + int(max_dist * np.sin(angle))
        
        cells = self._bresenham_line(start_x, start_y, end_x, end_y)
        
        for i, (x, y) in enumerate(cells[:-1]):
            if (x, y) in self.grid:
                cell = self.grid[(x, y)]
                
                # Mark as free
                if cell.status == CellStatus.UNKNOWN:
                    self.explored_cells += 1
                
                cell.update(CellStatus.FREE, 0.9, time)
        
        # Mark endpoint as occupied (if in range)
        if len(cells) > 0 and cells[-1] in self.grid:
            if max_dist < sensor_data.get('max_range', 30.0) - 1:
                end_cell = self.grid[cells[-1]]
                end_cell.update(CellStatus.OCCUPIED, 0.8, time)
    
    def _bresenham_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int
    ) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm
        
        Args:
            x0, y0: Start point
            x1, y1: End point
            
        Returns:
            List of cells along line
        """
        cells = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            cells.append((x0, y0))
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return cells
    
    def _update_frontiers(self):
        """Update frontier cells"""
        self.frontiers.clear()
        
        for (x, y), cell in self.grid.items():
            if cell.status == CellStatus.FREE:
                # Check neighbors
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    
                    if (nx, ny) in self.grid:
                        neighbor = self.grid[(nx, ny)]
                        
                        if neighbor.status == CellStatus.UNKNOWN:
                            self.frontiers.add((x, y))
                            break
    
    def get_frontiers(self) -> List[Tuple[int, int]]:
        """Get list of frontier cells
        
        Returns:
            List of frontier cells
        """
        return list(self.frontiers)
    
    def get_exploration_rate(self) -> float:
        """Get exploration completion rate
        
        Returns:
            Exploration rate (0-1)
        """
        return self.explored_cells / self.total_cells
    
    def get_map_image(self) -> np.ndarray:
        """Get map as image
        
        Returns:
            Map image
        """
        img = np.zeros((self.map_size[1], self.map_size[0], 3), dtype=np.uint8)
        
        for (x, y), cell in self.grid.items():
            if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                if cell.status == CellStatus.UNKNOWN:
                    img[y, x] = [128, 128, 128]  # Gray
                elif cell.status == CellStatus.FREE:
                    img[y, x] = [255, 255, 255]  # White
                elif cell.status == CellStatus.OCCUPIED:
                    img[y, x] = [0, 0, 0]  # Black
                
                # Highlight frontiers
                if (x, y) in self.frontiers:
                    img[y, x] = [0, 255, 0]  # Green
        
        return img
    
    def merge_map(self, other_map: 'MapBuilder', confidence_weight: float = 0.5):
        """Merge another map into this one
        
        Args:
            other_map: Map to merge
            confidence_weight: Weight for other map's confidence
        """
        for (x, y), other_cell in other_map.grid.items():
            if (x, y) in self.grid:
                my_cell = self.grid[(x, y)]
                
                # Weighted confidence update
                if other_cell.status != CellStatus.UNKNOWN:
                    combined_confidence = (
                        my_cell.confidence * (1 - confidence_weight) +
                        other_cell.confidence * confidence_weight
                    )
                    
                    # Use cell with higher confidence
                    if other_cell.confidence > my_cell.confidence:
                        my_cell.status = other_cell.status
                    
                    my_cell.confidence = combined_confidence
                    my_cell.last_update = max(
                        my_cell.last_update,
                        other_cell.last_update
                    )


class FrontierExploration:
    """Frontier-based exploration strategy"""
    
    def __init__(self, map_builder: MapBuilder):
        """Initialize frontier exploration
        
        Args:
            map_builder: Map builder instance
        """
        self.map_builder = map_builder
        self.assigned_frontiers: Dict[str, Tuple[int, int]] = {}
        
    def assign_frontiers(
        self,
        agents: List['ExplorationAgent']
    ) -> Dict[str, Tuple[int, int]]:
        """Assign frontiers to agents
        
        Args:
            agents: List of exploration agents
            
        Returns:
            Agent-frontier assignments
        """
        frontiers = self.map_builder.get_frontiers()
        assignments = {}
        
        if not frontiers:
            return assignments
        
        # Cluster frontiers
        frontier_clusters = self._cluster_frontiers(frontiers)
        
        # Assign clusters to agents
        for i, agent in enumerate(agents):
            if i < len(frontier_clusters):
                # Find centroid of cluster
                cluster = frontier_clusters[i % len(frontier_clusters)]
                centroid = self._compute_centroid(cluster)
                
                assignments[agent.agent_id] = centroid
        
        self.assigned_frontiers = assignments
        return assignments
    
    def _cluster_frontiers(
        self,
        frontiers: List[Tuple[int, int]],
        max_clusters: int = 10
    ) -> List[List[Tuple[int, int]]]:
        """Cluster frontiers using simple clustering
        
        Args:
            frontiers: List of frontier cells
            max_clusters: Maximum number of clusters
            
        Returns:
            List of frontier clusters
        """
        if len(frontiers) <= max_clusters:
            return [[f] for f in frontiers]
        
        # Simple k-means style clustering
        clusters = []
        remaining = frontiers.copy()
        
        while remaining and len(clusters) < max_clusters:
            # Start new cluster
            seed = remaining.pop(0)
            cluster = [seed]
            
            # Add nearby frontiers
            i = 0
            while i < len(remaining):
                fx, fy = remaining[i]
                
                # Check distance to cluster
                min_dist = min(
                    abs(fx - cx) + abs(fy - cy)
                    for cx, cy in cluster
                )
                
                if min_dist < 10:  # Cluster radius
                    cluster.append(remaining.pop(i))
                else:
                    i += 1
            
            clusters.append(cluster)
        
        return clusters
    
    def _compute_centroid(
        self,
        cells: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Compute centroid of cells
        
        Args:
            cells: List of cells
            
        Returns:
            Centroid cell
        """
        if not cells:
            return (0, 0)
        
        avg_x = sum(x for x, y in cells) / len(cells)
        avg_y = sum(y for x, y in cells) / len(cells)
        
        return (int(avg_x), int(avg_y))


class AreaCoverage:
    """Area coverage exploration strategy"""
    
    def __init__(self, map_size: Tuple[int, int], cell_size: int = 10):
        """Initialize area coverage
        
        Args:
            map_size: Map dimensions
            cell_size: Coverage cell size
        """
        self.map_size = map_size
        self.cell_size = cell_size
        
        # Coverage grid
        self.coverage_grid = {}
        self.grid_width = map_size[0] // cell_size
        self.grid_height = map_size[1] // cell_size
        
        for gx in range(self.grid_width):
            for gy in range(self.grid_height):
                self.coverage_grid[(gx, gy)] = {
                    'visited': False,
                    'visit_count': 0,
                    'last_visit': 0.0
                }
    
    def get_coverage_targets(
        self,
        num_agents: int,
        current_time: float
    ) -> List[Tuple[int, int]]:
        """Get coverage targets for agents
        
        Args:
            num_agents: Number of agents
            current_time: Current time
            
        Returns:
            List of target positions
        """
        # Sort cells by priority (least visited, oldest visit)
        unvisited = []
        visited = []
        
        for (gx, gy), cell_info in self.coverage_grid.items():
            priority = (
                cell_info['visit_count'] * 1000 +
                (current_time - cell_info['last_visit'])
            )
            
            center_x = gx * self.cell_size + self.cell_size // 2
            center_y = gy * self.cell_size + self.cell_size // 2
            
            if not cell_info['visited']:
                unvisited.append((priority, (center_x, center_y)))
            else:
                visited.append((priority, (center_x, center_y)))
        
        # Sort by priority
        unvisited.sort()
        visited.sort()
        
        # Select targets
        targets = []
        
        # Prioritize unvisited cells
        for _, pos in unvisited[:num_agents]:
            targets.append(pos)
        
        # Add visited cells if needed
        remaining = num_agents - len(targets)
        for _, pos in visited[:remaining]:
            targets.append(pos)
        
        return targets
    
    def update_coverage(self, position: Tuple[float, float], time: float):
        """Update coverage information
        
        Args:
            position: Agent position
            time: Current time
        """
        gx = int(position[0] / self.cell_size)
        gy = int(position[1] / self.cell_size)
        
        if (gx, gy) in self.coverage_grid:
            cell = self.coverage_grid[(gx, gy)]
            cell['visited'] = True
            cell['visit_count'] += 1
            cell['last_visit'] = time


class ExplorationAgent:
    """Individual exploration agent"""
    
    def __init__(
        self,
        agent_id: str,
        initial_position: np.ndarray,
        sensor_range: float = 10.0,
        move_speed: float = 2.0,
        communication_range: float = 30.0
    ):
        """Initialize exploration agent
        
        Args:
            agent_id: Unique agent identifier
            initial_position: Starting position
            sensor_range: Sensor detection range
            move_speed: Movement speed
            communication_range: Communication range
        """
        self.agent_id = agent_id
        self.position = initial_position.copy()
        self.orientation = 0.0
        self.sensor_range = sensor_range
        self.move_speed = move_speed
        self.communication_range = communication_range
        
        # Local map
        self.local_map = None
        
        # Exploration state
        self.target_position = None
        self.path = []
        self.strategy = ExplorationStrategy.FRONTIER
        
        # Communication
        self.last_map_share = 0.0
        self.map_share_interval = 5.0  # seconds
        
        logger.info(f"Initialized ExplorationAgent {agent_id}")
    
    def sense_environment(
        self,
        true_map: np.ndarray,
        time: float
    ) -> Dict[str, Any]:
        """Sense the environment
        
        Args:
            true_map: True environment map
            time: Current time
            
        Returns:
            Sensor data
        """
        # Simulate laser scanner
        num_rays = 36  # 10 degree resolution
        angles = np.linspace(-np.pi, np.pi, num_rays)
        max_range = self.sensor_range
        
        ranges = []
        
        for angle in angles:
            # Cast ray
            ray_angle = self.orientation + angle
            
            # Simple ray casting
            for dist in np.linspace(0, max_range, 100):
                x = self.position[0] + dist * np.cos(ray_angle)
                y = self.position[1] + dist * np.sin(ray_angle)
                
                # Check map
                map_x = int(x)
                map_y = int(y)
                
                if (0 <= map_x < true_map.shape[1] and
                    0 <= map_y < true_map.shape[0]):
                    
                    if true_map[map_y, map_x] == 0:  # Obstacle
                        ranges.append(dist)
                        break
                else:
                    ranges.append(max_range)
                    break
            else:
                ranges.append(max_range)
        
        return {
            'ranges': ranges,
            'angles': angles,
            'max_range': max_range,
            'position': self.position.copy(),
            'orientation': self.orientation
        }
    
    def update_local_map(self, sensor_data: Dict[str, Any], time: float):
        """Update local map with sensor data
        
        Args:
            sensor_data: Sensor observations
            time: Current time
        """
        if self.local_map is None:
            # Initialize local map centered on agent
            map_size = (200, 200)  # Local map size
            self.local_map = MapBuilder(map_size, resolution=1.0)
        
        # Update map
        pose = (self.position[0], self.position[1], self.orientation)
        self.local_map.update_from_sensor(sensor_data, pose, time)
    
    def plan_exploration(
        self,
        shared_map: MapBuilder,
        other_agents: List['ExplorationAgent']
    ) -> Optional[np.ndarray]:
        """Plan exploration target
        
        Args:
            shared_map: Shared global map
            other_agents: Other agents in swarm
            
        Returns:
            Target position
        """
        if self.strategy == ExplorationStrategy.FRONTIER:
            # Get frontiers
            frontiers = shared_map.get_frontiers()
            
            if not frontiers:
                return None
            
            # Find nearest unassigned frontier
            best_frontier = None
            best_distance = float('inf')
            
            for fx, fy in frontiers:
                # Convert to world coordinates
                world_x = fx * shared_map.resolution
                world_y = fy * shared_map.resolution
                
                # Check if assigned to another agent
                assigned = False
                for agent in other_agents:
                    if agent.agent_id != self.agent_id and agent.target_position is not None:
                        dist_to_target = np.linalg.norm(
                            agent.target_position - np.array([world_x, world_y])
                        )
                        if dist_to_target < 5.0:  # Too close to another's target
                            assigned = True
                            break
                
                if not assigned:
                    distance = np.linalg.norm(
                        self.position - np.array([world_x, world_y])
                    )
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_frontier = (world_x, world_y)
            
            if best_frontier:
                return np.array([best_frontier[0], best_frontier[1], 0])
        
        return None
    
    def move_towards_target(self, dt: float):
        """Move towards target position
        
        Args:
            dt: Time step
        """
        if self.target_position is None:
            return
        
        # Simple movement
        direction = self.target_position[:2] - self.position[:2]
        distance = np.linalg.norm(direction)
        
        if distance > 0.5:  # Not at target
            # Normalize direction
            direction = direction / distance
            
            # Move
            move_distance = min(self.move_speed * dt, distance)
            self.position[:2] += direction * move_distance
            
            # Update orientation
            self.orientation = np.arctan2(direction[1], direction[0])
        else:
            # Reached target
            self.target_position = None
    
    def should_share_map(self, time: float) -> bool:
        """Check if should share map
        
        Args:
            time: Current time
            
        Returns:
            True if should share
        """
        return time - self.last_map_share > self.map_share_interval
    
    def share_map(self, time: float) -> MapBuilder:
        """Share local map
        
        Args:
            time: Current time
            
        Returns:
            Local map to share
        """
        self.last_map_share = time
        return self.local_map


class SwarmCoordinator:
    """Coordinates swarm exploration"""
    
    def __init__(self, num_agents: int, map_size: Tuple[int, int]):
        """Initialize swarm coordinator
        
        Args:
            num_agents: Number of agents
            map_size: Environment size
        """
        self.num_agents = num_agents
        self.map_size = map_size
        
        # Global map
        self.global_map = MapBuilder(map_size, resolution=1.0)
        
        # Exploration strategies
        self.frontier_exploration = FrontierExploration(self.global_map)
        self.area_coverage = AreaCoverage(map_size)
        
        # Coordination state
        self.agent_assignments: Dict[str, Any] = {}
        self.exploration_start_time = 0.0
        
        logger.info(f"Initialized SwarmCoordinator for {num_agents} agents")
    
    def coordinate_exploration(
        self,
        agents: List[ExplorationAgent],
        time: float
    ) -> Dict[str, Any]:
        """Coordinate agent exploration
        
        Args:
            agents: List of agents
            time: Current time
            
        Returns:
            Coordination commands
        """
        commands = {}
        
        # Update global map from agent maps
        for agent in agents:
            if agent.local_map and agent.should_share_map(time):
                self.global_map.merge_map(agent.local_map)
        
        # Assign exploration targets
        if self.global_map.get_exploration_rate() < 0.8:
            # Use frontier exploration
            assignments = self.frontier_exploration.assign_frontiers(agents)
            
            for agent_id, target in assignments.items():
                commands[agent_id] = {
                    'type': 'explore',
                    'target': np.array([target[0], target[1], 0]),
                    'strategy': ExplorationStrategy.FRONTIER
                }
        else:
            # Switch to coverage mode
            targets = self.area_coverage.get_coverage_targets(
                self.num_agents,
                time
            )
            
            for i, agent in enumerate(agents):
                if i < len(targets):
                    commands[agent.agent_id] = {
                        'type': 'explore',
                        'target': np.array([targets[i][0], targets[i][1], 0]),
                        'strategy': ExplorationStrategy.COVERAGE
                    }
        
        return commands
    
    def get_exploration_metrics(self) -> Dict[str, Any]:
        """Get exploration metrics
        
        Returns:
            Metrics dictionary
        """
        return {
            'exploration_rate': self.global_map.get_exploration_rate(),
            'frontiers_remaining': len(self.global_map.get_frontiers()),
            'map_confidence': np.mean([
                cell.confidence
                for cell in self.global_map.grid.values()
                if cell.status != CellStatus.UNKNOWN
            ])
        }


class SwarmExplorationScenario:
    """Main swarm exploration scenario"""
    
    def __init__(
        self,
        environment_size: Tuple[int, int] = (100, 100),
        num_agents: int = 5,
        obstacle_complexity: float = 0.2
    ):
        """Initialize swarm exploration scenario
        
        Args:
            environment_size: Size of environment
            num_agents: Number of agents
            obstacle_complexity: Complexity of obstacles (0-1)
        """
        self.environment_size = environment_size
        self.num_agents = num_agents
        
        # Create environment
        self.true_map = self._generate_environment(obstacle_complexity)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize coordinator
        self.coordinator = SwarmCoordinator(num_agents, environment_size)
        
        # Simulation state
        self.time = 0.0
        self.completed = False
        
        logger.info(
            f"Initialized SwarmExplorationScenario with {num_agents} agents"
        )
    
    def _generate_environment(
        self,
        complexity: float
    ) -> np.ndarray:
        """Generate random environment
        
        Args:
            complexity: Environment complexity
            
        Returns:
            Environment map (0=obstacle, 1=free)
        """
        env_map = np.ones(self.environment_size, dtype=np.uint8)
        
        # Add walls
        env_map[0, :] = 0
        env_map[-1, :] = 0
        env_map[:, 0] = 0
        env_map[:, -1] = 0
        
        # Add random obstacles
        num_obstacles = int(complexity * 20)
        
        for _ in range(num_obstacles):
            # Random rectangle
            x = np.random.randint(10, self.environment_size[0] - 10)
            y = np.random.randint(10, self.environment_size[1] - 10)
            w = np.random.randint(5, 15)
            h = np.random.randint(5, 15)
            
            env_map[y:y+h, x:x+w] = 0
        
        # Add some corridors
        for _ in range(3):
            if np.random.random() < 0.5:
                # Horizontal corridor
                y = np.random.randint(20, self.environment_size[1] - 20)
                env_map[y-1:y+2, :] = 1
            else:
                # Vertical corridor
                x = np.random.randint(20, self.environment_size[0] - 20)
                env_map[:, x-1:x+2] = 1
        
        return env_map
    
    def _initialize_agents(self) -> List[ExplorationAgent]:
        """Initialize exploration agents
        
        Returns:
            List of agents
        """
        agents = []
        
        # Find free starting positions
        free_positions = []
        for y in range(10, self.environment_size[1] - 10):
            for x in range(10, self.environment_size[0] - 10):
                if self.true_map[y, x] == 1:  # Free space
                    free_positions.append((x, y))
        
        # Sample starting positions
        import random
        start_positions = random.sample(free_positions, self.num_agents)
        
        for i in range(self.num_agents):
            position = np.array([
                start_positions[i][0],
                start_positions[i][1],
                0.0
            ])
            
            agent = ExplorationAgent(
                agent_id=f"agent_{i}",
                initial_position=position,
                sensor_range=15.0,
                move_speed=3.0,
                communication_range=40.0
            )
            
            agents.append(agent)
        
        return agents
    
    def step(self, dt: float = 0.1):
        """Run one simulation step
        
        Args:
            dt: Time step
        """
        self.time += dt
        
        # Sensing phase
        sensor_data = {}
        for agent in self.agents:
            data = agent.sense_environment(self.true_map, self.time)
            sensor_data[agent.agent_id] = data
            
            # Update local maps
            agent.update_local_map(data, self.time)
        
        # Coordination phase
        commands = self.coordinator.coordinate_exploration(
            self.agents,
            self.time
        )
        
        # Apply commands
        for agent in self.agents:
            if agent.agent_id in commands:
                command = commands[agent.agent_id]
                
                if command['type'] == 'explore':
                    agent.target_position = command['target']
                    agent.strategy = command['strategy']
        
        # Planning phase
        for agent in self.agents:
            if agent.target_position is None:
                # Plan new target
                target = agent.plan_exploration(
                    self.coordinator.global_map,
                    self.agents
                )
                
                if target is not None:
                    agent.target_position = target
        
        # Movement phase
        for agent in self.agents:
            agent.move_towards_target(dt)
            
            # Update coverage
            self.coordinator.area_coverage.update_coverage(
                (agent.position[0], agent.position[1]),
                self.time
            )
        
        # Check completion
        exploration_rate = self.coordinator.global_map.get_exploration_rate()
        if exploration_rate > 0.95:  # 95% explored
            self.completed = True
            logger.info(f"Exploration completed at time {self.time:.1f}s")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current scenario state
        
        Returns:
            State dictionary
        """
        metrics = self.coordinator.get_exploration_metrics()
        
        return {
            'time': self.time,
            'completed': self.completed,
            'exploration_rate': metrics['exploration_rate'],
            'frontiers_remaining': metrics['frontiers_remaining'],
            'agents': {
                agent.agent_id: {
                    'position': agent.position.tolist(),
                    'target': agent.target_position.tolist() if agent.target_position is not None else None,
                    'strategy': agent.strategy.value
                }
                for agent in self.agents
            }
        }
    
    def get_visualization(self) -> Dict[str, np.ndarray]:
        """Get visualization data
        
        Returns:
            Dictionary of images
        """
        # Global map
        global_map_img = self.coordinator.global_map.get_map_image()
        
        # True map
        true_map_img = np.zeros(
            (*self.true_map.shape, 3),
            dtype=np.uint8
        )
        true_map_img[self.true_map == 1] = [255, 255, 255]
        true_map_img[self.true_map == 0] = [0, 0, 0]
        
        # Add agent positions
        for agent in self.agents:
            x = int(agent.position[0])
            y = int(agent.position[1])
            
            if 0 <= x < global_map_img.shape[1] and 0 <= y < global_map_img.shape[0]:
                cv2.circle(global_map_img, (x, y), 3, (255, 0, 0), -1)
                cv2.circle(true_map_img, (x, y), 3, (255, 0, 0), -1)
        
        return {
            'global_map': global_map_img,
            'true_map': true_map_img
        }


# Example usage
def run_swarm_exploration():
    """Run swarm exploration scenario"""
    # Create scenario
    scenario = SwarmExplorationScenario(
        environment_size=(150, 150),
        num_agents=6,
        obstacle_complexity=0.3
    )
    
    # Run simulation
    max_time = 300.0  # 5 minutes
    dt = 0.1
    
    while scenario.time < max_time and not scenario.completed:
        scenario.step(dt)
        
        # Log state periodically
        if int(scenario.time) % 10 == 0:
            state = scenario.get_state()
            logger.info(
                f"Time: {state['time']:.1f}s, "
                f"Exploration: {state['exploration_rate']*100:.1f}%"
            )
    
    # Final state
    final_state = scenario.get_state()
    logger.info(f"Final state: {final_state}")