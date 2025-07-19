"""Utility functions for Multi-Agent Environment

This module provides helper functions and utilities for the multi-agent
environment.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import gymnasium as gym
from pathlib import Path
import json
import yaml
import logging

logger = logging.getLogger(__name__)


def create_formation_positions(
    num_agents: int,
    formation_type: str,
    spacing: float = 5.0,
    center: Optional[np.ndarray] = None,
    heading: float = 0.0
) -> List[np.ndarray]:
    """Create initial positions for a formation
    
    Args:
        num_agents: Number of agents
        formation_type: Type of formation (line, v, circle, grid)
        spacing: Spacing between agents
        center: Center position of formation
        heading: Formation heading in radians
        
    Returns:
        List of agent positions
    """
    if center is None:
        center = np.array([0.0, 0.0, 10.0])
    
    positions = []
    
    if formation_type == "line":
        # Line formation perpendicular to heading
        for i in range(num_agents):
            offset = (i - (num_agents - 1) / 2) * spacing
            x = center[0] + offset * np.sin(heading)
            y = center[1] - offset * np.cos(heading)
            z = center[2]
            positions.append(np.array([x, y, z]))
    
    elif formation_type == "v":
        # V formation
        if num_agents == 1:
            positions.append(center.copy())
        else:
            # Leader at front
            positions.append(center.copy())
            
            # Followers in V shape
            for i in range(1, num_agents):
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                
                # Position relative to leader
                back_offset = row * spacing
                side_offset = side * row * spacing * 0.7
                
                x = center[0] - back_offset * np.cos(heading) + side_offset * np.sin(heading)
                y = center[1] - back_offset * np.sin(heading) - side_offset * np.cos(heading)
                z = center[2]
                
                positions.append(np.array([x, y, z]))
    
    elif formation_type == "circle":
        # Circle formation
        radius = spacing * num_agents / (2 * np.pi)
        for i in range(num_agents):
            angle = 2 * np.pi * i / num_agents + heading
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            positions.append(np.array([x, y, z]))
    
    elif formation_type == "grid":
        # Grid formation
        grid_size = int(np.ceil(np.sqrt(num_agents)))
        for i in range(num_agents):
            row = i // grid_size
            col = i % grid_size
            
            # Center the grid
            x_offset = (col - (grid_size - 1) / 2) * spacing
            y_offset = (row - (grid_size - 1) / 2) * spacing
            
            # Rotate by heading
            x = center[0] + x_offset * np.cos(heading) - y_offset * np.sin(heading)
            y = center[1] + x_offset * np.sin(heading) + y_offset * np.cos(heading)
            z = center[2]
            
            positions.append(np.array([x, y, z]))
    
    else:
        # Default: random positions around center
        for i in range(num_agents):
            offset = np.random.randn(3) * spacing
            offset[2] = 0  # Keep same altitude
            positions.append(center + offset)
    
    return positions


def calculate_formation_metrics(
    positions: List[np.ndarray],
    formation_type: str,
    expected_spacing: float = 5.0
) -> Dict[str, float]:
    """Calculate metrics for formation quality
    
    Args:
        positions: List of agent positions
        formation_type: Type of formation
        expected_spacing: Expected spacing between agents
        
    Returns:
        Dictionary of formation metrics
    """
    metrics = {}
    
    if len(positions) < 2:
        return {"formation_error": 0.0, "spacing_error": 0.0}
    
    positions_array = np.array(positions)
    
    # Calculate centroid
    centroid = np.mean(positions_array, axis=0)
    metrics["centroid"] = centroid.tolist()
    
    # Calculate spread
    distances_from_centroid = np.linalg.norm(positions_array - centroid, axis=1)
    metrics["spread"] = float(np.mean(distances_from_centroid))
    
    # Formation-specific metrics
    if formation_type == "line":
        # Fit line and calculate deviations
        if len(positions) > 2:
            # Use PCA to find principal direction
            centered = positions_array - centroid
            cov = np.cov(centered[:, :2].T)  # Just x,y
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Project points onto line
            projections = centered[:, :2] @ main_direction
            perpendicular = centered[:, :2] - np.outer(projections, main_direction)
            
            # Average perpendicular distance
            metrics["formation_error"] = float(np.mean(np.linalg.norm(perpendicular, axis=1)))
            
            # Spacing along line
            sorted_projections = np.sort(projections)
            spacings = np.diff(sorted_projections)
            metrics["spacing_error"] = float(np.std(spacings))
        else:
            metrics["formation_error"] = 0.0
            metrics["spacing_error"] = 0.0
    
    elif formation_type == "circle":
        # Calculate radius variance
        radii = distances_from_centroid
        metrics["formation_error"] = float(np.std(radii))
        
        # Angular spacing
        angles = np.arctan2(
            positions_array[:, 1] - centroid[1],
            positions_array[:, 0] - centroid[0]
        )
        angles = np.sort(angles)
        angular_spacings = np.diff(np.append(angles, angles[0] + 2*np.pi))
        expected_angular_spacing = 2 * np.pi / len(positions)
        metrics["spacing_error"] = float(np.std(angular_spacings - expected_angular_spacing))
    
    else:
        # Generic: pairwise distance variance
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                distances.append(np.linalg.norm(positions[i] - positions[j]))
        
        if distances:
            metrics["formation_error"] = float(np.std(distances))
            metrics["spacing_error"] = float(np.abs(np.mean(distances) - expected_spacing))
        else:
            metrics["formation_error"] = 0.0
            metrics["spacing_error"] = 0.0
    
    return metrics


def check_collision(
    position1: np.ndarray,
    position2: np.ndarray,
    radius1: float = 1.0,
    radius2: float = 1.0
) -> bool:
    """Check if two spherical objects collide
    
    Args:
        position1: Position of first object
        position2: Position of second object
        radius1: Radius of first object
        radius2: Radius of second object
        
    Returns:
        True if collision detected
    """
    distance = np.linalg.norm(position1 - position2)
    return distance < (radius1 + radius2)


def calculate_communication_graph(
    positions: Dict[int, np.ndarray],
    communication_range: float
) -> Dict[int, List[int]]:
    """Calculate communication graph based on positions
    
    Args:
        positions: Dictionary of agent positions
        communication_range: Maximum communication range
        
    Returns:
        Adjacency list representation of communication graph
    """
    graph = {agent_id: [] for agent_id in positions}
    
    agent_ids = list(positions.keys())
    for i, agent1_id in enumerate(agent_ids):
        for agent2_id in agent_ids[i+1:]:
            distance = np.linalg.norm(
                positions[agent1_id] - positions[agent2_id]
            )
            
            if distance <= communication_range:
                graph[agent1_id].append(agent2_id)
                graph[agent2_id].append(agent1_id)
    
    return graph


def is_graph_connected(graph: Dict[int, List[int]]) -> bool:
    """Check if communication graph is connected
    
    Args:
        graph: Adjacency list representation
        
    Returns:
        True if graph is connected
    """
    if not graph:
        return True
    
    # BFS from first node
    start_node = next(iter(graph))
    visited = {start_node}
    queue = [start_node]
    
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(graph)


def calculate_coverage(
    positions: List[np.ndarray],
    world_size: Tuple[float, float],
    sensor_range: float = 10.0,
    resolution: float = 1.0
) -> float:
    """Calculate area coverage percentage
    
    Args:
        positions: List of agent positions
        world_size: Size of the world (x, y)
        sensor_range: Sensor detection range
        resolution: Grid resolution for coverage calculation
        
    Returns:
        Coverage percentage (0-1)
    """
    if not positions:
        return 0.0
    
    # Create coverage grid
    grid_x = int(world_size[0] / resolution)
    grid_y = int(world_size[1] / resolution)
    coverage_grid = np.zeros((grid_x, grid_y), dtype=bool)
    
    # Mark covered cells
    sensor_cells = int(sensor_range / resolution)
    
    for position in positions:
        # Convert to grid coordinates
        cx = int(position[0] / resolution)
        cy = int(position[1] / resolution)
        
        # Mark cells within sensor range
        for dx in range(-sensor_cells, sensor_cells + 1):
            for dy in range(-sensor_cells, sensor_cells + 1):
                gx, gy = cx + dx, cy + dy
                
                if 0 <= gx < grid_x and 0 <= gy < grid_y:
                    # Check if within circular sensor range
                    cell_x = gx * resolution
                    cell_y = gy * resolution
                    if np.linalg.norm([cell_x - position[0], cell_y - position[1]]) <= sensor_range:
                        coverage_grid[gx, gy] = True
    
    # Calculate coverage percentage
    return np.sum(coverage_grid) / coverage_grid.size


def save_episode_data(
    episode_data: Dict[str, Any],
    save_path: Union[str, Path],
    format: str = "json"
):
    """Save episode data to file
    
    Args:
        episode_data: Episode data to save
        save_path: Path to save file
        format: Save format (json or yaml)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for serialization
    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        return obj
    
    serializable_data = convert_arrays(episode_data)
    
    if format == "json":
        with open(save_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    elif format == "yaml":
        with open(save_path, 'w') as f:
            yaml.dump(serializable_data, f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Saved episode data to {save_path}")


def load_episode_data(
    load_path: Union[str, Path],
    format: str = "json"
) -> Dict[str, Any]:
    """Load episode data from file
    
    Args:
        load_path: Path to load file
        format: File format (json or yaml)
        
    Returns:
        Episode data
    """
    load_path = Path(load_path)
    
    if format == "json":
        with open(load_path, 'r') as f:
            data = json.load(f)
    elif format == "yaml":
        with open(load_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Loaded episode data from {load_path}")
    return data


def create_random_obstacles(
    num_obstacles: int,
    world_size: Tuple[float, float, float],
    min_size: float = 2.0,
    max_size: float = 10.0,
    obstacle_density: float = 0.1
) -> List[Dict[str, Any]]:
    """Create random obstacles in the environment
    
    Args:
        num_obstacles: Number of obstacles to create
        world_size: Size of the world
        min_size: Minimum obstacle size
        max_size: Maximum obstacle size
        obstacle_density: Fraction of world to fill with obstacles
        
    Returns:
        List of obstacle specifications
    """
    obstacles = []
    
    # Calculate total obstacle volume allowed
    world_volume = world_size[0] * world_size[1] * world_size[2]
    max_obstacle_volume = world_volume * obstacle_density
    total_volume = 0.0
    
    for i in range(num_obstacles):
        if total_volume >= max_obstacle_volume:
            break
        
        # Random size
        size = np.random.uniform(min_size, max_size, size=3)
        
        # Random position (keep away from boundaries)
        position = np.array([
            np.random.uniform(size[0], world_size[0] - size[0]),
            np.random.uniform(size[1], world_size[1] - size[1]),
            size[2] / 2  # Place on ground
        ])
        
        obstacle = {
            "position": position.tolist(),
            "size": size.tolist(),
            "type": "building",
            "static": True
        }
        
        obstacles.append(obstacle)
        total_volume += np.prod(size)
    
    return obstacles


class MultiAgentEnvWrapper(gym.Wrapper):
    """Gymnasium wrapper for multi-agent environment compatibility"""
    
    def __init__(self, env):
        """Initialize wrapper
        
        Args:
            env: Multi-agent environment instance
        """
        super().__init__(env)
        self.env = env
        
        # For single-agent compatibility
        self.single_agent_mode = False
        self.agent_id = None
    
    def reset(self, **kwargs):
        """Reset environment"""
        obs, info = self.env.reset(**kwargs)
        
        if self.single_agent_mode and self.agent_id is not None:
            return obs[self.agent_id], info[self.agent_id]
        
        return obs, info
    
    def step(self, action):
        """Step environment"""
        if self.single_agent_mode and self.agent_id is not None:
            # Convert single action to multi-agent format
            actions = {self.agent_id: action}
        else:
            actions = action
        
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        
        if self.single_agent_mode and self.agent_id is not None:
            return (
                obs[self.agent_id],
                rewards[self.agent_id],
                terminated[self.agent_id],
                truncated[self.agent_id],
                info[self.agent_id]
            )
        
        return obs, rewards, terminated, truncated, info
    
    def set_single_agent_mode(self, agent_id: int = 0):
        """Set wrapper to single-agent mode
        
        Args:
            agent_id: Agent ID to control
        """
        self.single_agent_mode = True
        self.agent_id = agent_id