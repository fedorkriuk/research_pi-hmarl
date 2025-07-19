"""Environment Visualization for Multi-Agent System

This module provides visualization capabilities for the multi-agent
environment using matplotlib and optional 3D rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EnvironmentVisualizer:
    """Visualizes the multi-agent environment"""
    
    def __init__(
        self,
        world_size: Tuple[float, float, float],
        render_mode: str = "human",
        figure_size: Tuple[int, int] = (10, 8),
        view_mode: str = "2d"  # 2d or 3d
    ):
        """Initialize environment visualizer
        
        Args:
            world_size: Size of the world (x, y, z)
            render_mode: Rendering mode (human, rgb_array)
            figure_size: Figure size in inches
            view_mode: 2D or 3D visualization
        """
        self.world_size = world_size
        self.render_mode = render_mode
        self.figure_size = figure_size
        self.view_mode = view_mode
        
        # Matplotlib setup
        self.fig = None
        self.ax = None
        self.agent_plots = {}
        self.obstacle_patches = []
        self.target_plots = []
        self.trail_plots = {}
        
        # Agent trails
        self.agent_trails = {}
        self.trail_length = 50
        
        # Colors for different agent types
        self.agent_colors = {
            "dji_mavic_3": "blue",
            "parrot_anafi": "green",
            "default": "red"
        }
        
        self._setup_figure()
        
        logger.info(f"Initialized EnvironmentVisualizer ({view_mode})")
    
    def _setup_figure(self):
        """Set up matplotlib figure"""
        plt.ion()  # Interactive mode
        
        self.fig = plt.figure(figsize=self.figure_size)
        
        if self.view_mode == "3d":
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(0, self.world_size[0])
            self.ax.set_ylim(0, self.world_size[1])
            self.ax.set_zlim(0, self.world_size[2])
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title('Multi-Agent Environment (3D)')
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.world_size[0])
            self.ax.set_ylim(0, self.world_size[1])
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_title('Multi-Agent Environment (2D Top View)')
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
    
    def render(
        self,
        agent_states: Dict[int, Dict[str, Any]],
        obstacles: List[Dict[str, Any]] = None,
        targets: List[Dict[str, Any]] = None,
        timestep: float = 0.0,
        communication_links: Optional[List[Tuple[int, int]]] = None
    ) -> Optional[np.ndarray]:
        """Render the current environment state
        
        Args:
            agent_states: Dictionary of agent states
            obstacles: List of obstacles
            targets: List of targets/waypoints
            timestep: Current simulation time
            communication_links: List of agent pairs with active communication
            
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.fig is None:
            self._setup_figure()
        
        # Clear dynamic elements
        self._clear_dynamic_elements()
        
        # Draw obstacles
        if obstacles:
            self._draw_obstacles(obstacles)
        
        # Draw targets
        if targets:
            self._draw_targets(targets)
        
        # Draw communication links
        if communication_links:
            self._draw_communication_links(agent_states, communication_links)
        
        # Draw agents
        self._draw_agents(agent_states)
        
        # Update info text
        self._update_info_text(agent_states, timestep)
        
        # Update trails
        self._update_trails(agent_states)
        
        # Draw trails
        self._draw_trails()
        
        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.001)
            return None
        elif self.render_mode == "rgb_array":
            # Convert to RGB array
            self.fig.canvas.draw()
            width, height = self.fig.canvas.get_width_height()
            buffer = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buffer = buffer.reshape((height, width, 3))
            return buffer
    
    def _clear_dynamic_elements(self):
        """Clear dynamic plot elements"""
        # Remove agent plots
        for plot in self.agent_plots.values():
            if hasattr(plot, 'remove'):
                plot.remove()
        self.agent_plots.clear()
        
        # Remove trail plots
        for plot in self.trail_plots.values():
            if hasattr(plot, 'remove'):
                plot.remove()
        self.trail_plots.clear()
        
        # Clear text
        for txt in self.ax.texts:
            txt.remove()
    
    def _draw_agents(self, agent_states: Dict[int, Dict[str, Any]]):
        """Draw agents on the plot
        
        Args:
            agent_states: Dictionary of agent states
        """
        for agent_id, state in agent_states.items():
            position = state["position"]
            
            # Get agent color
            agent_type = state.get("type", "default")
            color = self.agent_colors.get(agent_type, self.agent_colors["default"])
            
            # Adjust color based on battery
            battery_soc = state.get("battery_soc", 1.0)
            if battery_soc < 0.2:
                color = "orange"
            if battery_soc <= 0.0:
                color = "red"
            
            if self.view_mode == "3d":
                # 3D plot
                plot = self.ax.scatter(
                    position[0], position[1], position[2],
                    c=color, s=100, marker='o', edgecolors='black',
                    label=f"Agent {agent_id}"
                )
                
                # Add velocity vector
                velocity = state.get("velocity", np.zeros(3))
                if np.linalg.norm(velocity) > 0.1:
                    self.ax.quiver(
                        position[0], position[1], position[2],
                        velocity[0], velocity[1], velocity[2],
                        color=color, alpha=0.5, arrow_length_ratio=0.3
                    )
            else:
                # 2D plot
                plot = self.ax.scatter(
                    position[0], position[1],
                    c=color, s=150, marker='o', edgecolors='black',
                    label=f"Agent {agent_id}"
                )
                
                # Add agent ID
                self.ax.text(
                    position[0] + 1, position[1] + 1,
                    f"{agent_id}", fontsize=8
                )
                
                # Add altitude indicator
                altitude_text = f"{position[2]:.1f}m"
                self.ax.text(
                    position[0] + 1, position[1] - 2,
                    altitude_text, fontsize=6, alpha=0.7
                )
                
                # Add velocity vector
                velocity = state.get("velocity", np.zeros(3))
                if np.linalg.norm(velocity[:2]) > 0.1:
                    self.ax.arrow(
                        position[0], position[1],
                        velocity[0] * 2, velocity[1] * 2,
                        head_width=1.5, head_length=1.0,
                        fc=color, ec=color, alpha=0.5
                    )
            
            self.agent_plots[agent_id] = plot
    
    def _draw_obstacles(self, obstacles: List[Dict[str, Any]]):
        """Draw obstacles
        
        Args:
            obstacles: List of obstacle specifications
        """
        # Clear previous obstacles
        for patch in self.obstacle_patches:
            patch.remove()
        self.obstacle_patches.clear()
        
        for obstacle in obstacles:
            position = obstacle["position"]
            size = obstacle["size"]
            
            if self.view_mode == "3d":
                # Draw as wireframe box
                # This is simplified - proper 3D boxes would need more code
                pass
            else:
                # 2D rectangle
                rect = patches.Rectangle(
                    (position[0] - size[0]/2, position[1] - size[1]/2),
                    size[0], size[1],
                    linewidth=1, edgecolor='black',
                    facecolor='gray', alpha=0.5
                )
                self.ax.add_patch(rect)
                self.obstacle_patches.append(rect)
    
    def _draw_targets(self, targets: List[Dict[str, Any]]):
        """Draw targets or waypoints
        
        Args:
            targets: List of target specifications
        """
        # Clear previous targets
        for plot in self.target_plots:
            if hasattr(plot, 'remove'):
                plot.remove()
        self.target_plots.clear()
        
        for i, target in enumerate(targets):
            position = target["position"]
            radius = target.get("radius", 2.0)
            found = target.get("found", False)
            
            color = "green" if found else "yellow"
            marker = "^" if target.get("type") == "waypoint" else "*"
            
            if self.view_mode == "3d":
                plot = self.ax.scatter(
                    position[0], position[1], position[2],
                    c=color, s=200, marker=marker, edgecolors='black'
                )
            else:
                # Draw target
                plot = self.ax.scatter(
                    position[0], position[1],
                    c=color, s=200, marker=marker, edgecolors='black'
                )
                
                # Draw radius circle
                circle = patches.Circle(
                    (position[0], position[1]), radius,
                    fill=False, edgecolor=color, linestyle='--', alpha=0.5
                )
                self.ax.add_patch(circle)
                self.target_plots.append(circle)
                
                # Add label
                self.ax.text(
                    position[0], position[1] + radius + 2,
                    f"T{i}", fontsize=8, ha='center'
                )
            
            self.target_plots.append(plot)
    
    def _draw_communication_links(
        self,
        agent_states: Dict[int, Dict[str, Any]],
        communication_links: List[Tuple[int, int]]
    ):
        """Draw communication links between agents
        
        Args:
            agent_states: Agent states
            communication_links: List of connected agent pairs
        """
        for agent1_id, agent2_id in communication_links:
            if agent1_id in agent_states and agent2_id in agent_states:
                pos1 = agent_states[agent1_id]["position"]
                pos2 = agent_states[agent2_id]["position"]
                
                if self.view_mode == "3d":
                    self.ax.plot(
                        [pos1[0], pos2[0]],
                        [pos1[1], pos2[1]],
                        [pos1[2], pos2[2]],
                        'b--', alpha=0.3, linewidth=1
                    )
                else:
                    self.ax.plot(
                        [pos1[0], pos2[0]],
                        [pos1[1], pos2[1]],
                        'b--', alpha=0.3, linewidth=1
                    )
    
    def _update_trails(self, agent_states: Dict[int, Dict[str, Any]]):
        """Update agent position trails
        
        Args:
            agent_states: Current agent states
        """
        for agent_id, state in agent_states.items():
            if agent_id not in self.agent_trails:
                self.agent_trails[agent_id] = []
            
            # Add current position
            self.agent_trails[agent_id].append(state["position"].copy())
            
            # Limit trail length
            if len(self.agent_trails[agent_id]) > self.trail_length:
                self.agent_trails[agent_id].pop(0)
    
    def _draw_trails(self):
        """Draw agent trails"""
        for agent_id, trail in self.agent_trails.items():
            if len(trail) > 1:
                trail_array = np.array(trail)
                
                # Get agent color
                color = self.agent_colors.get("default", "red")
                if agent_id in self.agent_plots:
                    # Get color from current plot
                    pass
                
                if self.view_mode == "3d":
                    plot = self.ax.plot(
                        trail_array[:, 0],
                        trail_array[:, 1],
                        trail_array[:, 2],
                        color=color, alpha=0.3, linewidth=1
                    )
                else:
                    plot = self.ax.plot(
                        trail_array[:, 0],
                        trail_array[:, 1],
                        color=color, alpha=0.3, linewidth=1
                    )
                
                if len(plot) > 0:
                    self.trail_plots[agent_id] = plot[0]
    
    def _update_info_text(
        self,
        agent_states: Dict[int, Dict[str, Any]],
        timestep: float
    ):
        """Update information text on plot
        
        Args:
            agent_states: Agent states
            timestep: Current time
        """
        # Time info
        time_text = f"Time: {timestep:.1f}s"
        self.ax.text(
            0.02, 0.98, time_text,
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Agent count
        agent_text = f"Agents: {len(agent_states)}"
        self.ax.text(
            0.02, 0.93, agent_text,
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=10
        )
        
        # Average battery
        if agent_states:
            avg_battery = np.mean([
                state.get("battery_soc", 1.0) 
                for state in agent_states.values()
            ])
            battery_text = f"Avg Battery: {avg_battery:.1%}"
            self.ax.text(
                0.02, 0.88, battery_text,
                transform=self.ax.transAxes,
                verticalalignment='top',
                fontsize=10
            )
    
    def close(self):
        """Close the visualization window"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        self.agent_plots.clear()
        self.obstacle_patches.clear()
        self.target_plots.clear()
        self.trail_plots.clear()
        self.agent_trails.clear()
        
        logger.info("Visualization closed")
    
    def save_frame(self, filename: str):
        """Save current frame to file
        
        Args:
            filename: Output filename
        """
        if self.fig:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved frame to {filename}")