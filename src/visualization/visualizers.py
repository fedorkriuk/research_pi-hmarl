"""Visualization Components

This module implements various visualization components for the
monitoring dashboard.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from collections import deque
import colorsys
import logging

logger = logging.getLogger(__name__)


class MapVisualizer:
    """Visualizes agents and tasks on a map"""
    
    def __init__(
        self,
        map_bounds: Tuple[float, float, float, float],
        enable_3d: bool = True
    ):
        """Initialize map visualizer
        
        Args:
            map_bounds: Map boundaries (x_min, y_min, x_max, y_max)
            enable_3d: Enable 3D visualization
        """
        self.map_bounds = map_bounds
        self.enable_3d = enable_3d
        
        # Visual settings
        self.agent_colors = {}
        self.task_colors = {
            'surveillance': 'blue',
            'delivery': 'green',
            'search_rescue': 'red',
            'mapping': 'purple',
            'inspection': 'orange',
            'patrol': 'cyan'
        }
        
        # Trail storage
        self.agent_trails = {}
        self.trail_length = 50
    
    def create_map_figure(
        self,
        agents: Dict[int, Dict[str, Any]],
        tasks: Dict[str, Dict[str, Any]],
        show_trails: bool = True,
        show_communication: bool = False
    ) -> go.Figure:
        """Create map figure
        
        Args:
            agents: Agent data
            tasks: Task data
            show_trails: Show agent trails
            show_communication: Show communication links
            
        Returns:
            Plotly figure
        """
        if self.enable_3d:
            return self._create_3d_map(agents, tasks, show_trails, show_communication)
        else:
            return self._create_2d_map(agents, tasks, show_trails, show_communication)
    
    def _create_2d_map(
        self,
        agents: Dict[int, Dict[str, Any]],
        tasks: Dict[str, Dict[str, Any]],
        show_trails: bool,
        show_communication: bool
    ) -> go.Figure:
        """Create 2D map
        
        Args:
            agents: Agent data
            tasks: Task data
            show_trails: Show trails
            show_communication: Show communication
            
        Returns:
            2D map figure
        """
        fig = go.Figure()
        
        # Add map boundaries
        x_min, y_min, x_max, y_max = self.map_bounds
        fig.add_shape(
            type="rect",
            x0=x_min, y0=y_min, x1=x_max, y1=y_max,
            line=dict(color="gray", width=2),
            fillcolor="rgba(200, 200, 200, 0.1)"
        )
        
        # Add grid
        grid_spacing = 100
        for x in range(int(x_min), int(x_max) + 1, grid_spacing):
            fig.add_shape(
                type="line",
                x0=x, y0=y_min, x1=x, y1=y_max,
                line=dict(color="lightgray", width=0.5)
            )
        for y in range(int(y_min), int(y_max) + 1, grid_spacing):
            fig.add_shape(
                type="line",
                x0=x_min, y0=y, x1=x_max, y1=y,
                line=dict(color="lightgray", width=0.5)
            )
        
        # Add trails
        if show_trails:
            for agent_id, trail in self.agent_trails.items():
                if len(trail) > 1:
                    positions = np.array(trail)
                    fig.add_trace(go.Scatter(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        mode='lines',
                        line=dict(
                            color=self._get_agent_color(agent_id),
                            width=1
                        ),
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add agents
        for agent_id, agent in agents.items():
            pos = agent['position']
            color = self._get_agent_color(agent_id)
            
            # Agent marker
            fig.add_trace(go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=f"A{agent_id}",
                textposition="top center",
                name=f"Agent {agent_id}",
                hovertemplate=(
                    f"Agent {agent_id}<br>" +
                    "Position: (%{x:.1f}, %{y:.1f})<br>" +
                    f"Energy: {agent['energy']:.1f}%<br>" +
                    f"Status: {agent['status']}<extra></extra>"
                )
            ))
            
            # Direction indicator
            if 'velocity' in agent and np.linalg.norm(agent['velocity']) > 0.1:
                vel = agent['velocity']
                vel_norm = vel / (np.linalg.norm(vel) + 1e-6) * 20
                fig.add_annotation(
                    x=pos[0], y=pos[1],
                    ax=pos[0] + vel_norm[0],
                    ay=pos[1] + vel_norm[1],
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color
                )
        
        # Add tasks
        for task_id, task in tasks.items():
            if 'location' in task:
                loc = task['location']
                task_type = task.get('type', 'unknown')
                color = self.task_colors.get(task_type, 'gray')
                
                fig.add_trace(go.Scatter(
                    x=[loc[0]],
                    y=[loc[1]],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color=color,
                        symbol='square',
                        line=dict(color='black', width=1)
                    ),
                    name=f"Task {task_id}",
                    hovertemplate=(
                        f"Task {task_id}<br>" +
                        f"Type: {task_type}<br>" +
                        "Location: (%{x:.1f}, %{y:.1f})<br>" +
                        f"Progress: {task.get('progress', 0):.1f}%<extra></extra>"
                    )
                ))
        
        # Add communication links
        if show_communication:
            self._add_communication_links(fig, agents)
        
        # Update layout
        fig.update_layout(
            title="Agent and Task Map",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            hovermode='closest',
            showlegend=True,
            xaxis=dict(range=[x_min - 50, x_max + 50]),
            yaxis=dict(range=[y_min - 50, y_max + 50]),
            width=800,
            height=600
        )
        
        return fig
    
    def _create_3d_map(
        self,
        agents: Dict[int, Dict[str, Any]],
        tasks: Dict[str, Dict[str, Any]],
        show_trails: bool,
        show_communication: bool
    ) -> go.Figure:
        """Create 3D map
        
        Args:
            agents: Agent data
            tasks: Task data
            show_trails: Show trails
            show_communication: Show communication
            
        Returns:
            3D map figure
        """
        fig = go.Figure()
        
        # Add ground plane
        x_min, y_min, x_max, y_max = self.map_bounds
        xx, yy = np.meshgrid(
            [x_min, x_max],
            [y_min, y_max]
        )
        zz = np.zeros_like(xx)
        
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            showscale=False,
            opacity=0.3,
            name='Ground'
        ))
        
        # Add trails
        if show_trails:
            for agent_id, trail in self.agent_trails.items():
                if len(trail) > 1:
                    positions = np.array(trail)
                    # Add z-coordinate if missing
                    if positions.shape[1] == 2:
                        z = np.full(len(positions), 50)  # Default altitude
                        positions = np.column_stack([positions, z])
                    
                    fig.add_trace(go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                        mode='lines',
                        line=dict(
                            color=self._get_agent_color(agent_id),
                            width=3
                        ),
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add agents
        for agent_id, agent in agents.items():
            pos = agent['position']
            if len(pos) == 2:
                pos = [pos[0], pos[1], 50]  # Default altitude
            
            color = self._get_agent_color(agent_id)
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=f"A{agent_id}",
                textposition="top center",
                name=f"Agent {agent_id}",
                hovertemplate=(
                    f"Agent {agent_id}<br>" +
                    "Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>" +
                    f"Energy: {agent['energy']:.1f}%<br>" +
                    f"Status: {agent['status']}<extra></extra>"
                )
            ))
        
        # Add tasks
        for task_id, task in tasks.items():
            if 'location' in task:
                loc = task['location']
                if len(loc) == 2:
                    loc = [loc[0], loc[1], 0]  # Ground level
                
                task_type = task.get('type', 'unknown')
                color = self.task_colors.get(task_type, 'gray')
                
                # Task cylinder
                theta = np.linspace(0, 2*np.pi, 20)
                radius = 30
                x_cyl = loc[0] + radius * np.cos(theta)
                y_cyl = loc[1] + radius * np.sin(theta)
                z_bottom = np.zeros_like(theta)
                z_top = np.full_like(theta, 100)
                
                fig.add_trace(go.Scatter3d(
                    x=np.concatenate([x_cyl, x_cyl]),
                    y=np.concatenate([y_cyl, y_cyl]),
                    z=np.concatenate([z_bottom, z_top]),
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Update layout
        fig.update_layout(
            title="3D Agent and Task Map",
            scene=dict(
                xaxis_title="X Position (m)",
                yaxis_title="Y Position (m)",
                zaxis_title="Altitude (m)",
                xaxis=dict(range=[x_min - 50, x_max + 50]),
                yaxis=dict(range=[y_min - 50, y_max + 50]),
                zaxis=dict(range=[0, 200]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def update_trails(self, agents: Dict[int, Dict[str, Any]]):
        """Update agent trails
        
        Args:
            agents: Agent data
        """
        for agent_id, agent in agents.items():
            if agent_id not in self.agent_trails:
                self.agent_trails[agent_id] = deque(maxlen=self.trail_length)
            
            self.agent_trails[agent_id].append(agent['position'])
    
    def _get_agent_color(self, agent_id: int) -> str:
        """Get consistent color for agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Color string
        """
        if agent_id not in self.agent_colors:
            # Generate unique color
            hue = (agent_id * 137.5) % 360 / 360  # Golden angle
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            self.agent_colors[agent_id] = f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})"
        
        return self.agent_colors[agent_id]
    
    def _add_communication_links(
        self,
        fig: go.Figure,
        agents: Dict[int, Dict[str, Any]]
    ):
        """Add communication links between agents
        
        Args:
            fig: Figure to add to
            agents: Agent data
        """
        # Simple distance-based communication
        comm_range = 200  # meters
        
        agent_list = list(agents.items())
        for i, (id1, agent1) in enumerate(agent_list):
            for id2, agent2 in agent_list[i+1:]:
                pos1 = np.array(agent1['position'][:2])
                pos2 = np.array(agent2['position'][:2])
                
                dist = np.linalg.norm(pos1 - pos2)
                if dist <= comm_range:
                    fig.add_trace(go.Scatter(
                        x=[pos1[0], pos2[0]],
                        y=[pos1[1], pos2[1]],
                        mode='lines',
                        line=dict(
                            color='rgba(128, 128, 128, 0.3)',
                            width=1,
                            dash='dot'
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))


class TrajectoryVisualizer:
    """Visualizes agent trajectories and paths"""
    
    def __init__(self):
        """Initialize trajectory visualizer"""
        self.trajectory_history = {}
        self.planned_paths = {}
    
    def create_trajectory_figure(
        self,
        agent_id: int,
        trajectory: List[torch.Tensor],
        planned_path: Optional[List[torch.Tensor]] = None,
        obstacles: Optional[List[Dict[str, Any]]] = None
    ) -> go.Figure:
        """Create trajectory visualization
        
        Args:
            agent_id: Agent ID
            trajectory: Actual trajectory
            planned_path: Planned path
            obstacles: Obstacle list
            
        Returns:
            Trajectory figure
        """
        fig = go.Figure()
        
        # Convert trajectory
        if trajectory:
            traj_array = np.array([p.numpy() if isinstance(p, torch.Tensor) else p for p in trajectory])
            
            # Actual trajectory
            fig.add_trace(go.Scatter(
                x=traj_array[:, 0],
                y=traj_array[:, 1],
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=5),
                name='Actual Path',
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
            ))
            
            # Start and end markers
            fig.add_trace(go.Scatter(
                x=[traj_array[0, 0]],
                y=[traj_array[0, 1]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='circle'),
                name='Start',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[traj_array[-1, 0]],
                y=[traj_array[-1, 1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='square'),
                name='Current',
                showlegend=False
            ))
        
        # Planned path
        if planned_path:
            plan_array = np.array([p.numpy() if isinstance(p, torch.Tensor) else p for p in planned_path])
            
            fig.add_trace(go.Scatter(
                x=plan_array[:, 0],
                y=plan_array[:, 1],
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name='Planned Path',
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
            ))
        
        # Obstacles
        if obstacles:
            for obs in obstacles:
                if obs['type'] == 'circle':
                    theta = np.linspace(0, 2*np.pi, 50)
                    x = obs['center'][0] + obs['radius'] * np.cos(theta)
                    y = obs['center'][1] + obs['radius'] * np.sin(theta)
                    
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        line=dict(color='red', width=2),
                        name='Obstacle',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Update layout
        fig.update_layout(
            title=f"Agent {agent_id} Trajectory",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            hovermode='closest',
            showlegend=True,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            width=600,
            height=600
        )
        
        return fig
    
    def create_trajectory_comparison(
        self,
        trajectories: Dict[int, List[torch.Tensor]],
        labels: Optional[Dict[int, str]] = None
    ) -> go.Figure:
        """Compare multiple trajectories
        
        Args:
            trajectories: Dict of agent trajectories
            labels: Optional labels for agents
            
        Returns:
            Comparison figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (agent_id, trajectory) in enumerate(trajectories.items()):
            if trajectory:
                traj_array = np.array([p.numpy() if isinstance(p, torch.Tensor) else p for p in trajectory])
                
                label = labels.get(agent_id, f"Agent {agent_id}") if labels else f"Agent {agent_id}"
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=traj_array[:, 0],
                    y=traj_array[:, 1],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=label,
                    hovertemplate='Agent: ' + label + '<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
                ))
                
                # Start markers
                fig.add_trace(go.Scatter(
                    x=[traj_array[0, 0]],
                    y=[traj_array[0, 1]],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='circle'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title="Trajectory Comparison",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            hovermode='closest',
            showlegend=True,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            width=700,
            height=600
        )
        
        return fig


class EnergyVisualizer:
    """Visualizes energy consumption and battery states"""
    
    def __init__(self):
        """Initialize energy visualizer"""
        self.energy_history = {}
        self.power_history = {}
    
    def create_energy_figure(
        self,
        agents: Dict[int, Dict[str, Any]],
        show_predictions: bool = False
    ) -> go.Figure:
        """Create energy status figure
        
        Args:
            agents: Agent data
            show_predictions: Show energy predictions
            
        Returns:
            Energy figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Battery Levels', 'Power Consumption', 
                          'Temperature', 'Energy Distribution'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'pie'}]]
        )
        
        # Battery levels
        agent_ids = list(agents.keys())
        energies = [agents[id]['energy'] for id in agent_ids]
        colors = ['red' if e < 20 else 'yellow' if e < 50 else 'green' for e in energies]
        
        fig.add_trace(
            go.Bar(
                x=[f"Agent {id}" for id in agent_ids],
                y=energies,
                marker_color=colors,
                text=[f"{e:.1f}%" for e in energies],
                textposition='auto',
                name='Battery Level'
            ),
            row=1, col=1
        )
        
        # Power consumption history
        if agent_ids and agent_ids[0] in self.power_history:
            for agent_id in agent_ids[:5]:  # Limit to 5 agents
                if agent_id in self.power_history:
                    history = list(self.power_history[agent_id])
                    fig.add_trace(
                        go.Scatter(
                            y=history,
                            mode='lines',
                            name=f"Agent {agent_id}",
                            line=dict(width=2)
                        ),
                        row=1, col=2
                    )
        
        # Temperature
        temps = [agents[id].get('temperature', 25) for id in agent_ids]
        fig.add_trace(
            go.Scatter(
                x=[f"Agent {id}" for id in agent_ids],
                y=temps,
                mode='markers+lines',
                marker=dict(
                    size=10,
                    color=temps,
                    colorscale='Hot',
                    showscale=True,
                    colorbar=dict(title="°C", x=0.45, y=0.15)
                ),
                line=dict(color='gray', width=1),
                name='Temperature'
            ),
            row=2, col=1
        )
        
        # Energy distribution pie chart
        energy_ranges = {'Critical (<20%)': 0, 'Low (20-50%)': 0, 
                        'Good (50-80%)': 0, 'Full (>80%)': 0}
        
        for energy in energies:
            if energy < 20:
                energy_ranges['Critical (<20%)'] += 1
            elif energy < 50:
                energy_ranges['Low (20-50%)'] += 1
            elif energy < 80:
                energy_ranges['Good (50-80%)'] += 1
            else:
                energy_ranges['Full (>80%)'] += 1
        
        fig.add_trace(
            go.Pie(
                labels=list(energy_ranges.keys()),
                values=list(energy_ranges.values()),
                marker=dict(colors=['red', 'orange', 'yellow', 'green']),
                hole=0.3
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Energy Management Dashboard",
            showlegend=False,
            height=800,
            width=1000
        )
        
        fig.update_xaxes(title_text="Agent", row=1, col=1)
        fig.update_yaxes(title_text="Battery %", row=1, col=1)
        fig.update_xaxes(title_text="Time Steps", row=1, col=2)
        fig.update_yaxes(title_text="Power (W)", row=1, col=2)
        fig.update_xaxes(title_text="Agent", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        
        return fig
    
    def create_energy_timeline(
        self,
        agent_id: int,
        time_window: int = 300  # seconds
    ) -> go.Figure:
        """Create energy timeline for single agent
        
        Args:
            agent_id: Agent ID
            time_window: Time window in seconds
            
        Returns:
            Timeline figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Battery Level', 'Power Consumption', 'Temperature'),
            vertical_spacing=0.05
        )
        
        if agent_id in self.energy_history:
            history = self.energy_history[agent_id]
            
            # Time axis
            time_steps = list(range(len(history)))
            
            # Battery level
            energies = [h['energy'] for h in history]
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=energies,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='green', width=2),
                    name='Battery'
                ),
                row=1, col=1
            )
            
            # Add critical level line
            fig.add_hline(y=20, line_dash="dash", line_color="red", 
                         annotation_text="Critical", row=1, col=1)
            
            # Power consumption
            if 'power' in history[0]:
                powers = [h['power'] for h in history]
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=powers,
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Power'
                    ),
                    row=2, col=1
                )
            
            # Temperature
            if 'temperature' in history[0]:
                temps = [h['temperature'] for h in history]
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=temps,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Temperature'
                    ),
                    row=3, col=1
                )
                
                # Add temperature limit
                fig.add_hline(y=60, line_dash="dash", line_color="red", 
                             annotation_text="Limit", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Agent {agent_id} Energy Timeline",
            showlegend=False,
            height=600,
            width=800
        )
        
        fig.update_yaxes(title_text="Battery %", row=1, col=1)
        fig.update_yaxes(title_text="Power (W)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
        fig.update_xaxes(title_text="Time Steps", row=3, col=1)
        
        return fig
    
    def update_history(self, agents: Dict[int, Dict[str, Any]]):
        """Update energy history
        
        Args:
            agents: Agent data
        """
        for agent_id, agent in agents.items():
            if agent_id not in self.energy_history:
                self.energy_history[agent_id] = deque(maxlen=1000)
                self.power_history[agent_id] = deque(maxlen=100)
            
            self.energy_history[agent_id].append({
                'energy': agent['energy'],
                'temperature': agent.get('temperature', 25),
                'power': agent.get('power', 0),
                'timestamp': agent.get('timestamp', 0)
            })
            
            self.power_history[agent_id].append(agent.get('power', 0))


class TaskProgressVisualizer:
    """Visualizes task execution progress"""
    
    def __init__(self):
        """Initialize task progress visualizer"""
        self.task_history = {}
    
    def create_progress_figure(
        self,
        tasks: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create task progress figure
        
        Args:
            tasks: Task data
            
        Returns:
            Progress figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Task Progress', 'Task Timeline', 
                          'Task Distribution', 'Agent Utilization'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # Task progress bars
        task_ids = list(tasks.keys())[:10]  # Limit to 10 tasks
        progresses = [tasks[id].get('progress', 0) for id in task_ids]
        statuses = [tasks[id].get('status', 'unknown') for id in task_ids]
        
        colors = ['green' if s == 'completed' else 'blue' if s == 'active' else 'gray' 
                 for s in statuses]
        
        fig.add_trace(
            go.Bar(
                x=[f"Task {id[:8]}" for id in task_ids],
                y=progresses,
                marker_color=colors,
                text=[f"{p:.0f}%" for p in progresses],
                textposition='auto',
                name='Progress'
            ),
            row=1, col=1
        )
        
        # Task timeline (Gantt-like)
        active_tasks = [(id, t) for id, t in tasks.items() 
                       if t.get('status') in ['active', 'completed']][:5]
        
        for i, (task_id, task) in enumerate(active_tasks):
            start_time = task.get('start_time', 0)
            current_time = task.get('current_time', start_time + 100)
            
            fig.add_trace(
                go.Scatter(
                    x=[start_time, current_time],
                    y=[i, i],
                    mode='lines',
                    line=dict(
                        color='green' if task['status'] == 'completed' else 'blue',
                        width=20
                    ),
                    name=f"Task {task_id[:8]}",
                    showlegend=False,
                    hovertemplate=f"Task {task_id}<br>Progress: {task.get('progress', 0):.0f}%<extra></extra>"
                ),
                row=1, col=2
            )
        
        # Task type distribution
        task_types = {}
        for task in tasks.values():
            task_type = task.get('type', 'unknown')
            task_types[task_type] = task_types.get(task_type, 0) + 1
        
        if task_types:
            fig.add_trace(
                go.Pie(
                    labels=list(task_types.keys()),
                    values=list(task_types.values()),
                    hole=0.3
                ),
                row=2, col=1
            )
        
        # Agent utilization
        agent_tasks = {}
        for task in tasks.values():
            if task.get('status') == 'active':
                for agent_id in task.get('assigned_agents', []):
                    agent_tasks[agent_id] = agent_tasks.get(agent_id, 0) + 1
        
        if agent_tasks:
            fig.add_trace(
                go.Bar(
                    x=[f"Agent {id}" for id in agent_tasks.keys()],
                    y=list(agent_tasks.values()),
                    marker_color='purple',
                    name='Active Tasks'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Task Execution Dashboard",
            showlegend=False,
            height=800,
            width=1000
        )
        
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_yaxes(title_text="Progress %", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Task", row=1, col=2)
        fig.update_xaxes(title_text="Agent", row=2, col=2)
        fig.update_yaxes(title_text="Active Tasks", row=2, col=2)
        
        return fig
    
    def create_task_dependency_graph(
        self,
        tasks: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create task dependency graph
        
        Args:
            tasks: Task data with dependencies
            
        Returns:
            Dependency graph figure
        """
        # Build graph
        G = nx.DiGraph()
        
        for task_id, task in tasks.items():
            G.add_node(task_id, 
                      status=task.get('status', 'pending'),
                      progress=task.get('progress', 0))
            
            for dep in task.get('dependencies', []):
                if dep in tasks:
                    G.add_edge(dep, task_id)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            task = tasks.get(node, {})
            status = task.get('status', 'pending')
            progress = task.get('progress', 0)
            
            color = 'green' if status == 'completed' else 'blue' if status == 'active' else 'gray'
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=30,
                    color=color,
                    line=dict(color='white', width=2)
                ),
                text=f"{node[:8]}<br>{progress:.0f}%",
                textposition="middle center",
                hovertemplate=f"Task: {node}<br>Status: {status}<br>Progress: {progress:.0f}%<extra></extra>",
                showlegend=False
            ))
        
        fig.update_layout(
            title="Task Dependency Graph",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        return fig


class NetworkVisualizer:
    """Visualizes communication network and data flow"""
    
    def __init__(self):
        """Initialize network visualizer"""
        self.communication_history = deque(maxlen=1000)
        self.bandwidth_usage = {}
    
    def create_network_figure(
        self,
        agents: Dict[int, Dict[str, Any]],
        communication_links: Optional[List[Tuple[int, int, float]]] = None
    ) -> go.Figure:
        """Create network visualization
        
        Args:
            agents: Agent data
            communication_links: List of (agent1, agent2, signal_strength)
            
        Returns:
            Network figure
        """
        # Build network graph
        G = nx.Graph()
        
        for agent_id in agents:
            G.add_node(agent_id, pos=agents[agent_id]['position'][:2])
        
        if communication_links:
            for agent1, agent2, strength in communication_links:
                if agent1 in G and agent2 in G:
                    G.add_edge(agent1, agent2, weight=strength)
        else:
            # Create based on distance
            comm_range = 200
            agent_list = list(agents.items())
            for i, (id1, agent1) in enumerate(agent_list):
                for id2, agent2 in agent_list[i+1:]:
                    dist = np.linalg.norm(
                        np.array(agent1['position'][:2]) - 
                        np.array(agent2['position'][:2])
                    )
                    if dist <= comm_range:
                        strength = 1.0 - dist / comm_range
                        G.add_edge(id1, id2, weight=strength)
        
        # Layout
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create figure
        fig = go.Figure()
        
        # Add edges with varying width based on signal strength
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 0.5)
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=weight * 5,
                    color=f'rgba(0, 0, 255, {weight})'
                ),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            agent = agents[node]
            
            # Node color based on communication status
            degree = G.degree(node)
            color = 'green' if degree > 2 else 'yellow' if degree > 0 else 'red'
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=color,
                    line=dict(color='white', width=2)
                ),
                text=f"A{node}",
                textposition="top center",
                hovertemplate=(
                    f"Agent {node}<br>" +
                    f"Connections: {degree}<br>" +
                    f"Status: {agent['status']}<extra></extra>"
                ),
                showlegend=False
            ))
        
        # Add legend
        for color, label in [('green', 'Well Connected'), 
                           ('yellow', 'Limited Connection'), 
                           ('red', 'Isolated')]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=label
            ))
        
        fig.update_layout(
            title="Communication Network",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            hovermode='closest',
            showlegend=True,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            width=700,
            height=600
        )
        
        return fig
    
    def create_bandwidth_figure(
        self,
        time_window: int = 100
    ) -> go.Figure:
        """Create bandwidth usage figure
        
        Args:
            time_window: Time window for history
            
        Returns:
            Bandwidth figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Total Bandwidth Usage', 'Per-Agent Usage'),
            shared_xaxes=True
        )
        
        if self.communication_history:
            # Total bandwidth
            history = list(self.communication_history)[-time_window:]
            total_bandwidth = [h.get('total', 0) for h in history]
            
            fig.add_trace(
                go.Scatter(
                    y=total_bandwidth,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='blue', width=2),
                    name='Total'
                ),
                row=1, col=1
            )
            
            # Per-agent bandwidth
            if self.bandwidth_usage:
                for agent_id, usage in list(self.bandwidth_usage.items())[:5]:
                    fig.add_trace(
                        go.Scatter(
                            y=list(usage)[-time_window:],
                            mode='lines',
                            name=f"Agent {agent_id}",
                            line=dict(width=2)
                        ),
                        row=2, col=1
                    )
        
        fig.update_layout(
            title="Network Bandwidth Usage",
            showlegend=True,
            height=600,
            width=800
        )
        
        fig.update_yaxes(title_text="Bandwidth (Mbps)", row=1, col=1)
        fig.update_yaxes(title_text="Bandwidth (Mbps)", row=2, col=1)
        fig.update_xaxes(title_text="Time Steps", row=2, col=1)
        
        return fig
    
    def update_communication(
        self,
        communication_data: Dict[str, Any]
    ):
        """Update communication history
        
        Args:
            communication_data: Communication metrics
        """
        self.communication_history.append(communication_data)
        
        # Update per-agent bandwidth
        for agent_id, bandwidth in communication_data.get('per_agent', {}).items():
            if agent_id not in self.bandwidth_usage:
                self.bandwidth_usage[agent_id] = deque(maxlen=1000)
            self.bandwidth_usage[agent_id].append(bandwidth)