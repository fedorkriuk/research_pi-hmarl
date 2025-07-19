"""
Formation Flying Example

This example demonstrates coordinated formation flying with multiple drones:
- Various formation patterns (V-shape, line, circle, diamond)
- Dynamic formation switching
- Obstacle avoidance while maintaining formation
- Leader-follower dynamics
- Wind disturbance compensation
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import math

# Import PI-HMARL components
from src.environment import MultiAgentEnvironment
from src.agents.hierarchical_agent import HierarchicalAgent
from src.control.formation_controller import FormationController
from src.physics.aerodynamics import WindField
from src.communication.protocol import CommunicationProtocol, MessageType
from src.visualization.realtime_visualizer import RealtimeVisualizer


class FormationPattern:
    """Base class for formation patterns"""
    
    def get_positions(
        self,
        num_agents: int,
        leader_pos: np.ndarray,
        spacing: float = 10.0
    ) -> List[np.ndarray]:
        """Get agent positions for formation"""
        raise NotImplementedError


class VFormation(FormationPattern):
    """V-shaped formation like migrating birds"""
    
    def get_positions(
        self,
        num_agents: int,
        leader_pos: np.ndarray,
        spacing: float = 10.0
    ) -> List[np.ndarray]:
        positions = [leader_pos]
        
        # Angle of V-formation arms
        angle = 30 * np.pi / 180  # 30 degrees
        
        for i in range(1, num_agents):
            # Alternate between left and right wings
            side = 1 if i % 2 == 0 else -1
            row = (i + 1) // 2
            
            # Calculate offset
            dx = -spacing * row * np.cos(0)  # Behind leader
            dy = side * spacing * row * np.sin(angle)
            dz = 0  # Same altitude
            
            position = leader_pos + np.array([dx, dy, dz])
            positions.append(position)
        
        return positions


class LineFormation(FormationPattern):
    """Line formation"""
    
    def get_positions(
        self,
        num_agents: int,
        leader_pos: np.ndarray,
        spacing: float = 10.0
    ) -> List[np.ndarray]:
        positions = []
        
        for i in range(num_agents):
            offset = np.array([-i * spacing, 0, 0])
            positions.append(leader_pos + offset)
        
        return positions


class CircleFormation(FormationPattern):
    """Circular formation"""
    
    def get_positions(
        self,
        num_agents: int,
        leader_pos: np.ndarray,
        spacing: float = 10.0
    ) -> List[np.ndarray]:
        positions = []
        
        if num_agents == 1:
            return [leader_pos]
        
        # Calculate radius based on spacing
        radius = spacing * num_agents / (2 * np.pi)
        
        for i in range(num_agents):
            angle = 2 * np.pi * i / num_agents
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            positions.append(leader_pos + np.array([dx, dy, 0]))
        
        return positions


class DiamondFormation(FormationPattern):
    """Diamond formation"""
    
    def get_positions(
        self,
        num_agents: int,
        leader_pos: np.ndarray,
        spacing: float = 10.0
    ) -> List[np.ndarray]:
        positions = [leader_pos]
        
        if num_agents > 1:
            # Second drone behind leader
            positions.append(leader_pos + np.array([-spacing, 0, 0]))
        
        if num_agents > 2:
            # Third drone to the left
            positions.append(leader_pos + np.array([-spacing/2, -spacing, 0]))
        
        if num_agents > 3:
            # Fourth drone to the right
            positions.append(leader_pos + np.array([-spacing/2, spacing, 0]))
        
        # Additional drones form outer diamond
        for i in range(4, num_agents):
            angle = 2 * np.pi * (i - 4) / (num_agents - 4)
            radius = spacing * 2
            dx = -spacing + radius * np.cos(angle)
            dy = radius * np.sin(angle)
            positions.append(leader_pos + np.array([dx, dy, 0]))
        
        return positions


class FormationManager:
    """Manages formation flying operations"""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.formations = {
            'v_shape': VFormation(),
            'line': LineFormation(),
            'circle': CircleFormation(),
            'diamond': DiamondFormation()
        }
        self.current_formation = 'v_shape'
        self.formation_controller = FormationController()
        
        # Formation parameters
        self.spacing = 15.0  # meters
        self.position_tolerance = 2.0  # meters
        self.velocity_matching_gain = 0.5
        
    def switch_formation(self, new_formation: str):
        """Switch to a new formation"""
        if new_formation in self.formations:
            self.current_formation = new_formation
            print(f"Switching to {new_formation} formation")
    
    def get_desired_positions(self, leader_pos: np.ndarray) -> List[np.ndarray]:
        """Get desired positions for current formation"""
        formation = self.formations[self.current_formation]
        return formation.get_positions(self.num_agents, leader_pos, self.spacing)
    
    def compute_formation_error(
        self,
        current_positions: List[np.ndarray],
        desired_positions: List[np.ndarray]
    ) -> float:
        """Compute formation error metric"""
        if len(current_positions) != len(desired_positions):
            return float('inf')
        
        total_error = 0.0
        for current, desired in zip(current_positions, desired_positions):
            error = np.linalg.norm(current - desired)
            total_error += error
        
        return total_error / len(current_positions)
    
    def get_formation_feedback(
        self,
        agent_id: int,
        current_pos: np.ndarray,
        desired_pos: np.ndarray,
        leader_vel: np.ndarray,
        neighbors: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Get control feedback for formation keeping"""
        
        # Position error
        position_error = desired_pos - current_pos
        
        # Velocity matching with leader
        velocity_term = self.velocity_matching_gain * leader_vel
        
        # Collision avoidance with neighbors
        avoidance_term = np.zeros(3)
        for neighbor in neighbors:
            neighbor_pos = neighbor['position']
            distance = np.linalg.norm(current_pos - neighbor_pos)
            
            if distance < self.spacing * 0.8:  # Too close
                # Repulsion force
                direction = (current_pos - neighbor_pos) / (distance + 0.1)
                force = (self.spacing * 0.8 - distance) * 2.0
                avoidance_term += direction * force
        
        # Combined control
        control = position_error + velocity_term + avoidance_term
        
        return control


def create_obstacle_field(area_size: float, num_obstacles: int = 5) -> List[Dict[str, Any]]:
    """Create random obstacles in the environment"""
    obstacles = []
    
    for i in range(num_obstacles):
        obstacle = {
            'position': np.array([
                np.random.uniform(area_size * 0.3, area_size * 0.7),
                np.random.uniform(area_size * 0.3, area_size * 0.7),
                np.random.uniform(30, 70)
            ]),
            'radius': np.random.uniform(20, 40),
            'type': 'cylinder'
        }
        obstacles.append(obstacle)
    
    return obstacles


def run_formation_flying():
    """Run formation flying demonstration"""
    
    print("=== PI-HMARL Formation Flying Example ===\n")
    
    # Configuration
    num_drones = 7
    area_size = 1000.0  # 1km x 1km
    flight_altitude = 50.0
    mission_duration = 300  # 5 minutes
    
    # 1. Create Environment
    print("1. Setting up environment with wind...")
    
    # Create wind field
    wind_field = WindField(
        base_velocity=np.array([5.0, 2.0, 0.0]),  # 5 m/s east, 2 m/s north
        turbulence_intensity=0.2,
        gust_frequency=0.1
    )
    
    env = MultiAgentEnvironment(
        num_agents=num_drones,
        map_size=(area_size, area_size, 150),
        physics_config={
            'wind_field': wind_field,
            'turbulence_enabled': True
        },
        safety_config={
            'min_distance': 5.0,
            'max_altitude': 100.0
        }
    )
    
    # Add obstacles
    obstacles = create_obstacle_field(area_size, num_obstacles=8)
    for obstacle in obstacles:
        env.add_obstacle(obstacle)
    
    # 2. Create Agents
    print("2. Initializing drone agents...")
    agents = []
    
    for i in range(num_drones):
        agent = HierarchicalAgent(
            agent_id=i,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config={
                'control_frequency': 10.0,  # Hz
                'max_velocity': 20.0,  # m/s
                'max_acceleration': 5.0  # m/s^2
            }
        )
        agents.append(agent)
    
    # 3. Setup Formation Manager
    print("3. Configuring formation manager...")
    formation_manager = FormationManager(num_drones)
    
    # Formation switching schedule
    formation_schedule = [
        (0, 'v_shape'),
        (60, 'diamond'),
        (120, 'line'),
        (180, 'circle'),
        (240, 'v_shape')
    ]
    
    # 4. Setup Communication
    print("4. Establishing communication...")
    comm_protocols = []
    
    for i in range(num_drones):
        protocol = CommunicationProtocol(
            agent_id=i,
            num_agents=num_drones,
            range_limit=100.0  # Close formation communication
        )
        comm_protocols.append(protocol)
    
    # 5. Define Flight Path
    print("5. Planning flight path...")
    waypoints = [
        np.array([100, 100, flight_altitude]),
        np.array([800, 100, flight_altitude]),
        np.array([800, 800, flight_altitude]),
        np.array([100, 800, flight_altitude]),
        np.array([100, 100, flight_altitude])
    ]
    
    current_waypoint_idx = 0
    leader_id = 0  # Drone 0 is the leader
    
    # 6. Setup Visualization
    print("6. Starting visualization...")
    visualizer = RealtimeVisualizer(
        environment=env,
        config={
            'show_formation_lines': True,
            'show_obstacles': True,
            'show_wind_vectors': True,
            'update_rate': 30  # FPS
        }
    )
    visualizer.start()
    
    # 7. Metrics
    metrics = {
        'formation_error': [],
        'collision_events': 0,
        'obstacle_avoidance': 0,
        'wind_compensation': []
    }
    
    # 8. Run Formation Flying
    print("\n8. Starting formation flying mission...")
    print(f"   - Drones: {num_drones}")
    print(f"   - Formations: {[f[1] for f in formation_schedule]}")
    print(f"   - Wind: {wind_field.base_velocity[:2]} m/s\n")
    
    observations = env.reset()
    
    # Initial positions
    start_positions = formation_manager.get_desired_positions(waypoints[0])
    for i, pos in enumerate(start_positions):
        env.set_drone_position(i, pos)
    
    for step in range(mission_duration * 10):  # 10Hz
        time_seconds = step / 10.0
        
        # Check formation schedule
        for switch_time, formation_name in formation_schedule:
            if abs(time_seconds - switch_time) < 0.1:
                formation_manager.switch_formation(formation_name)
        
        # Leader navigation
        leader_pos = env.get_drone_position(leader_id)
        leader_target = waypoints[current_waypoint_idx]
        
        # Check if reached waypoint
        if np.linalg.norm(leader_pos - leader_target) < 20.0:
            current_waypoint_idx = (current_waypoint_idx + 1) % len(waypoints)
            print(f"Reached waypoint {current_waypoint_idx}")
        
        # Get leader velocity
        leader_vel = env.get_drone_velocity(leader_id)
        
        # Get desired formation positions
        desired_positions = formation_manager.get_desired_positions(leader_pos)
        
        # Current positions
        current_positions = [env.get_drone_position(i) for i in range(num_drones)]
        
        # Compute formation error
        formation_error = formation_manager.compute_formation_error(
            current_positions,
            desired_positions
        )
        metrics['formation_error'].append(formation_error)
        
        # Control each drone
        actions = {}
        
        for i in range(num_drones):
            if i == leader_id:
                # Leader follows waypoints
                action = agents[i].act(
                    observations[i],
                    target=waypoints[current_waypoint_idx]
                )
            else:
                # Followers maintain formation
                # Get neighbor information
                neighbors = []
                for j in range(num_drones):
                    if i != j:
                        neighbor_pos = current_positions[j]
                        distance = np.linalg.norm(current_positions[i] - neighbor_pos)
                        if distance < 50.0:  # Within communication range
                            neighbors.append({
                                'id': j,
                                'position': neighbor_pos,
                                'distance': distance
                            })
                
                # Get formation control feedback
                control_feedback = formation_manager.get_formation_feedback(
                    agent_id=i,
                    current_pos=current_positions[i],
                    desired_pos=desired_positions[i],
                    leader_vel=leader_vel,
                    neighbors=neighbors
                )
                
                # Apply control with wind compensation
                wind_at_position = wind_field.get_wind_at_position(
                    current_positions[i],
                    time_seconds
                )
                
                # Wind compensation
                wind_compensation = -wind_at_position * 0.3
                control_feedback += wind_compensation
                
                action = agents[i].act(
                    observations[i],
                    control_override=control_feedback
                )
            
            # Obstacle avoidance override
            for obstacle in obstacles:
                distance_to_obstacle = np.linalg.norm(
                    current_positions[i] - obstacle['position']
                )
                
                if distance_to_obstacle < obstacle['radius'] + 20.0:
                    # Emergency avoidance
                    avoidance_direction = (
                        current_positions[i] - obstacle['position']
                    ) / distance_to_obstacle
                    
                    avoidance_action = agents[i].act(
                        observations[i],
                        control_override=avoidance_direction * 10.0
                    )
                    action = avoidance_action
                    metrics['obstacle_avoidance'] += 1
            
            actions[i] = action
        
        # Communication
        for i in range(num_drones):
            # Share position and status
            message = {
                'type': MessageType.COORDINATION,
                'sender': i,
                'position': current_positions[i].tolist(),
                'velocity': env.get_drone_velocity(i).tolist(),
                'formation_error': formation_error
            }
            comm_protocols[i].broadcast(message)
        
        # Environment step
        observations, rewards, dones, info = env.step(actions)
        
        # Check for collisions
        if 'collision' in info:
            metrics['collision_events'] += 1
        
        # Update visualization
        visualizer.update({
            'formation_lines': list(zip(
                current_positions,
                desired_positions
            )),
            'formation_type': formation_manager.current_formation,
            'formation_error': formation_error,
            'wind_vectors': [
                wind_field.get_wind_at_position(pos, time_seconds)
                for pos in current_positions
            ]
        })
        
        # Progress update
        if step % 100 == 0:  # Every 10 seconds
            print(f"Time: {time_seconds:.0f}s | " +
                  f"Formation: {formation_manager.current_formation} | " +
                  f"Error: {formation_error:.2f}m | " +
                  f"Waypoint: {current_waypoint_idx+1}/{len(waypoints)}")
    
    # 9. Mission Summary
    print("\n\n=== Formation Flying Summary ===")
    print(f"Mission duration: {mission_duration}s")
    print(f"Average formation error: {np.mean(metrics['formation_error']):.2f}m")
    print(f"Collision events: {metrics['collision_events']}")
    print(f"Obstacle avoidance maneuvers: {metrics['obstacle_avoidance']}")
    print(f"Formations demonstrated: {len(formation_schedule)}")
    
    # 10. Generate Report
    generate_formation_report(metrics, formation_schedule)
    
    # Cleanup
    visualizer.stop()
    env.close()


def generate_formation_report(metrics: Dict, schedule: List[Tuple[int, str]]):
    """Generate formation flying report"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Formation Flying Mission Report', fontsize=16)
    
    # 1. Formation error over time
    ax = axes[0, 0]
    time_axis = np.arange(len(metrics['formation_error'])) / 10.0
    ax.plot(time_axis, metrics['formation_error'])
    
    # Mark formation switches
    for switch_time, formation_name in schedule:
        ax.axvline(x=switch_time, color='red', linestyle='--', alpha=0.5)
        ax.text(switch_time, ax.get_ylim()[1]*0.9, formation_name, 
                rotation=90, va='top', ha='right', fontsize=8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Formation Error (m)')
    ax.set_title('Formation Keeping Performance')
    ax.grid(True)
    
    # 2. Formation patterns visualization
    ax = axes[0, 1]
    ax.set_title('Formation Patterns')
    
    # Show different formations
    formations = {
        'V-Shape': [(0, 0), (-1, -1), (-1, 1), (-2, -2), (-2, 2)],
        'Diamond': [(0, 0), (-1, 0), (-0.5, -1), (-0.5, 1)],
        'Line': [(0, 0), (-1, 0), (-2, 0), (-3, 0)],
        'Circle': [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2*np.pi, 5)]
    }
    
    y_offset = 0
    for name, pattern in formations.items():
        xs = [p[0] for p in pattern]
        ys = [p[1] + y_offset for p in pattern]
        ax.scatter(xs, ys, s=50)
        ax.text(-3, y_offset, name, ha='right', va='center')
        y_offset += 3
    
    ax.set_xlim(-4, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 3. Error distribution
    ax = axes[1, 0]
    ax.hist(metrics['formation_error'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(metrics['formation_error']), 
               color='red', linestyle='--', 
               label=f'Mean: {np.mean(metrics['formation_error']):.2f}m')
    ax.set_xlabel('Formation Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Formation Error Distribution')
    ax.legend()
    ax.grid(True)
    
    # 4. Performance metrics
    ax = axes[1, 1]
    metrics_summary = {
        'Avg Error': f"{np.mean(metrics['formation_error']):.2f}m",
        'Min Error': f"{np.min(metrics['formation_error']):.2f}m",
        'Max Error': f"{np.max(metrics['formation_error']):.2f}m",
        'Collisions': metrics['collision_events'],
        'Avoidance': metrics['obstacle_avoidance']
    }
    
    y_pos = 0.9
    for key, value in metrics_summary.items():
        ax.text(0.1, y_pos, f"{key}:", fontweight='bold')
        ax.text(0.6, y_pos, str(value))
        y_pos -= 0.15
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Performance Summary')
    
    plt.tight_layout()
    
    # Save report
    report_path = Path('formation_flying_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"\nReport saved to: {report_path}")
    
    plt.close()


if __name__ == "__main__":
    run_formation_flying()