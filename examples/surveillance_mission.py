"""
Surveillance Mission Example

This example demonstrates how to use PI-HMARL for area surveillance
with multiple drones. The mission involves:
- Covering a specified area efficiently
- Detecting and tracking targets
- Coordinating between drones to avoid overlap
- Managing battery constraints
"""

import numpy as np
import torch
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

# Import PI-HMARL components
from src.environment import MultiAgentEnvironment
from src.agents.hierarchical_agent import HierarchicalAgent
from src.tasks.base_task import Task, TaskType, TaskStatus
from src.visualization.dashboard import Dashboard, DashboardConfig
from src.communication.protocol import CommunicationProtocol, MessageType
from src.energy.energy_optimizer import EnergyOptimizer


class SurveillanceTask(Task):
    """Surveillance task for area coverage"""
    
    def __init__(
        self,
        task_id: str,
        surveillance_area: List[tuple],
        coverage_requirement: float = 0.9,
        detection_range: float = 50.0,
        priority: str = "high"
    ):
        super().__init__(
            task_id=task_id,
            task_type=TaskType.SURVEILLANCE,
            priority=priority
        )
        
        self.surveillance_area = surveillance_area
        self.coverage_requirement = coverage_requirement
        self.detection_range = detection_range
        self.covered_cells = set()
        self.detected_targets = []
        
        # Discretize area into grid cells
        self.grid_resolution = detection_range / 2
        self._create_coverage_grid()
    
    def _create_coverage_grid(self):
        """Create grid for coverage tracking"""
        # Find bounding box
        xs = [p[0] for p in self.surveillance_area]
        ys = [p[1] for p in self.surveillance_area]
        
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)
        
        # Create grid
        self.grid_x = int((self.x_max - self.x_min) / self.grid_resolution) + 1
        self.grid_y = int((self.y_max - self.y_min) / self.grid_resolution) + 1
        self.total_cells = self.grid_x * self.grid_y
    
    def update_coverage(self, drone_positions: List[np.ndarray]):
        """Update coverage based on drone positions"""
        for pos in drone_positions:
            # Get cells within detection range
            grid_x = int((pos[0] - self.x_min) / self.grid_resolution)
            grid_y = int((pos[1] - self.y_min) / self.grid_resolution)
            
            # Mark cells as covered
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cell_x = grid_x + dx
                    cell_y = grid_y + dy
                    
                    if 0 <= cell_x < self.grid_x and 0 <= cell_y < self.grid_y:
                        self.covered_cells.add((cell_x, cell_y))
    
    def get_coverage_percentage(self) -> float:
        """Get current coverage percentage"""
        return len(self.covered_cells) / self.total_cells
    
    def get_uncovered_areas(self) -> List[tuple]:
        """Get list of uncovered grid cells"""
        uncovered = []
        
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                if (x, y) not in self.covered_cells:
                    # Convert back to world coordinates
                    world_x = self.x_min + x * self.grid_resolution
                    world_y = self.y_min + y * self.grid_resolution
                    uncovered.append((world_x, world_y))
        
        return uncovered


class SurveillanceCoordinator:
    """Coordinates multiple drones for surveillance mission"""
    
    def __init__(
        self,
        num_drones: int,
        communication_range: float = 1000.0
    ):
        self.num_drones = num_drones
        self.communication_range = communication_range
        self.sector_assignments = {}
        
    def assign_sectors(self, surveillance_area: List[tuple]) -> Dict[int, List[tuple]]:
        """Assign surveillance sectors to drones"""
        # Simple sector division - can be made more sophisticated
        xs = [p[0] for p in surveillance_area]
        ys = [p[1] for p in surveillance_area]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Divide into equal sectors
        sectors_per_side = int(np.sqrt(self.num_drones))
        x_step = (x_max - x_min) / sectors_per_side
        y_step = (y_max - y_min) / sectors_per_side
        
        drone_id = 0
        for i in range(sectors_per_side):
            for j in range(sectors_per_side):
                if drone_id < self.num_drones:
                    # Define sector boundaries
                    sector = [
                        (x_min + i * x_step, y_min + j * y_step),
                        (x_min + (i + 1) * x_step, y_min + j * y_step),
                        (x_min + (i + 1) * x_step, y_min + (j + 1) * y_step),
                        (x_min + i * x_step, y_min + (j + 1) * y_step)
                    ]
                    self.sector_assignments[drone_id] = sector
                    drone_id += 1
        
        return self.sector_assignments
    
    def generate_patrol_path(self, sector: List[tuple], altitude: float = 50.0) -> List[np.ndarray]:
        """Generate patrol path for a sector"""
        # Lawnmower pattern
        xs = [p[0] for p in sector]
        ys = [p[1] for p in sector]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        path = []
        y = y_min
        direction = 1
        spacing = 40.0  # Overlap for complete coverage
        
        while y <= y_max:
            if direction == 1:
                path.append(np.array([x_min, y, altitude]))
                path.append(np.array([x_max, y, altitude]))
            else:
                path.append(np.array([x_max, y, altitude]))
                path.append(np.array([x_min, y, altitude]))
            
            y += spacing
            direction *= -1
        
        return path


def run_surveillance_mission():
    """Run a complete surveillance mission"""
    
    print("=== PI-HMARL Surveillance Mission Example ===\n")
    
    # Configuration
    num_drones = 4
    surveillance_area = [
        (0, 0), (1000, 0), (1000, 1000), (0, 1000)
    ]  # 1km x 1km area
    mission_duration = 600  # 10 minutes
    
    # 1. Create Environment
    print("1. Setting up environment...")
    env_config = {
        'map_size': (1200, 1200, 200),
        'physics': {
            'wind_enabled': True,
            'wind_speed': 5.0,
            'turbulence': 0.1
        },
        'safety': {
            'min_distance': 10.0,
            'max_altitude': 150.0
        }
    }
    
    env = MultiAgentEnvironment(
        num_agents=num_drones,
        **env_config
    )
    
    # 2. Create Agents
    print("2. Initializing drone agents...")
    agents = []
    for i in range(num_drones):
        agent = HierarchicalAgent(
            agent_id=i,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config={
                'levels': ['strategic', 'tactical', 'operational'],
                'communication_enabled': True,
                'battery_capacity': 5000.0  # mAh
            }
        )
        agents.append(agent)
    
    # 3. Setup Communication
    print("3. Establishing communication network...")
    comm_protocols = []
    for i in range(num_drones):
        protocol = CommunicationProtocol(
            agent_id=i,
            num_agents=num_drones,
            range_limit=1000.0
        )
        comm_protocols.append(protocol)
    
    # 4. Create Surveillance Task
    print("4. Creating surveillance task...")
    surveillance_task = SurveillanceTask(
        task_id="SURV_001",
        surveillance_area=surveillance_area,
        coverage_requirement=0.95,
        detection_range=50.0
    )
    
    # 5. Setup Coordinator
    print("5. Initializing mission coordinator...")
    coordinator = SurveillanceCoordinator(num_drones)
    sector_assignments = coordinator.assign_sectors(surveillance_area)
    
    # Generate patrol paths
    patrol_paths = {}
    for drone_id, sector in sector_assignments.items():
        patrol_paths[drone_id] = coordinator.generate_patrol_path(sector)
    
    # 6. Setup Visualization
    print("6. Starting visualization dashboard...")
    dashboard_config = DashboardConfig(
        title="Surveillance Mission Monitor",
        update_interval=100,  # ms
        show_trajectories=True,
        show_communication=True,
        show_heatmap=True
    )
    
    dashboard = Dashboard(dashboard_config)
    dashboard.start()
    
    # 7. Initialize Energy Optimizer
    print("7. Configuring energy management...")
    energy_optimizer = EnergyOptimizer(
        num_agents=num_drones,
        charging_locations=[(600, 600, 0)]  # Central charging station
    )
    
    # 8. Run Mission
    print("\n8. Starting surveillance mission...")
    print(f"   - Area: {surveillance_area}")
    print(f"   - Drones: {num_drones}")
    print(f"   - Duration: {mission_duration}s")
    print(f"   - Coverage target: {surveillance_task.coverage_requirement*100}%\n")
    
    # Reset environment
    observations = env.reset()
    
    # Mission metrics
    metrics = {
        'coverage_history': [],
        'energy_usage': [],
        'targets_detected': [],
        'communication_stats': []
    }
    
    # Simulation loop
    current_waypoint = {i: 0 for i in range(num_drones)}
    
    for step in range(mission_duration * 10):  # 10Hz update rate
        # Get current positions
        drone_positions = []
        for i in range(num_drones):
            pos = env.get_drone_position(i)
            drone_positions.append(pos)
        
        # Update coverage
        surveillance_task.update_coverage(drone_positions)
        coverage = surveillance_task.get_coverage_percentage()
        
        # Agent decisions
        actions = {}
        for i in range(num_drones):
            # Get current waypoint
            if current_waypoint[i] < len(patrol_paths[i]):
                target = patrol_paths[i][current_waypoint[i]]
                
                # Check if reached waypoint
                distance = np.linalg.norm(drone_positions[i] - target)
                if distance < 5.0:
                    current_waypoint[i] += 1
                
                # Compute action to move toward target
                if current_waypoint[i] < len(patrol_paths[i]):
                    target = patrol_paths[i][current_waypoint[i]]
                    action = agents[i].act(observations[i], target=target)
                else:
                    # Return to base
                    action = agents[i].act(observations[i], target=np.array([600, 600, 50]))
            else:
                # Hover at base
                action = agents[i].act(observations[i], mode='hover')
            
            actions[i] = action
        
        # Check for low battery
        for i in range(num_drones):
            battery_level = env.get_battery_level(i)
            if battery_level < 0.3:  # 30% threshold
                # Override action to return to charging station
                charging_target = np.array([600, 600, 50])
                actions[i] = agents[i].act(observations[i], target=charging_target, priority='emergency')
        
        # Inter-drone communication
        for i in range(num_drones):
            # Share coverage information
            message = {
                'type': MessageType.COORDINATION,
                'sender': i,
                'coverage': coverage,
                'position': drone_positions[i].tolist(),
                'battery': env.get_battery_level(i)
            }
            comm_protocols[i].broadcast(message)
        
        # Environment step
        observations, rewards, dones, info = env.step(actions)
        
        # Update dashboard
        if step % 10 == 0:  # Update every second
            dashboard.update_data('drones', {
                'positions': drone_positions,
                'battery_levels': [env.get_battery_level(i) for i in range(num_drones)],
                'coverage': coverage
            })
            
            # Log metrics
            metrics['coverage_history'].append(coverage)
            metrics['energy_usage'].append(
                sum([env.get_battery_level(i) for i in range(num_drones)]) / num_drones
            )
        
        # Check mission completion
        if coverage >= surveillance_task.coverage_requirement:
            print(f"\n✓ Mission completed! Coverage: {coverage*100:.1f}%")
            break
        
        # Render if needed
        if step % 100 == 0:
            env.render()
    
    # 9. Mission Summary
    print("\n=== Mission Summary ===")
    print(f"Final coverage: {surveillance_task.get_coverage_percentage()*100:.1f}%")
    print(f"Average battery remaining: {np.mean([env.get_battery_level(i) for i in range(num_drones)])*100:.1f}%")
    print(f"Mission duration: {step/10:.1f}s")
    
    # 10. Generate Report
    generate_mission_report(metrics, surveillance_task, patrol_paths)
    
    # Cleanup
    dashboard.stop()
    env.close()


def generate_mission_report(metrics: Dict, task: SurveillanceTask, paths: Dict):
    """Generate mission report with visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Surveillance Mission Report', fontsize=16)
    
    # 1. Coverage progression
    ax = axes[0, 0]
    ax.plot(metrics['coverage_history'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage Progression')
    ax.grid(True)
    
    # 2. Energy usage
    ax = axes[0, 1]
    ax.plot(metrics['energy_usage'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average Battery Level (%)')
    ax.set_title('Energy Consumption')
    ax.grid(True)
    
    # 3. Patrol paths
    ax = axes[1, 0]
    colors = ['red', 'blue', 'green', 'orange']
    for drone_id, path in paths.items():
        if drone_id < len(colors):
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color=colors[drone_id], label=f'Drone {drone_id}', alpha=0.7)
    
    # Draw surveillance area
    area = task.surveillance_area + [task.surveillance_area[0]]
    xs = [p[0] for p in area]
    ys = [p[1] for p in area]
    ax.plot(xs, ys, 'k--', linewidth=2, label='Area boundary')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Patrol Paths')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # 4. Coverage heatmap
    ax = axes[1, 1]
    coverage_grid = np.zeros((task.grid_y, task.grid_x))
    for (x, y) in task.covered_cells:
        coverage_grid[y, x] = 1
    
    im = ax.imshow(coverage_grid, cmap='YlOrRd', origin='lower')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_title('Coverage Heatmap')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save report
    report_path = Path('surveillance_mission_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"\nReport saved to: {report_path}")
    
    plt.close()


if __name__ == "__main__":
    # Run the surveillance mission example
    run_surveillance_mission()