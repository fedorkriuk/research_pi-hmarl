"""
Search and Rescue Mission Example

This example demonstrates a coordinated search and rescue operation:
- Systematic area search with multiple drones
- Target detection using onboard sensors
- Dynamic task reallocation when targets are found
- Cooperative rescue operations
- Emergency response prioritization
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
import time

# Import PI-HMARL components
from src.environment import MultiAgentEnvironment
from src.agents.hierarchical_agent import HierarchicalAgent
from src.tasks.base_task import Task, TaskType, TaskStatus
from src.communication.protocol import CommunicationProtocol, MessageType, MessagePriority
from src.control.search_patterns import SpiralSearch, GridSearch, SectorSearch
from src.visualization.dashboard import Dashboard


class Victim:
    """Represents a person to be rescued"""
    
    def __init__(
        self,
        victim_id: str,
        position: np.ndarray,
        criticality: str = 'medium',  # 'low', 'medium', 'high', 'critical'
        detection_difficulty: float = 0.5,
        movement_pattern: str = 'stationary'
    ):
        self.id = victim_id
        self.position = position
        self.initial_position = position.copy()
        self.criticality = criticality
        self.detection_difficulty = detection_difficulty
        self.movement_pattern = movement_pattern
        self.detected = False
        self.rescued = False
        self.detection_time = None
        self.rescue_time = None
        
        # Health degradation
        self.health = 1.0
        self.health_decay_rate = {
            'low': 0.0001,
            'medium': 0.0005,
            'high': 0.001,
            'critical': 0.002
        }[criticality]
    
    def update(self, time_step: float):
        """Update victim state"""
        # Health degradation
        self.health = max(0, self.health - self.health_decay_rate * time_step)
        
        # Movement (if applicable)
        if self.movement_pattern == 'random_walk' and not self.detected:
            self.position += np.random.randn(3) * 0.5
            self.position[2] = 0  # Keep on ground
        elif self.movement_pattern == 'drift':
            # Simulate water drift
            self.position += np.array([0.2, 0.1, 0]) * time_step


class SearchArea:
    """Defines the search area and zones"""
    
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        terrain_type: str = 'mixed'
    ):
        self.bounds = bounds  # [(x_min, x_max), (y_min, y_max)]
        self.terrain_type = terrain_type
        self.searched_cells = set()
        self.cell_size = 20.0  # meters
        
        # Grid dimensions
        self.grid_x = int((bounds[0][1] - bounds[0][0]) / self.cell_size)
        self.grid_y = int((bounds[1][1] - bounds[1][0]) / self.cell_size)
        
        # Terrain difficulty affects search speed and detection
        self.terrain_difficulty = {
            'open': 0.2,
            'forest': 0.6,
            'urban': 0.4,
            'water': 0.8,
            'mixed': 0.5
        }[terrain_type]
        
        # Priority zones (e.g., last known positions)
        self.priority_zones = []
    
    def add_priority_zone(self, center: np.ndarray, radius: float, priority: float):
        """Add a high-priority search zone"""
        self.priority_zones.append({
            'center': center,
            'radius': radius,
            'priority': priority
        })
    
    def get_search_priority(self, position: np.ndarray) -> float:
        """Get search priority for a position"""
        base_priority = 1.0
        
        # Check priority zones
        for zone in self.priority_zones:
            distance = np.linalg.norm(position[:2] - zone['center'][:2])
            if distance < zone['radius']:
                base_priority = max(base_priority, zone['priority'])
        
        # Adjust for terrain
        base_priority *= (1.0 - self.terrain_difficulty * 0.5)
        
        return base_priority
    
    def mark_searched(self, position: np.ndarray, radius: float):
        """Mark area as searched"""
        grid_x = int((position[0] - self.bounds[0][0]) / self.cell_size)
        grid_y = int((position[1] - self.bounds[1][0]) / self.cell_size)
        
        cells_radius = int(radius / self.cell_size)
        
        for dx in range(-cells_radius, cells_radius + 1):
            for dy in range(-cells_radius, cells_radius + 1):
                cell_x = grid_x + dx
                cell_y = grid_y + dy
                
                if 0 <= cell_x < self.grid_x and 0 <= cell_y < self.grid_y:
                    self.searched_cells.add((cell_x, cell_y))
    
    def get_coverage(self) -> float:
        """Get search area coverage percentage"""
        total_cells = self.grid_x * self.grid_y
        return len(self.searched_cells) / total_cells if total_cells > 0 else 0


class SearchCoordinator:
    """Coordinates search and rescue operations"""
    
    def __init__(
        self,
        num_drones: int,
        search_area: SearchArea,
        sensor_range: float = 30.0
    ):
        self.num_drones = num_drones
        self.search_area = search_area
        self.sensor_range = sensor_range
        
        # Search patterns
        self.search_patterns = {
            'grid': GridSearch(),
            'spiral': SpiralSearch(),
            'sector': SectorSearch()
        }
        
        # Drone assignments
        self.drone_sectors = {}
        self.drone_patterns = {}
        self.drone_states = {}
        
        # Victim tracking
        self.detected_victims = []
        self.rescue_assignments = {}
        
        self._assign_search_sectors()
    
    def _assign_search_sectors(self):
        """Assign initial search sectors to drones"""
        # Divide search area into sectors
        x_min, x_max = self.search_area.bounds[0]
        y_min, y_max = self.search_area.bounds[1]
        
        # Simple grid division
        sectors_per_side = int(np.sqrt(self.num_drones))
        x_step = (x_max - x_min) / sectors_per_side
        y_step = (y_max - y_min) / sectors_per_side
        
        drone_id = 0
        for i in range(sectors_per_side):
            for j in range(sectors_per_side):
                if drone_id < self.num_drones:
                    sector_bounds = [
                        (x_min + i * x_step, x_min + (i + 1) * x_step),
                        (y_min + j * y_step, y_min + (j + 1) * y_step)
                    ]
                    
                    self.drone_sectors[drone_id] = sector_bounds
                    # Assign search pattern based on terrain
                    if self.search_area.terrain_type == 'urban':
                        self.drone_patterns[drone_id] = 'grid'
                    else:
                        self.drone_patterns[drone_id] = 'spiral'
                    
                    drone_id += 1
    
    def report_detection(
        self,
        drone_id: int,
        victim: Victim,
        confidence: float
    ):
        """Report victim detection"""
        if confidence > 0.7 and not victim.detected:
            victim.detected = True
            victim.detection_time = time.time()
            self.detected_victims.append(victim)
            
            print(f"Drone {drone_id} detected victim {victim.id} " +
                  f"({victim.criticality} priority) at {victim.position}")
            
            # Trigger rescue assignment
            self._assign_rescue(victim)
    
    def _assign_rescue(self, victim: Victim):
        """Assign drones for rescue operation"""
        # Number of drones needed based on criticality
        drones_needed = {
            'low': 1,
            'medium': 1,
            'high': 2,
            'critical': 3
        }[victim.criticality]
        
        # Find closest available drones
        available_drones = []
        for drone_id in range(self.num_drones):
            if drone_id not in self.rescue_assignments:
                available_drones.append(drone_id)
        
        if len(available_drones) >= drones_needed:
            # Sort by distance to victim
            drone_distances = []
            for drone_id in available_drones:
                if drone_id in self.drone_states:
                    distance = np.linalg.norm(
                        self.drone_states[drone_id]['position'] - victim.position
                    )
                    drone_distances.append((drone_id, distance))
            
            drone_distances.sort(key=lambda x: x[1])
            
            # Assign closest drones
            assigned_drones = []
            for i in range(min(drones_needed, len(drone_distances))):
                drone_id = drone_distances[i][0]
                self.rescue_assignments[drone_id] = victim
                assigned_drones.append(drone_id)
            
            print(f"Assigned drones {assigned_drones} to rescue {victim.id}")
    
    def update_drone_state(self, drone_id: int, state: Dict[str, Any]):
        """Update drone state information"""
        self.drone_states[drone_id] = state
    
    def get_search_waypoint(
        self,
        drone_id: int,
        current_position: np.ndarray
    ) -> Optional[np.ndarray]:
        """Get next search waypoint for drone"""
        if drone_id not in self.drone_sectors:
            return None
        
        sector = self.drone_sectors[drone_id]
        pattern = self.drone_patterns[drone_id]
        
        # Generate waypoint using search pattern
        search_pattern = self.search_patterns[pattern]
        waypoint = search_pattern.get_next_waypoint(
            current_position,
            sector,
            self.search_area.searched_cells
        )
        
        return waypoint


def detect_victim(
    drone_pos: np.ndarray,
    victim: Victim,
    sensor_range: float,
    terrain_difficulty: float
) -> float:
    """Calculate detection probability"""
    distance = np.linalg.norm(drone_pos - victim.position)
    
    if distance > sensor_range:
        return 0.0
    
    # Base detection probability
    base_prob = 1.0 - (distance / sensor_range)
    
    # Adjust for victim detection difficulty
    base_prob *= (1.0 - victim.detection_difficulty)
    
    # Adjust for terrain
    base_prob *= (1.0 - terrain_difficulty * 0.5)
    
    # Add some randomness
    base_prob *= np.random.uniform(0.8, 1.2)
    
    return np.clip(base_prob, 0.0, 1.0)


def run_search_rescue_mission():
    """Run search and rescue mission"""
    
    print("=== PI-HMARL Search and Rescue Mission ===\n")
    
    # Configuration
    num_drones = 8
    search_area_bounds = [(0, 1500), (0, 1500)]  # 1.5km x 1.5km
    sensor_range = 40.0  # meters
    mission_duration = 900  # 15 minutes
    
    # 1. Create Search Area
    print("1. Defining search area...")
    search_area = SearchArea(
        bounds=search_area_bounds,
        terrain_type='mixed'
    )
    
    # Add priority zones (last known positions)
    search_area.add_priority_zone(
        center=np.array([400, 600, 0]),
        radius=200,
        priority=2.0
    )
    search_area.add_priority_zone(
        center=np.array([1000, 1000, 0]),
        radius=150,
        priority=1.5
    )
    
    # 2. Create Victims
    print("2. Generating victim scenarios...")
    victims = []
    
    # Critical victims in priority zones
    victims.append(Victim(
        victim_id="V001",
        position=np.array([420, 580, 0]),
        criticality='critical',
        detection_difficulty=0.3
    ))
    
    victims.append(Victim(
        victim_id="V002",
        position=np.array([980, 1020, 0]),
        criticality='high',
        detection_difficulty=0.4
    ))
    
    # Random victims
    for i in range(3, 10):
        position = np.array([
            np.random.uniform(*search_area_bounds[0]),
            np.random.uniform(*search_area_bounds[1]),
            0
        ])
        
        victims.append(Victim(
            victim_id=f"V{i:03d}",
            position=position,
            criticality=np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
            detection_difficulty=np.random.uniform(0.3, 0.7),
            movement_pattern=np.random.choice(['stationary', 'random_walk'], p=[0.8, 0.2])
        ))
    
    print(f"   - Total victims: {len(victims)}")
    print(f"   - Critical: {sum(1 for v in victims if v.criticality == 'critical')}")
    print(f"   - High priority: {sum(1 for v in victims if v.criticality == 'high')}")
    
    # 3. Setup Environment
    print("3. Initializing environment...")
    env = MultiAgentEnvironment(
        num_agents=num_drones,
        map_size=(1600, 1600, 150),
        physics_config={
            'wind_enabled': True,
            'wind_speed': 4.0,
            'visibility': 0.8  # Reduced visibility
        }
    )
    
    # 4. Create Agents
    print("4. Deploying search and rescue drones...")
    agents = []
    
    for i in range(num_drones):
        agent = HierarchicalAgent(
            agent_id=i,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config={
                'sensor_range': sensor_range,
                'sensor_type': 'thermal_camera',
                'max_carry_weight': 50.0,  # kg
                'battery_capacity': 6000.0  # mAh
            }
        )
        agents.append(agent)
    
    # 5. Setup Coordinator
    print("5. Initializing search coordinator...")
    coordinator = SearchCoordinator(
        num_drones=num_drones,
        search_area=search_area,
        sensor_range=sensor_range
    )
    
    # 6. Setup Communication
    print("6. Establishing emergency communication network...")
    comm_protocols = []
    
    for i in range(num_drones):
        protocol = CommunicationProtocol(
            agent_id=i,
            num_agents=num_drones,
            range_limit=1000.0,
            priority_routing=True  # Emergency priority
        )
        comm_protocols.append(protocol)
    
    # 7. Setup Dashboard
    print("7. Starting mission dashboard...")
    dashboard = Dashboard(config={
        'title': 'Search & Rescue Operations',
        'show_heatmap': True,
        'show_victims': True,
        'show_alerts': True
    })
    dashboard.start()
    
    # 8. Mission Metrics
    metrics = {
        'victims_detected': 0,
        'victims_rescued': 0,
        'detection_times': [],
        'rescue_times': [],
        'area_coverage': [],
        'false_positives': 0,
        'battery_usage': []
    }
    
    # 9. Run Mission
    print("\n9. Commencing search and rescue operations...")
    observations = env.reset()
    
    # Deploy drones from base
    base_position = np.array([800, 800, 0])
    for i in range(num_drones):
        start_pos = base_position + np.array([
            (i % 4) * 20 - 30,
            (i // 4) * 20 - 10,
            50
        ])
        env.set_drone_position(i, start_pos)
    
    for step in range(mission_duration * 10):  # 10Hz
        time_seconds = step / 10.0
        
        # Update victims
        for victim in victims:
            if not victim.rescued:
                victim.update(1.0)
        
        # Get drone positions
        drone_positions = [env.get_drone_position(i) for i in range(num_drones)]
        
        # Update coordinator
        for i in range(num_drones):
            coordinator.update_drone_state(i, {
                'position': drone_positions[i],
                'battery': env.get_battery_level(i),
                'status': 'searching' if i not in coordinator.rescue_assignments else 'rescuing'
            })
        
        # Execute drone actions
        actions = {}
        
        for i in range(num_drones):
            # Check if assigned to rescue
            if i in coordinator.rescue_assignments:
                victim = coordinator.rescue_assignments[i]
                
                if not victim.rescued:
                    # Move to victim
                    target = victim.position + np.array([0, 0, 30])
                    distance = np.linalg.norm(drone_positions[i] - target)
                    
                    if distance < 5.0:
                        # Perform rescue
                        victim.rescued = True
                        victim.rescue_time = time.time()
                        metrics['victims_rescued'] += 1
                        
                        rescue_duration = victim.rescue_time - victim.detection_time
                        metrics['rescue_times'].append(rescue_duration)
                        
                        print(f"Drone {i} rescued {victim.id}! " +
                              f"Rescue time: {rescue_duration:.1f}s")
                        
                        # Return to base
                        del coordinator.rescue_assignments[i]
                        target = base_position + np.array([0, 0, 50])
                    
                    actions[i] = agents[i].act(observations[i], target=target)
                else:
                    # Return to base after rescue
                    target = base_position + np.array([0, 0, 50])
                    actions[i] = agents[i].act(observations[i], target=target)
            else:
                # Search pattern
                waypoint = coordinator.get_search_waypoint(i, drone_positions[i])
                
                if waypoint is not None:
                    # Add altitude
                    target = np.append(waypoint[:2], 40.0)
                    actions[i] = agents[i].act(observations[i], target=target)
                    
                    # Mark area as searched
                    search_area.mark_searched(drone_positions[i], sensor_range)
                else:
                    # Hover in place
                    actions[i] = agents[i].act(observations[i], mode='hover')
            
            # Victim detection
            for victim in victims:
                if not victim.detected:
                    detection_prob = detect_victim(
                        drone_positions[i],
                        victim,
                        sensor_range,
                        search_area.terrain_difficulty
                    )
                    
                    if detection_prob > 0.7:
                        coordinator.report_detection(i, victim, detection_prob)
                        metrics['victims_detected'] += 1
                        
                        detection_time = time_seconds
                        metrics['detection_times'].append(detection_time)
                        
                        # Send high-priority message
                        message = {
                            'type': MessageType.EMERGENCY,
                            'priority': MessagePriority.HIGH,
                            'sender': i,
                            'victim_id': victim.id,
                            'position': victim.position.tolist(),
                            'criticality': victim.criticality
                        }
                        comm_protocols[i].broadcast(message)
        
        # Environment step
        observations, rewards, dones, info = env.step(actions)
        
        # Update metrics
        if step % 10 == 0:  # Every second
            coverage = search_area.get_coverage()
            metrics['area_coverage'].append(coverage)
            
            avg_battery = np.mean([env.get_battery_level(i) for i in range(num_drones)])
            metrics['battery_usage'].append(avg_battery)
        
        # Update dashboard
        if step % 50 == 0:  # Every 5 seconds
            dashboard.update_data('search_status', {
                'coverage': search_area.get_coverage() * 100,
                'victims_detected': metrics['victims_detected'],
                'victims_rescued': metrics['victims_rescued'],
                'drone_positions': drone_positions,
                'victim_positions': [v.position for v in victims],
                'victim_status': [(v.detected, v.rescued) for v in victims]
            })
        
        # Progress report
        if step % 600 == 0:  # Every minute
            print(f"\nTime: {time_seconds:.0f}s | " +
                  f"Coverage: {search_area.get_coverage()*100:.1f}% | " +
                  f"Detected: {metrics['victims_detected']}/{len(victims)} | " +
                  f"Rescued: {metrics['victims_rescued']}")
        
        # Check mission completion
        if metrics['victims_rescued'] == len(victims):
            print("\n✓ All victims rescued! Mission complete.")
            break
    
    # 10. Mission Summary
    print("\n\n=== Search and Rescue Mission Summary ===")
    print(f"Mission duration: {time_seconds:.1f}s")
    print(f"Victims found: {metrics['victims_detected']}/{len(victims)}")
    print(f"Victims rescued: {metrics['victims_rescued']}/{len(victims)}")
    print(f"Area covered: {search_area.get_coverage()*100:.1f}%")
    
    if metrics['detection_times']:
        print(f"Average detection time: {np.mean(metrics['detection_times']):.1f}s")
    
    if metrics['rescue_times']:
        print(f"Average rescue time: {np.mean(metrics['rescue_times']):.1f}s")
    
    # Victim health status
    print("\nVictim Status:")
    for victim in victims:
        status = "Rescued" if victim.rescued else ("Detected" if victim.detected else "Missing")
        print(f"  {victim.id} ({victim.criticality}): {status}, Health: {victim.health*100:.0f}%")
    
    # 11. Generate Report
    generate_rescue_report(metrics, victims, search_area)
    
    # Cleanup
    dashboard.stop()
    env.close()


def generate_rescue_report(
    metrics: Dict,
    victims: List[Victim],
    search_area: SearchArea
):
    """Generate search and rescue mission report"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Search and Rescue Mission Report', fontsize=16)
    
    # 1. Search coverage over time
    ax = axes[0, 0]
    time_axis = np.arange(len(metrics['area_coverage']))
    ax.plot(time_axis, np.array(metrics['area_coverage']) * 100)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Area Coverage (%)')
    ax.set_title('Search Progress')
    ax.grid(True)
    
    # 2. Victim detection and rescue timeline
    ax = axes[0, 1]
    
    # Detection times
    if metrics['detection_times']:
        ax.scatter(metrics['detection_times'], 
                  range(len(metrics['detection_times'])),
                  c='orange', s=50, label='Detected')
    
    # Rescue times (simplified)
    rescue_times_cumulative = []
    for i, victim in enumerate(victims):
        if victim.rescued and victim.rescue_time and victim.detection_time:
            rescue_time = (victim.rescue_time - victim.detection_time) + metrics['detection_times'][i]
            rescue_times_cumulative.append(rescue_time)
    
    if rescue_times_cumulative:
        ax.scatter(rescue_times_cumulative,
                  range(len(rescue_times_cumulative)),
                  c='green', s=50, label='Rescued')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Victim Count')
    ax.set_title('Detection and Rescue Timeline')
    ax.legend()
    ax.grid(True)
    
    # 3. Search area map
    ax = axes[1, 0]
    
    # Draw search area
    x_min, x_max = search_area.bounds[0]
    y_min, y_max = search_area.bounds[1]
    
    # Searched cells (heatmap)
    search_grid = np.zeros((search_area.grid_y, search_area.grid_x))
    for (x, y) in search_area.searched_cells:
        if 0 <= x < search_area.grid_x and 0 <= y < search_area.grid_y:
            search_grid[y, x] = 1
    
    im = ax.imshow(search_grid, cmap='YlOrRd', alpha=0.5, 
                   extent=[x_min, x_max, y_min, y_max],
                   origin='lower')
    
    # Plot victims
    for victim in victims:
        color = 'green' if victim.rescued else ('orange' if victim.detected else 'red')
        marker = 'o' if victim.criticality != 'critical' else '*'
        size = 50 if victim.criticality == 'low' else (100 if victim.criticality == 'medium' else 150)
        
        ax.scatter(victim.position[0], victim.position[1],
                  c=color, marker=marker, s=size, edgecolors='black')
    
    # Priority zones
    for zone in search_area.priority_zones:
        circle = Circle(zone['center'][:2], zone['radius'],
                       fill=False, edgecolor='blue', linewidth=2, linestyle='--')
        ax.add_patch(circle)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Search Area and Victim Locations')
    
    # 4. Performance metrics
    ax = axes[1, 1]
    
    # Metrics by priority
    priority_stats = {'critical': [0, 0], 'high': [0, 0], 'medium': [0, 0], 'low': [0, 0]}
    
    for victim in victims:
        priority_stats[victim.criticality][0] += 1  # Total
        if victim.rescued:
            priority_stats[victim.criticality][1] += 1  # Rescued
    
    priorities = list(priority_stats.keys())
    totals = [priority_stats[p][0] for p in priorities]
    rescued = [priority_stats[p][1] for p in priorities]
    
    x = np.arange(len(priorities))
    width = 0.35
    
    ax.bar(x - width/2, totals, width, label='Total', alpha=0.8)
    ax.bar(x + width/2, rescued, width, label='Rescued', alpha=0.8)
    
    ax.set_xlabel('Priority Level')
    ax.set_ylabel('Number of Victims')
    ax.set_title('Rescue Performance by Priority')
    ax.set_xticks(x)
    ax.set_xticklabels(priorities)
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    
    # Save report
    report_path = Path('search_rescue_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"\nReport saved to: {report_path}")
    
    plt.close()


if __name__ == "__main__":
    run_search_rescue_mission()