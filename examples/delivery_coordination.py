"""
Multi-Agent Delivery Coordination Example

This example demonstrates package delivery coordination with multiple drones:
- Dynamic package assignment based on location and capacity
- Energy-aware route planning
- Collision avoidance in shared airspace
- Real-time re-routing for new packages
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import heapq
from pathlib import Path

# Import PI-HMARL components
from src.environment import MultiAgentEnvironment
from src.agents.hierarchical_agent import HierarchicalAgent
from src.tasks.base_task import Task, TaskType
from src.task_decomposition.assignment_optimizer import AssignmentOptimizer
from src.energy.energy_optimizer import EnergyOptimizer
from src.physics.aerodynamics import DroneAerodynamics
from src.communication.protocol import CommunicationProtocol, MessageType


@dataclass
class Package:
    """Package to be delivered"""
    id: str
    pickup_location: np.ndarray
    delivery_location: np.ndarray
    weight: float  # kg
    priority: str  # 'low', 'medium', 'high'
    deadline: float  # seconds
    assigned_drone: int = -1
    picked_up: bool = False
    delivered: bool = False


@dataclass
class DeliveryHub:
    """Delivery hub/warehouse"""
    id: str
    location: np.ndarray
    packages: List[Package]
    charging_stations: int = 4


class DeliveryOptimizer:
    """Optimizes package assignments and routes"""
    
    def __init__(
        self,
        num_drones: int,
        drone_capacity: float = 2.0,  # kg
        energy_model: EnergyOptimizer = None
    ):
        self.num_drones = num_drones
        self.drone_capacity = drone_capacity
        self.energy_model = energy_model
        self.assignment_optimizer = AssignmentOptimizer()
    
    def assign_packages(
        self,
        packages: List[Package],
        drone_states: Dict[int, Dict[str, Any]]
    ) -> Dict[int, List[Package]]:
        """Assign packages to drones optimally"""
        
        # Filter unassigned packages
        unassigned = [p for p in packages if p.assigned_drone == -1 and not p.delivered]
        
        if not unassigned:
            return {}
        
        # Build cost matrix
        cost_matrix = np.zeros((self.num_drones, len(unassigned)))
        
        for i in range(self.num_drones):
            drone_state = drone_states[i]
            drone_pos = drone_state['position']
            drone_battery = drone_state['battery']
            current_load = drone_state['current_load']
            
            for j, package in enumerate(unassigned):
                # Check capacity constraint
                if current_load + package.weight > self.drone_capacity:
                    cost_matrix[i, j] = float('inf')
                    continue
                
                # Calculate cost factors
                pickup_distance = np.linalg.norm(drone_pos - package.pickup_location)
                delivery_distance = np.linalg.norm(
                    package.pickup_location - package.delivery_location
                )
                
                # Energy cost estimation
                if self.energy_model:
                    energy_required = self.energy_model.estimate_energy(
                        drone_pos,
                        package.pickup_location,
                        package.delivery_location,
                        package.weight
                    )
                    
                    if energy_required > drone_battery * 0.8:  # Safety margin
                        cost_matrix[i, j] = float('inf')
                        continue
                else:
                    energy_required = (pickup_distance + delivery_distance) * 0.01
                
                # Priority factor
                priority_multiplier = {
                    'low': 1.0,
                    'medium': 0.7,
                    'high': 0.4
                }[package.priority]
                
                # Total cost
                cost = (pickup_distance + delivery_distance) * priority_multiplier
                cost_matrix[i, j] = cost
        
        # Solve assignment problem
        assignments = self.assignment_optimizer.optimize_assignment(
            cost_matrix=cost_matrix,
            method='hungarian'
        )
        
        # Convert to package assignments
        drone_packages = {i: [] for i in range(self.num_drones)}
        
        for drone_id, package_idx in assignments.items():
            if package_idx < len(unassigned):
                package = unassigned[package_idx]
                package.assigned_drone = drone_id
                drone_packages[drone_id].append(package)
        
        return drone_packages
    
    def plan_delivery_route(
        self,
        packages: List[Package],
        start_position: np.ndarray,
        battery_level: float
    ) -> List[Tuple[np.ndarray, str]]:
        """Plan optimal delivery route for assigned packages"""
        
        if not packages:
            return []
        
        # Sort packages by priority and deadline
        sorted_packages = sorted(
            packages,
            key=lambda p: (
                0 if p.priority == 'high' else (1 if p.priority == 'medium' else 2),
                p.deadline
            )
        )
        
        route = []
        current_pos = start_position
        
        for package in sorted_packages:
            # Add pickup waypoint
            route.append((package.pickup_location, f'pickup_{package.id}'))
            
            # Add delivery waypoint
            route.append((package.delivery_location, f'deliver_{package.id}'))
            
            current_pos = package.delivery_location
        
        # Optimize route using TSP solver (simplified)
        optimized_route = self._optimize_route_tsp(route, start_position)
        
        return optimized_route
    
    def _optimize_route_tsp(
        self,
        waypoints: List[Tuple[np.ndarray, str]],
        start: np.ndarray
    ) -> List[Tuple[np.ndarray, str]]:
        """Simple TSP optimization (nearest neighbor heuristic)"""
        
        if len(waypoints) <= 2:
            return waypoints
        
        optimized = []
        remaining = waypoints.copy()
        current_pos = start
        
        while remaining:
            # Find nearest waypoint
            min_dist = float('inf')
            nearest_idx = 0
            
            for i, (pos, label) in enumerate(remaining):
                dist = np.linalg.norm(current_pos - pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            # Add to route
            waypoint = remaining.pop(nearest_idx)
            optimized.append(waypoint)
            current_pos = waypoint[0]
        
        return optimized


class DeliveryCoordinator:
    """Coordinates multi-drone delivery operations"""
    
    def __init__(
        self,
        num_drones: int,
        hubs: List[DeliveryHub],
        delivery_area: List[tuple]
    ):
        self.num_drones = num_drones
        self.hubs = hubs
        self.delivery_area = delivery_area
        
        # Airspace management
        self.flight_corridors = self._create_flight_corridors()
        self.altitude_layers = {
            'low': 30.0,    # Return flights
            'medium': 50.0,  # Empty drones
            'high': 70.0     # Loaded drones
        }
    
    def _create_flight_corridors(self) -> Dict[str, List[np.ndarray]]:
        """Create flight corridors for traffic management"""
        corridors = {}
        
        # Create grid-based corridors
        xs = [p[0] for p in self.delivery_area]
        ys = [p[1] for p in self.delivery_area]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # North-South corridors
        for i in range(5):
            x = x_min + (x_max - x_min) * i / 4
            corridors[f'NS_{i}'] = [
                np.array([x, y_min, 0]),
                np.array([x, y_max, 0])
            ]
        
        # East-West corridors
        for i in range(5):
            y = y_min + (y_max - y_min) * i / 4
            corridors[f'EW_{i}'] = [
                np.array([x_min, y, 0]),
                np.array([x_max, y, 0])
            ]
        
        return corridors
    
    def get_safe_altitude(self, drone_state: Dict[str, Any]) -> float:
        """Get safe altitude based on drone state"""
        if drone_state['current_load'] > 0:
            return self.altitude_layers['high']
        elif drone_state['returning_to_base']:
            return self.altitude_layers['low']
        else:
            return self.altitude_layers['medium']


def generate_delivery_scenario(
    num_packages: int = 20,
    area_size: float = 2000.0
) -> Tuple[List[DeliveryHub], List[Package]]:
    """Generate random delivery scenario"""
    
    # Create hubs
    hubs = [
        DeliveryHub(
            id="HUB_CENTRAL",
            location=np.array([area_size/2, area_size/2, 0]),
            packages=[],
            charging_stations=6
        ),
        DeliveryHub(
            id="HUB_NORTH",
            location=np.array([area_size/2, area_size*0.8, 0]),
            packages=[],
            charging_stations=4
        )
    ]
    
    # Generate packages
    packages = []
    priorities = ['low', 'medium', 'high']
    priority_weights = [0.5, 0.3, 0.2]
    
    for i in range(num_packages):
        # Random pickup from hubs
        hub = np.random.choice(hubs)
        
        # Random delivery location
        delivery_loc = np.array([
            np.random.uniform(100, area_size-100),
            np.random.uniform(100, area_size-100),
            0
        ])
        
        package = Package(
            id=f"PKG_{i:03d}",
            pickup_location=hub.location.copy(),
            delivery_location=delivery_loc,
            weight=np.random.uniform(0.5, 2.0),
            priority=np.random.choice(priorities, p=priority_weights),
            deadline=np.random.uniform(300, 1200)  # 5-20 minutes
        )
        
        packages.append(package)
        hub.packages.append(package)
    
    return hubs, packages


def run_delivery_mission():
    """Run multi-drone delivery coordination example"""
    
    print("=== PI-HMARL Multi-Drone Delivery Example ===\n")
    
    # Configuration
    num_drones = 6
    drone_capacity = 2.0  # kg
    delivery_area = [(0, 0), (2000, 0), (2000, 2000), (0, 2000)]  # 2km x 2km
    mission_duration = 1800  # 30 minutes
    
    # 1. Setup Environment
    print("1. Creating delivery environment...")
    env = MultiAgentEnvironment(
        num_agents=num_drones,
        map_size=(2200, 2200, 150),
        physics_config={
            'wind_enabled': True,
            'wind_speed': 3.0
        },
        safety_config={
            'min_distance': 15.0,
            'max_altitude': 100.0
        }
    )
    
    # 2. Create Delivery Scenario
    print("2. Generating delivery scenario...")
    hubs, packages = generate_delivery_scenario(num_packages=30)
    print(f"   - Hubs: {len(hubs)}")
    print(f"   - Packages: {len(packages)}")
    print(f"   - High priority: {sum(1 for p in packages if p.priority == 'high')}")
    
    # 3. Initialize Agents
    print("3. Initializing delivery drones...")
    agents = []
    drone_states = {}
    
    for i in range(num_drones):
        agent = HierarchicalAgent(
            agent_id=i,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config={
                'battery_capacity': 5000.0,
                'payload_capacity': drone_capacity,
                'cruise_speed': 15.0  # m/s
            }
        )
        agents.append(agent)
        
        # Initialize drone state
        drone_states[i] = {
            'position': hubs[0].location.copy() + np.array([i*10, 0, 30]),
            'battery': 1.0,
            'current_load': 0.0,
            'assigned_packages': [],
            'current_route': [],
            'returning_to_base': False,
            'status': 'idle'
        }
    
    # 4. Setup Optimization
    print("4. Initializing delivery optimizer...")
    energy_model = EnergyOptimizer(num_agents=num_drones)
    delivery_optimizer = DeliveryOptimizer(
        num_drones=num_drones,
        drone_capacity=drone_capacity,
        energy_model=energy_model
    )
    
    coordinator = DeliveryCoordinator(
        num_drones=num_drones,
        hubs=hubs,
        delivery_area=delivery_area
    )
    
    # 5. Setup Communication
    print("5. Establishing communication network...")
    comm_protocols = []
    for i in range(num_drones):
        protocol = CommunicationProtocol(
            agent_id=i,
            num_agents=num_drones,
            range_limit=2000.0
        )
        comm_protocols.append(protocol)
    
    # 6. Metrics Tracking
    metrics = {
        'packages_delivered': 0,
        'total_distance': 0.0,
        'average_delivery_time': [],
        'energy_efficiency': [],
        'priority_performance': {'high': 0, 'medium': 0, 'low': 0}
    }
    
    # 7. Run Delivery Operations
    print("\n7. Starting delivery operations...")
    print(f"   - Duration: {mission_duration}s")
    print(f"   - Update rate: 10Hz\n")
    
    observations = env.reset()
    
    for step in range(mission_duration * 10):  # 10Hz
        
        # Update drone positions from environment
        for i in range(num_drones):
            drone_states[i]['position'] = env.get_drone_position(i)
            drone_states[i]['battery'] = env.get_battery_level(i)
        
        # Package assignment (every 30 seconds)
        if step % 300 == 0:
            undelivered = [p for p in packages if not p.delivered]
            if undelivered:
                assignments = delivery_optimizer.assign_packages(
                    undelivered,
                    drone_states
                )
                
                # Update drone routes
                for drone_id, assigned_packages in assignments.items():
                    if assigned_packages:
                        route = delivery_optimizer.plan_delivery_route(
                            assigned_packages,
                            drone_states[drone_id]['position'],
                            drone_states[drone_id]['battery']
                        )
                        drone_states[drone_id]['current_route'] = route
                        drone_states[drone_id]['assigned_packages'] = assigned_packages
                        drone_states[drone_id]['status'] = 'delivering'
                        
                        print(f"Drone {drone_id} assigned {len(assigned_packages)} packages")
        
        # Execute drone actions
        actions = {}
        for i in range(num_drones):
            state = drone_states[i]
            
            # Check battery level
            if state['battery'] < 0.3 and state['status'] != 'charging':
                # Return to nearest hub for charging
                nearest_hub = min(hubs, key=lambda h: np.linalg.norm(state['position'] - h.location))
                state['status'] = 'returning_to_charge'
                target = nearest_hub.location + np.array([0, 0, 30])
                actions[i] = agents[i].act(observations[i], target=target, priority='high')
                continue
            
            # Execute delivery route
            if state['current_route'] and state['status'] == 'delivering':
                target_pos, target_label = state['current_route'][0]
                target_3d = np.append(target_pos[:2], coordinator.get_safe_altitude(state))
                
                # Check if reached waypoint
                distance = np.linalg.norm(state['position'] - target_3d)
                if distance < 5.0:
                    # Process waypoint
                    if 'pickup' in target_label:
                        package_id = target_label.split('_')[1]
                        for p in packages:
                            if p.id == package_id:
                                p.picked_up = True
                                state['current_load'] += p.weight
                                print(f"Drone {i} picked up {package_id}")
                                break
                    
                    elif 'deliver' in target_label:
                        package_id = target_label.split('_')[1]
                        for p in packages:
                            if p.id == package_id:
                                p.delivered = True
                                state['current_load'] -= p.weight
                                metrics['packages_delivered'] += 1
                                metrics['priority_performance'][p.priority] += 1
                                print(f"Drone {i} delivered {package_id} ({p.priority} priority)")
                                break
                    
                    # Remove completed waypoint
                    state['current_route'].pop(0)
                    
                    # Check if route completed
                    if not state['current_route']:
                        state['status'] = 'returning'
                        state['assigned_packages'] = []
                
                # Move toward target
                actions[i] = agents[i].act(observations[i], target=target_3d)
            
            # Return to base
            elif state['status'] == 'returning':
                target = hubs[0].location + np.array([i*10, 0, 30])
                distance = np.linalg.norm(state['position'] - target)
                
                if distance < 5.0:
                    state['status'] = 'idle'
                else:
                    actions[i] = agents[i].act(observations[i], target=target)
            
            # Charging
            elif state['status'] == 'charging':
                if state['battery'] > 0.9:
                    state['status'] = 'idle'
                    print(f"Drone {i} finished charging")
                else:
                    actions[i] = agents[i].act(observations[i], mode='hover')
            
            # Idle
            else:
                actions[i] = agents[i].act(observations[i], mode='hover')
        
        # Communication between drones
        for i in range(num_drones):
            # Share status and position
            message = {
                'type': MessageType.COORDINATION,
                'drone_id': i,
                'position': drone_states[i]['position'].tolist(),
                'status': drone_states[i]['status'],
                'battery': drone_states[i]['battery']
            }
            comm_protocols[i].broadcast(message)
        
        # Environment step
        observations, rewards, dones, info = env.step(actions)
        
        # Update metrics
        if step % 100 == 0:  # Every 10 seconds
            total_battery = sum(drone_states[i]['battery'] for i in range(num_drones))
            metrics['energy_efficiency'].append(
                metrics['packages_delivered'] / (num_drones - total_battery + 0.1)
            )
        
        # Progress update
        if step % 600 == 0:  # Every minute
            delivered = sum(1 for p in packages if p.delivered)
            print(f"\nTime: {step/10:.0f}s | Delivered: {delivered}/{len(packages)} | " +
                  f"Active drones: {sum(1 for d in drone_states.values() if d['status'] == 'delivering')}")
    
    # 8. Mission Summary
    print("\n\n=== Delivery Mission Summary ===")
    print(f"Total packages: {len(packages)}")
    print(f"Delivered: {metrics['packages_delivered']}")
    print(f"Delivery rate: {metrics['packages_delivered']/len(packages)*100:.1f}%")
    print(f"\nPriority performance:")
    for priority in ['high', 'medium', 'low']:
        total = sum(1 for p in packages if p.priority == priority)
        delivered = metrics['priority_performance'][priority]
        print(f"  {priority}: {delivered}/{total} ({delivered/total*100:.1f}%)")
    
    # 9. Generate Report
    generate_delivery_report(metrics, packages, drone_states, hubs)
    
    env.close()


def generate_delivery_report(
    metrics: Dict,
    packages: List[Package],
    drone_states: Dict,
    hubs: List[DeliveryHub]
):
    """Generate delivery mission report"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Delivery Mission Report', fontsize=16)
    
    # 1. Delivery timeline
    ax = axes[0, 0]
    delivered_times = []
    for i, p in enumerate(packages):
        if p.delivered:
            delivered_times.append(i)  # Simplified
    
    ax.hist(delivered_times, bins=20, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Packages Delivered')
    ax.set_title('Delivery Timeline')
    ax.grid(True)
    
    # 2. Energy efficiency
    ax = axes[0, 1]
    if metrics['energy_efficiency']:
        ax.plot(metrics['energy_efficiency'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Packages/Energy')
        ax.set_title('Energy Efficiency')
        ax.grid(True)
    
    # 3. Delivery map
    ax = axes[1, 0]
    
    # Plot hubs
    for hub in hubs:
        ax.scatter(hub.location[0], hub.location[1], s=200, c='red', marker='s', label='Hub')
    
    # Plot deliveries
    for p in packages:
        if p.delivered:
            ax.scatter(p.delivery_location[0], p.delivery_location[1], 
                      c='green', s=30, alpha=0.6)
        else:
            ax.scatter(p.delivery_location[0], p.delivery_location[1], 
                      c='orange', s=30, alpha=0.6)
    
    # Plot drone positions
    for i, state in drone_states.items():
        ax.scatter(state['position'][0], state['position'][1], 
                  c='blue', s=50, marker='^')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Delivery Locations')
    ax.grid(True)
    ax.axis('equal')
    
    # 4. Drone utilization
    ax = axes[1, 1]
    statuses = [state['status'] for state in drone_states.values()]
    status_counts = {s: statuses.count(s) for s in set(statuses)}
    
    ax.bar(status_counts.keys(), status_counts.values())
    ax.set_xlabel('Status')
    ax.set_ylabel('Number of Drones')
    ax.set_title('Drone Utilization')
    
    plt.tight_layout()
    
    # Save report
    report_path = Path('delivery_mission_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"\nReport saved to: {report_path}")
    
    plt.close()


if __name__ == "__main__":
    run_delivery_mission()