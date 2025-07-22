"""
Multi-Domain Scenario Generator for Q1 Publication Standards
Creates diverse, challenging scenarios across multiple physical domains
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import json
from abc import ABC, abstractmethod
from enum import Enum

class PhysicalDomain(Enum):
    """Physical domains for evaluation"""
    AERIAL = "aerial"
    GROUND = "ground"
    UNDERWATER = "underwater"
    SPACE = "space"
    HYBRID = "hybrid"

@dataclass
class DomainPhysics:
    """Physics constraints for a specific domain"""
    domain: PhysicalDomain
    gravity: float  # m/s²
    medium_density: float  # kg/m³
    drag_coefficient: float
    max_velocity: float  # m/s
    max_acceleration: float  # m/s²
    communication_range: float  # meters
    communication_bandwidth: float  # bits/second
    sensor_noise_std: float
    actuator_noise_std: float
    energy_consumption_model: str
    environmental_factors: List[str]

@dataclass
class ScenarioConfig:
    """Configuration for a specific scenario"""
    name: str
    domain: PhysicalDomain
    num_agents: int
    mission_duration: int  # timesteps
    objectives: List[str]
    physics: DomainPhysics
    difficulty_level: float  # 0-1
    noise_level: float  # 0-1
    failure_rate: float  # 0-1
    communication_constraints: Dict[str, Any]
    adversarial_elements: bool
    heterogeneous_agents: bool
    dynamic_environment: bool

class MultiDomainScenarioGenerator:
    """
    Generate challenging scenarios across multiple physical domains
    Meeting Q1 publication standards for comprehensive evaluation
    """
    
    def __init__(self):
        self.physics_configs = self._initialize_domain_physics()
        self.scenario_templates = self._initialize_scenario_templates()
        
    def _initialize_domain_physics(self) -> Dict[PhysicalDomain, DomainPhysics]:
        """Initialize realistic physics for each domain"""
        return {
            PhysicalDomain.AERIAL: DomainPhysics(
                domain=PhysicalDomain.AERIAL,
                gravity=9.81,
                medium_density=1.225,  # Air at sea level
                drag_coefficient=0.47,
                max_velocity=30.0,  # m/s for quadcopters
                max_acceleration=10.0,
                communication_range=1000.0,
                communication_bandwidth=1e6,
                sensor_noise_std=0.1,
                actuator_noise_std=0.05,
                energy_consumption_model="quadratic_velocity",
                environmental_factors=["wind", "turbulence", "obstacles", "no_fly_zones"]
            ),
            PhysicalDomain.GROUND: DomainPhysics(
                domain=PhysicalDomain.GROUND,
                gravity=9.81,
                medium_density=1.0,  # Simplified
                drag_coefficient=0.3,
                max_velocity=10.0,  # m/s for ground robots
                max_acceleration=5.0,
                communication_range=500.0,
                communication_bandwidth=5e5,
                sensor_noise_std=0.15,
                actuator_noise_std=0.08,
                energy_consumption_model="linear_velocity_friction",
                environmental_factors=["terrain", "obstacles", "slopes", "weather"]
            ),
            PhysicalDomain.UNDERWATER: DomainPhysics(
                domain=PhysicalDomain.UNDERWATER,
                gravity=9.81,
                medium_density=1000.0,  # Water
                drag_coefficient=0.8,
                max_velocity=2.0,  # m/s for AUVs
                max_acceleration=1.0,
                communication_range=100.0,  # Acoustic communication
                communication_bandwidth=1e4,  # Much lower underwater
                sensor_noise_std=0.2,
                actuator_noise_std=0.1,
                energy_consumption_model="cubic_velocity_drag",
                environmental_factors=["currents", "pressure", "visibility", "temperature"]
            ),
            PhysicalDomain.SPACE: DomainPhysics(
                domain=PhysicalDomain.SPACE,
                gravity=0.0,  # Microgravity
                medium_density=0.0,  # Vacuum
                drag_coefficient=0.0,
                max_velocity=100.0,  # m/s for spacecraft
                max_acceleration=0.5,  # Limited by fuel
                communication_range=10000.0,
                communication_bandwidth=1e7,
                sensor_noise_std=0.05,
                actuator_noise_std=0.02,
                energy_consumption_model="impulse_based",
                environmental_factors=["radiation", "debris", "eclipse", "thermal"]
            )
        }
    
    def _initialize_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize scenario templates for each domain"""
        return {
            'aerial_search_rescue': {
                'domain': PhysicalDomain.AERIAL,
                'objectives': ['search_area', 'locate_victims', 'coordinate_rescue', 'avoid_obstacles'],
                'base_agents': 10,
                'base_duration': 2000,
                'difficulty_range': (0.3, 0.9)
            },
            'ground_exploration': {
                'domain': PhysicalDomain.GROUND,
                'objectives': ['map_terrain', 'collect_samples', 'avoid_hazards', 'return_to_base'],
                'base_agents': 8,
                'base_duration': 1500,
                'difficulty_range': (0.4, 0.85)
            },
            'underwater_inspection': {
                'domain': PhysicalDomain.UNDERWATER,
                'objectives': ['inspect_structure', 'detect_anomalies', 'maintain_formation', 'surface_periodically'],
                'base_agents': 6,
                'base_duration': 1800,
                'difficulty_range': (0.5, 0.95)
            },
            'space_formation': {
                'domain': PhysicalDomain.SPACE,
                'objectives': ['maintain_orbit', 'form_constellation', 'avoid_debris', 'minimize_fuel'],
                'base_agents': 12,
                'base_duration': 3000,
                'difficulty_range': (0.4, 0.9)
            }
        }
    
    def create_q1_evaluation_suite(self) -> Dict[str, List[ScenarioConfig]]:
        """
        Create comprehensive evaluation suite meeting Q1 standards
        """
        evaluation_suite = {}
        
        # Generate scenarios for each domain
        for domain in PhysicalDomain:
            if domain == PhysicalDomain.HYBRID:
                continue  # Handle separately
            
            domain_scenarios = []
            
            # Easy scenarios (baseline)
            domain_scenarios.extend(self._generate_domain_scenarios(
                domain, difficulty=0.3, num_scenarios=2
            ))
            
            # Medium scenarios (standard evaluation)
            domain_scenarios.extend(self._generate_domain_scenarios(
                domain, difficulty=0.6, num_scenarios=3
            ))
            
            # Hard scenarios (stress testing)
            domain_scenarios.extend(self._generate_domain_scenarios(
                domain, difficulty=0.9, num_scenarios=2
            ))
            
            # Adversarial scenarios
            domain_scenarios.extend(self._generate_adversarial_scenarios(
                domain, num_scenarios=2
            ))
            
            # Long-duration scenarios
            domain_scenarios.extend(self._generate_long_duration_scenarios(
                domain, num_scenarios=1
            ))
            
            evaluation_suite[domain.value] = domain_scenarios
        
        # Add cross-domain scenarios
        evaluation_suite['cross_domain'] = self._generate_cross_domain_scenarios()
        
        return evaluation_suite
    
    def _generate_domain_scenarios(self,
                                 domain: PhysicalDomain,
                                 difficulty: float,
                                 num_scenarios: int) -> List[ScenarioConfig]:
        """Generate scenarios for a specific domain and difficulty"""
        scenarios = []
        physics = self.physics_configs[domain]
        
        for i in range(num_scenarios):
            # Vary agent count
            base_agents = 10
            num_agents = int(base_agents * (1 + (difficulty - 0.5) * 0.5))
            
            # Set noise and failure rates based on difficulty
            noise_level = 0.05 + difficulty * 0.15  # 5-20% noise
            failure_rate = difficulty * 0.3  # Up to 30% failure
            
            # Communication constraints
            comm_constraints = {
                'bandwidth_reduction': difficulty * 0.5,
                'latency_ms': 10 + difficulty * 90,  # 10-100ms
                'packet_loss_rate': difficulty * 0.1,  # Up to 10%
                'range_reduction': difficulty * 0.3
            }
            
            config = ScenarioConfig(
                name=f"{domain.value}_d{difficulty:.1f}_s{i}",
                domain=domain,
                num_agents=num_agents,
                mission_duration=int(1000 * (1 + difficulty)),
                objectives=self._get_domain_objectives(domain),
                physics=physics,
                difficulty_level=difficulty,
                noise_level=noise_level,
                failure_rate=failure_rate,
                communication_constraints=comm_constraints,
                adversarial_elements=False,
                heterogeneous_agents=difficulty > 0.5,
                dynamic_environment=difficulty > 0.3
            )
            
            scenarios.append(config)
        
        return scenarios
    
    def _generate_adversarial_scenarios(self,
                                      domain: PhysicalDomain,
                                      num_scenarios: int) -> List[ScenarioConfig]:
        """Generate adversarial scenarios with antagonistic elements"""
        scenarios = []
        physics = self.physics_configs[domain]
        
        for i in range(num_scenarios):
            config = ScenarioConfig(
                name=f"{domain.value}_adversarial_{i}",
                domain=domain,
                num_agents=15,
                mission_duration=2000,
                objectives=self._get_domain_objectives(domain) + ['evade_adversaries', 'protect_assets'],
                physics=physics,
                difficulty_level=0.95,
                noise_level=0.15,
                failure_rate=0.2,
                communication_constraints={
                    'jamming_probability': 0.3,
                    'false_message_rate': 0.1,
                    'byzantine_agents': 2,
                    'cyber_attacks': True
                },
                adversarial_elements=True,
                heterogeneous_agents=True,
                dynamic_environment=True
            )
            
            scenarios.append(config)
        
        return scenarios
    
    def _generate_long_duration_scenarios(self,
                                        domain: PhysicalDomain,
                                        num_scenarios: int) -> List[ScenarioConfig]:
        """Generate long-duration scenarios for endurance testing"""
        scenarios = []
        physics = self.physics_configs[domain]
        
        for i in range(num_scenarios):
            config = ScenarioConfig(
                name=f"{domain.value}_long_duration_{i}",
                domain=domain,
                num_agents=20,
                mission_duration=5000,  # 5x normal duration
                objectives=self._get_domain_objectives(domain) + ['energy_management', 'fault_recovery'],
                physics=physics,
                difficulty_level=0.8,
                noise_level=0.1,
                failure_rate=0.15,
                communication_constraints={
                    'bandwidth_reduction': 0.3,
                    'intermittent_connectivity': True,
                    'connection_dropout_rate': 0.05
                },
                adversarial_elements=False,
                heterogeneous_agents=True,
                dynamic_environment=True
            )
            
            scenarios.append(config)
        
        return scenarios
    
    def _generate_cross_domain_scenarios(self) -> List[ScenarioConfig]:
        """Generate scenarios that span multiple domains"""
        scenarios = []
        
        # Air-Ground coordination
        air_ground_physics = DomainPhysics(
            domain=PhysicalDomain.HYBRID,
            gravity=9.81,
            medium_density=1.0,
            drag_coefficient=0.4,
            max_velocity=20.0,
            max_acceleration=7.5,
            communication_range=750.0,
            communication_bandwidth=7.5e5,
            sensor_noise_std=0.12,
            actuator_noise_std=0.06,
            energy_consumption_model="mixed",
            environmental_factors=["wind", "terrain", "obstacles", "weather"]
        )
        
        scenarios.append(ScenarioConfig(
            name="air_ground_coordination",
            domain=PhysicalDomain.HYBRID,
            num_agents=16,  # 8 aerial + 8 ground
            mission_duration=2500,
            objectives=['coordinate_surveillance', 'track_targets', 'relay_communication', 'provide_support'],
            physics=air_ground_physics,
            difficulty_level=0.85,
            noise_level=0.12,
            failure_rate=0.15,
            communication_constraints={
                'cross_domain_latency': 50,
                'protocol_translation_overhead': 0.2
            },
            adversarial_elements=False,
            heterogeneous_agents=True,
            dynamic_environment=True
        ))
        
        return scenarios
    
    def _get_domain_objectives(self, domain: PhysicalDomain) -> List[str]:
        """Get standard objectives for a domain"""
        objectives_map = {
            PhysicalDomain.AERIAL: ['area_coverage', 'target_tracking', 'formation_flight', 'collision_avoidance'],
            PhysicalDomain.GROUND: ['path_planning', 'obstacle_avoidance', 'area_mapping', 'resource_collection'],
            PhysicalDomain.UNDERWATER: ['current_compensation', 'depth_control', 'sonar_mapping', 'communication_relay'],
            PhysicalDomain.SPACE: ['orbital_maintenance', 'attitude_control', 'debris_avoidance', 'power_management']
        }
        return objectives_map.get(domain, [])
    
    def generate_scenario_environment(self, config: ScenarioConfig) -> 'BaseEnvironment':
        """Generate actual environment from configuration"""
        # This would create the actual simulation environment
        # Simplified for demonstration
        
        class DomainEnvironment:
            def __init__(self, config):
                self.config = config
                self.num_agents = config.num_agents
                self.physics = config.physics
                self.timestep = 0
                self.max_timesteps = config.mission_duration
                
                # Initialize agent states
                self.agent_states = self._initialize_agents()
                
                # Initialize environment state
                self.environment_state = self._initialize_environment()
                
                # Failure tracking
                self.failed_agents = set()
                self.communication_graph = np.ones((self.num_agents, self.num_agents))
            
            def _initialize_agents(self):
                """Initialize agent states with domain-specific positions"""
                states = []
                for i in range(self.num_agents):
                    if self.config.heterogeneous_agents:
                        # Vary capabilities
                        capability_multiplier = 0.8 + 0.4 * np.random.rand()
                    else:
                        capability_multiplier = 1.0
                    
                    state = {
                        'position': np.random.randn(3) * 100,
                        'velocity': np.zeros(3),
                        'energy': 100.0,
                        'capability': capability_multiplier,
                        'sensors_active': True,
                        'communication_active': True
                    }
                    states.append(state)
                return states
            
            def _initialize_environment(self):
                """Initialize environment state"""
                return {
                    'wind_field': np.random.randn(3) * 5 if self.config.domain == PhysicalDomain.AERIAL else np.zeros(3),
                    'current_field': np.random.randn(3) * 2 if self.config.domain == PhysicalDomain.UNDERWATER else np.zeros(3),
                    'obstacles': self._generate_obstacles(),
                    'targets': self._generate_targets(),
                    'environmental_hazards': self._generate_hazards()
                }
            
            def _generate_obstacles(self):
                """Generate domain-appropriate obstacles"""
                num_obstacles = int(10 * self.config.difficulty_level)
                obstacles = []
                for _ in range(num_obstacles):
                    obstacles.append({
                        'position': np.random.randn(3) * 200,
                        'radius': 5 + np.random.rand() * 20
                    })
                return obstacles
            
            def _generate_targets(self):
                """Generate mission targets"""
                num_targets = len(self.config.objectives) * 3
                targets = []
                for _ in range(num_targets):
                    targets.append({
                        'position': np.random.randn(3) * 150,
                        'type': np.random.choice(self.config.objectives),
                        'completed': False
                    })
                return targets
            
            def _generate_hazards(self):
                """Generate environmental hazards"""
                if self.config.dynamic_environment:
                    return [{
                        'type': 'dynamic_obstacle',
                        'trajectory': lambda t: np.array([50 * np.cos(t/100), 50 * np.sin(t/100), 0])
                    }]
                return []
            
            def reset(self):
                """Reset environment"""
                self.timestep = 0
                self.agent_states = self._initialize_agents()
                self.environment_state = self._initialize_environment()
                self.failed_agents = set()
                return self._get_observations()
            
            def step(self, actions):
                """Execute environment step with physics"""
                self.timestep += 1
                
                # Apply physics
                self._apply_physics(actions)
                
                # Apply failures
                self._apply_failures()
                
                # Update environment
                if self.config.dynamic_environment:
                    self._update_environment()
                
                # Calculate rewards
                rewards = self._calculate_rewards()
                
                # Check termination
                done = self.timestep >= self.max_timesteps or self._mission_complete()
                
                # Get observations
                observations = self._get_observations()
                
                info = {
                    'timestep': self.timestep,
                    'active_agents': self.num_agents - len(self.failed_agents),
                    'objectives_completed': sum(1 for t in self.environment_state['targets'] if t['completed']),
                    'physics_violations': self._check_physics_violations()
                }
                
                return observations, rewards, done, info
            
            def _apply_physics(self, actions):
                """Apply domain-specific physics"""
                for i, (state, action) in enumerate(zip(self.agent_states, actions)):
                    if i in self.failed_agents:
                        continue
                    
                    # Add noise
                    action = action + np.random.randn(*action.shape) * self.config.noise_level
                    
                    # Apply physics constraints
                    physics = self.config.physics
                    
                    # Update velocity with acceleration limits
                    acceleration = action * physics.max_acceleration
                    state['velocity'] += acceleration * 0.1  # dt = 0.1
                    
                    # Apply drag
                    drag_force = -physics.drag_coefficient * physics.medium_density * state['velocity']**2 * np.sign(state['velocity'])
                    state['velocity'] += drag_force * 0.1
                    
                    # Velocity constraints
                    speed = np.linalg.norm(state['velocity'])
                    if speed > physics.max_velocity:
                        state['velocity'] *= physics.max_velocity / speed
                    
                    # Update position
                    state['position'] += state['velocity'] * 0.1
                    
                    # Energy consumption
                    energy_used = self._calculate_energy_consumption(state, action)
                    state['energy'] -= energy_used
                    
                    # Check energy depletion
                    if state['energy'] <= 0:
                        self.failed_agents.add(i)
            
            def _calculate_energy_consumption(self, state, action):
                """Calculate domain-specific energy consumption"""
                model = self.config.physics.energy_consumption_model
                
                if model == "quadratic_velocity":
                    return 0.01 * np.linalg.norm(state['velocity'])**2
                elif model == "linear_velocity_friction":
                    return 0.02 * np.linalg.norm(state['velocity']) + 0.005
                elif model == "cubic_velocity_drag":
                    return 0.001 * np.linalg.norm(state['velocity'])**3
                elif model == "impulse_based":
                    return 0.1 * np.linalg.norm(action)
                else:
                    return 0.01
            
            def _apply_failures(self):
                """Apply random failures based on failure rate"""
                for i in range(self.num_agents):
                    if i not in self.failed_agents and np.random.rand() < self.config.failure_rate / 1000:
                        self.failed_agents.add(i)
                        
                # Communication failures
                if self.config.communication_constraints:
                    for i in range(self.num_agents):
                        for j in range(self.num_agents):
                            if np.random.rand() < self.config.communication_constraints.get('packet_loss_rate', 0):
                                self.communication_graph[i, j] = 0
            
            def _update_environment(self):
                """Update dynamic environment elements"""
                # Update wind/currents
                if self.config.domain == PhysicalDomain.AERIAL:
                    self.environment_state['wind_field'] += np.random.randn(3) * 0.5
                elif self.config.domain == PhysicalDomain.UNDERWATER:
                    self.environment_state['current_field'] += np.random.randn(3) * 0.2
                
                # Update dynamic obstacles
                for hazard in self.environment_state['environmental_hazards']:
                    if hazard['type'] == 'dynamic_obstacle':
                        # Update based on trajectory function
                        pass
            
            def _calculate_rewards(self):
                """Calculate multi-objective rewards"""
                rewards = np.zeros(self.num_agents)
                
                for i, state in enumerate(self.agent_states):
                    if i in self.failed_agents:
                        rewards[i] = -10  # Failure penalty
                        continue
                    
                    # Distance to nearest target
                    min_dist = float('inf')
                    for target in self.environment_state['targets']:
                        if not target['completed']:
                            dist = np.linalg.norm(state['position'] - target['position'])
                            min_dist = min(min_dist, dist)
                    
                    # Basic reward
                    rewards[i] = -0.01 * min_dist
                    
                    # Energy efficiency bonus
                    rewards[i] += 0.001 * state['energy']
                    
                    # Collision penalty
                    for obstacle in self.environment_state['obstacles']:
                        dist = np.linalg.norm(state['position'] - obstacle['position'])
                        if dist < obstacle['radius']:
                            rewards[i] -= 5.0
                
                return rewards
            
            def _get_observations(self):
                """Get observations with noise and communication constraints"""
                observations = []
                
                for i, state in enumerate(self.agent_states):
                    if i in self.failed_agents:
                        observations.append(np.zeros(10))  # Dead agent
                        continue
                    
                    # Local observation
                    obs = np.concatenate([
                        state['position'],
                        state['velocity'],
                        [state['energy']],
                        self.environment_state['wind_field'] if self.config.domain == PhysicalDomain.AERIAL else np.zeros(3)
                    ])
                    
                    # Add sensor noise
                    obs += np.random.randn(len(obs)) * self.config.noise_level
                    
                    observations.append(obs)
                
                return np.array(observations)
            
            def _check_physics_violations(self):
                """Check for physics constraint violations"""
                violations = 0
                
                for i, state in enumerate(self.agent_states):
                    if i in self.failed_agents:
                        continue
                    
                    # Velocity violations
                    if np.linalg.norm(state['velocity']) > self.config.physics.max_velocity * 1.1:
                        violations += 1
                    
                    # Energy violations
                    if state['energy'] < 0:
                        violations += 1
                
                return violations
            
            def _mission_complete(self):
                """Check if mission objectives are complete"""
                completed_objectives = sum(1 for t in self.environment_state['targets'] if t['completed'])
                return completed_objectives >= len(self.environment_state['targets']) * 0.8
            
            def set_worst_case_scenario(self, enabled: bool):
                """Enable worst-case scenario for real-time testing"""
                if enabled:
                    self.config.noise_level = 0.2
                    self.config.failure_rate = 0.3
                    # Add more adversarial elements
        
        return DomainEnvironment(config)
    
    def validate_scenario_difficulty(self, config: ScenarioConfig) -> Dict[str, float]:
        """Validate and score scenario difficulty"""
        scores = {}
        
        # Physics complexity
        physics_score = 0.0
        if config.physics.drag_coefficient > 0:
            physics_score += 0.2
        if config.physics.communication_bandwidth < 1e6:
            physics_score += 0.2
        if len(config.physics.environmental_factors) > 3:
            physics_score += 0.2
        scores['physics_complexity'] = physics_score
        
        # Coordination complexity
        coord_score = config.num_agents / 50  # Normalize to 50 agents
        if config.heterogeneous_agents:
            coord_score += 0.2
        scores['coordination_complexity'] = min(coord_score, 1.0)
        
        # Environmental complexity
        env_score = config.noise_level + config.failure_rate
        if config.dynamic_environment:
            env_score += 0.3
        if config.adversarial_elements:
            env_score += 0.4
        scores['environmental_complexity'] = min(env_score, 1.0)
        
        # Communication complexity
        comm_score = 0.0
        constraints = config.communication_constraints
        if constraints.get('bandwidth_reduction', 0) > 0.3:
            comm_score += 0.3
        if constraints.get('latency_ms', 0) > 50:
            comm_score += 0.3
        if constraints.get('packet_loss_rate', 0) > 0.05:
            comm_score += 0.2
        scores['communication_complexity'] = comm_score
        
        # Overall difficulty
        scores['overall_difficulty'] = np.mean(list(scores.values()))
        scores['q1_suitable'] = scores['overall_difficulty'] > 0.5
        
        return scores
    
    def export_scenarios(self, output_path: str = 'scenarios/q1_evaluation_suite.json'):
        """Export scenario configurations for reproducibility"""
        suite = self.create_q1_evaluation_suite()
        
        # Convert to serializable format
        export_data = {}
        for domain, scenarios in suite.items():
            export_data[domain] = []
            for scenario in scenarios:
                scenario_dict = {
                    'name': scenario.name,
                    'domain': scenario.domain.value,
                    'num_agents': scenario.num_agents,
                    'mission_duration': scenario.mission_duration,
                    'objectives': scenario.objectives,
                    'difficulty_level': scenario.difficulty_level,
                    'noise_level': scenario.noise_level,
                    'failure_rate': scenario.failure_rate,
                    'communication_constraints': scenario.communication_constraints,
                    'adversarial_elements': scenario.adversarial_elements,
                    'heterogeneous_agents': scenario.heterogeneous_agents,
                    'dynamic_environment': scenario.dynamic_environment,
                    'physics': {
                        'gravity': scenario.physics.gravity,
                        'max_velocity': scenario.physics.max_velocity,
                        'max_acceleration': scenario.physics.max_acceleration,
                        'communication_range': scenario.physics.communication_range
                    }
                }
                
                # Add difficulty validation
                validation = self.validate_scenario_difficulty(scenario)
                scenario_dict['difficulty_validation'] = validation
                
                export_data[domain].append(scenario_dict)
        
        # Save to file
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Summary statistics
        total_scenarios = sum(len(scenarios) for scenarios in suite.values())
        print(f"Exported {total_scenarios} scenarios across {len(suite)} domains")
        print(f"Scenarios saved to: {output_path}")
        
        return export_data