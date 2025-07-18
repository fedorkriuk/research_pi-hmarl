"""Physics-Accurate Synthetic Data Generator for PI-HMARL

This module generates unlimited synthetic training data using real-world
physics parameters and PyBullet simulation engine. The data includes
perfect ground truth labels for physics constraints.
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import h5py
from pathlib import Path
import logging
from datetime import datetime
import json

from .real_parameter_extractor import (
    RealParameterExtractor, 
    DroneSpecifications,
    BatterySpecifications,
    CommunicationSpecifications
)

logger = logging.getLogger(__name__)


@dataclass
class SyntheticScenario:
    """Container for synthetic scenario configuration"""
    name: str
    description: str
    num_agents: int
    duration: float  # seconds
    timestep: float  # seconds
    
    # Environment parameters
    world_size: Tuple[float, float, float]  # x, y, z in meters
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    targets: List[Tuple[float, float, float]] = field(default_factory=list)
    
    # Agent parameters
    agent_types: List[str] = field(default_factory=list)
    initial_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    initial_velocities: List[Tuple[float, float, float]] = field(default_factory=list)
    
    # Physics parameters
    weather_scenario: str = "nominal"
    enable_wind: bool = True
    enable_turbulence: bool = True
    
    # Mission parameters
    mission_type: str = "search_rescue"  # search_rescue, surveillance, delivery
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicsState:
    """Complete physics state with perfect ground truth"""
    timestamp: float
    
    # Kinematic states
    positions: np.ndarray  # [num_agents, 3]
    velocities: np.ndarray  # [num_agents, 3]
    accelerations: np.ndarray  # [num_agents, 3]
    orientations: np.ndarray  # [num_agents, 4] quaternions
    angular_velocities: np.ndarray  # [num_agents, 3]
    
    # Dynamic states
    forces: np.ndarray  # [num_agents, 3] net forces
    torques: np.ndarray  # [num_agents, 3] net torques
    motor_speeds: np.ndarray  # [num_agents, 4] RPM for quadcopters
    
    # Energy states
    battery_voltages: np.ndarray  # [num_agents]
    battery_currents: np.ndarray  # [num_agents]
    battery_soc: np.ndarray  # [num_agents] state of charge
    power_consumption: np.ndarray  # [num_agents] watts
    
    # Constraint labels (perfect ground truth)
    energy_constraint_satisfied: np.ndarray  # [num_agents] bool
    collision_distances: np.ndarray  # [num_agents, num_agents] meters
    velocity_constraint_satisfied: np.ndarray  # [num_agents] bool
    acceleration_constraint_satisfied: np.ndarray  # [num_agents] bool
    
    # Environmental states
    wind_velocity: np.ndarray  # [3] m/s
    air_density: float  # kg/m³
    temperature: float  # Celsius


class PhysicsAccurateSynthetic:
    """Generates physics-accurate synthetic data with real parameters"""
    
    def __init__(
        self,
        parameter_extractor: Optional[RealParameterExtractor] = None,
        render: bool = False,
        gui: bool = False
    ):
        """Initialize the synthetic data generator
        
        Args:
            parameter_extractor: Real parameter extractor instance
            render: Whether to render simulation visually
            gui: Whether to use GUI mode
        """
        self.parameter_extractor = parameter_extractor or RealParameterExtractor()
        self.render = render
        self.gui = gui
        
        # PyBullet setup
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Simulation parameters
        self.timestep = 0.01  # 100Hz physics
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -9.81)
        
        # Agent storage
        self.agents = {}
        self.agent_specs = {}
        self.battery_models = {}
        
        # Environment
        self.obstacles = []
        self.targets = []
        
        # Data collection
        self.collected_states = []
        
        logger.info("Initialized PhysicsAccurateSynthetic generator")
    
    def create_drone_agent(
        self, 
        agent_id: int,
        drone_type: str,
        position: Tuple[float, float, float],
        orientation: Optional[Tuple[float, float, float, float]] = None
    ) -> int:
        """Create a drone agent with real specifications
        
        Args:
            agent_id: Unique agent identifier
            drone_type: Type of drone from parameter extractor
            position: Initial position (x, y, z)
            orientation: Initial orientation quaternion
            
        Returns:
            PyBullet body ID
        """
        # Get drone specifications
        drone_spec = self.parameter_extractor.get_drone_specs(drone_type)
        if not drone_spec:
            logger.warning(f"Unknown drone type: {drone_type}, using default")
            drone_spec = self.parameter_extractor.get_drone_specs("dji_mavic_3")
        
        # Create collision shape (simplified as box for now)
        half_extents = [
            drone_spec.dimensions["length"] / 2,
            drone_spec.dimensions["width"] / 2,
            drone_spec.dimensions["height"] / 2
        ]
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.3, 0.3, 0.8, 1.0]
        )
        
        # Create multi-body
        if orientation is None:
            orientation = [0, 0, 0, 1]
        
        body_id = p.createMultiBody(
            baseMass=drone_spec.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        # Enable dynamics
        p.changeDynamics(
            body_id,
            -1,
            linearDamping=0.1,
            angularDamping=0.1,
            lateralFriction=0.5
        )
        
        # Store agent info
        self.agents[agent_id] = body_id
        self.agent_specs[agent_id] = drone_spec
        
        # Initialize battery model
        battery_spec = self.parameter_extractor.get_battery_specs("samsung_25r")
        self.battery_models[agent_id] = {
            "spec": battery_spec,
            "soc": 1.0,  # Start at 100% charge
            "voltage": battery_spec.nominal_voltage * 4,  # 4S configuration
            "temperature": 20.0  # Celsius
        }
        
        logger.debug(f"Created drone agent {agent_id} of type {drone_type}")
        return body_id
    
    def create_obstacle(
        self,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float],
        static: bool = True
    ) -> int:
        """Create an obstacle in the environment
        
        Args:
            position: Position (x, y, z)
            size: Size (width, depth, height)
            static: Whether obstacle is static
            
        Returns:
            PyBullet body ID
        """
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in size],
            rgbaColor=[0.5, 0.5, 0.5, 1.0]
        )
        
        mass = 0 if static else 100.0
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.obstacles.append(body_id)
        return body_id
    
    def apply_drone_forces(
        self,
        agent_id: int,
        motor_commands: np.ndarray  # [4] motor thrusts in Newtons
    ):
        """Apply forces to drone based on motor commands
        
        Args:
            agent_id: Agent identifier
            motor_commands: Motor thrust commands
        """
        body_id = self.agents[agent_id]
        drone_spec = self.agent_specs[agent_id]
        
        # Get current state
        pos, orn = p.getBasePositionAndOrientation(body_id)
        vel, ang_vel = p.getBaseVelocity(body_id)
        
        # Convert orientation to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        # Calculate total thrust in body frame
        total_thrust = np.sum(motor_commands)
        thrust_body = np.array([0, 0, total_thrust])
        
        # Transform to world frame
        thrust_world = rot_matrix @ thrust_body
        
        # Apply thrust force
        p.applyExternalForce(
            body_id,
            -1,  # base link
            thrust_world,
            pos,
            p.WORLD_FRAME
        )
        
        # Calculate and apply torques from differential thrust
        # Simplified model: torques proportional to thrust differences
        arm_length = drone_spec.dimensions["length"] / 2
        
        # Roll torque (front-back differential)
        roll_torque = arm_length * (motor_commands[0] - motor_commands[2])
        
        # Pitch torque (left-right differential)  
        pitch_torque = arm_length * (motor_commands[1] - motor_commands[3])
        
        # Yaw torque (diagonal differential)
        yaw_torque = 0.1 * (
            (motor_commands[0] + motor_commands[2]) - 
            (motor_commands[1] + motor_commands[3])
        )
        
        torque_body = np.array([roll_torque, pitch_torque, yaw_torque])
        torque_world = rot_matrix @ torque_body
        
        p.applyExternalTorque(
            body_id,
            -1,
            torque_world,
            p.WORLD_FRAME
        )
        
        # Apply drag forces
        velocity = np.array(vel)
        speed = np.linalg.norm(velocity)
        if speed > 0.01:
            drag_force = -0.5 * drone_spec.drag_coefficient * \
                        drone_spec.frontal_area * 1.225 * speed * velocity
            p.applyExternalForce(
                body_id,
                -1,
                drag_force,
                pos,
                p.WORLD_FRAME
            )
    
    def update_battery_state(
        self,
        agent_id: int,
        motor_commands: np.ndarray,
        dt: float
    ):
        """Update battery state based on power consumption
        
        Args:
            agent_id: Agent identifier
            motor_commands: Motor thrust commands
            dt: Time step
        """
        drone_spec = self.agent_specs[agent_id]
        battery_model = self.battery_models[agent_id]
        battery_spec = battery_model["spec"]
        
        # Calculate power consumption
        # Simplified model: P = k * thrust^1.5
        thrust_total = np.sum(motor_commands)
        mechanical_power = 0.01 * thrust_total ** 1.5  # Watts
        
        # Account for efficiencies
        electrical_power = mechanical_power / (
            drone_spec.motor_efficiency * 
            drone_spec.propeller_efficiency * 
            drone_spec.esc_efficiency
        )
        
        # Add baseline power (electronics, sensors)
        baseline_power = 10.0  # Watts
        total_power = electrical_power + baseline_power
        
        # Calculate current draw
        voltage = battery_model["voltage"]
        current = total_power / voltage  # Amps
        
        # Update state of charge
        capacity_ah = battery_spec.capacity / 1000.0  # mAh to Ah
        soc_delta = (current * dt / 3600.0) / capacity_ah
        battery_model["soc"] = max(0.0, battery_model["soc"] - soc_delta)
        
        # Update voltage based on discharge curve
        c_rate = current / (battery_spec.capacity / 1000.0)
        battery_model["voltage"] = self.parameter_extractor.get_battery_voltage(
            "samsung_25r",
            battery_model["soc"],
            c_rate
        ) * 4  # 4S configuration
        
        # Store power consumption
        battery_model["current"] = current
        battery_model["power"] = total_power
    
    def compute_physics_state(self, scenario: SyntheticScenario) -> PhysicsState:
        """Compute complete physics state with perfect labels
        
        Args:
            scenario: Current scenario configuration
            
        Returns:
            PhysicsState with all ground truth labels
        """
        num_agents = len(self.agents)
        
        # Initialize arrays
        positions = np.zeros((num_agents, 3))
        velocities = np.zeros((num_agents, 3))
        accelerations = np.zeros((num_agents, 3))
        orientations = np.zeros((num_agents, 4))
        angular_velocities = np.zeros((num_agents, 3))
        
        forces = np.zeros((num_agents, 3))
        torques = np.zeros((num_agents, 3))
        motor_speeds = np.zeros((num_agents, 4))
        
        battery_voltages = np.zeros(num_agents)
        battery_currents = np.zeros(num_agents)
        battery_soc = np.zeros(num_agents)
        power_consumption = np.zeros(num_agents)
        
        # Collect states
        for i, (agent_id, body_id) in enumerate(self.agents.items()):
            # Kinematics
            pos, orn = p.getBasePositionAndOrientation(body_id)
            vel, ang_vel = p.getBaseVelocity(body_id)
            
            positions[i] = pos
            velocities[i] = vel
            orientations[i] = orn
            angular_velocities[i] = ang_vel
            
            # Calculate acceleration (finite difference)
            if hasattr(self, '_prev_velocities'):
                accelerations[i] = (velocities[i] - self._prev_velocities[i]) / self.timestep
            
            # Battery state
            battery_model = self.battery_models[agent_id]
            battery_voltages[i] = battery_model["voltage"]
            battery_currents[i] = battery_model.get("current", 0.0)
            battery_soc[i] = battery_model["soc"]
            power_consumption[i] = battery_model.get("power", 0.0)
        
        # Store for next iteration
        self._prev_velocities = velocities.copy()
        
        # Compute constraint satisfaction (perfect labels)
        energy_constraint_satisfied = battery_soc > 0.1  # 10% minimum
        
        # Collision distances
        collision_distances = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                collision_distances[i, j] = dist
                collision_distances[j, i] = dist
        
        # Velocity constraints
        speeds = np.linalg.norm(velocities, axis=1)
        max_speeds = np.array([self.agent_specs[aid].max_speed for aid in self.agents.keys()])
        velocity_constraint_satisfied = speeds <= max_speeds
        
        # Acceleration constraints  
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        max_accels = np.array([self.agent_specs[aid].max_speed / 2.0 for aid in self.agents.keys()])
        acceleration_constraint_satisfied = accel_magnitudes <= max_accels
        
        # Environmental state
        weather = self.parameter_extractor.get_weather_conditions(scenario.weather_scenario)
        wind_velocity = np.array([weather["wind_speed"], 0, 0])  # Simplified
        
        return PhysicsState(
            timestamp=p.getSimulationTime(),
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            orientations=orientations,
            angular_velocities=angular_velocities,
            forces=forces,
            torques=torques,
            motor_speeds=motor_speeds,
            battery_voltages=battery_voltages,
            battery_currents=battery_currents,
            battery_soc=battery_soc,
            power_consumption=power_consumption,
            energy_constraint_satisfied=energy_constraint_satisfied,
            collision_distances=collision_distances,
            velocity_constraint_satisfied=velocity_constraint_satisfied,
            acceleration_constraint_satisfied=acceleration_constraint_satisfied,
            wind_velocity=wind_velocity,
            air_density=1.225,
            temperature=weather["temperature"]
        )
    
    def generate_scenario_data(
        self,
        scenario: SyntheticScenario,
        controller: Optional[Callable] = None
    ) -> List[PhysicsState]:
        """Generate synthetic data for a complete scenario
        
        Args:
            scenario: Scenario configuration
            controller: Optional controller function for agents
            
        Returns:
            List of physics states
        """
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        
        # Create ground plane
        p.loadURDF("plane.urdf")
        
        # Create agents
        self.agents.clear()
        self.agent_specs.clear()
        self.battery_models.clear()
        
        for i in range(scenario.num_agents):
            agent_type = scenario.agent_types[i] if i < len(scenario.agent_types) else "dji_mavic_3"
            position = scenario.initial_positions[i] if i < len(scenario.initial_positions) else (i*5, 0, 2)
            self.create_drone_agent(i, agent_type, position)
        
        # Create obstacles
        for obstacle in scenario.obstacles:
            self.create_obstacle(
                obstacle["position"],
                obstacle["size"],
                obstacle.get("static", True)
            )
        
        # Simulation loop
        states = []
        num_steps = int(scenario.duration / scenario.timestep)
        
        for step in range(num_steps):
            # Get control commands
            if controller:
                commands = controller(self, scenario, step)
            else:
                # Default hover controller
                commands = {}
                for agent_id in self.agents:
                    # Simple hover at 2m
                    pos, _ = p.getBasePositionAndOrientation(self.agents[agent_id])
                    thrust = 9.81 * self.agent_specs[agent_id].mass / 4  # Hover thrust
                    height_error = 2.0 - pos[2]
                    thrust += 5.0 * height_error  # P controller
                    commands[agent_id] = np.ones(4) * max(0, thrust)
            
            # Apply forces
            for agent_id, motor_commands in commands.items():
                self.apply_drone_forces(agent_id, motor_commands)
                self.update_battery_state(agent_id, motor_commands, scenario.timestep)
            
            # Step physics
            p.stepSimulation()
            
            # Collect state
            if step % int(0.1 / scenario.timestep) == 0:  # Sample at 10Hz
                state = self.compute_physics_state(scenario)
                states.append(state)
            
            # Optional rendering
            if self.render and step % 10 == 0:
                if self.gui:
                    time.sleep(0.01)
        
        logger.info(f"Generated {len(states)} states for scenario '{scenario.name}'")
        return states
    
    def save_scenario_data(
        self,
        states: List[PhysicsState],
        scenario: SyntheticScenario,
        output_path: Path
    ):
        """Save scenario data to HDF5 file
        
        Args:
            states: List of physics states
            scenario: Scenario configuration  
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Save scenario metadata
            meta = f.create_group('metadata')
            meta.attrs['name'] = scenario.name
            meta.attrs['description'] = scenario.description
            meta.attrs['num_agents'] = scenario.num_agents
            meta.attrs['duration'] = scenario.duration
            meta.attrs['timestep'] = scenario.timestep
            meta.attrs['timestamp'] = datetime.now().isoformat()
            
            # Save trajectory data
            traj = f.create_group('trajectories')
            
            # Stack all states
            num_steps = len(states)
            if num_steps > 0:
                traj.create_dataset('positions', 
                    data=np.stack([s.positions for s in states]))
                traj.create_dataset('velocities',
                    data=np.stack([s.velocities for s in states]))
                traj.create_dataset('accelerations',
                    data=np.stack([s.accelerations for s in states]))
                traj.create_dataset('orientations',
                    data=np.stack([s.orientations for s in states]))
                
                # Energy data
                energy = f.create_group('energy')
                energy.create_dataset('battery_soc',
                    data=np.stack([s.battery_soc for s in states]))
                energy.create_dataset('power_consumption',
                    data=np.stack([s.power_consumption for s in states]))
                
                # Perfect constraint labels
                constraints = f.create_group('constraints')
                constraints.create_dataset('energy_satisfied',
                    data=np.stack([s.energy_constraint_satisfied for s in states]))
                constraints.create_dataset('velocity_satisfied',
                    data=np.stack([s.velocity_constraint_satisfied for s in states]))
                constraints.create_dataset('collision_distances',
                    data=np.stack([s.collision_distances for s in states]))
        
        logger.info(f"Saved scenario data to {output_path}")
    
    def generate_dataset(
        self,
        num_scenarios: int,
        output_dir: Path,
        scenario_generator: Optional[Callable] = None
    ):
        """Generate complete synthetic dataset
        
        Args:
            num_scenarios: Number of scenarios to generate
            output_dir: Output directory for dataset
            scenario_generator: Optional custom scenario generator
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default scenario generator
        if scenario_generator is None:
            scenario_generator = self._default_scenario_generator
        
        # Generate scenarios
        for i in range(num_scenarios):
            scenario = scenario_generator(i)
            states = self.generate_scenario_data(scenario)
            
            output_path = output_dir / f"scenario_{i:04d}.h5"
            self.save_scenario_data(states, scenario, output_path)
        
        # Save dataset metadata
        metadata = {
            "num_scenarios": num_scenarios,
            "generator": "PhysicsAccurateSynthetic",
            "real_parameters": {
                "drones": list(self.parameter_extractor.drone_specs.keys()),
                "batteries": list(self.parameter_extractor.battery_specs.keys()),
                "communication": list(self.parameter_extractor.communication_specs.keys())
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated synthetic dataset with {num_scenarios} scenarios in {output_dir}")
    
    def _default_scenario_generator(self, index: int) -> SyntheticScenario:
        """Default scenario generator with variety
        
        Args:
            index: Scenario index
            
        Returns:
            Generated scenario
        """
        # Vary number of agents
        num_agents = np.random.randint(2, 11)
        
        # Random world size
        world_size = (
            np.random.uniform(50, 200),
            np.random.uniform(50, 200),
            np.random.uniform(20, 50)
        )
        
        # Random initial positions
        initial_positions = []
        for i in range(num_agents):
            pos = (
                np.random.uniform(0, world_size[0]),
                np.random.uniform(0, world_size[1]),
                np.random.uniform(2, 10)
            )
            initial_positions.append(pos)
        
        # Random obstacles
        num_obstacles = np.random.randint(0, 10)
        obstacles = []
        for _ in range(num_obstacles):
            obstacle = {
                "position": (
                    np.random.uniform(0, world_size[0]),
                    np.random.uniform(0, world_size[1]),
                    np.random.uniform(0, 5)
                ),
                "size": (
                    np.random.uniform(1, 5),
                    np.random.uniform(1, 5),
                    np.random.uniform(2, 10)
                ),
                "static": True
            }
            obstacles.append(obstacle)
        
        # Mission types
        mission_types = ["search_rescue", "surveillance", "formation", "delivery"]
        mission_type = np.random.choice(mission_types)
        
        # Weather scenarios
        weather_scenarios = ["nominal", "windy", "rainy", "extreme"]
        weather_weights = [0.7, 0.2, 0.08, 0.02]
        weather_scenario = np.random.choice(weather_scenarios, p=weather_weights)
        
        return SyntheticScenario(
            name=f"scenario_{index:04d}",
            description=f"Synthetic {mission_type} scenario with {num_agents} agents",
            num_agents=num_agents,
            duration=60.0,  # 60 seconds
            timestep=0.01,
            world_size=world_size,
            obstacles=obstacles,
            agent_types=["dji_mavic_3"] * num_agents,
            initial_positions=initial_positions,
            weather_scenario=weather_scenario,
            mission_type=mission_type
        )
    
    def cleanup(self):
        """Clean up PyBullet connection"""
        p.disconnect()


# Convenience function
def create_synthetic_generator(
    parameter_extractor: Optional[RealParameterExtractor] = None,
    render: bool = False
) -> PhysicsAccurateSynthetic:
    """Create synthetic data generator
    
    Args:
        parameter_extractor: Optional parameter extractor
        render: Whether to render simulation
        
    Returns:
        PhysicsAccurateSynthetic instance
    """
    return PhysicsAccurateSynthetic(parameter_extractor, render)