"""Cooperative Manipulation Scenario

This module implements scenarios where multiple agents cooperate to
manipulate and transport objects.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class GraspType(Enum):
    """Types of grasps"""
    FORCE_CLOSURE = "force_closure"
    FORM_CLOSURE = "form_closure"
    FRICTION = "friction"
    MAGNETIC = "magnetic"


class ObjectShape(Enum):
    """Object shapes"""
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    IRREGULAR = "irregular"


@dataclass
class ManipulationObject:
    """Object to be manipulated"""
    object_id: str
    shape: ObjectShape
    dimensions: np.ndarray  # [length, width, height] or [radius, height]
    mass: float
    position: np.ndarray
    orientation: np.ndarray  # Quaternion [w, x, y, z]
    velocity: np.ndarray
    angular_velocity: np.ndarray
    friction_coefficient: float = 0.5
    is_grasped: bool = False
    grasping_agents: List[str] = None
    
    def __post_init__(self):
        if self.grasping_agents is None:
            self.grasping_agents = []
    
    def get_inertia_tensor(self) -> np.ndarray:
        """Calculate inertia tensor based on shape
        
        Returns:
            3x3 inertia tensor
        """
        if self.shape == ObjectShape.BOX:
            l, w, h = self.dimensions
            Ixx = self.mass * (w**2 + h**2) / 12
            Iyy = self.mass * (l**2 + h**2) / 12
            Izz = self.mass * (l**2 + w**2) / 12
            return np.diag([Ixx, Iyy, Izz])
        
        elif self.shape == ObjectShape.CYLINDER:
            r, h = self.dimensions[:2]
            Ixx = self.mass * (3 * r**2 + h**2) / 12
            Iyy = Ixx
            Izz = self.mass * r**2 / 2
            return np.diag([Ixx, Iyy, Izz])
        
        elif self.shape == ObjectShape.SPHERE:
            r = self.dimensions[0]
            I = 2 * self.mass * r**2 / 5
            return np.eye(3) * I
        
        else:  # Irregular
            # Approximate as box
            return self.mass * np.eye(3) * 0.1
    
    def apply_force(self, force: np.ndarray, contact_point: np.ndarray, dt: float):
        """Apply force to object
        
        Args:
            force: Force vector
            contact_point: Point where force is applied (relative to COM)
            dt: Time step
        """
        # Linear dynamics
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Angular dynamics
        torque = np.cross(contact_point, force)
        
        # Convert to body frame
        R = Rotation.from_quat(self.orientation[[1, 2, 3, 0]])  # scipy uses x,y,z,w
        torque_body = R.inv().apply(torque)
        
        # Angular acceleration
        I = self.get_inertia_tensor()
        angular_accel = np.linalg.inv(I) @ torque_body
        
        # Update angular velocity
        self.angular_velocity += R.apply(angular_accel) * dt
        
        # Update orientation
        omega_quat = np.array([0, *self.angular_velocity])
        q_dot = 0.5 * self._quaternion_multiply(self.orientation, omega_quat)
        self.orientation += q_dot * dt
        self.orientation /= np.linalg.norm(self.orientation)  # Normalize
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions
        
        Args:
            q1, q2: Quaternions [w, x, y, z]
            
        Returns:
            Product quaternion
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


class GraspPlanning:
    """Plans grasps for objects"""
    
    def __init__(self):
        """Initialize grasp planner"""
        self.min_grasp_force = 5.0  # Newtons
        self.max_grasp_force = 50.0
        self.safety_factor = 1.5
    
    def plan_grasp_points(
        self,
        obj: ManipulationObject,
        num_agents: int
    ) -> List[Dict[str, Any]]:
        """Plan grasp points for object
        
        Args:
            obj: Object to grasp
            num_agents: Number of agents available
            
        Returns:
            List of grasp configurations
        """
        grasp_points = []
        
        if obj.shape == ObjectShape.BOX:
            grasp_points = self._plan_box_grasp(obj, num_agents)
        elif obj.shape == ObjectShape.CYLINDER:
            grasp_points = self._plan_cylinder_grasp(obj, num_agents)
        elif obj.shape == ObjectShape.SPHERE:
            grasp_points = self._plan_sphere_grasp(obj, num_agents)
        else:
            grasp_points = self._plan_irregular_grasp(obj, num_agents)
        
        return grasp_points
    
    def _plan_box_grasp(
        self,
        obj: ManipulationObject,
        num_agents: int
    ) -> List[Dict[str, Any]]:
        """Plan grasp for box-shaped object
        
        Args:
            obj: Box object
            num_agents: Number of agents
            
        Returns:
            Grasp configurations
        """
        l, w, h = obj.dimensions
        grasps = []
        
        # Get object rotation
        R = Rotation.from_quat(obj.orientation[[1, 2, 3, 0]])
        
        if num_agents == 2:
            # Opposite sides grasp
            points = [
                np.array([l/2, 0, 0]),
                np.array([-l/2, 0, 0])
            ]
            normals = [
                np.array([-1, 0, 0]),
                np.array([1, 0, 0])
            ]
        elif num_agents == 4:
            # Four sides grasp
            points = [
                np.array([l/2, 0, 0]),
                np.array([-l/2, 0, 0]),
                np.array([0, w/2, 0]),
                np.array([0, -w/2, 0])
            ]
            normals = [
                np.array([-1, 0, 0]),
                np.array([1, 0, 0]),
                np.array([0, -1, 0]),
                np.array([0, 1, 0])
            ]
        else:
            # Distribute around perimeter
            angles = np.linspace(0, 2*np.pi, num_agents, endpoint=False)
            points = []
            normals = []
            
            for angle in angles:
                # Point on box perimeter
                x = l/2 * np.cos(angle)
                y = w/2 * np.sin(angle)
                
                # Snap to box surface
                if abs(x) > abs(y) * l/w:
                    x = l/2 * np.sign(x)
                    y = y * (l/2) / abs(x) if x != 0 else 0
                else:
                    y = w/2 * np.sign(y)
                    x = x * (w/2) / abs(y) if y != 0 else 0
                
                points.append(np.array([x, y, 0]))
                normals.append(-np.array([x, y, 0]) / np.linalg.norm([x, y, 0.001]))
        
        # Transform to world frame
        for i, (point, normal) in enumerate(zip(points, normals)):
            world_point = obj.position + R.apply(point)
            world_normal = R.apply(normal)
            
            grasps.append({
                'position': world_point,
                'normal': world_normal,
                'force_magnitude': self._calculate_required_force(obj, num_agents),
                'grasp_type': GraspType.FORCE_CLOSURE
            })
        
        return grasps
    
    def _plan_cylinder_grasp(
        self,
        obj: ManipulationObject,
        num_agents: int
    ) -> List[Dict[str, Any]]:
        """Plan grasp for cylindrical object"""
        r, h = obj.dimensions[:2]
        grasps = []
        
        # Distribute agents around circumference
        angles = np.linspace(0, 2*np.pi, num_agents, endpoint=False)
        
        R = Rotation.from_quat(obj.orientation[[1, 2, 3, 0]])
        
        for angle in angles:
            # Point on cylinder surface
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = 0  # Middle height
            
            point = np.array([x, y, z])
            normal = -np.array([x, y, 0]) / r  # Inward normal
            
            # Transform to world
            world_point = obj.position + R.apply(point)
            world_normal = R.apply(normal)
            
            grasps.append({
                'position': world_point,
                'normal': world_normal,
                'force_magnitude': self._calculate_required_force(obj, num_agents),
                'grasp_type': GraspType.FORCE_CLOSURE
            })
        
        return grasps
    
    def _plan_sphere_grasp(
        self,
        obj: ManipulationObject,
        num_agents: int
    ) -> List[Dict[str, Any]]:
        """Plan grasp for spherical object"""
        r = obj.dimensions[0]
        grasps = []
        
        # Distribute points on sphere using fibonacci spiral
        indices = np.arange(0, num_agents, dtype=float) + 0.5
        theta = np.arccos(1 - 2 * indices / num_agents)  # Polar angle
        phi = np.pi * (1 + 5**0.5) * indices  # Azimuthal angle
        
        for i in range(num_agents):
            # Point on sphere
            x = r * np.sin(theta[i]) * np.cos(phi[i])
            y = r * np.sin(theta[i]) * np.sin(phi[i])
            z = r * np.cos(theta[i])
            
            point = np.array([x, y, z])
            normal = -point / r  # Inward normal
            
            world_point = obj.position + point
            
            grasps.append({
                'position': world_point,
                'normal': normal,
                'force_magnitude': self._calculate_required_force(obj, num_agents),
                'grasp_type': GraspType.FORCE_CLOSURE
            })
        
        return grasps
    
    def _plan_irregular_grasp(
        self,
        obj: ManipulationObject,
        num_agents: int
    ) -> List[Dict[str, Any]]:
        """Plan grasp for irregular object (simplified)"""
        # Treat as sphere for simplicity
        return self._plan_sphere_grasp(obj, num_agents)
    
    def _calculate_required_force(
        self,
        obj: ManipulationObject,
        num_agents: int
    ) -> float:
        """Calculate required grasp force per agent
        
        Args:
            obj: Object to grasp
            num_agents: Number of agents
            
        Returns:
            Required force magnitude
        """
        # Force to overcome gravity
        weight = obj.mass * 9.81
        
        # Distribute among agents with safety factor
        force_per_agent = (weight * self.safety_factor) / num_agents
        
        # Consider friction
        if obj.friction_coefficient > 0:
            # With friction, less normal force needed
            force_per_agent /= obj.friction_coefficient
        
        return np.clip(force_per_agent, self.min_grasp_force, self.max_grasp_force)


class ManipulationAgent:
    """Agent capable of manipulation"""
    
    def __init__(
        self,
        agent_id: str,
        position: np.ndarray,
        max_force: float = 30.0,
        grasp_range: float = 1.0,
        move_speed: float = 2.0
    ):
        """Initialize manipulation agent
        
        Args:
            agent_id: Unique identifier
            position: Initial position
            max_force: Maximum applicable force
            grasp_range: Grasping range
            move_speed: Movement speed
        """
        self.agent_id = agent_id
        self.position = position.copy()
        self.velocity = np.zeros(3)
        self.max_force = max_force
        self.grasp_range = grasp_range
        self.move_speed = move_speed
        
        # Manipulation state
        self.is_grasping = False
        self.grasped_object = None
        self.grasp_point = None
        self.grasp_normal = None
        self.applied_force = np.zeros(3)
        
        # Control gains
        self.position_gain = 5.0
        self.force_gain = 0.8
        
        logger.info(f"Initialized ManipulationAgent {agent_id}")
    
    def plan_approach(
        self,
        grasp_config: Dict[str, Any]
    ) -> np.ndarray:
        """Plan approach to grasp point
        
        Args:
            grasp_config: Grasp configuration
            
        Returns:
            Target position for approach
        """
        grasp_pos = grasp_config['position']
        grasp_normal = grasp_config['normal']
        
        # Approach from normal direction
        approach_distance = self.grasp_range * 0.8
        approach_pos = grasp_pos - grasp_normal * approach_distance
        
        return approach_pos
    
    def attempt_grasp(
        self,
        obj: ManipulationObject,
        grasp_config: Dict[str, Any]
    ) -> bool:
        """Attempt to grasp object
        
        Args:
            obj: Object to grasp
            grasp_config: Grasp configuration
            
        Returns:
            Success status
        """
        grasp_pos = grasp_config['position']
        distance = np.linalg.norm(self.position - grasp_pos)
        
        if distance <= self.grasp_range:
            self.is_grasping = True
            self.grasped_object = obj.object_id
            self.grasp_point = grasp_pos - obj.position  # Relative to object
            self.grasp_normal = grasp_config['normal']
            
            # Add to object's grasping agents
            if self.agent_id not in obj.grasping_agents:
                obj.grasping_agents.append(self.agent_id)
                obj.is_grasped = len(obj.grasping_agents) > 0
            
            logger.info(f"Agent {self.agent_id} grasped object {obj.object_id}")
            return True
        
        return False
    
    def release_grasp(self, obj: ManipulationObject):
        """Release grasp on object
        
        Args:
            obj: Object to release
        """
        if self.is_grasping and self.grasped_object == obj.object_id:
            self.is_grasping = False
            self.grasped_object = None
            self.grasp_point = None
            self.grasp_normal = None
            self.applied_force = np.zeros(3)
            
            # Remove from object's grasping agents
            if self.agent_id in obj.grasping_agents:
                obj.grasping_agents.remove(self.agent_id)
                obj.is_grasped = len(obj.grasping_agents) > 0
            
            logger.info(f"Agent {self.agent_id} released object {obj.object_id}")
    
    def compute_manipulation_force(
        self,
        obj: ManipulationObject,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        other_agents: List['ManipulationAgent']
    ) -> np.ndarray:
        """Compute force to apply for manipulation
        
        Args:
            obj: Grasped object
            target_position: Target object position
            target_velocity: Target object velocity
            other_agents: Other manipulating agents
            
        Returns:
            Force vector to apply
        """
        if not self.is_grasping:
            return np.zeros(3)
        
        # Position error
        position_error = target_position - obj.position
        
        # Velocity error
        velocity_error = target_velocity - obj.velocity
        
        # Desired acceleration (PD control)
        desired_accel = (
            self.position_gain * position_error +
            2.0 * velocity_error
        )
        
        # Required force (distributed among agents)
        num_grasping = len(obj.grasping_agents)
        if num_grasping > 0:
            required_force = obj.mass * desired_accel / num_grasping
        else:
            required_force = np.zeros(3)
        
        # Add normal force component for secure grasp
        normal_force = self.grasp_normal * 20.0  # Inward force
        
        # Total force
        total_force = required_force + normal_force
        
        # Limit force magnitude
        force_mag = np.linalg.norm(total_force)
        if force_mag > self.max_force:
            total_force = total_force * self.max_force / force_mag
        
        return total_force
    
    def update_position(self, dt: float):
        """Update agent position
        
        Args:
            dt: Time step
        """
        self.position += self.velocity * dt
    
    def move_with_object(self, obj: ManipulationObject):
        """Update position to maintain grasp
        
        Args:
            obj: Grasped object
        """
        if self.is_grasping and self.grasp_point is not None:
            # Get object rotation
            R = Rotation.from_quat(obj.orientation[[1, 2, 3, 0]])
            
            # Update grasp position
            world_grasp_point = obj.position + R.apply(self.grasp_point)
            
            # Move to maintain grasp
            error = world_grasp_point - self.position
            self.velocity = error / 0.1  # Quick adjustment
            
            # Update grasp normal in world frame
            self.grasp_normal = R.apply(self.grasp_normal)


class ForceCoordination:
    """Coordinates forces among multiple agents"""
    
    def __init__(self):
        """Initialize force coordination"""
        self.force_distribution_method = "equal"  # or "optimal"
        self.coordination_gain = 0.5
    
    def coordinate_forces(
        self,
        agents: List[ManipulationAgent],
        obj: ManipulationObject,
        target_wrench: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Coordinate forces among agents
        
        Args:
            agents: List of manipulating agents
            obj: Object being manipulated
            target_wrench: Target force and torque
            
        Returns:
            Force commands for each agent
        """
        force_commands = {}
        
        grasping_agents = [a for a in agents if a.is_grasping]
        
        if not grasping_agents:
            return force_commands
        
        target_force = target_wrench.get('force', np.zeros(3))
        target_torque = target_wrench.get('torque', np.zeros(3))
        
        if self.force_distribution_method == "equal":
            # Simple equal distribution
            force_per_agent = target_force / len(grasping_agents)
            
            for agent in grasping_agents:
                force_commands[agent.agent_id] = force_per_agent
        
        else:  # optimal
            # Optimize force distribution to minimize internal forces
            forces = self._optimize_force_distribution(
                grasping_agents,
                obj,
                target_force,
                target_torque
            )
            
            for agent, force in zip(grasping_agents, forces):
                force_commands[agent.agent_id] = force
        
        return force_commands
    
    def _optimize_force_distribution(
        self,
        agents: List[ManipulationAgent],
        obj: ManipulationObject,
        target_force: np.ndarray,
        target_torque: np.ndarray
    ) -> List[np.ndarray]:
        """Optimize force distribution (simplified)
        
        Args:
            agents: Grasping agents
            obj: Object
            target_force: Target net force
            target_torque: Target net torque
            
        Returns:
            List of forces for each agent
        """
        # Simplified: distribute based on grasp positions
        forces = []
        
        # Get grasp points relative to COM
        grasp_points = []
        for agent in agents:
            if agent.grasp_point is not None:
                grasp_points.append(agent.grasp_point)
            else:
                grasp_points.append(np.zeros(3))
        
        # Equal force distribution with torque consideration
        base_force = target_force / len(agents)
        
        for i, (agent, grasp_point) in enumerate(zip(agents, grasp_points)):
            # Add contribution for torque
            if np.linalg.norm(grasp_point) > 0:
                # Simplified: add tangential force for torque
                r_cross = np.cross(grasp_point, target_torque)
                torque_force = r_cross / (np.linalg.norm(grasp_point)**2 + 0.1)
                torque_force = torque_force / len(agents)
            else:
                torque_force = np.zeros(3)
            
            total_force = base_force + torque_force
            
            # Limit to agent's capability
            force_mag = np.linalg.norm(total_force)
            if force_mag > agent.max_force:
                total_force = total_force * agent.max_force / force_mag
            
            forces.append(total_force)
        
        return forces


class ObjectTransport:
    """Manages object transport tasks"""
    
    def __init__(self):
        """Initialize object transport"""
        self.transport_tasks = []
        self.path_planner = None  # Would integrate path planning
        
    def create_transport_task(
        self,
        obj: ManipulationObject,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        waypoints: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Create transport task
        
        Args:
            obj: Object to transport
            start_pos: Starting position
            goal_pos: Goal position
            waypoints: Optional waypoints
            
        Returns:
            Transport task
        """
        if waypoints is None:
            # Simple straight line path
            waypoints = [start_pos, goal_pos]
        
        task = {
            'task_id': f"transport_{obj.object_id}_{len(self.transport_tasks)}",
            'object_id': obj.object_id,
            'start_position': start_pos,
            'goal_position': goal_pos,
            'waypoints': waypoints,
            'current_waypoint': 0,
            'status': 'active',
            'start_time': 0.0,
            'completion_time': None
        }
        
        self.transport_tasks.append(task)
        return task
    
    def get_transport_target(
        self,
        task: Dict[str, Any],
        current_pos: np.ndarray,
        lookahead: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get current transport target
        
        Args:
            task: Transport task
            current_pos: Current object position
            lookahead: Lookahead distance
            
        Returns:
            Target position and velocity
        """
        waypoints = task['waypoints']
        current_wp_idx = task['current_waypoint']
        
        if current_wp_idx >= len(waypoints) - 1:
            # At final waypoint
            target_pos = waypoints[-1]
            target_vel = np.zeros(3)
            task['status'] = 'completed'
        else:
            # Check if reached current waypoint
            wp = waypoints[current_wp_idx]
            distance = np.linalg.norm(current_pos - wp)
            
            if distance < 2.0:  # Waypoint reached threshold
                task['current_waypoint'] += 1
                current_wp_idx = task['current_waypoint']
            
            if current_wp_idx < len(waypoints) - 1:
                # Interpolate between waypoints
                wp1 = waypoints[current_wp_idx]
                wp2 = waypoints[current_wp_idx + 1]
                
                direction = wp2 - wp1
                dist = np.linalg.norm(direction)
                
                if dist > 0:
                    direction = direction / dist
                    
                    # Target position with lookahead
                    progress = np.dot(current_pos - wp1, direction)
                    target_pos = wp1 + direction * (progress + lookahead)
                    
                    # Clamp to waypoint
                    if np.linalg.norm(target_pos - wp1) > dist:
                        target_pos = wp2
                    
                    # Target velocity
                    target_vel = direction * 1.0  # 1 m/s transport speed
                else:
                    target_pos = wp2
                    target_vel = np.zeros(3)
            else:
                target_pos = waypoints[-1]
                target_vel = np.zeros(3)
        
        return target_pos, target_vel


class CooperativeManipulationScenario:
    """Main cooperative manipulation scenario"""
    
    def __init__(
        self,
        num_agents: int = 4,
        environment_size: Tuple[float, float, float] = (50.0, 50.0, 20.0)
    ):
        """Initialize cooperative manipulation scenario
        
        Args:
            num_agents: Number of agents
            environment_size: Size of environment
        """
        self.num_agents = num_agents
        self.environment_size = environment_size
        
        # Initialize components
        self.agents = self._initialize_agents()
        self.objects = self._initialize_objects()
        self.grasp_planner = GraspPlanning()
        self.force_coordinator = ForceCoordination()
        self.transport_manager = ObjectTransport()
        
        # Scenario state
        self.time = 0.0
        self.active_task = None
        self.performance_metrics = {
            'successful_grasps': 0,
            'dropped_objects': 0,
            'tasks_completed': 0,
            'total_transport_time': 0.0
        }
        
        logger.info(
            f"Initialized CooperativeManipulationScenario with {num_agents} agents"
        )
    
    def _initialize_agents(self) -> List[ManipulationAgent]:
        """Initialize manipulation agents
        
        Returns:
            List of agents
        """
        agents = []
        
        # Arrange agents in a circle
        center = np.array([self.environment_size[0] / 2, self.environment_size[1] / 2, 5.0])
        radius = 10.0
        
        for i in range(self.num_agents):
            angle = 2 * np.pi * i / self.num_agents
            position = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            
            agent = ManipulationAgent(
                agent_id=f"agent_{i}",
                position=position,
                max_force=30.0,
                grasp_range=1.5,
                move_speed=2.0
            )
            
            agents.append(agent)
        
        return agents
    
    def _initialize_objects(self) -> List[ManipulationObject]:
        """Initialize objects to manipulate
        
        Returns:
            List of objects
        """
        objects = []
        
        # Create different objects
        # Box
        box = ManipulationObject(
            object_id="box_1",
            shape=ObjectShape.BOX,
            dimensions=np.array([2.0, 1.5, 1.0]),
            mass=10.0,
            position=np.array([25.0, 25.0, 5.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            friction_coefficient=0.6
        )
        objects.append(box)
        
        # Cylinder
        cylinder = ManipulationObject(
            object_id="cylinder_1",
            shape=ObjectShape.CYLINDER,
            dimensions=np.array([0.8, 2.0]),
            mass=8.0,
            position=np.array([15.0, 35.0, 5.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            friction_coefficient=0.5
        )
        objects.append(cylinder)
        
        # Sphere
        sphere = ManipulationObject(
            object_id="sphere_1",
            shape=ObjectShape.SPHERE,
            dimensions=np.array([1.0]),
            mass=5.0,
            position=np.array([35.0, 15.0, 5.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            friction_coefficient=0.4
        )
        objects.append(sphere)
        
        return objects
    
    def create_manipulation_task(self):
        """Create a manipulation task"""
        # Select an object
        obj = self.objects[0]  # Use box for example
        
        # Create transport task
        start_pos = obj.position.copy()
        goal_pos = np.array([40.0, 40.0, 5.0])
        
        # Create waypoints for obstacle avoidance (simplified)
        waypoints = [
            start_pos,
            np.array([30.0, 30.0, 8.0]),  # Lift
            np.array([35.0, 35.0, 8.0]),  # Move
            goal_pos  # Place
        ]
        
        task = self.transport_manager.create_transport_task(
            obj,
            start_pos,
            goal_pos,
            waypoints
        )
        
        self.active_task = task
        logger.info(f"Created manipulation task: {task['task_id']}")
        
        return task
    
    def step(self, dt: float = 0.1):
        """Run one simulation step
        
        Args:
            dt: Time step
        """
        self.time += dt
        
        # If no active task, create one
        if self.active_task is None or self.active_task['status'] == 'completed':
            self.create_manipulation_task()
        
        # Get active object
        obj_id = self.active_task['object_id']
        obj = next(o for o in self.objects if o.object_id == obj_id)
        
        # Phase 1: Approach and grasp
        if not obj.is_grasped:
            # Plan grasps
            grasp_configs = self.grasp_planner.plan_grasp_points(obj, self.num_agents)
            
            # Assign agents to grasp points
            for i, (agent, grasp_config) in enumerate(zip(self.agents, grasp_configs)):
                if not agent.is_grasping:
                    # Approach grasp point
                    approach_pos = agent.plan_approach(grasp_config)
                    
                    # Move towards approach position
                    direction = approach_pos - agent.position
                    distance = np.linalg.norm(direction)
                    
                    if distance > 0.1:
                        agent.velocity = direction / distance * agent.move_speed
                        agent.update_position(dt)
                    else:
                        # Try to grasp
                        if agent.attempt_grasp(obj, grasp_config):
                            self.performance_metrics['successful_grasps'] += 1
        
        # Phase 2: Coordinated manipulation
        else:
            # Get transport target
            target_pos, target_vel = self.transport_manager.get_transport_target(
                self.active_task,
                obj.position,
                lookahead=3.0
            )
            
            # Compute target wrench
            position_error = target_pos - obj.position
            velocity_error = target_vel - obj.velocity
            
            target_force = obj.mass * (5.0 * position_error + 2.0 * velocity_error)
            target_torque = np.zeros(3)  # No rotation for now
            
            target_wrench = {
                'force': target_force,
                'torque': target_torque
            }
            
            # Coordinate forces
            force_commands = self.force_coordinator.coordinate_forces(
                self.agents,
                obj,
                target_wrench
            )
            
            # Apply forces
            total_force = np.zeros(3)
            
            for agent in self.agents:
                if agent.is_grasping:
                    # Get force command
                    if agent.agent_id in force_commands:
                        force = force_commands[agent.agent_id]
                    else:
                        # Compute individual force
                        force = agent.compute_manipulation_force(
                            obj,
                            target_pos,
                            target_vel,
                            self.agents
                        )
                    
                    agent.applied_force = force
                    
                    # Apply to object
                    contact_point = agent.grasp_point
                    if contact_point is not None:
                        obj.apply_force(force, contact_point, dt)
                        total_force += force
                    
                    # Update agent position to maintain grasp
                    agent.move_with_object(obj)
                    agent.update_position(dt)
            
            # Add gravity
            gravity_force = np.array([0, 0, -obj.mass * 9.81])
            obj.apply_force(gravity_force, np.zeros(3), dt)
            
            # Check if object dropped
            if obj.position[2] < 1.0:  # Hit ground
                logger.warning(f"Object {obj.object_id} dropped!")
                self.performance_metrics['dropped_objects'] += 1
                
                # Reset object
                obj.position[2] = 5.0
                obj.velocity = np.zeros(3)
                
                # Release grasps
                for agent in self.agents:
                    agent.release_grasp(obj)
        
        # Check task completion
        if self.active_task['status'] == 'completed':
            self.performance_metrics['tasks_completed'] += 1
            self.performance_metrics['total_transport_time'] += self.time
            
            logger.info(f"Task {self.active_task['task_id']} completed!")
            
            # Release object
            for agent in self.agents:
                agent.release_grasp(obj)
    
    def get_state(self) -> Dict[str, Any]:
        """Get scenario state
        
        Returns:
            State dictionary
        """
        return {
            'time': self.time,
            'active_task': self.active_task['task_id'] if self.active_task else None,
            'objects': {
                obj.object_id: {
                    'position': obj.position.tolist(),
                    'velocity': obj.velocity.tolist(),
                    'is_grasped': obj.is_grasped,
                    'grasping_agents': obj.grasping_agents
                }
                for obj in self.objects
            },
            'agents': {
                agent.agent_id: {
                    'position': agent.position.tolist(),
                    'is_grasping': agent.is_grasping,
                    'grasped_object': agent.grasped_object,
                    'applied_force': agent.applied_force.tolist()
                }
                for agent in self.agents
            },
            'performance': self.performance_metrics
        }
    
    def run_experiment(self, duration: float = 60.0):
        """Run manipulation experiment
        
        Args:
            duration: Experiment duration
        """
        dt = 0.1
        steps = int(duration / dt)
        
        for step in range(steps):
            self.step(dt)
            
            # Log periodically
            if step % 100 == 0:
                state = self.get_state()
                logger.info(
                    f"Time: {state['time']:.1f}s, "
                    f"Tasks completed: {state['performance']['tasks_completed']}"
                )
        
        # Final summary
        logger.info(f"Experiment completed. Performance: {self.performance_metrics}")


# Example usage
def run_cooperative_manipulation():
    """Run cooperative manipulation scenario"""
    scenario = CooperativeManipulationScenario(
        num_agents=4,
        environment_size=(50.0, 50.0, 20.0)
    )
    
    scenario.run_experiment(duration=120.0)