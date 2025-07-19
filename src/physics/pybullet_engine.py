"""PyBullet Physics Engine Implementation

This module implements the physics engine interface using PyBullet
for realistic physics simulation with real-world parameters.
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple, Any, Optional
import logging

from .base_physics import PhysicsEngine

logger = logging.getLogger(__name__)


class PyBulletEngine(PhysicsEngine):
    """PyBullet-based physics engine implementation"""
    
    def __init__(
        self,
        timestep: float = 0.01,
        gravity: np.ndarray = np.array([0, 0, -9.81]),
        enable_collision: bool = True,
        enable_aerodynamics: bool = True,
        enable_wind: bool = True,
        use_gui: bool = False
    ):
        """Initialize PyBullet engine
        
        Args:
            timestep: Physics simulation timestep
            gravity: Gravity vector
            enable_collision: Enable collision detection
            enable_aerodynamics: Enable aerodynamic forces
            enable_wind: Enable wind effects
            use_gui: Use GUI for visualization
        """
        super().__init__(timestep, gravity, enable_collision, enable_aerodynamics, enable_wind)
        
        self.use_gui = use_gui
        self.client_id = None
        self.plane_id = None
        
        # Object tracking
        self.bullet_ids: Dict[int, int] = {}  # object_id -> bullet_id
        self.object_ids: Dict[int, int] = {}  # bullet_id -> object_id
        
        # Constraint tracking
        self.bullet_constraints: Dict[int, int] = {}
        
        # Forces to apply
        self.pending_forces: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.pending_torques: Dict[int, np.ndarray] = {}
        
        self._initialize_bullet()
    
    def _initialize_bullet(self):
        """Initialize PyBullet simulation"""
        # Connect to PyBullet
        if self.use_gui:
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.client_id = p.connect(p.DIRECT)
        
        # Set data path for URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure simulation
        p.setTimeStep(self.timestep)
        p.setGravity(*self.gravity)
        
        # Add ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set collision detection parameters
        if self.enable_collision:
            p.setDefaultContactERP(0.9)
            p.changeDynamics(self.plane_id, -1, 
                           lateralFriction=1.0,
                           restitution=0.5)
        
        logger.info(f"Initialized PyBullet engine (GUI: {self.use_gui})")
    
    def reset(self):
        """Reset physics engine"""
        # Remove all objects except ground plane
        for bullet_id in list(self.bullet_ids.values()):
            if bullet_id != self.plane_id:
                p.removeBody(bullet_id)
        
        # Clear tracking dictionaries
        self.bullet_ids.clear()
        self.object_ids.clear()
        self.bullet_constraints.clear()
        self.pending_forces.clear()
        self.pending_torques.clear()
        
        # Reset simulation
        p.resetSimulation()
        
        # Reinitialize
        self._initialize_bullet()
        
        logger.debug("Reset PyBullet engine")
    
    def add_object(
        self,
        object_id: int,
        position: np.ndarray,
        orientation: np.ndarray,
        mass: float,
        inertia: np.ndarray,
        shape: str = "sphere",
        size: List[float] = None
    ) -> bool:
        """Add a physics object"""
        try:
            # Create collision shape
            if shape == "sphere":
                radius = size[0] if size else 0.5
                collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                                 rgbaColor=[0.5, 0.5, 0.5, 1])
            elif shape == "box":
                half_extents = size if size else [0.5, 0.5, 0.5]
                collision_shape = p.createCollisionShape(p.GEOM_BOX, 
                                                       halfExtents=half_extents)
                visual_shape = p.createVisualShape(p.GEOM_BOX, 
                                                 halfExtents=half_extents,
                                                 rgbaColor=[0.5, 0.5, 0.5, 1])
            elif shape == "cylinder":
                radius = size[0] if size else 0.5
                height = size[1] if size else 1.0
                collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                                       radius=radius, height=height)
                visual_shape = p.createVisualShape(p.GEOM_CYLINDER,
                                                 radius=radius, length=height,
                                                 rgbaColor=[0.5, 0.5, 0.5, 1])
            else:
                # Default to sphere
                collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.5,
                                                 rgbaColor=[0.5, 0.5, 0.5, 1])
            
            # Create multi-body
            bullet_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                baseOrientation=orientation,
                baseInertialFramePosition=[0, 0, 0],
                baseInertialFrameOrientation=[0, 0, 0, 1]
            )
            
            # Set inertia
            p.changeDynamics(bullet_id, -1, 
                           localInertiaDiagonal=np.diag(inertia).tolist(),
                           linearDamping=0.01,
                           angularDamping=0.01)
            
            # Enable collision if configured
            if not self.enable_collision:
                p.setCollisionFilterGroupMask(bullet_id, -1, 0, 0)
            
            # Track object
            self.bullet_ids[object_id] = bullet_id
            self.object_ids[bullet_id] = object_id
            self.objects[object_id] = {
                "mass": mass,
                "inertia": inertia,
                "shape": shape,
                "size": size
            }
            
            logger.debug(f"Added object {object_id} to PyBullet (ID: {bullet_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add object {object_id}: {e}")
            return False
    
    def remove_object(self, object_id: int) -> bool:
        """Remove a physics object"""
        if object_id not in self.bullet_ids:
            return False
        
        try:
            bullet_id = self.bullet_ids[object_id]
            p.removeBody(bullet_id)
            
            # Clean up tracking
            del self.bullet_ids[object_id]
            del self.object_ids[bullet_id]
            del self.objects[object_id]
            
            # Clean up pending forces
            if object_id in self.pending_forces:
                del self.pending_forces[object_id]
            if object_id in self.pending_torques:
                del self.pending_torques[object_id]
            
            logger.debug(f"Removed object {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove object {object_id}: {e}")
            return False
    
    def set_object_state(
        self,
        object_id: int,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None
    ) -> bool:
        """Set object state"""
        if object_id not in self.bullet_ids:
            return False
        
        try:
            bullet_id = self.bullet_ids[object_id]
            
            # Get current state if not all provided
            if position is None or orientation is None:
                pos, orn = p.getBasePositionAndOrientation(bullet_id)
                if position is None:
                    position = pos
                if orientation is None:
                    orientation = orn
            
            # Reset position and orientation
            p.resetBasePositionAndOrientation(bullet_id, position, orientation)
            
            # Set velocities
            if linear_velocity is not None or angular_velocity is not None:
                if linear_velocity is None:
                    linear_velocity, _ = p.getBaseVelocity(bullet_id)
                if angular_velocity is None:
                    _, angular_velocity = p.getBaseVelocity(bullet_id)
                
                p.resetBaseVelocity(bullet_id, linear_velocity, angular_velocity)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set state for object {object_id}: {e}")
            return False
    
    def get_object_state(self, object_id: int) -> Dict[str, np.ndarray]:
        """Get object state"""
        if object_id not in self.bullet_ids:
            return {}
        
        bullet_id = self.bullet_ids[object_id]
        
        # Get position and orientation
        position, orientation = p.getBasePositionAndOrientation(bullet_id)
        
        # Get velocities
        linear_velocity, angular_velocity = p.getBaseVelocity(bullet_id)
        
        return {
            "position": np.array(position),
            "orientation": np.array(orientation),
            "linear_velocity": np.array(linear_velocity),
            "angular_velocity": np.array(angular_velocity)
        }
    
    def apply_force(
        self,
        object_id: int,
        force: np.ndarray,
        position: Optional[np.ndarray] = None
    ):
        """Apply force to object"""
        if object_id not in self.bullet_ids:
            return
        
        # Store force to apply in next step
        if object_id not in self.pending_forces:
            self.pending_forces[object_id] = []
        
        if position is None:
            position = np.zeros(3)  # Apply at COM
        
        self.pending_forces[object_id].append((force.copy(), position.copy()))
    
    def apply_torque(self, object_id: int, torque: np.ndarray):
        """Apply torque to object"""
        if object_id not in self.bullet_ids:
            return
        
        # Accumulate torques
        if object_id not in self.pending_torques:
            self.pending_torques[object_id] = np.zeros(3)
        
        self.pending_torques[object_id] += torque
    
    def step(self) -> Dict[int, Dict[str, Any]]:
        """Advance physics simulation"""
        # Apply pending forces and torques
        for object_id, bullet_id in self.bullet_ids.items():
            # Apply forces
            if object_id in self.pending_forces:
                for force, position in self.pending_forces[object_id]:
                    p.applyExternalForce(
                        bullet_id, -1, force, position, p.WORLD_FRAME
                    )
            
            # Apply torques
            if object_id in self.pending_torques:
                p.applyExternalTorque(
                    bullet_id, -1, self.pending_torques[object_id], p.WORLD_FRAME
                )
            
            # Apply aerodynamic forces if enabled
            if self.enable_aerodynamics and object_id in self.objects:
                state = self.get_object_state(object_id)
                obj_info = self.objects[object_id]
                
                # Simple drag calculation
                drag_coeff = 0.5  # Default drag coefficient
                ref_area = 0.1    # Default reference area
                
                drag_force, lift_force = self.calculate_aerodynamic_forces(
                    state["linear_velocity"],
                    state["orientation"],
                    drag_coeff,
                    ref_area
                )
                
                if np.linalg.norm(drag_force) > 0:
                    p.applyExternalForce(
                        bullet_id, -1, drag_force, [0, 0, 0], p.WORLD_FRAME
                    )
                
                if np.linalg.norm(lift_force) > 0:
                    p.applyExternalForce(
                        bullet_id, -1, lift_force, [0, 0, 0], p.WORLD_FRAME
                    )
        
        # Clear pending forces
        self.pending_forces.clear()
        self.pending_torques.clear()
        
        # Step simulation
        p.stepSimulation()
        
        # Collect updated states
        states = {}
        for object_id in self.bullet_ids:
            states[object_id] = self.get_object_state(object_id)
        
        return states
    
    def check_collisions(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Check for collisions"""
        if not self.enable_collision:
            return []
        
        collisions = []
        
        # Get all contact points
        contact_points = p.getContactPoints()
        
        for contact in contact_points:
            body_a = contact[1]
            body_b = contact[2]
            
            # Skip ground plane collisions for now
            if body_a == self.plane_id or body_b == self.plane_id:
                continue
            
            # Convert bullet IDs to object IDs
            if body_a in self.object_ids and body_b in self.object_ids:
                object_a = self.object_ids[body_a]
                object_b = self.object_ids[body_b]
                
                contact_info = {
                    "position": np.array(contact[5]),
                    "normal": np.array(contact[7]),
                    "distance": contact[8],
                    "force": contact[9]
                }
                
                collisions.append((object_a, object_b, contact_info))
        
        return collisions
    
    def add_constraint(
        self,
        constraint_id: int,
        object1_id: int,
        object2_id: int,
        constraint_type: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """Add constraint between objects"""
        try:
            # Get bullet IDs
            body1 = self.bullet_ids.get(object1_id, -1)
            body2 = self.bullet_ids.get(object2_id, -1) if object2_id != -1 else -1
            
            if body1 == -1:
                return False
            
            # Create constraint based on type
            if constraint_type == "fixed":
                joint_id = p.createConstraint(
                    body1, -1, body2, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0], [0, 0, 0], [0, 0, 0]
                )
            elif constraint_type == "point2point":
                pivot1 = parameters.get("pivot1", [0, 0, 0])
                pivot2 = parameters.get("pivot2", [0, 0, 0])
                joint_id = p.createConstraint(
                    body1, -1, body2, -1,
                    p.JOINT_POINT2POINT,
                    [0, 0, 0], pivot1, pivot2
                )
            else:
                logger.warning(f"Unknown constraint type: {constraint_type}")
                return False
            
            self.bullet_constraints[constraint_id] = joint_id
            self.constraints[constraint_id] = {
                "type": constraint_type,
                "object1": object1_id,
                "object2": object2_id,
                "parameters": parameters
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add constraint {constraint_id}: {e}")
            return False
    
    def remove_constraint(self, constraint_id: int) -> bool:
        """Remove constraint"""
        if constraint_id not in self.bullet_constraints:
            return False
        
        try:
            p.removeConstraint(self.bullet_constraints[constraint_id])
            del self.bullet_constraints[constraint_id]
            del self.constraints[constraint_id]
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove constraint {constraint_id}: {e}")
            return False
    
    def visualize(self, mode: str = "human") -> Optional[np.ndarray]:
        """Visualize physics simulation"""
        if mode == "rgb_array":
            # Get camera image
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=10,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width/height,
                nearVal=0.1,
                farVal=100
            )
            
            _, _, rgb_img, _, _ = p.getCameraImage(
                width, height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix
            )
            
            return np.array(rgb_img)[:, :, :3]
        
        return None
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'client_id') and self.client_id is not None:
            p.disconnect()