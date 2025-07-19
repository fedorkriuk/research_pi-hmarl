"""Base Physics Engine Interface

This module defines the abstract base class for physics engines,
ensuring consistent interface across different physics backends.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PhysicsEngine(ABC):
    """Abstract base class for physics engines"""
    
    def __init__(
        self,
        timestep: float = 0.01,
        gravity: np.ndarray = np.array([0, 0, -9.81]),
        enable_collision: bool = True,
        enable_aerodynamics: bool = True,
        enable_wind: bool = True
    ):
        """Initialize physics engine
        
        Args:
            timestep: Physics simulation timestep
            gravity: Gravity vector (m/s²)
            enable_collision: Enable collision detection
            enable_aerodynamics: Enable aerodynamic forces
            enable_wind: Enable wind effects
        """
        self.timestep = timestep
        self.gravity = gravity
        self.enable_collision = enable_collision
        self.enable_aerodynamics = enable_aerodynamics
        self.enable_wind = enable_wind
        
        # Physics objects registry
        self.objects: Dict[int, Any] = {}
        self.constraints: Dict[int, Any] = {}
        
        # Environmental conditions
        self.wind_velocity = np.zeros(3)
        self.air_density = 1.225  # kg/m³ at sea level
        self.temperature = 20.0   # Celsius
        
        logger.info(f"Initialized {self.__class__.__name__} with timestep {timestep}s")
    
    @abstractmethod
    def reset(self):
        """Reset physics engine to initial state"""
        pass
    
    @abstractmethod
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
        """Add a physics object
        
        Args:
            object_id: Unique object identifier
            position: Initial position [x, y, z]
            orientation: Initial orientation (quaternion)
            mass: Object mass (kg)
            inertia: Inertia tensor (3x3 matrix)
            shape: Collision shape type
            size: Shape dimensions
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def remove_object(self, object_id: int) -> bool:
        """Remove a physics object
        
        Args:
            object_id: Object identifier
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def set_object_state(
        self,
        object_id: int,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None
    ) -> bool:
        """Set object state
        
        Args:
            object_id: Object identifier
            position: Position [x, y, z]
            orientation: Orientation (quaternion)
            linear_velocity: Linear velocity [vx, vy, vz]
            angular_velocity: Angular velocity [wx, wy, wz]
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def get_object_state(self, object_id: int) -> Dict[str, np.ndarray]:
        """Get object state
        
        Args:
            object_id: Object identifier
            
        Returns:
            Dictionary with position, orientation, velocities
        """
        pass
    
    @abstractmethod
    def apply_force(
        self,
        object_id: int,
        force: np.ndarray,
        position: Optional[np.ndarray] = None
    ):
        """Apply force to object
        
        Args:
            object_id: Object identifier
            force: Force vector [fx, fy, fz] in Newtons
            position: Application point (relative to COM)
        """
        pass
    
    @abstractmethod
    def apply_torque(self, object_id: int, torque: np.ndarray):
        """Apply torque to object
        
        Args:
            object_id: Object identifier
            torque: Torque vector [tx, ty, tz] in Nm
        """
        pass
    
    @abstractmethod
    def step(self) -> Dict[int, Dict[str, Any]]:
        """Advance physics simulation by one timestep
        
        Returns:
            Updated states for all objects
        """
        pass
    
    @abstractmethod
    def check_collisions(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Check for collisions between objects
        
        Returns:
            List of collision pairs with contact info
        """
        pass
    
    @abstractmethod
    def add_constraint(
        self,
        constraint_id: int,
        object1_id: int,
        object2_id: int,
        constraint_type: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """Add constraint between objects
        
        Args:
            constraint_id: Unique constraint identifier
            object1_id: First object ID
            object2_id: Second object ID (can be -1 for world)
            constraint_type: Type of constraint
            parameters: Constraint parameters
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def remove_constraint(self, constraint_id: int) -> bool:
        """Remove constraint
        
        Args:
            constraint_id: Constraint identifier
            
        Returns:
            Success status
        """
        pass
    
    def set_wind(self, wind_velocity: np.ndarray):
        """Set wind velocity
        
        Args:
            wind_velocity: Wind velocity vector [vx, vy, vz] m/s
        """
        self.wind_velocity = wind_velocity.copy()
        logger.debug(f"Set wind velocity to {wind_velocity}")
    
    def set_air_properties(self, density: float, temperature: float):
        """Set air properties
        
        Args:
            density: Air density (kg/m³)
            temperature: Temperature (Celsius)
        """
        self.air_density = density
        self.temperature = temperature
        logger.debug(f"Set air density={density} kg/m³, temperature={temperature}°C")
    
    def calculate_aerodynamic_forces(
        self,
        velocity: np.ndarray,
        orientation: np.ndarray,
        drag_coefficient: float,
        reference_area: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate aerodynamic forces
        
        Args:
            velocity: Object velocity
            orientation: Object orientation (quaternion)
            drag_coefficient: Drag coefficient
            reference_area: Reference area (m²)
            
        Returns:
            Drag force and lift force vectors
        """
        if not self.enable_aerodynamics:
            return np.zeros(3), np.zeros(3)
        
        # Relative velocity including wind
        v_rel = velocity - self.wind_velocity
        v_magnitude = np.linalg.norm(v_rel)
        
        if v_magnitude < 0.01:  # Nearly stationary
            return np.zeros(3), np.zeros(3)
        
        # Dynamic pressure
        q = 0.5 * self.air_density * v_magnitude**2
        
        # Drag force (opposite to velocity)
        drag_force = -drag_coefficient * reference_area * q * (v_rel / v_magnitude)
        
        # Simplified lift (perpendicular to velocity)
        # In reality, this depends on angle of attack
        lift_coefficient = 0.1  # Simplified
        lift_direction = np.cross([0, 0, 1], v_rel)
        if np.linalg.norm(lift_direction) > 0:
            lift_direction /= np.linalg.norm(lift_direction)
            lift_force = lift_coefficient * reference_area * q * lift_direction
        else:
            lift_force = np.zeros(3)
        
        return drag_force, lift_force
    
    def get_simulation_info(self) -> Dict[str, Any]:
        """Get simulation information
        
        Returns:
            Dictionary with simulation parameters and stats
        """
        return {
            "timestep": self.timestep,
            "gravity": self.gravity.tolist(),
            "num_objects": len(self.objects),
            "num_constraints": len(self.constraints),
            "collision_enabled": self.enable_collision,
            "aerodynamics_enabled": self.enable_aerodynamics,
            "wind_enabled": self.enable_wind,
            "wind_velocity": self.wind_velocity.tolist(),
            "air_density": self.air_density,
            "temperature": self.temperature
        }
    
    @abstractmethod
    def visualize(self, mode: str = "human") -> Optional[np.ndarray]:
        """Visualize physics simulation
        
        Args:
            mode: Visualization mode
            
        Returns:
            RGB array if mode is 'rgb_array', None otherwise
        """
        pass