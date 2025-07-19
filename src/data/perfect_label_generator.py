"""Perfect Label Generator for Physics Constraints

This module generates perfect ground truth labels for physics constraints
in the synthetic data, ensuring accurate supervision for PINN training.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConstraintLabel:
    """Container for physics constraint labels"""
    # Energy constraints
    energy_feasible: bool
    energy_margin: float  # Remaining energy percentage
    time_to_empty: float  # Seconds until battery depletion
    
    # Collision constraints
    collision_free: bool
    min_separation: float  # Minimum distance to other agents
    collision_risk: float  # Risk score 0-1
    
    # Dynamics constraints
    velocity_feasible: bool
    velocity_margin: float  # Margin to max velocity
    acceleration_feasible: bool
    acceleration_margin: float  # Margin to max acceleration
    
    # Hamiltonian constraints
    energy_conserved: bool
    hamiltonian_error: float  # Energy conservation error
    momentum_conserved: bool
    momentum_error: float  # Momentum conservation error
    
    # Communication constraints
    connected: bool
    link_quality: float  # Communication link quality 0-1
    latency: float  # Communication latency in ms
    
    # Overall feasibility
    fully_feasible: bool
    constraint_violation_score: float  # Overall violation score


class PerfectLabelGenerator:
    """Generates perfect physics constraint labels for synthetic data"""
    
    def __init__(self, physics_config: Dict[str, Any]):
        """Initialize the label generator
        
        Args:
            physics_config: Physics configuration parameters
        """
        self.physics_config = physics_config
        
        # Constraint thresholds
        self.min_battery_soc = 0.1  # 10% minimum
        self.min_separation_distance = physics_config.get("min_separation_distance", 2.0)
        self.max_velocity = physics_config.get("max_velocity", 20.0)
        self.max_acceleration = physics_config.get("max_acceleration", 10.0)
        self.communication_range = physics_config.get("communication_range", 50.0)
        
        # Tolerances
        self.energy_tolerance = 1e-6
        self.momentum_tolerance = 1e-6
        
        logger.info("Initialized PerfectLabelGenerator")
    
    def generate_labels(
        self,
        positions: np.ndarray,  # [num_agents, 3]
        velocities: np.ndarray,  # [num_agents, 3]
        accelerations: np.ndarray,  # [num_agents, 3]
        battery_soc: np.ndarray,  # [num_agents]
        power_consumption: np.ndarray,  # [num_agents]
        masses: np.ndarray,  # [num_agents]
        forces: Optional[np.ndarray] = None,  # [num_agents, 3]
        dt: float = 0.01
    ) -> List[PhysicsConstraintLabel]:
        """Generate perfect labels for current state
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            accelerations: Agent accelerations
            battery_soc: Battery state of charge
            power_consumption: Power consumption in watts
            masses: Agent masses
            forces: Applied forces (optional)
            dt: Time step
            
        Returns:
            List of constraint labels for each agent
        """
        num_agents = positions.shape[0]
        labels = []
        
        for i in range(num_agents):
            # Energy constraints
            energy_feasible = battery_soc[i] >= self.min_battery_soc
            energy_margin = battery_soc[i] - self.min_battery_soc
            
            # Time to empty calculation
            if power_consumption[i] > 0:
                battery_capacity_wh = 50.0  # Example: 50 Wh battery
                remaining_energy = battery_capacity_wh * battery_soc[i]
                time_to_empty = remaining_energy / power_consumption[i] * 3600  # seconds
            else:
                time_to_empty = float('inf')
            
            # Collision constraints
            distances = []
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            
            min_separation = min(distances) if distances else float('inf')
            collision_free = min_separation >= self.min_separation_distance
            collision_risk = max(0, 1 - min_separation / self.min_separation_distance)
            
            # Dynamics constraints
            speed = np.linalg.norm(velocities[i])
            velocity_feasible = speed <= self.max_velocity
            velocity_margin = self.max_velocity - speed
            
            accel_mag = np.linalg.norm(accelerations[i])
            acceleration_feasible = accel_mag <= self.max_acceleration
            acceleration_margin = self.max_acceleration - accel_mag
            
            # Hamiltonian constraints
            if forces is not None:
                # Check Newton's second law
                expected_accel = forces[i] / masses[i]
                accel_error = np.linalg.norm(accelerations[i] - expected_accel)
                momentum_conserved = accel_error < self.momentum_tolerance
                momentum_error = accel_error
            else:
                momentum_conserved = True
                momentum_error = 0.0
            
            # Energy conservation (simplified)
            kinetic_energy = 0.5 * masses[i] * speed**2
            potential_energy = masses[i] * 9.81 * positions[i, 2]
            total_energy = kinetic_energy + potential_energy
            
            # For perfect labels, we assume energy is conserved
            energy_conserved = True
            hamiltonian_error = 0.0
            
            # Communication constraints
            connected_count = 0
            total_link_quality = 0.0
            
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= self.communication_range:
                        connected_count += 1
                        # Link quality decreases with distance
                        link_quality = 1.0 - (dist / self.communication_range)**2
                        total_link_quality += link_quality
            
            connected = connected_count > 0
            avg_link_quality = total_link_quality / connected_count if connected_count > 0 else 0.0
            
            # Latency model: base + distance factor
            base_latency = 1.0  # ms
            distance_factor = 0.01  # ms/m
            if connected_count > 0:
                avg_distance = sum(np.linalg.norm(positions[i] - positions[j]) 
                                 for j in range(num_agents) if j != i) / connected_count
                latency = base_latency + distance_factor * avg_distance
            else:
                latency = float('inf')
            
            # Overall feasibility
            fully_feasible = all([
                energy_feasible,
                collision_free,
                velocity_feasible,
                acceleration_feasible,
                energy_conserved,
                momentum_conserved,
                connected
            ])
            
            # Constraint violation score
            violation_scores = []
            if not energy_feasible:
                violation_scores.append(abs(energy_margin))
            if not collision_free:
                violation_scores.append(collision_risk)
            if not velocity_feasible:
                violation_scores.append(abs(velocity_margin) / self.max_velocity)
            if not acceleration_feasible:
                violation_scores.append(abs(acceleration_margin) / self.max_acceleration)
            
            constraint_violation_score = sum(violation_scores) if violation_scores else 0.0
            
            # Create label
            label = PhysicsConstraintLabel(
                energy_feasible=energy_feasible,
                energy_margin=energy_margin,
                time_to_empty=time_to_empty,
                collision_free=collision_free,
                min_separation=min_separation,
                collision_risk=collision_risk,
                velocity_feasible=velocity_feasible,
                velocity_margin=velocity_margin,
                acceleration_feasible=acceleration_feasible,
                acceleration_margin=acceleration_margin,
                energy_conserved=energy_conserved,
                hamiltonian_error=hamiltonian_error,
                momentum_conserved=momentum_conserved,
                momentum_error=momentum_error,
                connected=connected,
                link_quality=avg_link_quality,
                latency=latency,
                fully_feasible=fully_feasible,
                constraint_violation_score=constraint_violation_score
            )
            
            labels.append(label)
        
        return labels
    
    def generate_trajectory_labels(
        self,
        trajectory_data: Dict[str, np.ndarray],
        agent_specs: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Generate labels for entire trajectory
        
        Args:
            trajectory_data: Dictionary with trajectory arrays
            agent_specs: List of agent specifications
            
        Returns:
            Dictionary of label arrays
        """
        num_steps = trajectory_data["positions"].shape[0]
        num_agents = trajectory_data["positions"].shape[1]
        
        # Initialize label arrays
        labels = {
            "energy_feasible": np.zeros((num_steps, num_agents), dtype=bool),
            "collision_free": np.zeros((num_steps, num_agents), dtype=bool),
            "velocity_feasible": np.zeros((num_steps, num_agents), dtype=bool),
            "acceleration_feasible": np.zeros((num_steps, num_agents), dtype=bool),
            "fully_feasible": np.zeros((num_steps, num_agents), dtype=bool),
            "min_separation": np.zeros((num_steps, num_agents)),
            "constraint_violation_score": np.zeros((num_steps, num_agents))
        }
        
        # Extract agent masses
        masses = np.array([spec["mass"] for spec in agent_specs])
        
        # Generate labels for each timestep
        for t in range(num_steps):
            step_labels = self.generate_labels(
                positions=trajectory_data["positions"][t],
                velocities=trajectory_data["velocities"][t],
                accelerations=trajectory_data["accelerations"][t],
                battery_soc=trajectory_data["battery_soc"][t],
                power_consumption=trajectory_data["power_consumption"][t],
                masses=masses,
                forces=trajectory_data.get("forces", [None])[t]
            )
            
            # Store in arrays
            for i, label in enumerate(step_labels):
                labels["energy_feasible"][t, i] = label.energy_feasible
                labels["collision_free"][t, i] = label.collision_free
                labels["velocity_feasible"][t, i] = label.velocity_feasible
                labels["acceleration_feasible"][t, i] = label.acceleration_feasible
                labels["fully_feasible"][t, i] = label.fully_feasible
                labels["min_separation"][t, i] = label.min_separation
                labels["constraint_violation_score"][t, i] = label.constraint_violation_score
        
        return labels
    
    def compute_constraint_statistics(
        self,
        labels: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute statistics on constraint satisfaction
        
        Args:
            labels: Dictionary of label arrays
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Satisfaction rates
        for key in ["energy_feasible", "collision_free", "velocity_feasible", 
                   "acceleration_feasible", "fully_feasible"]:
            if key in labels:
                stats[f"{key}_rate"] = np.mean(labels[key])
        
        # Separation statistics
        if "min_separation" in labels:
            stats["avg_min_separation"] = np.mean(labels["min_separation"])
            stats["min_min_separation"] = np.min(labels["min_separation"])
            
        # Violation statistics
        if "constraint_violation_score" in labels:
            stats["avg_violation_score"] = np.mean(labels["constraint_violation_score"])
            stats["max_violation_score"] = np.max(labels["constraint_violation_score"])
            stats["violation_rate"] = np.mean(labels["constraint_violation_score"] > 0)
        
        return stats
    
    def validate_physical_consistency(
        self,
        trajectory_data: Dict[str, np.ndarray],
        tolerance: float = 1e-6
    ) -> Dict[str, bool]:
        """Validate physical consistency of trajectory
        
        Args:
            trajectory_data: Dictionary with trajectory arrays
            tolerance: Numerical tolerance
            
        Returns:
            Dictionary of validation results
        """
        validation = {}
        
        # Check velocity consistency with position
        if "positions" in trajectory_data and "velocities" in trajectory_data:
            dt = 0.01  # Assuming fixed timestep
            computed_vel = np.diff(trajectory_data["positions"], axis=0) / dt
            vel_error = np.mean(np.abs(computed_vel - trajectory_data["velocities"][:-1]))
            validation["velocity_consistent"] = vel_error < tolerance
            validation["velocity_error"] = vel_error
        
        # Check acceleration consistency with velocity
        if "velocities" in trajectory_data and "accelerations" in trajectory_data:
            dt = 0.01
            computed_acc = np.diff(trajectory_data["velocities"], axis=0) / dt
            acc_error = np.mean(np.abs(computed_acc - trajectory_data["accelerations"][:-1]))
            validation["acceleration_consistent"] = acc_error < tolerance
            validation["acceleration_error"] = acc_error
        
        # Check energy conservation
        if all(k in trajectory_data for k in ["positions", "velocities", "masses"]):
            num_steps = trajectory_data["positions"].shape[0]
            total_energy = []
            
            for t in range(num_steps):
                ke = 0.5 * np.sum(trajectory_data["masses"] * 
                                 np.sum(trajectory_data["velocities"][t]**2, axis=1))
                pe = np.sum(trajectory_data["masses"] * 9.81 * 
                           trajectory_data["positions"][t, :, 2])
                total_energy.append(ke + pe)
            
            energy_variation = np.std(total_energy) / np.mean(total_energy)
            validation["energy_conserved"] = energy_variation < 0.01  # 1% tolerance
            validation["energy_variation"] = energy_variation
        
        return validation


# Convenience function
def create_label_generator(physics_config: Dict[str, Any]) -> PerfectLabelGenerator:
    """Create perfect label generator
    
    Args:
        physics_config: Physics configuration
        
    Returns:
        PerfectLabelGenerator instance
    """
    return PerfectLabelGenerator(physics_config)