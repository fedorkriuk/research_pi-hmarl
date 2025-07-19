"""Conservation Laws for Physics-Informed Learning

This module implements conservation laws (momentum, energy, angular momentum)
using real dynamics data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from .autodiff_physics import AutoDiffPhysics, PhysicsGradients

logger = logging.getLogger(__name__)


class ConservationLaws:
    """Enforces physical conservation laws"""
    
    def __init__(
        self,
        mass: float = 1.5,  # kg (drone mass)
        inertia_tensor: Optional[torch.Tensor] = None,
        gravity: float = 9.81,  # m/s²
        enable_relativistic: bool = False,
        c: float = 299792458.0  # Speed of light (m/s)
    ):
        """Initialize conservation laws
        
        Args:
            mass: System mass
            inertia_tensor: Moment of inertia tensor
            gravity: Gravitational acceleration
            enable_relativistic: Whether to include relativistic corrections
            c: Speed of light
        """
        self.mass = mass
        self.gravity = gravity
        self.enable_relativistic = enable_relativistic
        self.c = c
        
        # Default inertia tensor for drone (approximate)
        if inertia_tensor is None:
            # DJI Mavic 3 approximate inertia (kg⋅m²)
            self.inertia_tensor = torch.tensor([
                [0.02, 0.0, 0.0],
                [0.0, 0.02, 0.0],
                [0.0, 0.0, 0.03]
            ], dtype=torch.float32)
        else:
            self.inertia_tensor = inertia_tensor
        
        self.autodiff = AutoDiffPhysics()
        self.physics_grad = PhysicsGradients()
        
        logger.info(f"Initialized ConservationLaws with mass={mass}kg")
    
    def compute_all_conservation_losses(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        angular_velocities: Optional[torch.Tensor] = None,
        torques: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all conservation law violations
        
        Args:
            positions: Positions [batch, num_agents, 3]
            velocities: Velocities [batch, num_agents, 3]
            forces: Forces [batch, num_agents, 3]
            angular_velocities: Angular velocities [batch, num_agents, 3]
            torques: Torques [batch, num_agents, 3]
            time: Time tensor for derivatives
            
        Returns:
            Dictionary of conservation losses
        """
        losses = {}
        
        # Linear momentum conservation
        momentum_loss = self.linear_momentum_conservation(
            velocities, forces, time
        )
        losses["linear_momentum"] = momentum_loss
        
        # Energy conservation
        energy_loss = self.energy_conservation(
            positions, velocities, forces, time
        )
        losses["energy"] = energy_loss
        
        # Angular momentum conservation
        if angular_velocities is not None:
            angular_loss = self.angular_momentum_conservation(
                positions, velocities, angular_velocities, 
                forces, torques, time
            )
            losses["angular_momentum"] = angular_loss
        
        # Newton's third law
        action_reaction_loss = self.action_reaction_symmetry(forces)
        losses["action_reaction"] = action_reaction_loss
        
        return losses
    
    def linear_momentum_conservation(
        self,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Linear momentum conservation: dp/dt = F_ext
        
        Args:
            velocities: Velocities
            forces: External forces
            time: Time for derivatives
            
        Returns:
            Conservation loss
        """
        batch_size, num_agents, _ = velocities.shape
        
        # Compute momentum
        if self.enable_relativistic:
            # Relativistic momentum: p = γmv
            gamma = self._lorentz_factor(velocities)
            momentum = self.mass * gamma.unsqueeze(-1) * velocities
        else:
            # Classical momentum: p = mv
            momentum = self.mass * velocities
        
        # Total momentum
        total_momentum = momentum.sum(dim=1)
        
        if time is not None and time.requires_grad:
            # Time derivative of momentum
            dp_dt = self.autodiff.compute_derivatives(total_momentum, time)
            
            # Total external force
            total_force = forces.sum(dim=1)
            
            # Conservation: dp/dt = F_ext
            residual = dp_dt - total_force
        else:
            # For closed system, total momentum should be constant
            # Check variation in total momentum
            momentum_mean = total_momentum.mean(dim=0, keepdim=True)
            residual = total_momentum - momentum_mean
        
        return F.mse_loss(residual, torch.zeros_like(residual))
    
    def energy_conservation(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Energy conservation: dE/dt = P_ext
        
        Args:
            positions: Positions
            velocities: Velocities
            forces: Forces
            time: Time
            
        Returns:
            Conservation loss
        """
        # Kinetic energy
        if self.enable_relativistic:
            # Relativistic kinetic energy: K = (γ - 1)mc²
            gamma = self._lorentz_factor(velocities)
            kinetic_energy = (gamma - 1) * self.mass * self.c**2
            kinetic_energy = kinetic_energy.sum(dim=1)
        else:
            # Classical kinetic energy: K = 0.5 * m * v²
            kinetic_energy = 0.5 * self.mass * (velocities ** 2).sum(dim=(1, 2))
        
        # Potential energy (gravitational)
        height = positions[:, :, 2]  # z-coordinate
        potential_energy = self.mass * self.gravity * height.sum(dim=1)
        
        # Total energy
        total_energy = kinetic_energy + potential_energy
        
        if time is not None and time.requires_grad:
            # Time derivative of energy
            dE_dt = self.autodiff.compute_derivatives(total_energy, time)
            
            # Power from external forces: P = F·v
            power = (forces * velocities).sum(dim=(1, 2))
            
            # Conservation: dE/dt = P_ext
            residual = dE_dt - power
        else:
            # Energy should be conserved
            energy_mean = total_energy.mean(dim=0, keepdim=True)
            residual = total_energy - energy_mean
        
        return F.mse_loss(residual, torch.zeros_like(residual))
    
    def angular_momentum_conservation(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
        forces: torch.Tensor,
        torques: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Angular momentum conservation: dL/dt = τ_ext
        
        Args:
            positions: Positions relative to center
            velocities: Linear velocities
            angular_velocities: Angular velocities
            forces: Forces
            torques: External torques
            time: Time
            
        Returns:
            Conservation loss
        """
        batch_size, num_agents, _ = positions.shape
        
        # Orbital angular momentum: L_orbital = r × p
        if self.enable_relativistic:
            gamma = self._lorentz_factor(velocities)
            momentum = self.mass * gamma.unsqueeze(-1) * velocities
        else:
            momentum = self.mass * velocities
        
        L_orbital = torch.cross(positions, momentum, dim=-1)
        
        # Spin angular momentum: L_spin = I·ω
        # Expand inertia tensor for batch and agents
        I = self.inertia_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_agents, 3, 3
        ).to(angular_velocities.device)
        
        L_spin = torch.bmm(
            I.view(-1, 3, 3),
            angular_velocities.view(-1, 3, 1)
        ).view(batch_size, num_agents, 3)
        
        # Total angular momentum
        L_total = (L_orbital + L_spin).sum(dim=1)
        
        if time is not None and time.requires_grad:
            # Time derivative
            dL_dt = self.autodiff.compute_derivatives(L_total, time)
            
            # External torques
            tau_orbital = torch.cross(positions, forces, dim=-1).sum(dim=1)
            
            if torques is not None:
                tau_total = tau_orbital + torques.sum(dim=1)
            else:
                tau_total = tau_orbital
            
            # Conservation: dL/dt = τ_ext
            residual = dL_dt - tau_total
        else:
            # Angular momentum should be conserved
            L_mean = L_total.mean(dim=0, keepdim=True)
            residual = L_total - L_mean
        
        return F.mse_loss(residual, torch.zeros_like(residual))
    
    def action_reaction_symmetry(self, forces: torch.Tensor) -> torch.Tensor:
        """Newton's third law: F_ij = -F_ji
        
        Args:
            forces: Pairwise forces [batch, num_agents, num_agents, 3]
            
        Returns:
            Symmetry loss
        """
        if forces.dim() == 3:
            # Convert to pairwise format if needed
            return torch.tensor(0.0, device=forces.device)
        
        # Check F_ij + F_ji = 0
        forces_transpose = forces.transpose(1, 2)
        residual = forces + forces_transpose
        
        return F.mse_loss(residual, torch.zeros_like(residual))
    
    def _lorentz_factor(self, velocities: torch.Tensor) -> torch.Tensor:
        """Compute Lorentz factor γ = 1/√(1 - v²/c²)
        
        Args:
            velocities: Velocities
            
        Returns:
            Lorentz factor
        """
        v_squared = (velocities ** 2).sum(dim=-1)
        beta_squared = v_squared / (self.c ** 2)
        
        # Ensure numerical stability
        beta_squared = torch.clamp(beta_squared, max=0.999)
        
        gamma = 1.0 / torch.sqrt(1.0 - beta_squared)
        
        return gamma


class MomentumConservation(nn.Module):
    """Neural network module for momentum conservation"""
    
    def __init__(
        self,
        num_agents: int,
        mass: float = 1.5,
        enforce_closed_system: bool = True
    ):
        """Initialize momentum conservation module
        
        Args:
            num_agents: Number of agents
            mass: Agent mass
            enforce_closed_system: Whether to enforce closed system
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.mass = mass
        self.enforce_closed_system = enforce_closed_system
        
        # Learnable interaction forces
        self.interaction_net = nn.Sequential(
            nn.Linear(6, 64),  # Relative position and velocity
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # Force vector
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute forces ensuring momentum conservation
        
        Args:
            positions: Agent positions
            velocities: Agent velocities
            
        Returns:
            Forces and momentum
        """
        batch_size = positions.shape[0]
        forces = torch.zeros_like(positions)
        
        # Compute pairwise forces
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                # Relative state
                rel_pos = positions[:, j] - positions[:, i]
                rel_vel = velocities[:, j] - velocities[:, i]
                rel_state = torch.cat([rel_pos, rel_vel], dim=-1)
                
                # Interaction force
                f_ij = self.interaction_net(rel_state)
                
                # Newton's third law
                forces[:, i] += f_ij
                forces[:, j] -= f_ij
        
        # Compute momentum
        momentum = self.mass * velocities
        
        if self.enforce_closed_system:
            # Remove any net force
            net_force = forces.mean(dim=1, keepdim=True)
            forces = forces - net_force
        
        return forces, momentum


class EnergyConservation(nn.Module):
    """Neural network module for energy conservation"""
    
    def __init__(
        self,
        potential_func: Optional[Callable] = None,
        mass: float = 1.5,
        gravity: float = 9.81
    ):
        """Initialize energy conservation module
        
        Args:
            potential_func: Potential energy function
            mass: System mass
            gravity: Gravitational acceleration
        """
        super().__init__()
        
        self.mass = mass
        self.gravity = gravity
        
        if potential_func is None:
            # Default gravitational potential
            self.potential_net = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            self.potential_func = potential_func
    
    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute energy components
        
        Args:
            positions: Positions
            velocities: Velocities
            
        Returns:
            Energy components
        """
        # Kinetic energy
        kinetic = 0.5 * self.mass * (velocities ** 2).sum(dim=-1)
        
        # Potential energy
        if hasattr(self, 'potential_net'):
            potential = self.potential_net(positions).squeeze(-1)
            # Add gravitational potential
            potential = potential + self.mass * self.gravity * positions[:, :, 2]
        else:
            potential = self.potential_func(positions)
        
        # Total energy
        total = kinetic + potential
        
        return {
            "kinetic_energy": kinetic,
            "potential_energy": potential,
            "total_energy": total
        }


class AngularMomentumConservation(nn.Module):
    """Neural network module for angular momentum conservation"""
    
    def __init__(
        self,
        inertia_tensor: torch.Tensor,
        mass: float = 1.5
    ):
        """Initialize angular momentum conservation
        
        Args:
            inertia_tensor: Moment of inertia tensor
            mass: System mass
        """
        super().__init__()
        
        self.register_buffer('inertia_tensor', inertia_tensor)
        self.mass = mass
        
        # Torque network
        self.torque_net = nn.Sequential(
            nn.Linear(9, 64),  # State: pos, vel, ang_vel
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # Torque vector
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        angular_velocities: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute angular momentum and torques
        
        Args:
            positions: Positions
            velocities: Linear velocities
            angular_velocities: Angular velocities
            
        Returns:
            Angular momentum components
        """
        batch_size, num_agents = positions.shape[:2]
        
        # Orbital angular momentum
        momentum = self.mass * velocities
        L_orbital = torch.cross(positions, momentum, dim=-1)
        
        # Spin angular momentum
        I_expanded = self.inertia_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_agents, 3, 3
        )
        
        L_spin = torch.bmm(
            I_expanded.view(-1, 3, 3),
            angular_velocities.view(-1, 3, 1)
        ).view(batch_size, num_agents, 3)
        
        # Total angular momentum
        L_total = L_orbital + L_spin
        
        # Compute torques
        state = torch.cat([positions, velocities, angular_velocities], dim=-1)
        torques = self.torque_net(state.view(-1, 9)).view(batch_size, num_agents, 3)
        
        return {
            "orbital_angular_momentum": L_orbital,
            "spin_angular_momentum": L_spin,
            "total_angular_momentum": L_total,
            "torques": torques
        }


class ThermodynamicConstraints:
    """Thermodynamic constraints for energy systems"""
    
    def __init__(self, temperature: float = 293.15):  # 20°C
        """Initialize thermodynamic constraints
        
        Args:
            temperature: System temperature in Kelvin
        """
        self.temperature = temperature
        self.k_B = 1.380649e-23  # Boltzmann constant
    
    def entropy_production_loss(
        self,
        heat_flow: torch.Tensor,
        temperature_gradient: torch.Tensor
    ) -> torch.Tensor:
        """Ensure non-negative entropy production
        
        Args:
            heat_flow: Heat flow vector
            temperature_gradient: Temperature gradient
            
        Returns:
            Entropy production loss
        """
        # Entropy production: σ = -q·∇T/T² ≥ 0
        entropy_production = -(heat_flow * temperature_gradient).sum(dim=-1) / (self.temperature ** 2)
        
        # Penalize negative entropy production
        loss = F.relu(-entropy_production).mean()
        
        return loss