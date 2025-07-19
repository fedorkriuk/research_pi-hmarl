"""Port-Hamiltonian Neural Networks for Energy Conservation

This module implements port-Hamiltonian formulations for guaranteed
energy conservation using real energy specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging

from .base_pinn import PhysicsInformedNetwork
from .autodiff_physics import AutoDiffPhysics

logger = logging.getLogger(__name__)


class PortHamiltonianNetwork(PhysicsInformedNetwork):
    """Port-Hamiltonian neural network for energy-conserving dynamics"""
    
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dims: List[int] = [64, 64],
        dissipation: bool = True,
        external_forces: bool = True,
        **kwargs
    ):
        """Initialize Port-Hamiltonian network
        
        Args:
            state_dim: State dimension (q and p combined)
            control_dim: Control input dimension
            hidden_dims: Hidden layer dimensions
            dissipation: Whether to include dissipation
            external_forces: Whether to include external forces
            **kwargs: Additional arguments for base class
        """
        # Port-Hamiltonian formulation: ẋ = [J(x) - R(x)]∇H + g(x)u
        super().__init__(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,  # Hamiltonian is scalar
            physics_dim=state_dim,
            **kwargs
        )
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.q_dim = state_dim // 2  # Position dimension
        self.p_dim = state_dim // 2  # Momentum dimension
        
        # Hamiltonian network H(q, p)
        self.hamiltonian = self._create_hamiltonian_network(hidden_dims)
        
        # Interconnection matrix J(x) - skew-symmetric
        self.J_net = SkewSymmetricMatrix(state_dim, hidden_dims[0])
        
        # Dissipation matrix R(x) - positive semi-definite
        if dissipation:
            self.R_net = PositiveSemiDefiniteMatrix(state_dim, hidden_dims[0])
        else:
            self.R_net = None
        
        # Control matrix g(x)
        if external_forces:
            self.g_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.Tanh(),
                nn.Linear(hidden_dims[0], state_dim * control_dim)
            )
        else:
            self.g_net = None
        
        # Automatic differentiation
        self.autodiff = AutoDiffPhysics()
        
        logger.info(f"Initialized PortHamiltonianNetwork with state_dim={state_dim}")
    
    def _create_hamiltonian_network(self, hidden_dims: List[int]) -> nn.Module:
        """Create Hamiltonian function network
        
        Args:
            hidden_dims: Hidden dimensions
            
        Returns:
            Hamiltonian network
        """
        layers = []
        input_dim = self.state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        return nn.Sequential(*layers)
    
    def compute_hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian energy
        
        Args:
            state: State vector [batch, state_dim]
            
        Returns:
            Hamiltonian [batch, 1]
        """
        return self.hamiltonian(state)
    
    def forward(
        self,
        state: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        return_energy: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward dynamics using port-Hamiltonian formulation
        
        Args:
            state: Current state [batch, state_dim]
            control: Control input [batch, control_dim]
            return_energy: Whether to return energy
            
        Returns:
            Dictionary with dynamics and optional energy
        """
        batch_size = state.shape[0]
        state.requires_grad_(True)
        
        # Compute Hamiltonian
        H = self.compute_hamiltonian(state)
        
        # Compute gradient ∇H
        grad_H = self.autodiff.compute_derivatives(H, state)
        
        # Get interconnection matrix J(x)
        J = self.J_net(state)
        
        # Compute conservative dynamics: J∇H
        conservative_dynamics = torch.bmm(J, grad_H.unsqueeze(-1)).squeeze(-1)
        
        # Add dissipation if enabled
        if self.R_net is not None:
            R = self.R_net(state)
            dissipative_dynamics = -torch.bmm(R, grad_H.unsqueeze(-1)).squeeze(-1)
            dynamics = conservative_dynamics + dissipative_dynamics
        else:
            dynamics = conservative_dynamics
        
        # Add control if provided
        if control is not None and self.g_net is not None:
            g = self.g_net(state).view(batch_size, self.state_dim, self.control_dim)
            control_dynamics = torch.bmm(g, control.unsqueeze(-1)).squeeze(-1)
            dynamics = dynamics + control_dynamics
        
        result = {"dynamics": dynamics}
        
        if return_energy:
            # Decompose into kinetic and potential energy
            q = state[:, :self.q_dim]
            p = state[:, self.p_dim:]
            
            # Kinetic energy: T = p^T M^{-1} p / 2 (assuming unit mass)
            kinetic_energy = 0.5 * (p ** 2).sum(dim=1, keepdim=True)
            
            # Potential energy: V = H - T
            potential_energy = H - kinetic_energy
            
            result.update({
                "hamiltonian": H,
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
                "total_energy": H
            })
        
        return result
    
    def compute_physics_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        grad_x: Optional[torch.Tensor] = None,
        grad_xx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-based losses
        
        Args:
            x: Input states
            y: Network output (not used directly)
            grad_x: First derivatives
            grad_xx: Second derivatives
            
        Returns:
            Physics losses
        """
        losses = {}
        
        # Energy conservation loss
        # For autonomous system: dH/dt = -∇H^T R ∇H ≤ 0
        H = self.compute_hamiltonian(x)
        grad_H = self.autodiff.compute_derivatives(H, x)
        
        if self.R_net is not None:
            R = self.R_net(x)
            # Compute dissipation rate
            dissipation = torch.bmm(
                grad_H.unsqueeze(1),
                torch.bmm(R, grad_H.unsqueeze(-1))
            ).squeeze()
            
            # Dissipation should be non-negative
            losses["dissipation_positive"] = F.relu(-dissipation).mean()
        
        # Verify skew-symmetry of J
        J = self.J_net(x)
        J_transpose = J.transpose(-1, -2)
        skew_symmetry_error = torch.norm(J + J_transpose, dim=(-2, -1))
        losses["skew_symmetry"] = skew_symmetry_error.mean()
        
        # Physical energy constraints
        # Kinetic energy should be non-negative
        p = x[:, self.p_dim:]
        kinetic_energy = 0.5 * (p ** 2).sum(dim=1)
        losses["kinetic_positive"] = F.relu(-kinetic_energy).mean()
        
        return losses


class HamiltonianDynamics:
    """Hamiltonian dynamics solver"""
    
    def __init__(
        self,
        hamiltonian_func: Callable,
        state_dim: int,
        dt: float = 0.01
    ):
        """Initialize Hamiltonian dynamics
        
        Args:
            hamiltonian_func: Function that computes Hamiltonian
            state_dim: State dimension
            dt: Time step
        """
        self.hamiltonian_func = hamiltonian_func
        self.state_dim = state_dim
        self.q_dim = state_dim // 2
        self.p_dim = state_dim // 2
        self.dt = dt
        self.autodiff = AutoDiffPhysics()
    
    def symplectic_euler_step(
        self,
        state: torch.Tensor,
        control: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Symplectic Euler integration step
        
        Args:
            state: Current state [q, p]
            control: Control input
            
        Returns:
            Next state
        """
        q = state[:, :self.q_dim]
        p = state[:, self.p_dim:]
        
        # Compute Hamiltonian
        H = self.hamiltonian_func(state)
        
        # Compute derivatives
        dH_dq = self.autodiff.compute_derivatives(H, q)
        dH_dp = self.autodiff.compute_derivatives(H, p)
        
        # Symplectic Euler update
        # p_{n+1} = p_n - dt * dH/dq
        p_next = p - self.dt * dH_dq
        
        # Update state for q computation
        state_mid = torch.cat([q, p_next], dim=1)
        H_mid = self.hamiltonian_func(state_mid)
        dH_dp_mid = self.autodiff.compute_derivatives(H_mid, p_next)
        
        # q_{n+1} = q_n + dt * dH/dp(p_{n+1})
        q_next = q + self.dt * dH_dp_mid
        
        return torch.cat([q_next, p_next], dim=1)
    
    def leapfrog_step(
        self,
        state: torch.Tensor,
        control: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Leapfrog integration step (2nd order symplectic)
        
        Args:
            state: Current state
            control: Control input
            
        Returns:
            Next state
        """
        q = state[:, :self.q_dim]
        p = state[:, self.p_dim:]
        
        # Half step for momentum
        H = self.hamiltonian_func(state)
        dH_dq = self.autodiff.compute_derivatives(H, q)
        p_half = p - 0.5 * self.dt * dH_dq
        
        # Full step for position
        state_half = torch.cat([q, p_half], dim=1)
        H_half = self.hamiltonian_func(state_half)
        dH_dp_half = self.autodiff.compute_derivatives(H_half, p_half)
        q_next = q + self.dt * dH_dp_half
        
        # Half step for momentum
        state_next_q = torch.cat([q_next, p_half], dim=1)
        H_next = self.hamiltonian_func(state_next_q)
        dH_dq_next = self.autodiff.compute_derivatives(H_next, q_next)
        p_next = p_half - 0.5 * self.dt * dH_dq_next
        
        return torch.cat([q_next, p_next], dim=1)


class SkewSymmetricMatrix(nn.Module):
    """Learnable skew-symmetric matrix J(x)"""
    
    def __init__(self, dim: int, hidden_dim: int = 32):
        """Initialize skew-symmetric matrix
        
        Args:
            dim: Matrix dimension
            hidden_dim: Hidden dimension for parameterization
        """
        super().__init__()
        
        self.dim = dim
        
        # Parameterize upper triangular part
        n_params = dim * (dim - 1) // 2
        
        self.param_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_params)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute skew-symmetric matrix
        
        Args:
            x: State [batch, dim]
            
        Returns:
            Skew-symmetric matrix [batch, dim, dim]
        """
        batch_size = x.shape[0]
        
        # Get parameters
        params = self.param_net(x)
        
        # Build skew-symmetric matrix
        J = torch.zeros(batch_size, self.dim, self.dim, device=x.device)
        
        # Fill upper triangular
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                J[:, i, j] = params[:, idx]
                J[:, j, i] = -params[:, idx]  # Skew-symmetry
                idx += 1
        
        return J


class PositiveSemiDefiniteMatrix(nn.Module):
    """Learnable positive semi-definite matrix R(x)"""
    
    def __init__(self, dim: int, hidden_dim: int = 32, min_eigenvalue: float = 0.0):
        """Initialize PSD matrix
        
        Args:
            dim: Matrix dimension
            hidden_dim: Hidden dimension
            min_eigenvalue: Minimum eigenvalue for stability
        """
        super().__init__()
        
        self.dim = dim
        self.min_eigenvalue = min_eigenvalue
        
        # Parameterize via Cholesky decomposition
        n_params = dim * (dim + 1) // 2
        
        self.param_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_params)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute PSD matrix
        
        Args:
            x: State [batch, dim]
            
        Returns:
            PSD matrix [batch, dim, dim]
        """
        batch_size = x.shape[0]
        
        # Get parameters
        params = self.param_net(x)
        
        # Build lower triangular matrix L
        L = torch.zeros(batch_size, self.dim, self.dim, device=x.device)
        
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                if i == j:
                    # Diagonal elements must be positive
                    L[:, i, j] = F.softplus(params[:, idx]) + self.min_eigenvalue
                else:
                    L[:, i, j] = params[:, idx]
                idx += 1
        
        # R = L L^T
        R = torch.bmm(L, L.transpose(-1, -2))
        
        return R


class EnergyBasedModel(PortHamiltonianNetwork):
    """Energy-based model with real physical parameters"""
    
    def __init__(
        self,
        mass: float = 1.5,  # kg (drone mass)
        gravity: float = 9.81,  # m/s²
        drag_coefficient: float = 0.5,  # Aerodynamic drag
        **kwargs
    ):
        """Initialize energy-based model
        
        Args:
            mass: System mass
            gravity: Gravitational acceleration
            drag_coefficient: Drag coefficient
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        
        self.mass = mass
        self.gravity = gravity
        self.drag_coefficient = drag_coefficient
    
    def compute_hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        """Compute physical Hamiltonian
        
        Args:
            state: State [q, p]
            
        Returns:
            Hamiltonian energy
        """
        q = state[:, :self.q_dim]  # Position
        p = state[:, self.p_dim:]  # Momentum
        
        # Kinetic energy: T = p^T p / (2m)
        kinetic = 0.5 * (p ** 2).sum(dim=1, keepdim=True) / self.mass
        
        # Potential energy: V = mgh (height is last component of q)
        if self.q_dim >= 3:
            height = q[:, 2:3]  # z-coordinate
            potential = self.mass * self.gravity * height
        else:
            potential = torch.zeros_like(kinetic)
        
        # Neural network correction
        nn_correction = self.hamiltonian(state)
        
        # Total Hamiltonian
        H = kinetic + potential + 0.1 * nn_correction
        
        return H