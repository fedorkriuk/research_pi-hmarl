"""Base Physics-Informed Neural Network Implementation

This module provides the foundation for physics-informed neural networks
with automatic differentiation and constraint enforcement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PhysicsInformedNetwork(nn.Module, ABC):
    """Base class for physics-informed neural networks"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        physics_dim: int,
        activation: str = "tanh",
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        physics_weight: float = 1.0
    ):
        """Initialize PINN
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            physics_dim: Physics state dimension
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
            physics_weight: Weight for physics loss
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_dim = physics_dim
        self.physics_weight = physics_weight
        
        # Build network layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layer = PINNLayer(
                dims[i], dims[i+1],
                activation=activation if i < len(dims) - 2 else "none",
                use_batch_norm=use_batch_norm and i < len(dims) - 2,
                dropout=dropout if i < len(dims) - 2 else 0.0
            )
            self.layers.append(layer)
        
        # Physics constraint modules
        self.physics_modules = nn.ModuleDict()
        
        # Initialize weights using Xavier/Glorot
        self._initialize_weights()
        
        logger.info(f"Initialized PhysicsInformedNetwork with {len(self.layers)} layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    @abstractmethod
    def compute_physics_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        grad_x: Optional[torch.Tensor] = None,
        grad_xx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-based loss terms
        
        Args:
            x: Input positions/states
            y: Network output
            grad_x: First derivatives
            grad_xx: Second derivatives
            
        Returns:
            Dictionary of physics loss terms
        """
        pass
    
    def forward_with_physics(
        self,
        x: torch.Tensor,
        compute_gradients: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with physics computations
        
        Args:
            x: Input tensor
            compute_gradients: Whether to compute gradients
            
        Returns:
            Dictionary with output and physics quantities
        """
        x.requires_grad_(True)
        
        # Forward pass
        y = self.forward(x)
        
        result = {"output": y}
        
        if compute_gradients:
            # Compute gradients
            grad_x = self._compute_gradient(y, x, create_graph=True)
            grad_xx = self._compute_gradient(grad_x, x, create_graph=True)
            
            result["grad_x"] = grad_x
            result["grad_xx"] = grad_xx
            
            # Compute physics losses
            physics_losses = self.compute_physics_loss(x, y, grad_x, grad_xx)
            result.update(physics_losses)
        
        return result
    
    def _compute_gradient(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        grad_outputs: Optional[torch.Tensor] = None,
        create_graph: bool = True
    ) -> torch.Tensor:
        """Compute gradients using automatic differentiation
        
        Args:
            outputs: Output tensor
            inputs: Input tensor
            grad_outputs: Gradient w.r.t. outputs
            create_graph: Whether to create computation graph
            
        Returns:
            Gradient tensor
        """
        if grad_outputs is None:
            grad_outputs = torch.ones_like(outputs)
        
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        return gradients
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for tanh activation
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def add_physics_module(self, name: str, module: nn.Module):
        """Add a physics constraint module
        
        Args:
            name: Module name
            module: Physics module
        """
        self.physics_modules[name] = module
    
    def get_physics_parameters(self) -> Dict[str, Any]:
        """Get physics-related parameters
        
        Returns:
            Dictionary of physics parameters
        """
        params = {
            "physics_weight": self.physics_weight,
            "physics_dim": self.physics_dim,
            "num_physics_modules": len(self.physics_modules)
        }
        
        # Add module-specific parameters
        for name, module in self.physics_modules.items():
            if hasattr(module, "get_parameters"):
                params[f"{name}_params"] = module.get_parameters()
        
        return params


class PINNLayer(nn.Module):
    """Single layer for PINN with optional normalization and activation"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "tanh",
        use_batch_norm: bool = False,
        dropout: float = 0.0
    ):
        """Initialize PINN layer
        
        Args:
            in_features: Input features
            out_features: Output features
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(out_features) if use_batch_norm else None
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.linear(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x
    
    def _get_activation(self, name: str) -> Optional[nn.Module]:
        """Get activation function by name
        
        Args:
            name: Activation name
            
        Returns:
            Activation module
        """
        if name == "none":
            return None
        elif name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "elu":
            return nn.ELU()
        elif name == "silu":
            return nn.SiLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {name}")


class DynamicsNetwork(PhysicsInformedNetwork):
    """PINN for learning dynamics with physics constraints"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128, 128],
        dt: float = 0.05,  # Time step
        mass: float = 1.5,  # kg (drone mass)
        **kwargs
    ):
        """Initialize dynamics network
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            dt: Time step
            mass: Agent mass
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=state_dim,
            physics_dim=state_dim,
            **kwargs
        )
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt
        self.mass = mass
    
    def compute_physics_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        grad_x: Optional[torch.Tensor] = None,
        grad_xx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute dynamics physics losses
        
        Args:
            x: Input states and actions
            y: Predicted next states
            grad_x: First derivatives
            grad_xx: Second derivatives
            
        Returns:
            Physics loss terms
        """
        losses = {}
        
        # Extract current state and action
        state = x[:, :self.state_dim]
        action = x[:, self.state_dim:]
        
        # Extract positions and velocities (assuming state = [pos, vel])
        pos_dim = self.state_dim // 2
        position = state[:, :pos_dim]
        velocity = state[:, pos_dim:]
        
        # Predicted next state
        next_position = y[:, :pos_dim]
        next_velocity = y[:, pos_dim:]
        
        # Kinematic consistency
        # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        expected_position = position + velocity * self.dt + 0.5 * action[:, :pos_dim] * self.dt**2
        kinematic_loss = F.mse_loss(next_position, expected_position)
        losses["kinematic_consistency"] = kinematic_loss
        
        # Velocity update
        # v(t+dt) = v(t) + a(t)*dt
        expected_velocity = velocity + action[:, :pos_dim] * self.dt
        velocity_loss = F.mse_loss(next_velocity, expected_velocity)
        losses["velocity_consistency"] = velocity_loss
        
        # Energy conservation (simplified)
        # E = 0.5 * m * v^2
        current_energy = 0.5 * self.mass * (velocity ** 2).sum(dim=1)
        next_energy = 0.5 * self.mass * (next_velocity ** 2).sum(dim=1)
        
        # Work done by forces
        work = (action[:, :pos_dim] * velocity).sum(dim=1) * self.dt
        expected_energy = current_energy + work
        
        energy_loss = F.mse_loss(next_energy, expected_energy)
        losses["energy_consistency"] = energy_loss
        
        return losses


class ConstrainedPINN(PhysicsInformedNetwork):
    """PINN with explicit constraint enforcement"""
    
    def __init__(
        self,
        *args,
        constraint_functions: Optional[List[Callable]] = None,
        lagrange_multipliers: Optional[List[float]] = None,
        **kwargs
    ):
        """Initialize constrained PINN
        
        Args:
            constraint_functions: List of constraint functions
            lagrange_multipliers: Initial Lagrange multipliers
            *args, **kwargs: Arguments for base class
        """
        super().__init__(*args, **kwargs)
        
        self.constraint_functions = constraint_functions or []
        
        # Learnable Lagrange multipliers
        if lagrange_multipliers is None:
            lagrange_multipliers = [1.0] * len(self.constraint_functions)
        
        self.lagrange_multipliers = nn.ParameterList([
            nn.Parameter(torch.tensor(lm, dtype=torch.float32))
            for lm in lagrange_multipliers
        ])
    
    def compute_constraint_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute constraint violation losses
        
        Args:
            x: Input
            y: Output
            
        Returns:
            Constraint losses
        """
        losses = {}
        
        for i, (constraint_fn, lm) in enumerate(
            zip(self.constraint_functions, self.lagrange_multipliers)
        ):
            violation = constraint_fn(x, y)
            
            # Augmented Lagrangian
            loss = lm * violation + 0.5 * violation ** 2
            losses[f"constraint_{i}"] = loss.mean()
            losses[f"constraint_{i}_violation"] = violation.mean()
        
        return losses
    
    def update_lagrange_multipliers(self, violations: List[float], lr: float = 0.01):
        """Update Lagrange multipliers based on constraint violations
        
        Args:
            violations: List of constraint violations
            lr: Learning rate for multiplier update
        """
        with torch.no_grad():
            for i, (lm, violation) in enumerate(
                zip(self.lagrange_multipliers, violations)
            ):
                # Gradient ascent on Lagrange multipliers
                lm.data += lr * violation


class ResidualPINN(PhysicsInformedNetwork):
    """PINN with residual connections for better gradient flow"""
    
    def __init__(self, *args, **kwargs):
        """Initialize residual PINN"""
        super().__init__(*args, **kwargs)
        
        # Add residual connections
        self.residual_layers = nn.ModuleList()
        
        # Create residual connections every 2 layers
        for i in range(0, len(self.layers) - 1, 2):
            if i + 2 < len(self.layers):
                in_dim = self.layers[i].linear.in_features
                out_dim = self.layers[i + 2].linear.out_features
                
                if in_dim == out_dim:
                    # Identity residual
                    self.residual_layers.append(nn.Identity())
                else:
                    # Projection residual
                    self.residual_layers.append(nn.Linear(in_dim, out_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        residual_idx = 0
        
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and residual_idx < len(self.residual_layers):
                # Store residual
                residual = self.residual_layers[residual_idx](x)
            
            x = layer(x)
            
            if i % 2 == 1 and i < len(self.layers) - 1 and residual_idx < len(self.residual_layers):
                # Add residual
                x = x + residual
                residual_idx += 1
        
        return x


def create_pinn(
    network_type: str = "standard",
    **kwargs
) -> PhysicsInformedNetwork:
    """Factory function to create PINN
    
    Args:
        network_type: Type of PINN to create
        **kwargs: Arguments for specific PINN type
        
    Returns:
        PINN instance
    """
    if network_type == "standard":
        return DynamicsNetwork(**kwargs)
    elif network_type == "constrained":
        return ConstrainedPINN(**kwargs)
    elif network_type == "residual":
        return ResidualPINN(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")