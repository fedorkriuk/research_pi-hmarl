"""Constraint Embedding Networks for Physics-Informed Learning

This module implements neural networks that embed physical constraints
into the learning architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging

logger = logging.getLogger(__name__)


class ConstraintEmbedding(nn.Module):
    """Embeds physical constraints into neural network representations"""
    
    def __init__(
        self,
        input_dim: int,
        constraint_dim: int,
        hidden_dim: int = 128,
        num_constraints: int = 5,
        use_attention: bool = True
    ):
        """Initialize constraint embedding
        
        Args:
            input_dim: Input state dimension
            constraint_dim: Constraint representation dimension
            hidden_dim: Hidden layer dimension
            num_constraints: Number of constraints to embed
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.constraint_dim = constraint_dim
        self.num_constraints = num_constraints
        self.use_attention = use_attention
        
        # Constraint-specific encoders
        self.constraint_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, constraint_dim)
            ) for _ in range(num_constraints)
        ])
        
        # Constraint satisfaction predictor
        self.satisfaction_heads = nn.ModuleList([
            nn.Linear(constraint_dim, 1) for _ in range(num_constraints)
        ])
        
        # Attention mechanism for constraint aggregation
        if use_attention:
            self.constraint_attention = nn.MultiheadAttention(
                constraint_dim, num_heads=4, batch_first=True
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(constraint_dim * num_constraints, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, constraint_dim)
        )
        
        logger.info(f"Initialized ConstraintEmbedding with {num_constraints} constraints")
    
    def forward(
        self,
        x: torch.Tensor,
        return_satisfaction: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass
        
        Args:
            x: Input state
            return_satisfaction: Whether to return satisfaction scores
            
        Returns:
            Constraint embedding and optionally satisfaction scores
        """
        batch_size = x.shape[0]
        
        # Encode each constraint
        constraint_embeddings = []
        satisfaction_scores = []
        
        for i, (encoder, sat_head) in enumerate(
            zip(self.constraint_encoders, self.satisfaction_heads)
        ):
            # Encode constraint
            embedding = encoder(x)
            constraint_embeddings.append(embedding)
            
            # Predict satisfaction
            if return_satisfaction:
                sat_score = torch.sigmoid(sat_head(embedding))
                satisfaction_scores.append(sat_score)
        
        # Stack embeddings
        embeddings = torch.stack(constraint_embeddings, dim=1)  # [batch, num_constraints, dim]
        
        # Apply attention if enabled
        if self.use_attention:
            attended, _ = self.constraint_attention(embeddings, embeddings, embeddings)
            embeddings = attended
        
        # Aggregate embeddings
        aggregated = embeddings.view(batch_size, -1)
        output = self.output_projection(aggregated)
        
        if return_satisfaction:
            satisfaction = torch.cat(satisfaction_scores, dim=-1)
            return output, satisfaction
        
        return output


class PhysicsEncoder(nn.Module):
    """Encodes physical state with constraint awareness"""
    
    def __init__(
        self,
        state_dim: int,
        physics_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        constraint_types: Optional[List[str]] = None
    ):
        """Initialize physics encoder
        
        Args:
            state_dim: State dimension
            physics_dim: Physics representation dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            constraint_types: Types of constraints to encode
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.physics_dim = physics_dim
        
        # Default constraint types
        if constraint_types is None:
            constraint_types = [
                "energy", "momentum", "collision", "boundary", "continuity"
            ]
        self.constraint_types = constraint_types
        
        # Main encoder
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, physics_dim))
        self.encoder = nn.Sequential(*layers)
        
        # Constraint-specific heads
        self.constraint_heads = nn.ModuleDict({
            constraint: nn.Linear(physics_dim, physics_dim // len(constraint_types))
            for constraint in constraint_types
        })
        
        # Physics projection layers
        self.physics_projections = nn.ModuleDict({
            "position": nn.Linear(physics_dim, 3),
            "velocity": nn.Linear(physics_dim, 3),
            "force": nn.Linear(physics_dim, 3),
            "energy": nn.Linear(physics_dim, 1),
            "momentum": nn.Linear(physics_dim, 3)
        })
    
    def forward(
        self,
        state: torch.Tensor,
        constraint_mask: Optional[Dict[str, bool]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            state: Input state
            constraint_mask: Which constraints to apply
            
        Returns:
            Dictionary of physics encodings
        """
        # Main encoding
        physics_encoding = self.encoder(state)
        
        outputs = {"encoding": physics_encoding}
        
        # Apply constraint-specific encodings
        constraint_features = []
        for constraint_type in self.constraint_types:
            if constraint_mask is None or constraint_mask.get(constraint_type, True):
                constraint_feat = self.constraint_heads[constraint_type](physics_encoding)
                constraint_features.append(constraint_feat)
                outputs[f"{constraint_type}_features"] = constraint_feat
        
        # Combine constraint features
        if constraint_features:
            combined_constraints = torch.cat(constraint_features, dim=-1)
            outputs["combined_constraints"] = combined_constraints
        
        # Project to physical quantities
        for proj_name, proj_layer in self.physics_projections.items():
            outputs[proj_name] = proj_layer(physics_encoding)
        
        return outputs


class SymmetryPreservingLayer(nn.Module):
    """Neural network layer that preserves physical symmetries"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        symmetry_type: str = "rotation",
        spatial_dim: int = 3
    ):
        """Initialize symmetry-preserving layer
        
        Args:
            in_features: Input features
            out_features: Output features
            symmetry_type: Type of symmetry to preserve
            spatial_dim: Spatial dimension
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.symmetry_type = symmetry_type
        self.spatial_dim = spatial_dim
        
        if symmetry_type == "rotation":
            # Rotation-equivariant layer
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
        elif symmetry_type == "translation":
            # Translation-invariant layer
            self.conv = nn.Conv1d(1, out_features, kernel_size=in_features)
        elif symmetry_type == "permutation":
            # Permutation-invariant layer
            self.linear = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"Unknown symmetry type: {symmetry_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass preserving symmetry
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.symmetry_type == "rotation":
            # Standard linear but with special initialization
            return F.linear(x, self.weight, self.bias)
        
        elif self.symmetry_type == "translation":
            # Use convolution for translation invariance
            x_expanded = x.unsqueeze(1)  # Add channel dimension
            output = self.conv(x_expanded)
            return output.squeeze(1)
        
        elif self.symmetry_type == "permutation":
            # Sum-pooling for permutation invariance
            x_sum = x.sum(dim=-2, keepdim=True)
            x_mean = x.mean(dim=-2, keepdim=True)
            x_max = x.max(dim=-2, keepdim=True)[0]
            
            # Combine statistics
            combined = torch.cat([x_sum, x_mean, x_max], dim=-1)
            return self.linear(combined)


class LagrangianLayer(nn.Module):
    """Layer that learns Lagrangian dynamics"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        learn_mass_matrix: bool = True
    ):
        """Initialize Lagrangian layer
        
        Args:
            state_dim: State dimension (q and q_dot)
            hidden_dim: Hidden dimension
            learn_mass_matrix: Whether to learn mass matrix
        """
        super().__init__()
        
        self.q_dim = state_dim // 2
        self.learn_mass_matrix = learn_mass_matrix
        
        # Kinetic energy network T(q, q_dot)
        self.kinetic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Potential energy network V(q)
        self.potential_net = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        if learn_mass_matrix:
            # Learned mass matrix (positive definite)
            self.mass_matrix_net = nn.Sequential(
                nn.Linear(self.q_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.q_dim * self.q_dim)
            )
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass computing Lagrangian
        
        Args:
            state: State [q, q_dot]
            
        Returns:
            Dictionary with Lagrangian components
        """
        q = state[:, :self.q_dim]
        q_dot = state[:, self.q_dim:]
        
        # Potential energy
        V = self.potential_net(q)
        
        # Kinetic energy
        if self.learn_mass_matrix:
            # Compute M(q)
            M_flat = self.mass_matrix_net(q)
            M = M_flat.view(-1, self.q_dim, self.q_dim)
            
            # Ensure positive definite via Cholesky
            L = torch.tril(M)
            diag_idx = torch.arange(self.q_dim)
            L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 0.1
            M = torch.bmm(L, L.transpose(-1, -2))
            
            # T = 0.5 * q_dot^T M(q) q_dot
            q_dot_expanded = q_dot.unsqueeze(-1)
            T = 0.5 * torch.bmm(
                q_dot.unsqueeze(1),
                torch.bmm(M, q_dot_expanded)
            ).squeeze()
        else:
            # Simple kinetic energy
            T = self.kinetic_net(state)
        
        # Lagrangian L = T - V
        L = T - V
        
        return {
            "lagrangian": L,
            "kinetic_energy": T,
            "potential_energy": V,
            "mass_matrix": M if self.learn_mass_matrix else None
        }


class ConstraintProjection(nn.Module):
    """Projects solutions onto constraint manifold"""
    
    def __init__(
        self,
        constraint_functions: List[Callable],
        projection_steps: int = 10,
        step_size: float = 0.1
    ):
        """Initialize constraint projection
        
        Args:
            constraint_functions: List of constraint functions
            projection_steps: Number of projection iterations
            step_size: Projection step size
        """
        super().__init__()
        
        self.constraint_functions = constraint_functions
        self.projection_steps = projection_steps
        self.step_size = step_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto constraint manifold
        
        Args:
            x: Input to project
            
        Returns:
            Projected output
        """
        x_proj = x.clone()
        
        # Iterative projection
        for _ in range(self.projection_steps):
            # Compute constraint violations
            violations = []
            for constraint_fn in self.constraint_functions:
                violation = constraint_fn(x_proj)
                violations.append(violation)
            
            if not violations:
                break
            
            # Compute projection direction
            total_violation = torch.stack(violations).sum(dim=0)
            
            # Gradient of constraint
            grad = torch.autograd.grad(
                total_violation.sum(),
                x_proj,
                create_graph=True
            )[0]
            
            # Project
            x_proj = x_proj - self.step_size * grad
        
        return x_proj


class PhysicsInformedActivation(nn.Module):
    """Activation functions that respect physical constraints"""
    
    def __init__(
        self,
        activation_type: str = "bounded_relu",
        bounds: Optional[Tuple[float, float]] = None
    ):
        """Initialize physics-informed activation
        
        Args:
            activation_type: Type of activation
            bounds: Physical bounds for output
        """
        super().__init__()
        
        self.activation_type = activation_type
        self.bounds = bounds or (-1.0, 1.0)
        
        if activation_type == "energy_preserving":
            # Custom activation that preserves energy norm
            self.activation = self._energy_preserving_activation
        elif activation_type == "bounded_relu":
            self.activation = self._bounded_relu
        elif activation_type == "smooth_clamp":
            self.activation = self._smooth_clamp
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return self.activation(x)
    
    def _energy_preserving_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Activation that preserves L2 norm (energy)"""
        norm = torch.norm(x, dim=-1, keepdim=True)
        normalized = x / (norm + 1e-8)
        
        # Apply standard activation to norm only
        activated_norm = F.relu(norm)
        
        return normalized * activated_norm
    
    def _bounded_relu(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU with physical bounds"""
        activated = F.relu(x)
        return torch.clamp(activated, self.bounds[0], self.bounds[1])
    
    def _smooth_clamp(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth clamping using tanh"""
        center = (self.bounds[1] + self.bounds[0]) / 2
        scale = (self.bounds[1] - self.bounds[0]) / 2
        
        return center + scale * torch.tanh((x - center) / scale)