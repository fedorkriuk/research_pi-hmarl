"""Physics Loss Functions for Multi-Constraint Optimization

This module implements physics-informed loss functions that combine
multiple physical constraints with adaptive weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class PhysicsLossCalculator:
    """Calculates and combines multiple physics-based losses"""
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        adaptive_weighting: bool = True,
        grad_norm_weighting: bool = False,
        uncertainty_weighting: bool = False
    ):
        """Initialize physics loss calculator
        
        Args:
            loss_weights: Initial weights for each loss term
            adaptive_weighting: Whether to use adaptive loss weighting
            grad_norm_weighting: Whether to use gradient normalization
            uncertainty_weighting: Whether to use uncertainty-based weighting
        """
        self.loss_weights = loss_weights or {
            "energy_conservation": 1.0,
            "momentum_conservation": 1.0,
            "collision_avoidance": 10.0,
            "pde_residual": 1.0,
            "boundary_condition": 1.0
        }
        
        self.adaptive_weighting = adaptive_weighting
        self.grad_norm_weighting = grad_norm_weighting
        self.uncertainty_weighting = uncertainty_weighting
        
        # For adaptive weighting
        if adaptive_weighting:
            self.loss_history = {key: [] for key in self.loss_weights}
            self.weight_history = {key: [w] for key, w in self.loss_weights.items()}
        
        # For uncertainty weighting
        if uncertainty_weighting:
            self.log_variances = nn.ParameterDict({
                key: nn.Parameter(torch.zeros(1))
                for key in self.loss_weights
            })
        
        logger.info(f"Initialized PhysicsLossCalculator with {len(self.loss_weights)} terms")
    
    def compute_total_loss(
        self,
        loss_dict: Dict[str, torch.Tensor],
        step: Optional[int] = None,
        return_weighted: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted total loss
        
        Args:
            loss_dict: Dictionary of individual losses
            step: Current training step
            return_weighted: Whether to return weighted losses
            
        Returns:
            Total loss and weighted loss components
        """
        # Update adaptive weights if enabled
        if self.adaptive_weighting and step is not None:
            self._update_adaptive_weights(loss_dict, step)
        
        # Compute weighted losses
        weighted_losses = {}
        total_loss = torch.tensor(0.0, device=list(loss_dict.values())[0].device)
        
        for key, loss in loss_dict.items():
            if key in self.loss_weights:
                weight = self._get_weight(key, loss)
                weighted_loss = weight * loss
                weighted_losses[key] = weighted_loss
                total_loss = total_loss + weighted_loss
            else:
                # Use default weight of 1.0 for unknown losses
                weighted_losses[key] = loss
                total_loss = total_loss + loss
        
        # Record history
        if self.adaptive_weighting:
            for key, loss in loss_dict.items():
                if key in self.loss_history:
                    self.loss_history[key].append(loss.detach().item())
        
        if return_weighted:
            return total_loss, weighted_losses
        else:
            return total_loss, loss_dict
    
    def _get_weight(self, key: str, loss: torch.Tensor) -> torch.Tensor:
        """Get weight for a specific loss term
        
        Args:
            key: Loss key
            loss: Loss value
            
        Returns:
            Weight
        """
        if self.uncertainty_weighting and key in self.log_variances:
            # Uncertainty-based weighting: w = 1/(2*σ²)
            log_var = self.log_variances[key]
            weight = 0.5 * torch.exp(-log_var)
            
            # Add regularization term: log(σ)
            weight = weight + 0.5 * log_var
        else:
            weight = self.loss_weights.get(key, 1.0)
        
        return weight
    
    def _update_adaptive_weights(
        self,
        loss_dict: Dict[str, torch.Tensor],
        step: int,
        update_freq: int = 100
    ):
        """Update weights adaptively based on loss magnitudes
        
        Args:
            loss_dict: Current losses
            step: Training step
            update_freq: How often to update weights
        """
        if step % update_freq != 0:
            return
        
        if self.grad_norm_weighting:
            # Gradient normalization approach
            self._update_grad_norm_weights(loss_dict)
        else:
            # Loss magnitude balancing
            self._update_magnitude_weights(loss_dict)
    
    def _update_magnitude_weights(self, loss_dict: Dict[str, torch.Tensor]):
        """Update weights based on loss magnitudes"""
        # Compute average loss magnitudes
        avg_losses = {}
        for key in self.loss_weights:
            if key in self.loss_history and len(self.loss_history[key]) > 10:
                avg_losses[key] = np.mean(self.loss_history[key][-100:])
        
        if len(avg_losses) < 2:
            return
        
        # Compute target (geometric mean)
        target = np.exp(np.mean([np.log(v + 1e-8) for v in avg_losses.values()]))
        
        # Update weights
        for key, avg_loss in avg_losses.items():
            if avg_loss > 0:
                # Scale weight to balance losses
                scale = target / (avg_loss + 1e-8)
                new_weight = self.loss_weights[key] * (0.9 + 0.1 * scale)
                
                # Clamp to reasonable range
                new_weight = np.clip(new_weight, 0.1, 100.0)
                
                self.loss_weights[key] = new_weight
                self.weight_history[key].append(new_weight)
    
    def _update_grad_norm_weights(self, loss_dict: Dict[str, torch.Tensor]):
        """Update weights based on gradient norms"""
        # This requires access to model parameters
        # Simplified version - normalize by loss variance
        loss_vars = {}
        
        for key in self.loss_weights:
            if key in self.loss_history and len(self.loss_history[key]) > 10:
                loss_vars[key] = np.var(self.loss_history[key][-100:])
        
        if len(loss_vars) < 2:
            return
        
        # Normalize weights by inverse variance
        total_inv_var = sum(1.0 / (v + 1e-8) for v in loss_vars.values())
        
        for key, var in loss_vars.items():
            inv_var = 1.0 / (var + 1e-8)
            self.loss_weights[key] = inv_var / total_inv_var * len(loss_vars)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loss statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "current_weights": dict(self.loss_weights),
            "loss_history_length": {
                key: len(history) for key, history in self.loss_history.items()
            } if self.adaptive_weighting else {}
        }
        
        # Add recent loss averages
        if self.adaptive_weighting:
            recent_avgs = {}
            for key, history in self.loss_history.items():
                if len(history) > 0:
                    recent_avgs[key] = np.mean(history[-100:])
            stats["recent_averages"] = recent_avgs
        
        return stats


class MultiPhysicsLoss(nn.Module):
    """Multi-physics loss combining different physical constraints"""
    
    def __init__(
        self,
        physics_modules: Dict[str, nn.Module],
        loss_calculator: Optional[PhysicsLossCalculator] = None
    ):
        """Initialize multi-physics loss
        
        Args:
            physics_modules: Dictionary of physics constraint modules
            loss_calculator: Loss calculator instance
        """
        super().__init__()
        
        self.physics_modules = nn.ModuleDict(physics_modules)
        self.loss_calculator = loss_calculator or PhysicsLossCalculator()
        
        # Loss function registry
        self.loss_functions = {
            "mse": F.mse_loss,
            "l1": F.l1_loss,
            "huber": F.smooth_l1_loss,
            "relative": self._relative_loss
        }
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        physics_state: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-physics loss
        
        Args:
            predictions: Model predictions
            targets: Target values
            physics_state: Physical state variables
            step: Training step
            
        Returns:
            Total loss and individual components
        """
        losses = {}
        
        # Data fitting losses
        for key in predictions:
            if key in targets:
                data_loss = self.loss_functions["mse"](
                    predictions[key], targets[key]
                )
                losses[f"data_{key}"] = data_loss
        
        # Physics constraint losses
        for name, module in self.physics_modules.items():
            if hasattr(module, "compute_loss"):
                physics_loss = module.compute_loss(physics_state)
                if isinstance(physics_loss, dict):
                    for loss_key, loss_val in physics_loss.items():
                        losses[f"{name}_{loss_key}"] = loss_val
                else:
                    losses[name] = physics_loss
        
        # Compute weighted total
        total_loss, weighted_losses = self.loss_calculator.compute_total_loss(
            losses, step
        )
        
        return total_loss, weighted_losses
    
    def _relative_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Relative error loss
        
        Args:
            pred: Predictions
            target: Targets
            epsilon: Small value for stability
            
        Returns:
            Relative loss
        """
        return torch.mean(
            torch.abs(pred - target) / (torch.abs(target) + epsilon)
        )


class PDEResidualLoss:
    """Computes PDE residual losses"""
    
    def __init__(
        self,
        pde_type: str = "heat",
        domain_size: Tuple[float, ...] = (1.0, 1.0),
        coefficients: Optional[Dict[str, float]] = None
    ):
        """Initialize PDE residual loss
        
        Args:
            pde_type: Type of PDE
            domain_size: Physical domain size
            coefficients: PDE coefficients
        """
        self.pde_type = pde_type
        self.domain_size = domain_size
        self.coefficients = coefficients or {}
        
        # PDE-specific defaults
        if pde_type == "heat":
            self.coefficients.setdefault("diffusivity", 0.1)
        elif pde_type == "wave":
            self.coefficients.setdefault("wave_speed", 1.0)
        elif pde_type == "navier_stokes":
            self.coefficients.setdefault("viscosity", 0.01)
            self.coefficients.setdefault("density", 1.0)
    
    def compute_residual(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        derivatives: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute PDE residual
        
        Args:
            u: Solution
            x: Spatial coordinates
            t: Time
            derivatives: Dictionary of derivatives
            
        Returns:
            PDE residual
        """
        if self.pde_type == "heat":
            return self._heat_equation_residual(u, derivatives)
        elif self.pde_type == "wave":
            return self._wave_equation_residual(u, derivatives)
        elif self.pde_type == "navier_stokes":
            return self._navier_stokes_residual(u, derivatives)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")
    
    def _heat_equation_residual(
        self,
        u: torch.Tensor,
        derivatives: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Heat equation: ∂u/∂t - α∇²u = 0"""
        du_dt = derivatives.get("u_t", torch.zeros_like(u))
        laplacian_u = derivatives.get("u_xx", torch.zeros_like(u))
        
        if laplacian_u.dim() > u.dim():
            laplacian_u = laplacian_u.sum(dim=-1)
        
        alpha = self.coefficients["diffusivity"]
        residual = du_dt - alpha * laplacian_u
        
        return residual
    
    def _wave_equation_residual(
        self,
        u: torch.Tensor,
        derivatives: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Wave equation: ∂²u/∂t² - c²∇²u = 0"""
        d2u_dt2 = derivatives.get("u_tt", torch.zeros_like(u))
        laplacian_u = derivatives.get("u_xx", torch.zeros_like(u))
        
        if laplacian_u.dim() > u.dim():
            laplacian_u = laplacian_u.sum(dim=-1)
        
        c = self.coefficients["wave_speed"]
        residual = d2u_dt2 - c**2 * laplacian_u
        
        return residual
    
    def _navier_stokes_residual(
        self,
        u: torch.Tensor,
        derivatives: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Navier-Stokes (simplified): ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u"""
        # This is a simplified version
        # Full implementation would require velocity components and pressure
        du_dt = derivatives.get("u_t", torch.zeros_like(u))
        grad_u = derivatives.get("u_x", torch.zeros_like(u))
        laplacian_u = derivatives.get("u_xx", torch.zeros_like(u))
        
        if laplacian_u.dim() > u.dim():
            laplacian_u = laplacian_u.sum(dim=-1)
        
        nu = self.coefficients["viscosity"]
        
        # Simplified: ignore pressure and nonlinear terms
        residual = du_dt - nu * laplacian_u
        
        return residual


class BoundaryConditionLoss:
    """Computes boundary condition losses"""
    
    def __init__(
        self,
        bc_type: str = "dirichlet",
        bc_weight: float = 100.0
    ):
        """Initialize boundary condition loss
        
        Args:
            bc_type: Type of boundary condition
            bc_weight: Weight for boundary loss
        """
        self.bc_type = bc_type
        self.bc_weight = bc_weight
    
    def compute_loss(
        self,
        u_boundary: torch.Tensor,
        boundary_values: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        normal_derivatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute boundary condition loss
        
        Args:
            u_boundary: Solution at boundary
            boundary_values: Target boundary values
            boundary_mask: Mask for boundary points
            normal_derivatives: Normal derivatives (for Neumann)
            
        Returns:
            Boundary loss
        """
        if self.bc_type == "dirichlet":
            # u = g on boundary
            loss = F.mse_loss(u_boundary, boundary_values)
        
        elif self.bc_type == "neumann":
            # ∂u/∂n = g on boundary
            if normal_derivatives is None:
                raise ValueError("Normal derivatives required for Neumann BC")
            loss = F.mse_loss(normal_derivatives, boundary_values)
        
        elif self.bc_type == "robin":
            # au + b∂u/∂n = g
            # Requires both u and ∂u/∂n
            if normal_derivatives is None:
                raise ValueError("Normal derivatives required for Robin BC")
            
            # Default coefficients
            a, b = 1.0, 1.0
            robin_value = a * u_boundary + b * normal_derivatives
            loss = F.mse_loss(robin_value, boundary_values)
        
        elif self.bc_type == "periodic":
            # u(x=0) = u(x=L)
            # Assumes first half is left boundary, second half is right
            n = u_boundary.shape[0] // 2
            loss = F.mse_loss(u_boundary[:n], u_boundary[n:])
        
        else:
            raise ValueError(f"Unknown BC type: {self.bc_type}")
        
        if boundary_mask is not None:
            loss = loss * boundary_mask.float().mean()
        
        return self.bc_weight * loss


class RegularizationLoss:
    """Physics-based regularization losses"""
    
    def __init__(self):
        """Initialize regularization losses"""
        pass
    
    def smoothness_loss(
        self,
        field: torch.Tensor,
        gradients: torch.Tensor,
        beta: float = 0.01
    ) -> torch.Tensor:
        """Smoothness regularization
        
        Args:
            field: Field values
            gradients: Field gradients
            beta: Smoothness weight
            
        Returns:
            Smoothness loss
        """
        # L2 norm of gradients
        grad_norm = (gradients ** 2).sum(dim=-1).mean()
        
        return beta * grad_norm
    
    def sparsity_loss(
        self,
        field: torch.Tensor,
        p: float = 1.0,
        beta: float = 0.01
    ) -> torch.Tensor:
        """Sparsity regularization
        
        Args:
            field: Field values
            p: Norm power (1 for L1, 2 for L2)
            beta: Sparsity weight
            
        Returns:
            Sparsity loss
        """
        if p == 1:
            return beta * torch.abs(field).mean()
        elif p == 2:
            return beta * (field ** 2).mean()
        else:
            return beta * (torch.abs(field) ** p).mean()
    
    def physics_informed_regularization(
        self,
        energy: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
        beta_energy: float = 0.01,
        beta_entropy: float = 0.01
    ) -> torch.Tensor:
        """Physics-informed regularization
        
        Args:
            energy: Energy values
            entropy: Entropy values
            beta_energy: Energy weight
            beta_entropy: Entropy weight
            
        Returns:
            Regularization loss
        """
        reg_loss = torch.tensor(0.0, device=energy.device)
        
        # Energy should be bounded
        energy_penalty = F.relu(energy - 1e6)  # Penalize very large energy
        reg_loss = reg_loss + beta_energy * energy_penalty.mean()
        
        # Entropy should be non-negative
        if entropy is not None:
            entropy_penalty = F.relu(-entropy)  # Penalize negative entropy
            reg_loss = reg_loss + beta_entropy * entropy_penalty.mean()
        
        return reg_loss