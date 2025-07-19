"""Automatic Differentiation for Physics Computations

This module provides automatic differentiation utilities for computing
physics gradients and enforcing physical laws.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
import logging

logger = logging.getLogger(__name__)


class AutoDiffPhysics:
    """Automatic differentiation for physics computations"""
    
    def __init__(
        self,
        enable_second_order: bool = True,
        enable_mixed_derivatives: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize auto-differentiation physics
        
        Args:
            enable_second_order: Whether to compute second derivatives
            enable_mixed_derivatives: Whether to compute mixed partials
            dtype: Data type for computations
        """
        self.enable_second_order = enable_second_order
        self.enable_mixed_derivatives = enable_mixed_derivatives
        self.dtype = dtype
        
        logger.info("Initialized AutoDiffPhysics")
    
    def compute_derivatives(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        order: int = 1,
        dim: Optional[int] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute derivatives using automatic differentiation
        
        Args:
            outputs: Output tensor
            inputs: Input tensor (must have requires_grad=True)
            order: Derivative order (1 or 2)
            dim: Specific dimension to differentiate (None for all)
            
        Returns:
            Derivatives or dictionary of derivatives
        """
        if not inputs.requires_grad:
            inputs.requires_grad_(True)
        
        if order == 1:
            return self._compute_first_order(outputs, inputs, dim)
        elif order == 2:
            return self._compute_second_order(outputs, inputs, dim)
        else:
            raise ValueError(f"Unsupported derivative order: {order}")
    
    def _compute_first_order(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """Compute first-order derivatives
        
        Args:
            outputs: Output tensor
            inputs: Input tensor
            dim: Dimension to differentiate
            
        Returns:
            First derivatives
        """
        grad_outputs = torch.ones_like(outputs, dtype=self.dtype)
        
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if dim is not None:
            return gradients[..., dim]
        
        return gradients
    
    def _compute_second_order(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        dim: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute second-order derivatives
        
        Args:
            outputs: Output tensor
            inputs: Input tensor
            dim: Dimension to differentiate
            
        Returns:
            Dictionary of second derivatives
        """
        # First derivatives
        first_grads = self._compute_first_order(outputs, inputs)
        
        derivatives = {"first": first_grads}
        
        if not self.enable_second_order:
            return derivatives
        
        # Second derivatives
        if dim is not None:
            # Single dimension
            second_grad = self._compute_first_order(
                first_grads[..., dim], inputs, dim
            )
            derivatives["second"] = second_grad
        else:
            # All dimensions
            input_dim = inputs.shape[-1]
            second_grads = []
            
            for i in range(input_dim):
                grad_i = self._compute_first_order(
                    first_grads[..., i], inputs, i
                )
                second_grads.append(grad_i)
            
            derivatives["second"] = torch.stack(second_grads, dim=-1)
            
            # Mixed derivatives if enabled
            if self.enable_mixed_derivatives:
                mixed_grads = torch.zeros(
                    *inputs.shape[:-1], input_dim, input_dim,
                    dtype=self.dtype, device=inputs.device
                )
                
                for i in range(input_dim):
                    for j in range(input_dim):
                        if i != j:
                            mixed_grad = self._compute_first_order(
                                first_grads[..., i], inputs, j
                            )
                            mixed_grads[..., i, j] = mixed_grad
                
                derivatives["mixed"] = mixed_grads
        
        return derivatives
    
    def compute_jacobian(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute Jacobian matrix
        
        Args:
            outputs: Output tensor [batch, output_dim]
            inputs: Input tensor [batch, input_dim]
            
        Returns:
            Jacobian [batch, output_dim, input_dim]
        """
        batch_size = outputs.shape[0]
        output_dim = outputs.shape[-1]
        input_dim = inputs.shape[-1]
        
        jacobian = torch.zeros(
            batch_size, output_dim, input_dim,
            dtype=self.dtype, device=outputs.device
        )
        
        for i in range(output_dim):
            grad_i = torch.autograd.grad(
                outputs=outputs[:, i].sum(),
                inputs=inputs,
                create_graph=True,
                retain_graph=True
            )[0]
            jacobian[:, i, :] = grad_i
        
        return jacobian
    
    def compute_hessian(
        self,
        output: torch.Tensor,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hessian matrix
        
        Args:
            output: Scalar output tensor
            inputs: Input tensor [batch, input_dim]
            
        Returns:
            Hessian [batch, input_dim, input_dim]
        """
        if output.numel() != 1:
            raise ValueError("Hessian requires scalar output")
        
        input_dim = inputs.shape[-1]
        batch_size = inputs.shape[0]
        
        # First derivatives
        first_grads = torch.autograd.grad(
            outputs=output,
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivatives
        hessian = torch.zeros(
            batch_size, input_dim, input_dim,
            dtype=self.dtype, device=inputs.device
        )
        
        for i in range(input_dim):
            second_grads = torch.autograd.grad(
                outputs=first_grads[:, i].sum(),
                inputs=inputs,
                create_graph=True,
                retain_graph=True
            )[0]
            hessian[:, i, :] = second_grads
        
        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
        
        return hessian


class PhysicsGradients:
    """Compute physics-specific gradients"""
    
    def __init__(self, spatial_dim: int = 3):
        """Initialize physics gradients
        
        Args:
            spatial_dim: Spatial dimension (2D or 3D)
        """
        self.spatial_dim = spatial_dim
        self.autodiff = AutoDiffPhysics()
    
    def gradient(
        self,
        scalar_field: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient of scalar field
        
        Args:
            scalar_field: Scalar field values [batch, 1]
            positions: Positions [batch, spatial_dim]
            
        Returns:
            Gradient vector [batch, spatial_dim]
        """
        return self.autodiff.compute_derivatives(scalar_field, positions, order=1)
    
    def divergence(
        self,
        vector_field: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute divergence of vector field
        
        Args:
            vector_field: Vector field [batch, spatial_dim]
            positions: Positions [batch, spatial_dim]
            
        Returns:
            Divergence [batch, 1]
        """
        div = torch.zeros(positions.shape[0], 1, device=positions.device)
        
        for i in range(self.spatial_dim):
            grad_i = self.autodiff.compute_derivatives(
                vector_field[:, i:i+1], positions, dim=i
            )
            div += grad_i
        
        return div
    
    def curl(
        self,
        vector_field: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute curl of vector field (3D only)
        
        Args:
            vector_field: Vector field [batch, 3]
            positions: Positions [batch, 3]
            
        Returns:
            Curl [batch, 3]
        """
        if self.spatial_dim != 3:
            raise ValueError("Curl is only defined in 3D")
        
        # Compute all partial derivatives
        dvx_dy = self.autodiff.compute_derivatives(
            vector_field[:, 0:1], positions, dim=1
        )
        dvx_dz = self.autodiff.compute_derivatives(
            vector_field[:, 0:1], positions, dim=2
        )
        dvy_dx = self.autodiff.compute_derivatives(
            vector_field[:, 1:2], positions, dim=0
        )
        dvy_dz = self.autodiff.compute_derivatives(
            vector_field[:, 1:2], positions, dim=2
        )
        dvz_dx = self.autodiff.compute_derivatives(
            vector_field[:, 2:3], positions, dim=0
        )
        dvz_dy = self.autodiff.compute_derivatives(
            vector_field[:, 2:3], positions, dim=1
        )
        
        # Curl components
        curl_x = dvy_dz - dvz_dy
        curl_y = dvz_dx - dvx_dz
        curl_z = dvx_dy - dvy_dx
        
        return torch.cat([curl_x, curl_y, curl_z], dim=1)
    
    def laplacian(
        self,
        field: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute Laplacian of field
        
        Args:
            field: Scalar or vector field
            positions: Positions
            
        Returns:
            Laplacian
        """
        derivatives = self.autodiff.compute_derivatives(
            field, positions, order=2
        )
        
        # Sum of second derivatives
        laplacian = derivatives["second"].sum(dim=-1, keepdim=True)
        
        return laplacian


class PhysicsOperators(nn.Module):
    """Neural network operators for physics computations"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1
    ):
        """Initialize physics operators
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        # Gradient operator network
        self.gradient_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim * input_dim)
        )
        
        # Laplacian operator network
        self.laplacian_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def compute_gradient(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute learned gradient
        
        Args:
            x: Positions
            u: Field values
            
        Returns:
            Gradient
        """
        # Concatenate position and field
        inp = torch.cat([x, u], dim=-1)
        
        # Compute gradient
        grad = self.gradient_net(inp)
        grad = grad.view(-1, self.output_dim, self.input_dim)
        
        return grad
    
    def compute_laplacian(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute learned Laplacian
        
        Args:
            x: Positions
            u: Field values
            
        Returns:
            Laplacian
        """
        inp = torch.cat([x, u], dim=-1)
        return self.laplacian_net(inp)


class AutoDiffLoss:
    """Loss functions using automatic differentiation"""
    
    def __init__(self):
        """Initialize autodiff loss"""
        self.autodiff = AutoDiffPhysics()
        self.physics_grad = PhysicsGradients()
    
    def pde_loss(
        self,
        pde_func: Callable,
        u: torch.Tensor,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute PDE residual loss
        
        Args:
            pde_func: PDE function that returns residual
            u: Solution
            x: Spatial coordinates
            t: Time coordinate
            
        Returns:
            PDE loss
        """
        # Compute derivatives
        if t is not None:
            # Time-dependent PDE
            du_dt = self.autodiff.compute_derivatives(u, t)
            du_dx = self.autodiff.compute_derivatives(u, x)
            d2u_dx2 = self.autodiff.compute_derivatives(u, x, order=2)["second"]
            
            residual = pde_func(u, x, t, du_dt, du_dx, d2u_dx2)
        else:
            # Steady-state PDE
            du_dx = self.autodiff.compute_derivatives(u, x)
            d2u_dx2 = self.autodiff.compute_derivatives(u, x, order=2)["second"]
            
            residual = pde_func(u, x, du_dx, d2u_dx2)
        
        return F.mse_loss(residual, torch.zeros_like(residual))
    
    def boundary_loss(
        self,
        u: torch.Tensor,
        x_boundary: torch.Tensor,
        boundary_values: torch.Tensor,
        boundary_type: str = "dirichlet"
    ) -> torch.Tensor:
        """Compute boundary condition loss
        
        Args:
            u: Solution at boundary
            x_boundary: Boundary positions
            boundary_values: Target boundary values
            boundary_type: Type of boundary condition
            
        Returns:
            Boundary loss
        """
        if boundary_type == "dirichlet":
            # u = g on boundary
            return F.mse_loss(u, boundary_values)
        
        elif boundary_type == "neumann":
            # du/dn = g on boundary
            # Compute normal derivative
            du_dx = self.autodiff.compute_derivatives(u, x_boundary)
            
            # Assume last dimension is normal direction
            normal_derivative = du_dx[..., -1:]
            
            return F.mse_loss(normal_derivative, boundary_values)
        
        elif boundary_type == "robin":
            # au + b*du/dn = g
            # Need coefficients a, b
            raise NotImplementedError("Robin boundary conditions not implemented")
        
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    def conservation_loss(
        self,
        quantity: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute conservation law loss
        
        Args:
            quantity: Conserved quantity
            x: Spatial coordinates
            t: Time
            
        Returns:
            Conservation loss
        """
        # Time derivative
        dq_dt = self.autodiff.compute_derivatives(quantity, t)
        
        # Spatial flux divergence
        # Assuming quantity has flux in last dimensions
        if quantity.shape[-1] > 1:
            # Vector quantity
            div_flux = self.physics_grad.divergence(quantity, x)
        else:
            # Scalar quantity - compute gradient flux
            grad_q = self.physics_grad.gradient(quantity, x)
            div_flux = self.physics_grad.divergence(grad_q, x)
        
        # Conservation: dq/dt + div(flux) = 0
        residual = dq_dt + div_flux
        
        return F.mse_loss(residual, torch.zeros_like(residual))