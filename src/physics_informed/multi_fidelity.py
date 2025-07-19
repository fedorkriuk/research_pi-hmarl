"""Multi-Fidelity Physics Modeling

This module implements multi-fidelity physics models that combine
high and low-fidelity simulations for efficient learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class MultiFidelityPhysics(nn.Module):
    """Combines multiple fidelity levels of physics models"""
    
    def __init__(
        self,
        low_fidelity_model: nn.Module,
        high_fidelity_model: Optional[nn.Module] = None,
        num_fidelity_levels: int = 3,
        correlation_type: str = "nonlinear",
        adaptive_sampling: bool = True
    ):
        """Initialize multi-fidelity physics
        
        Args:
            low_fidelity_model: Low-fidelity (fast) model
            high_fidelity_model: High-fidelity (accurate) model
            num_fidelity_levels: Number of fidelity levels
            correlation_type: Type of correlation between levels
            adaptive_sampling: Whether to use adaptive sampling
        """
        super().__init__()
        
        self.num_levels = num_fidelity_levels
        self.correlation_type = correlation_type
        self.adaptive_sampling = adaptive_sampling
        
        # Fidelity models
        self.models = nn.ModuleList()
        self.models.append(low_fidelity_model)
        
        if high_fidelity_model is not None:
            self.models.append(high_fidelity_model)
        
        # Correlation models between fidelity levels
        self.correlation_models = nn.ModuleList()
        
        for i in range(self.num_levels - 1):
            if correlation_type == "linear":
                corr_model = LinearCorrelation()
            elif correlation_type == "nonlinear":
                corr_model = NonlinearCorrelation(
                    input_dim=low_fidelity_model.output_dim,
                    hidden_dim=128
                )
            else:
                corr_model = nn.Identity()
            
            self.correlation_models.append(corr_model)
        
        # Uncertainty quantification
        self.uncertainty_models = nn.ModuleList([
            UncertaintyEstimator(model.output_dim) 
            for model in self.models
        ])
        
        # Fidelity selector
        self.fidelity_selector = FidelitySelector(
            num_levels=num_fidelity_levels,
            adaptive=adaptive_sampling
        )
        
        logger.info(f"Initialized MultiFidelityPhysics with {num_fidelity_levels} levels")
    
    def forward(
        self,
        x: torch.Tensor,
        fidelity_level: Optional[int] = None,
        return_all_levels: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-fidelity model
        
        Args:
            x: Input
            fidelity_level: Specific fidelity level to use
            return_all_levels: Whether to return all fidelity outputs
            
        Returns:
            Dictionary of outputs
        """
        if fidelity_level is None:
            # Adaptive selection
            fidelity_level = self.fidelity_selector(x)
        
        outputs = {}
        
        # Low-fidelity prediction (always computed)
        y_low = self.models[0](x)
        uncertainty_low = self.uncertainty_models[0](x)
        
        outputs["low_fidelity"] = y_low
        outputs["low_uncertainty"] = uncertainty_low
        
        if fidelity_level > 0 or return_all_levels:
            # Higher fidelity predictions
            y_prev = y_low
            
            for level in range(1, min(fidelity_level + 1, len(self.models))):
                # Correlation model
                if level - 1 < len(self.correlation_models):
                    correlation = self.correlation_models[level - 1](x, y_prev)
                else:
                    correlation = y_prev
                
                # High-fidelity correction
                if level < len(self.models):
                    y_high = self.models[level](x)
                    y_corrected = y_high + correlation
                else:
                    # Extrapolate using correlation
                    y_corrected = y_prev + correlation
                
                # Uncertainty
                uncertainty = self.uncertainty_models[min(level, len(self.uncertainty_models) - 1)](x)
                
                outputs[f"level_{level}"] = y_corrected
                outputs[f"uncertainty_{level}"] = uncertainty
                
                y_prev = y_corrected
        
        # Final output
        outputs["prediction"] = outputs.get(f"level_{fidelity_level}", y_low)
        outputs["fidelity_level"] = fidelity_level
        
        return outputs
    
    def compute_correlation_loss(
        self,
        x: torch.Tensor,
        y_low: torch.Tensor,
        y_high: torch.Tensor
    ) -> torch.Tensor:
        """Compute correlation loss between fidelity levels
        
        Args:
            x: Input
            y_low: Low-fidelity output
            y_high: High-fidelity output
            
        Returns:
            Correlation loss
        """
        # Predict correlation
        if len(self.correlation_models) > 0:
            correlation = self.correlation_models[0](x, y_low)
            y_corrected = y_low + correlation
            
            # Loss
            loss = F.mse_loss(y_corrected, y_high)
        else:
            # Direct difference
            loss = F.mse_loss(y_low, y_high)
        
        return loss


class FidelitySelector(nn.Module):
    """Selects appropriate fidelity level based on input"""
    
    def __init__(
        self,
        num_levels: int,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        adaptive: bool = True
    ):
        """Initialize fidelity selector
        
        Args:
            num_levels: Number of fidelity levels
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            adaptive: Whether to use adaptive selection
        """
        super().__init__()
        
        self.num_levels = num_levels
        self.adaptive = adaptive
        
        if adaptive and input_dim is not None:
            # Neural network selector
            self.selector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_levels)
            )
        else:
            self.selector = None
        
        # Cost model for each fidelity level
        self.computational_costs = torch.tensor([1.0, 10.0, 100.0])[:num_levels]
        
        # Accuracy model
        self.expected_accuracy = torch.tensor([0.7, 0.9, 0.99])[:num_levels]
    
    def forward(
        self,
        x: torch.Tensor,
        accuracy_requirement: float = 0.9,
        budget: Optional[float] = None
    ) -> int:
        """Select fidelity level
        
        Args:
            x: Input
            accuracy_requirement: Required accuracy
            budget: Computational budget
            
        Returns:
            Selected fidelity level
        """
        if self.adaptive and self.selector is not None:
            # Neural selection
            logits = self.selector(x)
            
            # Apply constraints
            if budget is not None:
                # Mask out expensive options
                cost_mask = self.computational_costs <= budget
                logits = logits.masked_fill(~cost_mask, float('-inf'))
            
            # Select level
            level = torch.argmax(logits, dim=-1).item()
        else:
            # Rule-based selection
            # Find minimum level meeting accuracy requirement
            valid_levels = torch.where(self.expected_accuracy >= accuracy_requirement)[0]
            
            if len(valid_levels) > 0:
                if budget is not None:
                    # Filter by budget
                    affordable = valid_levels[
                        self.computational_costs[valid_levels] <= budget
                    ]
                    if len(affordable) > 0:
                        level = affordable[0].item()
                    else:
                        level = 0  # Fallback to lowest
                else:
                    level = valid_levels[0].item()
            else:
                level = self.num_levels - 1  # Use highest if none meet requirement
        
        return level


class LinearCorrelation(nn.Module):
    """Linear correlation model between fidelity levels"""
    
    def __init__(self):
        """Initialize linear correlation"""
        super().__init__()
        
        # Multiplicative and additive factors
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        x: torch.Tensor,
        y_low: torch.Tensor
    ) -> torch.Tensor:
        """Compute linear correlation
        
        Args:
            x: Input (not used in linear case)
            y_low: Low-fidelity output
            
        Returns:
            Correlation term
        """
        return self.scale * y_low + self.bias


class NonlinearCorrelation(nn.Module):
    """Nonlinear correlation model between fidelity levels"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        """Initialize nonlinear correlation
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim * 2  # Input + low-fidelity output
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.correlation_net = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        y_low: torch.Tensor
    ) -> torch.Tensor:
        """Compute nonlinear correlation
        
        Args:
            x: Input
            y_low: Low-fidelity output
            
        Returns:
            Correlation term
        """
        # Concatenate input and low-fidelity output
        combined = torch.cat([x, y_low], dim=-1)
        
        # Predict correction
        correction = self.correlation_net(combined)
        
        return correction


class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty for each fidelity level"""
    
    def __init__(
        self,
        input_dim: int,
        uncertainty_type: str = "aleatoric",
        hidden_dim: int = 64
    ):
        """Initialize uncertainty estimator
        
        Args:
            input_dim: Input dimension
            uncertainty_type: Type of uncertainty to model
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.uncertainty_type = uncertainty_type
        
        if uncertainty_type == "aleatoric":
            # Data uncertainty
            self.uncertainty_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Ensure positive
            )
        elif uncertainty_type == "epistemic":
            # Model uncertainty (using dropout)
            self.uncertainty_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            )
        else:
            # Combined uncertainty
            self.aleatoric_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            )
            self.epistemic_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            )
    
    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """Estimate uncertainty
        
        Args:
            x: Input
            num_samples: Number of samples for epistemic uncertainty
            
        Returns:
            Uncertainty estimate
        """
        if self.uncertainty_type in ["aleatoric", "epistemic"]:
            return self.uncertainty_net(x)
        else:
            # Combined uncertainty
            aleatoric = self.aleatoric_net(x)
            
            # Sample epistemic uncertainty
            epistemic_samples = []
            for _ in range(num_samples):
                epistemic_samples.append(self.epistemic_net(x))
            
            epistemic = torch.stack(epistemic_samples).var(dim=0)
            
            # Total uncertainty
            total_uncertainty = aleatoric + epistemic
            
            return total_uncertainty


class CoKriging(nn.Module):
    """Co-Kriging for multi-fidelity Gaussian process regression"""
    
    def __init__(
        self,
        kernel_type: str = "rbf",
        num_inducing_points: int = 100
    ):
        """Initialize Co-Kriging
        
        Args:
            kernel_type: Type of kernel function
            num_inducing_points: Number of inducing points
        """
        super().__init__()
        
        self.kernel_type = kernel_type
        self.num_inducing_points = num_inducing_points
        
        # Kernel parameters
        self.length_scale = nn.Parameter(torch.ones(1))
        self.variance = nn.Parameter(torch.ones(1))
        self.noise = nn.Parameter(torch.ones(1) * 0.01)
        
        # Cross-correlation parameter
        self.rho = nn.Parameter(torch.ones(1) * 0.9)
    
    def kernel(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> torch.Tensor:
        """Compute kernel matrix
        
        Args:
            x1: First input set
            x2: Second input set
            
        Returns:
            Kernel matrix
        """
        if self.kernel_type == "rbf":
            # RBF kernel
            dist = torch.cdist(x1, x2)
            K = self.variance * torch.exp(-0.5 * dist**2 / self.length_scale**2)
        elif self.kernel_type == "matern":
            # Matern 5/2 kernel
            dist = torch.cdist(x1, x2)
            sqrt5 = np.sqrt(5)
            K = self.variance * (1 + sqrt5 * dist / self.length_scale + 
                                5 * dist**2 / (3 * self.length_scale**2)) * \
                torch.exp(-sqrt5 * dist / self.length_scale)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return K
    
    def forward(
        self,
        x_low: torch.Tensor,
        y_low: torch.Tensor,
        x_high: torch.Tensor,
        y_high: torch.Tensor,
        x_test: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Co-Kriging prediction
        
        Args:
            x_low: Low-fidelity inputs
            y_low: Low-fidelity outputs
            x_high: High-fidelity inputs
            y_high: High-fidelity outputs
            x_test: Test inputs
            
        Returns:
            Mean and variance predictions
        """
        # Build co-Kriging covariance matrix
        n_low = x_low.shape[0]
        n_high = x_high.shape[0]
        
        # Auto-covariances
        K_ll = self.kernel(x_low, x_low) + self.noise * torch.eye(n_low)
        K_hh = self.kernel(x_high, x_high) + self.noise * torch.eye(n_high)
        
        # Cross-covariances
        K_lh = self.rho * self.kernel(x_low, x_high)
        K_hl = K_lh.T
        
        # Full covariance matrix
        K = torch.block_diag(K_ll, K_hh)
        K[:n_low, n_low:] = K_lh
        K[n_low:, :n_low] = K_hl
        
        # Observations
        y = torch.cat([y_low, y_high])
        
        # Test covariances
        k_star_l = self.kernel(x_test, x_low)
        k_star_h = self.rho * self.kernel(x_test, x_high)
        k_star = torch.cat([k_star_l, k_star_h], dim=1)
        
        # Prediction
        K_inv = torch.linalg.inv(K + 1e-6 * torch.eye(K.shape[0]))
        mean = k_star @ K_inv @ y
        
        # Variance
        k_star_star = self.kernel(x_test, x_test)
        var = torch.diag(k_star_star - k_star @ K_inv @ k_star.T)
        
        return mean, var


class AdaptiveSampling:
    """Adaptive sampling strategy for multi-fidelity models"""
    
    def __init__(
        self,
        acquisition_type: str = "expected_improvement",
        budget_allocation: str = "dynamic"
    ):
        """Initialize adaptive sampling
        
        Args:
            acquisition_type: Type of acquisition function
            budget_allocation: Budget allocation strategy
        """
        self.acquisition_type = acquisition_type
        self.budget_allocation = budget_allocation
    
    def select_next_sample(
        self,
        model: MultiFidelityPhysics,
        x_candidates: torch.Tensor,
        budget_remaining: float
    ) -> Tuple[int, int]:
        """Select next sample location and fidelity
        
        Args:
            model: Multi-fidelity model
            x_candidates: Candidate locations
            budget_remaining: Remaining computational budget
            
        Returns:
            Sample index and fidelity level
        """
        # Evaluate acquisition function
        acquisition_values = []
        
        for i, x in enumerate(x_candidates):
            # Get predictions at all fidelity levels
            outputs = model(x.unsqueeze(0), return_all_levels=True)
            
            # Compute acquisition value
            if self.acquisition_type == "expected_improvement":
                # Expected improvement considering cost
                uncertainty = outputs.get("uncertainty_2", outputs["low_uncertainty"])
                mean = outputs["prediction"]
                
                # Standard EI calculation
                z = mean / (uncertainty + 1e-6)
                ei = mean * torch.distributions.Normal(0, 1).cdf(z) + \
                     uncertainty * torch.distributions.Normal(0, 1).log_prob(z).exp()
                
                acquisition_values.append(ei.item())
            
            elif self.acquisition_type == "uncertainty":
                # Pure uncertainty sampling
                uncertainty = outputs.get("uncertainty_2", outputs["low_uncertainty"])
                acquisition_values.append(uncertainty.item())
        
        # Select best candidate
        best_idx = np.argmax(acquisition_values)
        
        # Select fidelity level based on budget
        if self.budget_allocation == "dynamic":
            # Allocate more budget to promising areas
            if acquisition_values[best_idx] > np.percentile(acquisition_values, 90):
                fidelity = 2  # High fidelity for promising areas
            elif acquisition_values[best_idx] > np.percentile(acquisition_values, 70):
                fidelity = 1  # Medium fidelity
            else:
                fidelity = 0  # Low fidelity for exploration
        else:
            # Fixed allocation
            fidelity = 0 if budget_remaining < 100 else 1
        
        return best_idx, fidelity