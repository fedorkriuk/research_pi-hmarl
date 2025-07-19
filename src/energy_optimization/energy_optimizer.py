"""Energy-Aware Optimization for Multi-Agent Systems

This module implements energy-aware optimization algorithms that balance
task performance with energy consumption.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class EnergyAwareOptimizer:
    """Optimizes actions considering energy constraints"""
    
    def __init__(
        self,
        action_dim: int,
        energy_weight: float = 0.3,
        min_energy_reserve: float = 0.2,  # 20% reserve
        optimization_method: str = "gradient",
        planning_horizon: int = 10
    ):
        """Initialize energy-aware optimizer
        
        Args:
            action_dim: Action dimension
            energy_weight: Weight for energy in objective
            min_energy_reserve: Minimum energy reserve ratio
            optimization_method: Optimization method
            planning_horizon: Planning horizon for optimization
        """
        self.action_dim = action_dim
        self.energy_weight = energy_weight
        self.min_energy_reserve = min_energy_reserve
        self.optimization_method = optimization_method
        self.planning_horizon = planning_horizon
        
        # Power consumption model
        self.power_model = PowerConsumptionModel(action_dim)
        
        # Value function approximator
        self.value_net = nn.Sequential(
            nn.Linear(action_dim + 2, 64),  # action + energy state
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Constraint handler
        self.constraint_handler = EnergyConstraintHandler(
            min_reserve=min_energy_reserve
        )
        
        logger.info(f"Initialized EnergyAwareOptimizer with {energy_weight:.2f} energy weight")
    
    def optimize_action(
        self,
        state: torch.Tensor,
        energy_state: Dict[str, torch.Tensor],
        task_value: Callable,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Optimize action considering energy
        
        Args:
            state: Current state
            energy_state: Energy state (soc, power, temperature)
            task_value: Function to evaluate task value
            constraints: Additional constraints
            
        Returns:
            Optimized action and metrics
        """
        if self.optimization_method == "gradient":
            return self._gradient_optimization(
                state, energy_state, task_value, constraints
            )
        elif self.optimization_method == "sampling":
            return self._sampling_optimization(
                state, energy_state, task_value, constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _gradient_optimization(
        self,
        state: torch.Tensor,
        energy_state: Dict[str, torch.Tensor],
        task_value: Callable,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Gradient-based optimization
        
        Args:
            state: Current state
            energy_state: Energy state
            task_value: Task value function
            constraints: Constraints
            
        Returns:
            Optimized action and metrics
        """
        # Initialize action
        action = torch.zeros(self.action_dim, requires_grad=True)
        
        # Optimizer
        optimizer = optim.Adam([action], lr=0.1)
        
        best_action = action.clone()
        best_value = float('-inf')
        
        metrics = {
            'iterations': 0,
            'task_values': [],
            'energy_costs': [],
            'total_values': []
        }
        
        # Optimization loop
        for i in range(50):
            optimizer.zero_grad()
            
            # Compute task value
            task_val = task_value(state, action)
            
            # Compute energy cost
            power = self.power_model(action)
            energy_cost = self._compute_energy_cost(
                power, energy_state, self.planning_horizon
            )
            
            # Total objective (maximize)
            total_value = task_val - self.energy_weight * energy_cost
            
            # Apply constraints
            if constraints:
                constraint_penalty = self.constraint_handler.compute_penalty(
                    action, energy_state, constraints
                )
                total_value = total_value - constraint_penalty
            
            # Gradient step
            loss = -total_value  # Minimize negative value
            loss.backward()
            optimizer.step()
            
            # Clamp action to valid range
            with torch.no_grad():
                action.clamp_(-1.0, 1.0)
            
            # Track best
            if total_value.item() > best_value:
                best_value = total_value.item()
                best_action = action.clone().detach()
            
            # Record metrics
            metrics['iterations'] = i + 1
            metrics['task_values'].append(task_val.item())
            metrics['energy_costs'].append(energy_cost.item())
            metrics['total_values'].append(total_value.item())
        
        # Final metrics
        final_power = self.power_model(best_action)
        metrics['final_power'] = final_power.item()
        metrics['energy_efficiency'] = task_val.item() / (final_power.item() + 1e-6)
        
        return best_action, metrics
    
    def _sampling_optimization(
        self,
        state: torch.Tensor,
        energy_state: Dict[str, torch.Tensor],
        task_value: Callable,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sampling-based optimization
        
        Args:
            state: Current state
            energy_state: Energy state
            task_value: Task value function
            constraints: Constraints
            
        Returns:
            Optimized action and metrics
        """
        num_samples = 100
        
        # Sample actions
        actions = torch.randn(num_samples, self.action_dim)
        actions = torch.tanh(actions)  # Bound to [-1, 1]
        
        # Evaluate all samples
        task_values = []
        energy_costs = []
        total_values = []
        
        for i in range(num_samples):
            action = actions[i]
            
            # Task value
            task_val = task_value(state, action)
            task_values.append(task_val)
            
            # Energy cost
            power = self.power_model(action)
            energy_cost = self._compute_energy_cost(
                power, energy_state, self.planning_horizon
            )
            energy_costs.append(energy_cost)
            
            # Total value
            total_val = task_val - self.energy_weight * energy_cost
            
            # Constraints
            if constraints:
                penalty = self.constraint_handler.compute_penalty(
                    action, energy_state, constraints
                )
                total_val = total_val - penalty
            
            total_values.append(total_val)
        
        # Select best action
        task_values = torch.stack(task_values)
        energy_costs = torch.stack(energy_costs)
        total_values = torch.stack(total_values)
        
        best_idx = torch.argmax(total_values)
        best_action = actions[best_idx]
        
        # Metrics
        metrics = {
            'num_samples': num_samples,
            'best_task_value': task_values[best_idx].item(),
            'best_energy_cost': energy_costs[best_idx].item(),
            'best_total_value': total_values[best_idx].item(),
            'mean_task_value': task_values.mean().item(),
            'mean_energy_cost': energy_costs.mean().item(),
            'final_power': self.power_model(best_action).item()
        }
        
        return best_action, metrics
    
    def _compute_energy_cost(
        self,
        power: torch.Tensor,
        energy_state: Dict[str, torch.Tensor],
        horizon: int
    ) -> torch.Tensor:
        """Compute energy cost over planning horizon
        
        Args:
            power: Power consumption
            energy_state: Current energy state
            horizon: Planning horizon
            
        Returns:
            Energy cost
        """
        soc = energy_state['soc']
        capacity = energy_state.get('capacity', 18.5)  # Wh
        
        # Energy consumed over horizon
        energy_consumed = power * horizon * 0.1 / 3600.0  # Wh
        
        # Remaining energy
        current_energy = soc * capacity
        remaining_energy = current_energy - energy_consumed
        
        # Cost increases as we approach minimum reserve
        min_energy = self.min_energy_reserve * capacity
        
        if remaining_energy < min_energy:
            # Heavy penalty for depleting reserve
            cost = 100.0 * (min_energy - remaining_energy) / capacity
        else:
            # Normal cost based on energy usage
            cost = energy_consumed / capacity
            
            # Add small penalty as SOC decreases
            soc_penalty = 0.1 * (1.0 - soc) ** 2
            cost = cost + soc_penalty
        
        return cost


class PowerConsumptionModel(nn.Module):
    """Models power consumption based on actions"""
    
    def __init__(
        self,
        action_dim: int,
        base_power: float = 50.0,  # W - hovering power
        max_power: float = 200.0   # W - maximum power
    ):
        """Initialize power consumption model
        
        Args:
            action_dim: Action dimension
            base_power: Base power consumption
            max_power: Maximum power consumption
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.base_power = base_power
        self.max_power = max_power
        
        # Neural network for complex power modeling
        self.power_net = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1
        )
        
        # Action-specific power coefficients
        self.register_buffer(
            'power_coefficients',
            torch.tensor([
                1.0,   # Forward/backward
                0.8,   # Left/right  
                1.2,   # Up/down (more power)
                0.6    # Rotation
            ])[:action_dim]
        )
    
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """Compute power consumption from action
        
        Args:
            action: Action vector
            
        Returns:
            Power consumption in Watts
        """
        # Base hovering power
        power = self.base_power
        
        # Action magnitude contribution
        action_magnitude = torch.norm(action, dim=-1, keepdim=True)
        
        # Weighted action power
        if action.dim() == 1:
            weighted_action = action * self.power_coefficients
        else:
            weighted_action = action * self.power_coefficients.unsqueeze(0)
        
        # Neural network prediction
        power_factor = self.power_net(action)
        
        # Total power
        additional_power = power_factor * (self.max_power - self.base_power)
        
        if action.dim() == 1:
            total_power = power + additional_power.squeeze()
        else:
            total_power = power + additional_power
        
        # Add quadratic term for aggressive maneuvers
        aggressive_penalty = 10.0 * action_magnitude ** 2
        total_power = total_power + aggressive_penalty.squeeze()
        
        return total_power


class EnergyObjective(nn.Module):
    """Multi-objective function for task and energy"""
    
    def __init__(
        self,
        task_weight: float = 0.7,
        energy_weight: float = 0.3,
        safety_weight: float = 0.1
    ):
        """Initialize energy objective
        
        Args:
            task_weight: Weight for task performance
            energy_weight: Weight for energy efficiency
            safety_weight: Weight for safety margins
        """
        super().__init__()
        
        self.task_weight = task_weight
        self.energy_weight = energy_weight
        self.safety_weight = safety_weight
        
        # Normalize weights
        total_weight = task_weight + energy_weight + safety_weight
        self.task_weight /= total_weight
        self.energy_weight /= total_weight
        self.safety_weight /= total_weight
    
    def forward(
        self,
        task_reward: torch.Tensor,
        energy_cost: torch.Tensor,
        safety_margin: torch.Tensor
    ) -> torch.Tensor:
        """Compute multi-objective value
        
        Args:
            task_reward: Task performance reward
            energy_cost: Energy consumption cost
            safety_margin: Safety margin (0-1)
            
        Returns:
            Combined objective value
        """
        # Normalize components
        task_term = self.task_weight * task_reward
        energy_term = -self.energy_weight * energy_cost
        safety_term = self.safety_weight * safety_margin
        
        # Combined objective
        objective = task_term + energy_term + safety_term
        
        return objective


class TaskEnergyTradeoff(nn.Module):
    """Learns task-energy tradeoff curves"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64
    ):
        """Initialize tradeoff model
        
        Args:
            state_dim: State dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        # Pareto front predictor
        self.pareto_net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # state + energy budget
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [achievable_task_value, min_energy]
        )
        
        # Tradeoff curve parameters
        self.tradeoff_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [a, b, c] for curve: task = a*exp(-b*energy) + c
        )
    
    def forward(
        self,
        state: torch.Tensor,
        energy_budget: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict task-energy tradeoff
        
        Args:
            state: Current state
            energy_budget: Available energy
            
        Returns:
            Tradeoff predictions
        """
        # Predict Pareto front point
        x = torch.cat([state, energy_budget.unsqueeze(-1)], dim=-1)
        pareto_point = self.pareto_net(x)
        
        achievable_task = pareto_point[..., 0]
        min_energy = pareto_point[..., 1]
        
        # Predict tradeoff curve parameters
        curve_params = self.tradeoff_net(state)
        a, b, c = curve_params[..., 0], curve_params[..., 1], curve_params[..., 2]
        
        # Ensure positive parameters
        a = F.softplus(a)
        b = F.softplus(b)
        
        return {
            'achievable_task_value': achievable_task,
            'minimum_energy': F.softplus(min_energy),
            'curve_a': a,
            'curve_b': b,
            'curve_c': c,
            'efficiency': achievable_task / (F.softplus(min_energy) + 1e-6)
        }
    
    def get_optimal_energy(
        self,
        state: torch.Tensor,
        task_requirement: torch.Tensor
    ) -> torch.Tensor:
        """Get optimal energy allocation for task requirement
        
        Args:
            state: Current state
            task_requirement: Required task performance
            
        Returns:
            Optimal energy allocation
        """
        # Get curve parameters
        outputs = self.forward(state, torch.tensor(1.0))  # Dummy budget
        a = outputs['curve_a']
        b = outputs['curve_b'] 
        c = outputs['curve_c']
        
        # Solve for energy: task = a*exp(-b*energy) + c
        # energy = -ln((task - c) / a) / b
        
        energy = -torch.log((task_requirement - c) / a + 1e-6) / b
        
        # Clamp to reasonable range
        energy = torch.clamp(energy, outputs['minimum_energy'], 1.0)
        
        return energy


class EnergyConstraint:
    """Handles energy-related constraints"""
    
    def __init__(
        self,
        min_landing_energy: float = 0.15,  # 15% for safe landing
        min_return_energy: float = 0.25,   # 25% for return to base
        temperature_limit: float = 60.0    # Max temperature in Celsius
    ):
        """Initialize energy constraints
        
        Args:
            min_landing_energy: Minimum SOC for emergency landing
            min_return_energy: Minimum SOC for return to base
            temperature_limit: Maximum safe temperature
        """
        self.min_landing_energy = min_landing_energy
        self.min_return_energy = min_return_energy
        self.temperature_limit = temperature_limit
    
    def check_constraints(
        self,
        energy_state: Dict[str, torch.Tensor],
        mission_phase: str = "operation"
    ) -> Dict[str, bool]:
        """Check if energy constraints are satisfied
        
        Args:
            energy_state: Current energy state
            mission_phase: Current mission phase
            
        Returns:
            Constraint satisfaction status
        """
        soc = energy_state['soc']
        temperature = energy_state.get('temperature', 25.0)
        
        constraints_met = {
            'landing_energy': soc >= self.min_landing_energy,
            'return_energy': soc >= self.min_return_energy,
            'temperature': temperature <= self.temperature_limit,
            'critical': soc >= 0.05  # 5% absolute minimum
        }
        
        # Phase-specific constraints
        if mission_phase == "return":
            constraints_met['mission'] = soc >= self.min_landing_energy
        elif mission_phase == "operation":
            constraints_met['mission'] = soc >= self.min_return_energy
        
        return constraints_met
    
    def get_safe_power_limit(
        self,
        energy_state: Dict[str, torch.Tensor],
        time_horizon: float = 60.0  # seconds
    ) -> torch.Tensor:
        """Get safe power limit to maintain constraints
        
        Args:
            energy_state: Current energy state
            time_horizon: Time horizon for power limit
            
        Returns:
            Maximum safe power
        """
        soc = energy_state['soc']
        capacity = energy_state.get('capacity', 18.5)  # Wh
        
        # Available energy above minimum reserve
        available_energy = (soc - self.min_return_energy) * capacity
        
        # Maximum power over time horizon
        max_power = available_energy * 3600.0 / time_horizon  # W
        
        # Temperature constraint
        temperature = energy_state.get('temperature', 25.0)
        temp_margin = self.temperature_limit - temperature
        
        # Reduce power if approaching temperature limit
        if temp_margin < 10.0:
            temp_factor = temp_margin / 10.0
            max_power = max_power * temp_factor
        
        return torch.clamp(max_power, min=0.0)


class PowerAllocation(nn.Module):
    """Allocates power among different subsystems"""
    
    def __init__(
        self,
        num_subsystems: int = 4,
        priority_learning: bool = True
    ):
        """Initialize power allocation
        
        Args:
            num_subsystems: Number of subsystems
            priority_learning: Whether to learn priorities
        """
        super().__init__()
        
        self.num_subsystems = num_subsystems
        
        # Default priorities
        self.default_priorities = torch.tensor([
            1.0,   # Propulsion
            0.8,   # Navigation
            0.6,   # Communication
            0.4    # Auxiliary
        ])[:num_subsystems]
        
        if priority_learning:
            # Learnable priority network
            self.priority_net = nn.Sequential(
                nn.Linear(num_subsystems + 2, 32),  # power requests + total available + SOC
                nn.ReLU(),
                nn.Linear(32, num_subsystems),
                nn.Softmax(dim=-1)
            )
        else:
            self.priority_net = None
    
    def forward(
        self,
        power_requests: torch.Tensor,
        available_power: torch.Tensor,
        soc: torch.Tensor
    ) -> torch.Tensor:
        """Allocate power to subsystems
        
        Args:
            power_requests: Power requested by each subsystem
            available_power: Total available power
            soc: State of charge
            
        Returns:
            Power allocated to each subsystem
        """
        total_request = power_requests.sum()
        
        if total_request <= available_power:
            # Can satisfy all requests
            return power_requests
        
        # Need to allocate based on priorities
        if self.priority_net is not None:
            # Learn priorities based on context
            x = torch.cat([
                power_requests,
                available_power.unsqueeze(-1),
                soc.unsqueeze(-1)
            ], dim=-1)
            priorities = self.priority_net(x)
        else:
            priorities = self.default_priorities
        
        # Weighted allocation
        weighted_requests = power_requests * priorities
        total_weighted = weighted_requests.sum()
        
        if total_weighted > 0:
            allocation = weighted_requests * available_power / total_weighted
        else:
            # Equal allocation if no weights
            allocation = torch.ones_like(power_requests) * available_power / self.num_subsystems
        
        # Ensure we don't exceed requests
        allocation = torch.minimum(allocation, power_requests)
        
        return allocation