"""Reward Calculator for Multi-Agent Environment

This module implements multi-objective reward calculation including
task performance, energy efficiency, and physics constraint satisfaction.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # Task rewards
    task_completion_reward: float = 100.0
    progress_reward_scale: float = 10.0
    target_reached_reward: float = 50.0
    formation_maintained_reward: float = 5.0
    
    # Energy rewards/penalties  
    energy_efficiency_weight: float = 0.1
    battery_depletion_penalty: float = -100.0
    low_battery_penalty_threshold: float = 0.2
    
    # Physics constraint penalties
    collision_penalty: float = -50.0
    velocity_violation_penalty: float = -5.0
    acceleration_violation_penalty: float = -5.0
    boundary_violation_penalty: float = -10.0
    
    # Communication rewards
    connectivity_reward: float = 1.0
    information_sharing_reward: float = 2.0
    
    # Cooperation rewards
    coordination_reward: float = 5.0
    helping_reward: float = 10.0
    
    # General parameters
    time_penalty: float = -0.01
    distance_penalty_weight: float = 0.01
    sparse_rewards: bool = False
    normalize_rewards: bool = True
    
    # Multi-objective weights
    task_weight: float = 0.6
    energy_weight: float = 0.2
    safety_weight: float = 0.15
    cooperation_weight: float = 0.05


class RewardCalculator:
    """Calculates rewards for multi-agent scenarios"""
    
    def __init__(
        self,
        reward_type: str = "multi_objective",
        sparse_rewards: bool = False,
        config: Optional[RewardConfig] = None
    ):
        """Initialize reward calculator
        
        Args:
            reward_type: Type of reward (multi_objective, task_only, shaped)
            sparse_rewards: Whether to use sparse rewards
            config: Reward configuration
        """
        self.reward_type = reward_type
        self.sparse_rewards = sparse_rewards
        self.config = config or RewardConfig(sparse_rewards=sparse_rewards)
        
        # Reward history for normalization
        self.reward_history = []
        self.reward_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": -100.0,
            "max": 100.0
        }
        
        logger.info(f"Initialized RewardCalculator (type: {reward_type})")
    
    def calculate_rewards(
        self,
        state_manager: Any,
        agent_manager: Any,
        episode_manager: Any,
        actions: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        """Calculate rewards for all agents
        
        Args:
            state_manager: State manager instance
            agent_manager: Agent manager instance
            episode_manager: Episode manager instance
            actions: Agent actions taken
            
        Returns:
            Dictionary mapping agent_id to reward
        """
        rewards = {}
        
        for agent_id in agent_manager.active_agents:
            if self.reward_type == "multi_objective":
                reward = self._calculate_multi_objective_reward(
                    agent_id, state_manager, agent_manager, episode_manager, actions
                )
            elif self.reward_type == "task_only":
                reward = self._calculate_task_reward(
                    agent_id, state_manager, agent_manager, episode_manager
                )
            elif self.reward_type == "shaped":
                reward = self._calculate_shaped_reward(
                    agent_id, state_manager, agent_manager, episode_manager, actions
                )
            else:
                reward = 0.0
            
            # Apply normalization if configured
            if self.config.normalize_rewards:
                reward = self._normalize_reward(reward)
            
            rewards[agent_id] = reward
        
        # Update reward statistics
        self._update_reward_stats(list(rewards.values()))
        
        return rewards
    
    def _calculate_multi_objective_reward(
        self,
        agent_id: int,
        state_manager: Any,
        agent_manager: Any,
        episode_manager: Any,
        actions: Dict[int, np.ndarray]
    ) -> float:
        """Calculate multi-objective reward
        
        Args:
            agent_id: Agent ID
            state_manager: State manager
            agent_manager: Agent manager
            episode_manager: Episode manager
            actions: Actions taken
            
        Returns:
            Total reward
        """
        # Get individual reward components
        task_reward = self._calculate_task_reward(
            agent_id, state_manager, agent_manager, episode_manager
        )
        
        energy_reward = self._calculate_energy_reward(
            agent_id, state_manager, actions
        )
        
        safety_reward = self._calculate_safety_reward(
            agent_id, state_manager
        )
        
        cooperation_reward = self._calculate_cooperation_reward(
            agent_id, state_manager, agent_manager
        )
        
        # Weighted combination
        total_reward = (
            self.config.task_weight * task_reward +
            self.config.energy_weight * energy_reward +
            self.config.safety_weight * safety_reward +
            self.config.cooperation_weight * cooperation_reward
        )
        
        return total_reward
    
    def _calculate_task_reward(
        self,
        agent_id: int,
        state_manager: Any,
        agent_manager: Any,
        episode_manager: Any
    ) -> float:
        """Calculate task-specific reward
        
        Args:
            agent_id: Agent ID
            state_manager: State manager
            agent_manager: Agent manager
            episode_manager: Episode manager
            
        Returns:
            Task reward
        """
        reward = 0.0
        
        # Get agent state
        agent_state = state_manager.get_agent_state(agent_id)
        position = agent_state["position"]
        
        # Get scenario-specific rewards from episode manager
        scenario_rewards = episode_manager.get_agent_rewards(agent_id)
        
        if scenario_rewards:
            # Target reached
            if scenario_rewards.get("target_reached", False):
                reward += self.config.target_reached_reward
            
            # Progress towards target
            if "distance_to_target" in scenario_rewards:
                prev_distance = scenario_rewards.get("prev_distance_to_target", float('inf'))
                curr_distance = scenario_rewards["distance_to_target"]
                progress = prev_distance - curr_distance
                
                if not self.sparse_rewards:
                    reward += self.config.progress_reward_scale * progress
            
            # Formation maintenance
            if "formation_error" in scenario_rewards:
                formation_error = scenario_rewards["formation_error"]
                if formation_error < 2.0:  # Within tolerance
                    reward += self.config.formation_maintained_reward
            
            # Mission completion
            if scenario_rewards.get("mission_completed", False):
                reward += self.config.task_completion_reward
        
        # Time penalty (encourage efficiency)
        if not self.sparse_rewards:
            reward += self.config.time_penalty
        
        return reward
    
    def _calculate_energy_reward(
        self,
        agent_id: int,
        state_manager: Any,
        actions: Dict[int, np.ndarray]
    ) -> float:
        """Calculate energy-related reward
        
        Args:
            agent_id: Agent ID
            state_manager: State manager
            actions: Actions taken
            
        Returns:
            Energy reward
        """
        reward = 0.0
        
        agent_state = state_manager.get_agent_state(agent_id)
        battery_soc = agent_state["battery_soc"]
        power_consumption = agent_state["power_consumption"]
        
        # Battery depletion penalty
        if battery_soc <= 0.0:
            reward += self.config.battery_depletion_penalty
        elif battery_soc < self.config.low_battery_penalty_threshold:
            # Progressive penalty as battery depletes
            penalty_factor = (self.config.low_battery_penalty_threshold - battery_soc) / \
                           self.config.low_battery_penalty_threshold
            reward -= 10.0 * penalty_factor
        
        # Energy efficiency reward
        if not self.sparse_rewards and agent_id in actions:
            # Reward efficient actions (low power for given velocity)
            velocity = np.linalg.norm(agent_state["velocity"])
            if velocity > 0.1:
                efficiency = velocity / (power_consumption + 1.0)
                reward += self.config.energy_efficiency_weight * efficiency
        
        return reward
    
    def _calculate_safety_reward(
        self,
        agent_id: int,
        state_manager: Any
    ) -> float:
        """Calculate safety-related reward
        
        Args:
            agent_id: Agent ID
            state_manager: State manager
            
        Returns:
            Safety reward
        """
        reward = 0.0
        
        # Get constraint violations
        violations = state_manager.get_constraint_violations(agent_id)
        agent_state = state_manager.get_agent_state(agent_id)
        
        # Collision penalty
        if violations.get("collision", False) or agent_state["collision"]:
            reward += self.config.collision_penalty
        
        # Velocity violation penalty
        if violations.get("velocity", False):
            reward += self.config.velocity_violation_penalty
        
        # Acceleration violation penalty
        if violations.get("acceleration", False):
            reward += self.config.acceleration_violation_penalty
        
        # Boundary violation penalty
        if violations.get("altitude", False):
            reward += self.config.boundary_violation_penalty
        
        # Reward maintaining safe separation
        if not self.sparse_rewards:
            min_separation = agent_state["min_separation"]
            if 2.0 < min_separation < 10.0:  # Safe range
                reward += 0.1 * (min_separation - 2.0)
        
        return reward
    
    def _calculate_cooperation_reward(
        self,
        agent_id: int,
        state_manager: Any,
        agent_manager: Any
    ) -> float:
        """Calculate cooperation-related reward
        
        Args:
            agent_id: Agent ID
            state_manager: State manager
            agent_manager: Agent manager
            
        Returns:
            Cooperation reward
        """
        reward = 0.0
        
        if self.sparse_rewards:
            return reward
        
        # Get nearby agents
        position = state_manager.get_agent_state(agent_id)["position"]
        nearby_agents = agent_manager.get_nearby_agents(agent_id, radius=20.0)
        
        # Connectivity reward
        if len(nearby_agents) > 0:
            reward += self.config.connectivity_reward * min(len(nearby_agents), 3)
        
        # Formation coordination
        if len(nearby_agents) >= 2:
            # Check if maintaining formation
            positions = [position] + [
                state_manager.get_agent_state(aid)["position"]
                for aid in nearby_agents[:2]
            ]
            
            # Simple formation check (triangle)
            distances = []
            for i in range(3):
                for j in range(i+1, 3):
                    distances.append(np.linalg.norm(positions[i] - positions[j]))
            
            if all(3.0 < d < 10.0 for d in distances):
                reward += self.config.coordination_reward
        
        return reward
    
    def _calculate_shaped_reward(
        self,
        agent_id: int,
        state_manager: Any,
        agent_manager: Any,
        episode_manager: Any,
        actions: Dict[int, np.ndarray]
    ) -> float:
        """Calculate shaped reward with dense feedback
        
        Args:
            agent_id: Agent ID
            state_manager: State manager
            agent_manager: Agent manager
            episode_manager: Episode manager
            actions: Actions taken
            
        Returns:
            Shaped reward
        """
        # Start with multi-objective reward
        reward = self._calculate_multi_objective_reward(
            agent_id, state_manager, agent_manager, episode_manager, actions
        )
        
        # Add additional shaping terms
        agent_state = state_manager.get_agent_state(agent_id)
        
        # Velocity shaping (encourage movement)
        velocity = np.linalg.norm(agent_state["velocity"])
        if velocity > 0.5:
            reward += 0.01 * velocity
        
        # Height maintenance (for aerial vehicles)
        height = agent_state["position"][2]
        if 5.0 < height < 20.0:  # Preferred altitude range
            reward += 0.1
        
        # Action smoothness
        if agent_id in actions and hasattr(self, "_prev_actions"):
            if agent_id in self._prev_actions:
                action_change = np.linalg.norm(
                    actions[agent_id] - self._prev_actions[agent_id]
                )
                reward -= 0.01 * action_change
        
        # Store actions for next step
        if not hasattr(self, "_prev_actions"):
            self._prev_actions = {}
        if agent_id in actions:
            self._prev_actions[agent_id] = actions[agent_id].copy()
        
        return reward
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward based on statistics
        
        Args:
            reward: Raw reward value
            
        Returns:
            Normalized reward
        """
        # Clip extreme values
        reward = np.clip(reward, self.reward_stats["min"], self.reward_stats["max"])
        
        # Z-score normalization
        if self.reward_stats["std"] > 0:
            normalized = (reward - self.reward_stats["mean"]) / self.reward_stats["std"]
        else:
            normalized = reward
        
        # Scale to reasonable range
        return np.clip(normalized, -10.0, 10.0)
    
    def _update_reward_stats(self, rewards: List[float]):
        """Update reward statistics for normalization
        
        Args:
            rewards: List of rewards from current step
        """
        self.reward_history.extend(rewards)
        
        # Keep limited history
        if len(self.reward_history) > 10000:
            self.reward_history = self.reward_history[-10000:]
        
        # Update statistics
        if len(self.reward_history) > 100:
            self.reward_stats["mean"] = np.mean(self.reward_history)
            self.reward_stats["std"] = np.std(self.reward_history)
            self.reward_stats["min"] = np.percentile(self.reward_history, 1)
            self.reward_stats["max"] = np.percentile(self.reward_history, 99)
    
    def get_reward_info(self, agent_id: int) -> Dict[str, float]:
        """Get detailed reward breakdown for an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dictionary of reward components
        """
        # This would be filled during reward calculation
        # For now, return empty dict
        return {
            "task_reward": 0.0,
            "energy_reward": 0.0,
            "safety_reward": 0.0,
            "cooperation_reward": 0.0,
            "total_reward": 0.0
        }