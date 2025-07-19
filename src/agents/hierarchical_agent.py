"""Hierarchical Agent Base Class

This module implements the base hierarchical agent framework with
meta-controller and execution policies using real-world constraints.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import logging

from .meta_controller import MetaController
from .execution_policy import ExecutionPolicy
from .temporal_abstraction import TemporalAbstraction
from .action_decomposer import ActionDecomposer
from .hierarchical_state import HierarchicalStateEncoder
from .communication_interfaces import MessagePassing, FeedbackLoop
from .skill_library import SkillLibrary

logger = logging.getLogger(__name__)


class HierarchicalAgent(nn.Module):
    """Base class for hierarchical agents with real-world constraints"""
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        meta_action_dim: int,
        primitive_action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        planning_horizon: float = 30.0,  # seconds (real mission planning)
        control_frequency: float = 20.0,  # Hz (real flight controller)
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize hierarchical agent
        
        Args:
            agent_id: Unique agent identifier
            state_dim: Dimension of state space
            meta_action_dim: Dimension of meta-controller action space
            primitive_action_dim: Dimension of execution policy action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of neural network layers
            planning_horizon: Meta-controller planning horizon (seconds)
            control_frequency: Execution policy control frequency (Hz)
            device: Computation device
        """
        super().__init__()
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.meta_action_dim = meta_action_dim
        self.primitive_action_dim = primitive_action_dim
        self.device = torch.device(device)
        
        # Real-world timing constraints
        self.planning_horizon = planning_horizon
        self.control_frequency = control_frequency
        self.meta_update_frequency = 1.0  # Hz (real command frequency)
        
        # Hierarchical components
        self.state_encoder = HierarchicalStateEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.meta_controller = MetaController(
            state_dim=hidden_dim,
            action_dim=meta_action_dim,
            hidden_dim=hidden_dim,
            planning_horizon=planning_horizon
        )
        
        self.execution_policy = ExecutionPolicy(
            state_dim=hidden_dim,
            action_dim=primitive_action_dim,
            hidden_dim=hidden_dim,
            control_frequency=control_frequency
        )
        
        self.temporal_abstraction = TemporalAbstraction(
            state_dim=hidden_dim,
            num_options=meta_action_dim
        )
        
        self.action_decomposer = ActionDecomposer(
            meta_action_dim=meta_action_dim,
            primitive_action_dim=primitive_action_dim
        )
        
        # Communication interfaces
        self.message_passing = MessagePassing(
            agent_id=agent_id,
            hidden_dim=hidden_dim
        )
        
        self.feedback_loop = FeedbackLoop(
            hidden_dim=hidden_dim
        )
        
        # Skill library
        self.skill_library = SkillLibrary()
        self._initialize_basic_skills()
        
        # Internal state
        self.current_option = None
        self.option_start_time = 0.0
        self.execution_buffer = []
        self.meta_state_history = []
        
        # Performance tracking
        self.meta_decision_count = 0
        self.execution_step_count = 0
        
        self.to(self.device)
        logger.info(f"Initialized HierarchicalAgent {agent_id}")
    
    def forward(
        self,
        state: torch.Tensor,
        messages: Optional[Dict[int, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through hierarchical agent
        
        Args:
            state: Current state observation
            messages: Messages from other agents
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and additional info
        """
        # Encode state hierarchically
        local_features, tactical_features, strategic_features = \
            self.state_encoder(state)
        
        # Process messages if available
        if messages:
            tactical_features = self.message_passing.process_messages(
                tactical_features, messages
            )
        
        # Meta-controller decision (at lower frequency)
        if self._should_update_meta_controller():
            meta_action, meta_info = self.meta_controller(
                strategic_features,
                deterministic=deterministic
            )
            
            # Select option based on meta-action
            self.current_option = self.temporal_abstraction.select_option(
                strategic_features, meta_action
            )
            self.option_start_time = self.execution_step_count / self.control_frequency
            
            self.meta_decision_count += 1
            self.meta_state_history.append({
                "features": strategic_features.detach(),
                "action": meta_action.detach(),
                "option": self.current_option
            })
        
        # Execution policy (at high frequency)
        if self.current_option is not None:
            # Get option-conditioned policy
            execution_features = self.temporal_abstraction.get_option_features(
                tactical_features, self.current_option
            )
            
            # Execute primitive action
            primitive_action, exec_info = self.execution_policy(
                execution_features,
                deterministic=deterministic
            )
            
            # Decompose action based on current option
            final_action = self.action_decomposer(
                primitive_action, self.current_option
            )
            
            # Send feedback to meta-controller
            self.feedback_loop.update(
                execution_features,
                primitive_action,
                exec_info.get("value", 0.0)
            )
        else:
            # Default safe action if no option selected
            final_action = torch.zeros(
                state.shape[0], self.primitive_action_dim,
                device=self.device
            )
            exec_info = {}
        
        self.execution_step_count += 1
        
        # Compile info
        info = {
            "meta_info": meta_info if self._should_update_meta_controller() else {},
            "exec_info": exec_info,
            "current_option": self.current_option,
            "option_progress": self._get_option_progress(),
            "hierarchy_level": "execution",
            "meta_decisions": self.meta_decision_count,
            "execution_steps": self.execution_step_count
        }
        
        return final_action, info
    
    def act(
        self,
        state: np.ndarray,
        messages: Optional[Dict[int, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action given state (numpy interface)
        
        Args:
            state: Current state observation
            messages: Messages from other agents
            
        Returns:
            Action and additional info
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if messages:
            message_tensors = {
                agent_id: torch.FloatTensor(msg).unsqueeze(0).to(self.device)
                for agent_id, msg in messages.items()
            }
        else:
            message_tensors = None
        
        # Get action
        with torch.no_grad():
            action_tensor, info = self.forward(
                state_tensor, message_tensors, deterministic=True
            )
        
        # Convert back to numpy
        action = action_tensor.squeeze(0).cpu().numpy()
        
        return action, info
    
    def update_option_status(self, state: torch.Tensor, reward: float):
        """Update option termination status
        
        Args:
            state: Current state
            reward: Current reward
        """
        if self.current_option is not None:
            # Check termination condition
            _, _, strategic_features = self.state_encoder(state)
            
            should_terminate = self.temporal_abstraction.check_termination(
                strategic_features,
                self.current_option,
                self.option_start_time,
                self.execution_step_count / self.control_frequency
            )
            
            if should_terminate:
                logger.info(f"Agent {self.agent_id}: Option {self.current_option} terminated")
                self.current_option = None
                
                # Update skill library with experience
                if len(self.meta_state_history) > 0:
                    self.skill_library.update_skill_performance(
                        self.meta_state_history[-1]["option"],
                        reward
                    )
    
    def get_communication_message(self) -> torch.Tensor:
        """Generate message for other agents
        
        Returns:
            Message tensor
        """
        if len(self.meta_state_history) > 0:
            # Share strategic features and current option
            latest_state = self.meta_state_history[-1]
            message = self.message_passing.create_message(
                latest_state["features"],
                self.current_option
            )
        else:
            # Empty message if no history
            message = torch.zeros(self.message_passing.message_dim, device=self.device)
        
        return message
    
    def _should_update_meta_controller(self) -> bool:
        """Check if meta-controller should be updated
        
        Returns:
            Whether to update meta-controller
        """
        # Update based on frequency
        steps_per_meta_update = int(self.control_frequency / self.meta_update_frequency)
        
        # Also update if no current option
        return (self.execution_step_count % steps_per_meta_update == 0 or
                self.current_option is None)
    
    def _get_option_progress(self) -> float:
        """Get progress of current option
        
        Returns:
            Progress percentage (0-1)
        """
        if self.current_option is None:
            return 0.0
        
        elapsed_time = self.execution_step_count / self.control_frequency - self.option_start_time
        expected_duration = self.temporal_abstraction.get_expected_duration(self.current_option)
        
        return min(1.0, elapsed_time / expected_duration)
    
    def _initialize_basic_skills(self):
        """Initialize basic skills in library"""
        # Takeoff skill
        self.skill_library.add_skill("takeoff", {
            "description": "Vertical takeoff to target altitude",
            "preconditions": ["on_ground", "motors_armed"],
            "postconditions": ["airborne", "stable_hover"],
            "expected_duration": 10.0,  # seconds
            "energy_cost": 50.0  # Joules
        })
        
        # Landing skill
        self.skill_library.add_skill("landing", {
            "description": "Controlled landing at target position",
            "preconditions": ["airborne", "low_altitude"],
            "postconditions": ["on_ground", "motors_idle"],
            "expected_duration": 15.0,
            "energy_cost": 30.0
        })
        
        # Formation skills
        self.skill_library.add_skill("join_formation", {
            "description": "Join formation at assigned position",
            "preconditions": ["airborne", "stable_flight"],
            "postconditions": ["in_formation"],
            "expected_duration": 20.0,
            "energy_cost": 40.0
        })
        
        # Navigation skills
        self.skill_library.add_skill("goto_waypoint", {
            "description": "Navigate to specified waypoint",
            "preconditions": ["airborne"],
            "postconditions": ["at_waypoint"],
            "expected_duration": 30.0,
            "energy_cost": 60.0
        })
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "agent_id": self.agent_id,
            "state_dict": self.state_dict(),
            "meta_decisions": self.meta_decision_count,
            "execution_steps": self.execution_step_count,
            "skill_library": self.skill_library.get_skills()
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint for agent {self.agent_id} to {path}")
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        self.meta_decision_count = checkpoint["meta_decisions"]
        self.execution_step_count = checkpoint["execution_steps"]
        
        # Restore skills
        for skill_name, skill_data in checkpoint["skill_library"].items():
            self.skill_library.add_skill(skill_name, skill_data)
        
        logger.info(f"Loaded checkpoint for agent {self.agent_id} from {path}")