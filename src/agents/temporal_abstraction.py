"""Temporal Abstraction with Options and Skills

This module implements temporal abstraction using the options framework
for hierarchical reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Option:
    """Represents a temporal abstraction option"""
    
    def __init__(
        self,
        option_id: int,
        name: str,
        initiation_set: Optional[callable] = None,
        policy: Optional[nn.Module] = None,
        termination_condition: Optional[callable] = None,
        expected_duration: float = 10.0
    ):
        """Initialize option
        
        Args:
            option_id: Unique option identifier
            name: Option name
            initiation_set: Function to check if option can be initiated
            policy: Option-specific policy network
            termination_condition: Function to check termination
            expected_duration: Expected duration in seconds
        """
        self.option_id = option_id
        self.name = name
        self.initiation_set = initiation_set or (lambda s: True)
        self.policy = policy
        self.termination_condition = termination_condition or (lambda s, t: t > expected_duration)
        self.expected_duration = expected_duration
        
        # Statistics
        self.execution_count = 0
        self.success_count = 0
        self.total_duration = 0.0
    
    def can_initiate(self, state: torch.Tensor) -> bool:
        """Check if option can be initiated in current state
        
        Args:
            state: Current state
            
        Returns:
            Whether option can be initiated
        """
        return self.initiation_set(state)
    
    def should_terminate(
        self,
        state: torch.Tensor,
        elapsed_time: float
    ) -> bool:
        """Check if option should terminate
        
        Args:
            state: Current state
            elapsed_time: Time since option started
            
        Returns:
            Whether option should terminate
        """
        return self.termination_condition(state, elapsed_time)
    
    def update_statistics(self, duration: float, success: bool):
        """Update option statistics
        
        Args:
            duration: Execution duration
            success: Whether execution was successful
        """
        self.execution_count += 1
        if success:
            self.success_count += 1
        self.total_duration += duration
    
    @property
    def success_rate(self) -> float:
        """Get option success rate"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    @property
    def average_duration(self) -> float:
        """Get average execution duration"""
        if self.execution_count == 0:
            return self.expected_duration
        return self.total_duration / self.execution_count


class TemporalAbstraction(nn.Module):
    """Temporal abstraction layer for hierarchical RL"""
    
    def __init__(
        self,
        state_dim: int,
        num_options: int,
        hidden_dim: int = 128,
        use_learned_termination: bool = True
    ):
        """Initialize temporal abstraction
        
        Args:
            state_dim: State dimension
            num_options: Number of options
            hidden_dim: Hidden dimension
            use_learned_termination: Whether to learn termination conditions
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_options = num_options
        self.use_learned_termination = use_learned_termination
        
        # Option selection network
        self.option_selector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Option-specific feature extractors
        self.option_features = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            ) for _ in range(num_options)
        ])
        
        # Learned termination functions
        if use_learned_termination:
            self.termination_functions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                ) for _ in range(num_options)
            ])
        
        # Option value functions
        self.option_values = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_options)
        ])
        
        # Create default options
        self.options = self._create_default_options()
        
        logger.info(f"Initialized TemporalAbstraction with {num_options} options")
    
    def select_option(
        self,
        state: torch.Tensor,
        meta_action: torch.Tensor
    ) -> int:
        """Select option based on state and meta-action
        
        Args:
            state: Current state
            meta_action: Meta-controller action
            
        Returns:
            Selected option ID
        """
        # For now, meta_action directly corresponds to option
        # In practice, this could be more complex
        option_id = meta_action.item() if meta_action.dim() == 0 else meta_action[0].item()
        
        # Ensure option is valid
        option_id = min(option_id, self.num_options - 1)
        
        return option_id
    
    def get_option_features(
        self,
        state: torch.Tensor,
        option_id: int
    ) -> torch.Tensor:
        """Get option-specific features
        
        Args:
            state: Current state
            option_id: Current option
            
        Returns:
            Option-conditioned features
        """
        # Transform state based on current option
        option_features = self.option_features[option_id](state)
        
        # Combine with original state (residual connection)
        combined_features = state + 0.5 * option_features
        
        return combined_features
    
    def check_termination(
        self,
        state: torch.Tensor,
        option_id: int,
        start_time: float,
        current_time: float
    ) -> bool:
        """Check if option should terminate
        
        Args:
            state: Current state
            option_id: Current option
            start_time: Option start time
            current_time: Current time
            
        Returns:
            Whether to terminate
        """
        elapsed_time = current_time - start_time
        
        if self.use_learned_termination:
            # Prepare input with time
            time_tensor = torch.tensor(
                [[elapsed_time]], 
                device=state.device,
                dtype=state.dtype
            )
            
            # Expand dimensions if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            termination_input = torch.cat([state, time_tensor], dim=1)
            
            # Get termination probability
            termination_prob = self.termination_functions[option_id](termination_input)
            
            # Sample termination decision
            should_terminate = torch.bernoulli(termination_prob).item() > 0.5
        else:
            # Use predefined termination
            option = self.options[option_id]
            should_terminate = option.should_terminate(state, elapsed_time)
        
        return should_terminate
    
    def get_option_value(
        self,
        state: torch.Tensor,
        option_id: int
    ) -> torch.Tensor:
        """Get value of executing option in state
        
        Args:
            state: Current state
            option_id: Option to evaluate
            
        Returns:
            Option value
        """
        return self.option_values[option_id](state).squeeze(-1)
    
    def get_expected_duration(self, option_id: int) -> float:
        """Get expected duration of option
        
        Args:
            option_id: Option ID
            
        Returns:
            Expected duration in seconds
        """
        if option_id < len(self.options):
            return self.options[option_id].expected_duration
        else:
            return 10.0  # Default
    
    def _create_default_options(self) -> List[Option]:
        """Create default options for common behaviors"""
        options = []
        
        # Navigation options
        options.append(Option(
            option_id=0,
            name="hover",
            expected_duration=5.0,
            termination_condition=lambda s, t: t > 5.0
        ))
        
        options.append(Option(
            option_id=1,
            name="move_forward",
            expected_duration=10.0,
            termination_condition=lambda s, t: t > 10.0
        ))
        
        options.append(Option(
            option_id=2,
            name="circle_search",
            expected_duration=20.0,
            termination_condition=lambda s, t: t > 20.0
        ))
        
        # Formation options
        options.append(Option(
            option_id=3,
            name="form_line",
            expected_duration=15.0,
            termination_condition=lambda s, t: t > 15.0
        ))
        
        options.append(Option(
            option_id=4,
            name="form_v",
            expected_duration=15.0,
            termination_condition=lambda s, t: t > 15.0
        ))
        
        # Mission-specific options
        options.append(Option(
            option_id=5,
            name="search_pattern",
            expected_duration=30.0,
            termination_condition=lambda s, t: t > 30.0
        ))
        
        options.append(Option(
            option_id=6,
            name="return_to_base",
            expected_duration=60.0,
            termination_condition=lambda s, t: t > 60.0
        ))
        
        # Emergency options
        options.append(Option(
            option_id=7,
            name="emergency_land",
            expected_duration=10.0,
            termination_condition=lambda s, t: t > 10.0
        ))
        
        # Pad with generic options
        while len(options) < self.num_options:
            options.append(Option(
                option_id=len(options),
                name=f"option_{len(options)}",
                expected_duration=15.0
            ))
        
        return options[:self.num_options]
    
    def forward(
        self,
        state: torch.Tensor,
        available_options: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get option values and termination
        
        Args:
            state: Current state
            available_options: List of available options
            
        Returns:
            Option values and termination probabilities
        """
        # Get option selection logits
        option_logits = self.option_selector(state)
        
        # Mask unavailable options
        if available_options is not None:
            mask = torch.ones_like(option_logits) * float('-inf')
            mask[:, available_options] = 0
            option_logits = option_logits + mask
        
        # Get option values
        option_values = []
        for i in range(self.num_options):
            value = self.option_values[i](state)
            option_values.append(value)
        
        option_values = torch.cat(option_values, dim=-1)
        
        return option_logits, option_values


class SkillChaining:
    """Manages skill chaining and composition"""
    
    def __init__(self, skill_library: Dict[str, Option]):
        """Initialize skill chaining
        
        Args:
            skill_library: Dictionary of available skills
        """
        self.skill_library = skill_library
        self.skill_chains = {}
        self._initialize_skill_chains()
    
    def _initialize_skill_chains(self):
        """Initialize common skill chains"""
        # Search and rescue chain
        self.skill_chains["search_and_rescue"] = [
            "takeoff",
            "search_pattern",
            "hover",
            "return_to_base",
            "landing"
        ]
        
        # Formation flight chain
        self.skill_chains["formation_flight"] = [
            "takeoff",
            "join_formation",
            "maintain_formation",
            "break_formation",
            "landing"
        ]
        
        # Emergency response chain
        self.skill_chains["emergency_response"] = [
            "emergency_ascent",
            "hover",
            "emergency_land"
        ]
    
    def get_skill_sequence(
        self,
        objective: str,
        current_state: Dict[str, Any]
    ) -> List[str]:
        """Get skill sequence for objective
        
        Args:
            objective: Mission objective
            current_state: Current system state
            
        Returns:
            Sequence of skill names
        """
        # Get base chain
        base_chain = self.skill_chains.get(objective, [])
        
        # Adapt based on current state
        adapted_chain = []
        for skill in base_chain:
            # Check if skill can be executed
            if skill in self.skill_library:
                option = self.skill_library[skill]
                if option.can_initiate(current_state):
                    adapted_chain.append(skill)
            else:
                # Find alternative skill
                alternative = self._find_alternative_skill(skill, current_state)
                if alternative:
                    adapted_chain.append(alternative)
        
        return adapted_chain
    
    def _find_alternative_skill(
        self,
        skill: str,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """Find alternative skill if primary is unavailable
        
        Args:
            skill: Original skill name
            state: Current state
            
        Returns:
            Alternative skill name or None
        """
        alternatives = {
            "landing": ["emergency_land", "controlled_descent"],
            "takeoff": ["emergency_ascent", "gradual_climb"],
            "search_pattern": ["random_search", "spiral_search"]
        }
        
        for alternative in alternatives.get(skill, []):
            if alternative in self.skill_library:
                option = self.skill_library[alternative]
                if option.can_initiate(state):
                    return alternative
        
        return None