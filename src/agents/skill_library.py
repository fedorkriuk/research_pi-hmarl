"""Skill Library for Common Drone Maneuvers

This module implements a library of reusable skills based on real drone
operations and maneuvers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Represents a reusable skill"""
    
    name: str
    description: str
    skill_id: int
    required_sensors: List[str]
    preconditions: Callable[[Dict[str, Any]], bool]
    postconditions: Callable[[Dict[str, Any]], bool]
    expected_duration: float
    energy_cost: float  # Joules
    risk_level: int  # 1-5
    

class SkillLibrary(nn.Module):
    """Library of learned and predefined skills"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_skills: int = 20
    ):
        """Initialize skill library
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            num_skills: Number of skills to maintain
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_skills = num_skills
        
        # Skill encoders (one per skill)
        self.skill_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            ) for _ in range(num_skills)
        ])
        
        # Skill value functions
        self.skill_values = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_skills)
        ])
        
        # Skill termination conditions
        self.skill_terminations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + 1, hidden_dim),  # +1 for elapsed time
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_skills)
        ])
        
        # Initialize predefined skills
        self.skills = self._initialize_skills()
        
        # Skill composition network
        self.composition_net = SkillComposer(
            num_skills=num_skills,
            hidden_dim=hidden_dim
        )
        
        logger.info(f"Initialized SkillLibrary with {num_skills} skills")
    
    def execute_skill(
        self,
        skill_id: int,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute a skill
        
        Args:
            skill_id: Skill to execute
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and info dict
        """
        if skill_id >= self.num_skills:
            skill_id = self.num_skills - 1
        
        # Get skill-specific action
        action = self.skill_encoders[skill_id](state)
        
        # Get skill value
        value = self.skill_values[skill_id](state).squeeze(-1)
        
        # Add noise if not deterministic
        if not deterministic:
            noise = torch.randn_like(action) * 0.1
            action = action + noise
            action = torch.tanh(action)  # Re-normalize
        
        info = {
            "skill_id": skill_id,
            "skill_name": self.skills[skill_id].name if skill_id < len(self.skills) else f"skill_{skill_id}",
            "value": value,
            "expected_duration": self.skills[skill_id].expected_duration if skill_id < len(self.skills) else 10.0,
            "energy_cost": self.skills[skill_id].energy_cost if skill_id < len(self.skills) else 100.0
        }
        
        return action, info
    
    def check_skill_termination(
        self,
        skill_id: int,
        state: torch.Tensor,
        elapsed_time: float
    ) -> bool:
        """Check if skill should terminate
        
        Args:
            skill_id: Current skill
            state: Current state
            elapsed_time: Time since skill started
            
        Returns:
            Whether to terminate
        """
        # Prepare input
        time_tensor = torch.tensor([[elapsed_time]], device=state.device, dtype=state.dtype)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        termination_input = torch.cat([state, time_tensor], dim=1)
        
        # Get termination probability
        term_prob = self.skill_terminations[skill_id](termination_input)
        
        # Sample termination
        should_terminate = torch.bernoulli(term_prob).item() > 0.5
        
        # Also check predefined conditions
        if skill_id < len(self.skills):
            skill = self.skills[skill_id]
            # Maximum duration check
            if elapsed_time > skill.expected_duration * 2:
                should_terminate = True
        
        return should_terminate
    
    def get_skill_value(
        self,
        skill_id: int,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Get value of executing skill in state
        
        Args:
            skill_id: Skill to evaluate
            state: Current state
            
        Returns:
            Skill value
        """
        return self.skill_values[skill_id](state).squeeze(-1)
    
    def compose_skills(
        self,
        skill_sequence: List[int],
        state: torch.Tensor
    ) -> torch.Tensor:
        """Compose multiple skills into a sequence
        
        Args:
            skill_sequence: Sequence of skill IDs
            state: Initial state
            
        Returns:
            Composed action sequence
        """
        return self.composition_net(skill_sequence, state)
    
    def _initialize_skills(self) -> List[Skill]:
        """Initialize predefined skills"""
        skills = []
        
        # Basic navigation skills
        skills.append(Skill(
            name="hover",
            description="Maintain position with minimal drift",
            skill_id=0,
            required_sensors=["gps", "imu"],
            preconditions=lambda s: s.get("altitude", 0) > 0.5,
            postconditions=lambda s: s.get("velocity", 0) < 0.5,
            expected_duration=10.0,
            energy_cost=50.0,  # Joules
            risk_level=1
        ))
        
        skills.append(Skill(
            name="takeoff",
            description="Vertical takeoff to target altitude",
            skill_id=1,
            required_sensors=["gps", "imu", "barometer"],
            preconditions=lambda s: s.get("altitude", 0) < 1.0 and s.get("battery_level", 0) > 0.3,
            postconditions=lambda s: s.get("altitude", 0) > 5.0,
            expected_duration=5.0,
            energy_cost=200.0,
            risk_level=2
        ))
        
        skills.append(Skill(
            name="landing",
            description="Controlled descent and landing",
            skill_id=2,
            required_sensors=["gps", "imu", "lidar"],
            preconditions=lambda s: s.get("altitude", 0) > 1.0,
            postconditions=lambda s: s.get("altitude", 0) < 0.5 and s.get("velocity", 0) < 0.1,
            expected_duration=8.0,
            energy_cost=100.0,
            risk_level=3
        ))
        
        # Advanced navigation skills
        skills.append(Skill(
            name="waypoint_navigation",
            description="Navigate to specific waypoint",
            skill_id=3,
            required_sensors=["gps", "imu", "compass"],
            preconditions=lambda s: s.get("gps_fix", False) and s.get("altitude", 0) > 5.0,
            postconditions=lambda s: s.get("distance_to_waypoint", float('inf')) < 2.0,
            expected_duration=30.0,
            energy_cost=500.0,
            risk_level=2
        ))
        
        skills.append(Skill(
            name="orbit",
            description="Circular orbit around point",
            skill_id=4,
            required_sensors=["gps", "imu", "compass"],
            preconditions=lambda s: s.get("altitude", 0) > 10.0,
            postconditions=lambda s: True,
            expected_duration=60.0,
            energy_cost=800.0,
            risk_level=2
        ))
        
        # Search patterns
        skills.append(Skill(
            name="spiral_search",
            description="Expanding spiral search pattern",
            skill_id=5,
            required_sensors=["gps", "imu", "camera"],
            preconditions=lambda s: s.get("altitude", 0) > 20.0 and s.get("battery_level", 0) > 0.5,
            postconditions=lambda s: s.get("search_complete", False),
            expected_duration=120.0,
            energy_cost=1500.0,
            risk_level=2
        ))
        
        skills.append(Skill(
            name="grid_search",
            description="Grid-based search pattern",
            skill_id=6,
            required_sensors=["gps", "imu", "camera"],
            preconditions=lambda s: s.get("altitude", 0) > 20.0 and s.get("battery_level", 0) > 0.5,
            postconditions=lambda s: s.get("search_complete", False),
            expected_duration=180.0,
            energy_cost=2000.0,
            risk_level=2
        ))
        
        # Formation flying skills
        skills.append(Skill(
            name="join_formation",
            description="Join multi-agent formation",
            skill_id=7,
            required_sensors=["gps", "imu", "communication"],
            preconditions=lambda s: s.get("formation_available", False) and s.get("altitude", 0) > 10.0,
            postconditions=lambda s: s.get("in_formation", False),
            expected_duration=15.0,
            energy_cost=300.0,
            risk_level=3
        ))
        
        skills.append(Skill(
            name="maintain_formation",
            description="Maintain position in formation",
            skill_id=8,
            required_sensors=["gps", "imu", "communication"],
            preconditions=lambda s: s.get("in_formation", False),
            postconditions=lambda s: s.get("formation_error", float('inf')) < 2.0,
            expected_duration=60.0,
            energy_cost=600.0,
            risk_level=2
        ))
        
        # Emergency skills
        skills.append(Skill(
            name="emergency_land",
            description="Emergency landing procedure",
            skill_id=9,
            required_sensors=["imu", "barometer"],
            preconditions=lambda s: True,  # Always available
            postconditions=lambda s: s.get("altitude", 0) < 0.5,
            expected_duration=10.0,
            energy_cost=50.0,
            risk_level=5
        ))
        
        skills.append(Skill(
            name="return_to_home",
            description="Return to launch point",
            skill_id=10,
            required_sensors=["gps", "imu", "compass"],
            preconditions=lambda s: s.get("home_position_set", False) and s.get("battery_level", 0) > 0.2,
            postconditions=lambda s: s.get("distance_to_home", float('inf')) < 5.0,
            expected_duration=90.0,
            energy_cost=1000.0,
            risk_level=3
        ))
        
        # Inspection skills
        skills.append(Skill(
            name="point_inspection",
            description="Inspect specific point of interest",
            skill_id=11,
            required_sensors=["gps", "imu", "camera", "gimbal"],
            preconditions=lambda s: s.get("target_visible", False) and s.get("altitude", 0) > 5.0,
            postconditions=lambda s: s.get("inspection_complete", False),
            expected_duration=20.0,
            energy_cost=250.0,
            risk_level=2
        ))
        
        skills.append(Skill(
            name="perimeter_patrol",
            description="Patrol along defined perimeter",
            skill_id=12,
            required_sensors=["gps", "imu", "camera"],
            preconditions=lambda s: s.get("perimeter_defined", False) and s.get("battery_level", 0) > 0.4,
            postconditions=lambda s: s.get("patrol_complete", False),
            expected_duration=300.0,
            energy_cost=3000.0,
            risk_level=2
        ))
        
        # Pad with generic skills
        while len(skills) < self.num_skills:
            skills.append(Skill(
                name=f"skill_{len(skills)}",
                description=f"Generic skill {len(skills)}",
                skill_id=len(skills),
                required_sensors=["imu"],
                preconditions=lambda s: True,
                postconditions=lambda s: True,
                expected_duration=15.0,
                energy_cost=200.0,
                risk_level=2
            ))
        
        return skills[:self.num_skills]


class SkillComposer(nn.Module):
    """Composes multiple skills into sequences"""
    
    def __init__(self, num_skills: int, hidden_dim: int = 128):
        """Initialize skill composer
        
        Args:
            num_skills: Number of skills
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.num_skills = num_skills
        
        # Skill embedding
        self.skill_embedding = nn.Embedding(num_skills, hidden_dim)
        
        # Sequence encoder (LSTM)
        self.sequence_encoder = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Transition predictor
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
        )
    
    def forward(
        self,
        skill_sequence: List[int],
        state: torch.Tensor
    ) -> torch.Tensor:
        """Compose skill sequence
        
        Args:
            skill_sequence: List of skill IDs
            state: Initial state
            
        Returns:
            Next skill probabilities
        """
        # Convert to tensor
        sequence = torch.tensor(skill_sequence, device=state.device)
        
        # Get skill embeddings
        embeddings = self.skill_embedding(sequence)
        
        # Add batch dimension if needed
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        # Encode sequence
        output, (hidden, _) = self.sequence_encoder(embeddings)
        
        # Get last hidden state
        last_hidden = hidden[-1]
        
        # Predict next skill
        combined = torch.cat([last_hidden, state], dim=-1)
        next_skill_logits = self.transition_net(combined)
        
        return next_skill_logits


class SkillTransfer:
    """Handles skill transfer between different drone types"""
    
    def __init__(self):
        """Initialize skill transfer"""
        # Drone capability mappings
        self.drone_capabilities = {
            "mavic_3": {
                "max_speed": 21.0,
                "max_altitude": 6000.0,
                "flight_time": 46.0,  # minutes
                "sensors": ["gps", "imu", "camera", "lidar", "gimbal"],
                "special_features": ["obstacle_avoidance", "return_to_home"]
            },
            "anafi": {
                "max_speed": 16.0,
                "max_altitude": 4500.0,
                "flight_time": 25.0,
                "sensors": ["gps", "imu", "camera"],
                "special_features": ["return_to_home"]
            },
            "phantom_4": {
                "max_speed": 20.0,
                "max_altitude": 6000.0,
                "flight_time": 30.0,
                "sensors": ["gps", "imu", "camera", "lidar"],
                "special_features": ["obstacle_avoidance", "return_to_home"]
            }
        }
    
    def adapt_skill(
        self,
        skill: Skill,
        source_drone: str,
        target_drone: str
    ) -> Skill:
        """Adapt skill from source to target drone
        
        Args:
            skill: Skill to adapt
            source_drone: Source drone type
            target_drone: Target drone type
            
        Returns:
            Adapted skill
        """
        source_cap = self.drone_capabilities.get(source_drone, {})
        target_cap = self.drone_capabilities.get(target_drone, {})
        
        # Adapt based on capabilities
        adapted_skill = Skill(
            name=skill.name,
            description=skill.description,
            skill_id=skill.skill_id,
            required_sensors=self._adapt_sensors(
                skill.required_sensors,
                target_cap.get("sensors", [])
            ),
            preconditions=skill.preconditions,
            postconditions=skill.postconditions,
            expected_duration=self._adapt_duration(
                skill.expected_duration,
                source_cap,
                target_cap
            ),
            energy_cost=self._adapt_energy(
                skill.energy_cost,
                source_cap,
                target_cap
            ),
            risk_level=self._adapt_risk(
                skill.risk_level,
                source_cap,
                target_cap
            )
        )
        
        return adapted_skill
    
    def _adapt_sensors(
        self,
        required: List[str],
        available: List[str]
    ) -> List[str]:
        """Adapt required sensors to available ones"""
        # Filter to available sensors
        adapted = [s for s in required if s in available]
        
        # Add alternatives for missing sensors
        if "lidar" in required and "lidar" not in available:
            if "camera" in available:
                adapted.append("camera")  # Use camera for obstacle detection
        
        return adapted
    
    def _adapt_duration(
        self,
        duration: float,
        source_cap: Dict[str, Any],
        target_cap: Dict[str, Any]
    ) -> float:
        """Adapt expected duration based on capabilities"""
        # Scale by speed ratio
        speed_ratio = target_cap.get("max_speed", 15.0) / source_cap.get("max_speed", 20.0)
        return duration / speed_ratio
    
    def _adapt_energy(
        self,
        energy: float,
        source_cap: Dict[str, Any],
        target_cap: Dict[str, Any]
    ) -> float:
        """Adapt energy cost based on capabilities"""
        # Scale by flight time ratio (inverse - longer flight time = more efficient)
        efficiency_ratio = source_cap.get("flight_time", 30.0) / target_cap.get("flight_time", 30.0)
        return energy * efficiency_ratio
    
    def _adapt_risk(
        self,
        risk: int,
        source_cap: Dict[str, Any],
        target_cap: Dict[str, Any]
    ) -> int:
        """Adapt risk level based on capabilities"""
        # Increase risk if missing safety features
        adapted_risk = risk
        
        source_features = set(source_cap.get("special_features", []))
        target_features = set(target_cap.get("special_features", []))
        
        if "obstacle_avoidance" in source_features and "obstacle_avoidance" not in target_features:
            adapted_risk = min(5, adapted_risk + 1)
        
        return adapted_risk