"""Skill Matching and Capability Management

This module implements skill-based matching between agents and tasks,
including capability learning and evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SkillLevel(Enum):
    """Skill proficiency levels"""
    NOVICE = 1
    BEGINNER = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


@dataclass
class Skill:
    """Individual skill representation"""
    name: str
    level: SkillLevel
    experience: float  # Hours of experience
    success_rate: float
    last_used: Optional[float]
    certifications: List[str]


@dataclass
class AgentCapability:
    """Agent capability profile"""
    agent_id: int
    skills: Dict[str, Skill]
    specializations: List[str]
    performance_history: List[float]
    learning_rate: float
    adaptability: float


@dataclass
class TaskRequirement:
    """Task skill requirements"""
    task_id: str
    required_skills: Dict[str, SkillLevel]
    preferred_skills: Dict[str, SkillLevel]
    skill_importance: Dict[str, float]
    complexity: float
    flexibility: float  # How flexible the requirements are


class SkillMatcher:
    """Matches agent skills to task requirements"""
    
    def __init__(
        self,
        skill_categories: Optional[List[str]] = None,
        learning_enabled: bool = True
    ):
        """Initialize skill matcher
        
        Args:
            skill_categories: Skill category list
            learning_enabled: Whether to enable skill learning
        """
        if skill_categories is None:
            skill_categories = [
                'navigation', 'perception', 'manipulation',
                'planning', 'communication', 'computation',
                'coordination', 'emergency_response', 'maintenance',
                'data_analysis', 'surveillance', 'cargo_handling'
            ]
        
        self.skill_categories = skill_categories
        self.learning_enabled = learning_enabled
        
        # Components
        self.capability_manager = AgentCapabilities(skill_categories)
        self.requirement_analyzer = TaskRequirements()
        self.skill_learner = SkillLearning() if learning_enabled else None
        self.capability_evolution = CapabilityEvolution()
        
        # Matching network
        self.matching_net = nn.Sequential(
            nn.Linear(len(skill_categories) * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Match score
        )
        
        logger.info(f"Initialized SkillMatcher with {len(skill_categories)} categories")
    
    def match_agents_to_task(
        self,
        task_req: TaskRequirement,
        agent_capabilities: Dict[int, AgentCapability],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float]]:
        """Match agents to task based on skills
        
        Args:
            task_req: Task requirements
            agent_capabilities: Agent capabilities
            constraints: Additional constraints
            
        Returns:
            List of (agent_id, match_score) sorted by score
        """
        matches = []
        
        for agent_id, agent_cap in agent_capabilities.items():
            # Compute match score
            score = self._compute_match_score(task_req, agent_cap)
            
            # Apply constraints
            if constraints:
                score = self._apply_constraints(score, agent_cap, constraints)
            
            matches.append((agent_id, score))
        
        # Sort by score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def _compute_match_score(
        self,
        task_req: TaskRequirement,
        agent_cap: AgentCapability
    ) -> float:
        """Compute skill match score
        
        Args:
            task_req: Task requirements
            agent_cap: Agent capability
            
        Returns:
            Match score (0-1)
        """
        # Extract features
        features = self._extract_match_features(task_req, agent_cap)
        
        # Neural match score
        neural_score = torch.sigmoid(self.matching_net(features)).item()
        
        # Rule-based components
        requirement_score = self._compute_requirement_match(task_req, agent_cap)
        experience_score = self._compute_experience_match(task_req, agent_cap)
        
        # Combine scores
        total_score = (
            0.5 * neural_score +
            0.3 * requirement_score +
            0.2 * experience_score
        )
        
        # Boost for specialization match
        if any(spec in task_req.required_skills for spec in agent_cap.specializations):
            total_score *= 1.2
        
        return min(total_score, 1.0)
    
    def _extract_match_features(
        self,
        task_req: TaskRequirement,
        agent_cap: AgentCapability
    ) -> torch.Tensor:
        """Extract features for matching
        
        Args:
            task_req: Task requirements
            agent_cap: Agent capability
            
        Returns:
            Feature vector
        """
        features = []
        
        # For each skill category
        for skill_name in self.skill_categories:
            # Required level
            req_level = task_req.required_skills.get(skill_name, SkillLevel.NOVICE)
            features.append(req_level.value / 5.0)
            
            # Agent level
            if skill_name in agent_cap.skills:
                agent_level = agent_cap.skills[skill_name].level.value
                agent_exp = min(agent_cap.skills[skill_name].experience / 100, 1.0)
                agent_success = agent_cap.skills[skill_name].success_rate
            else:
                agent_level = 0
                agent_exp = 0
                agent_success = 0
            
            features.append(agent_level / 5.0)
            features.append(agent_exp)
            features.append(agent_success)
        
        return torch.tensor(features)
    
    def _compute_requirement_match(
        self,
        task_req: TaskRequirement,
        agent_cap: AgentCapability
    ) -> float:
        """Compute requirement satisfaction score
        
        Args:
            task_req: Task requirements
            agent_cap: Agent capability
            
        Returns:
            Requirement match score
        """
        total_importance = 0.0
        satisfied_importance = 0.0
        
        for skill_name, req_level in task_req.required_skills.items():
            importance = task_req.skill_importance.get(skill_name, 1.0)
            total_importance += importance
            
            if skill_name in agent_cap.skills:
                agent_skill = agent_cap.skills[skill_name]
                if agent_skill.level.value >= req_level.value:
                    # Full credit if meets requirement
                    satisfied_importance += importance
                else:
                    # Partial credit based on level difference
                    level_diff = req_level.value - agent_skill.level.value
                    partial_credit = max(0, 1 - level_diff * 0.25)
                    satisfied_importance += importance * partial_credit
        
        if total_importance > 0:
            return satisfied_importance / total_importance
        else:
            return 1.0
    
    def _compute_experience_match(
        self,
        task_req: TaskRequirement,
        agent_cap: AgentCapability
    ) -> float:
        """Compute experience-based match score
        
        Args:
            task_req: Task requirements
            agent_cap: Agent capability
            
        Returns:
            Experience score
        """
        # Average performance history
        if agent_cap.performance_history:
            avg_performance = np.mean(agent_cap.performance_history[-10:])
        else:
            avg_performance = 0.5
        
        # Complexity match
        complexity_diff = abs(task_req.complexity - avg_performance)
        complexity_score = 1.0 - complexity_diff
        
        # Recent experience bonus
        recent_experience = 0.0
        for skill_name in task_req.required_skills:
            if skill_name in agent_cap.skills:
                skill = agent_cap.skills[skill_name]
                if skill.last_used:
                    # Bonus for recent use (within last 24 hours)
                    recency = 1.0  # Simplified
                    recent_experience += recency * skill.success_rate
        
        recent_bonus = recent_experience / max(len(task_req.required_skills), 1)
        
        return 0.7 * complexity_score + 0.3 * recent_bonus
    
    def _apply_constraints(
        self,
        score: float,
        agent_cap: AgentCapability,
        constraints: Dict[str, Any]
    ) -> float:
        """Apply constraints to match score
        
        Args:
            score: Base match score
            agent_cap: Agent capability
            constraints: Constraints
            
        Returns:
            Adjusted score
        """
        # Minimum skill level constraint
        if 'min_skill_level' in constraints:
            min_level = constraints['min_skill_level']
            avg_level = np.mean([
                skill.level.value for skill in agent_cap.skills.values()
            ])
            if avg_level < min_level:
                score *= 0.5
        
        # Certification requirements
        if 'required_certifications' in constraints:
            required_certs = set(constraints['required_certifications'])
            agent_certs = set()
            for skill in agent_cap.skills.values():
                agent_certs.update(skill.certifications)
            
            if not required_certs.issubset(agent_certs):
                score *= 0.3
        
        return score
    
    def update_capabilities(
        self,
        agent_id: int,
        task_id: str,
        performance: Dict[str, float],
        agent_cap: AgentCapability,
        task_req: TaskRequirement
    ) -> AgentCapability:
        """Update agent capabilities based on task performance
        
        Args:
            agent_id: Agent ID
            task_id: Task ID
            performance: Task performance metrics
            agent_cap: Current agent capability
            task_req: Task requirements
            
        Returns:
            Updated agent capability
        """
        if not self.learning_enabled or self.skill_learner is None:
            return agent_cap
        
        # Learn from experience
        skill_updates = self.skill_learner.learn_from_task(
            agent_cap, task_req, performance
        )
        
        # Apply updates
        for skill_name, updates in skill_updates.items():
            if skill_name in agent_cap.skills:
                skill = agent_cap.skills[skill_name]
                
                # Update experience
                skill.experience += updates.get('experience_gain', 0)
                
                # Update success rate
                old_success = skill.success_rate
                new_success = updates.get('success_rate', old_success)
                skill.success_rate = 0.9 * old_success + 0.1 * new_success
                
                # Update level if threshold reached
                if skill.experience > skill.level.value * 50:
                    if skill.level.value < 5:
                        skill.level = SkillLevel(skill.level.value + 1)
                
                # Update last used
                skill.last_used = updates.get('timestamp', 0)
        
        # Update performance history
        overall_perf = performance.get('overall', 0.5)
        agent_cap.performance_history.append(overall_perf)
        
        # Evolve capabilities
        agent_cap = self.capability_evolution.evolve(agent_cap, performance)
        
        return agent_cap


class AgentCapabilities:
    """Manages agent capability profiles"""
    
    def __init__(self, skill_categories: List[str]):
        """Initialize capability manager
        
        Args:
            skill_categories: List of skill categories
        """
        self.skill_categories = skill_categories
        self.capability_db = {}
        
        # Skill embedding network
        self.skill_embedder = nn.Embedding(len(skill_categories), 32)
    
    def create_agent_profile(
        self,
        agent_id: int,
        initial_skills: Optional[Dict[str, int]] = None,
        specialization: Optional[str] = None
    ) -> AgentCapability:
        """Create new agent capability profile
        
        Args:
            agent_id: Agent ID
            initial_skills: Initial skill levels
            specialization: Agent specialization
            
        Returns:
            Agent capability profile
        """
        skills = {}
        
        # Initialize skills
        for skill_name in self.skill_categories:
            if initial_skills and skill_name in initial_skills:
                level = SkillLevel(initial_skills[skill_name])
            else:
                level = SkillLevel.NOVICE
            
            skills[skill_name] = Skill(
                name=skill_name,
                level=level,
                experience=0.0,
                success_rate=0.7,  # Initial success rate
                last_used=None,
                certifications=[]
            )
        
        # Set specialization
        specializations = []
        if specialization:
            specializations.append(specialization)
            # Boost specialized skill
            if specialization in skills:
                skills[specialization].level = SkillLevel.INTERMEDIATE
        
        # Create profile
        profile = AgentCapability(
            agent_id=agent_id,
            skills=skills,
            specializations=specializations,
            performance_history=[],
            learning_rate=0.1,
            adaptability=0.5
        )
        
        # Store in database
        self.capability_db[agent_id] = profile
        
        return profile
    
    def get_skill_embedding(
        self,
        agent_cap: AgentCapability
    ) -> torch.Tensor:
        """Get skill embedding for agent
        
        Args:
            agent_cap: Agent capability
            
        Returns:
            Skill embedding vector
        """
        # Get skill indices and levels
        skill_indices = []
        skill_levels = []
        
        for i, skill_name in enumerate(self.skill_categories):
            if skill_name in agent_cap.skills:
                skill_indices.append(i)
                skill_levels.append(agent_cap.skills[skill_name].level.value)
        
        if not skill_indices:
            return torch.zeros(32)
        
        # Get embeddings
        indices_tensor = torch.tensor(skill_indices)
        embeddings = self.skill_embedder(indices_tensor)
        
        # Weight by skill level
        levels_tensor = torch.tensor(skill_levels, dtype=torch.float32)
        weighted_embeddings = embeddings * levels_tensor.unsqueeze(1)
        
        # Average pooling
        skill_embedding = weighted_embeddings.mean(dim=0)
        
        return skill_embedding
    
    def recommend_training(
        self,
        agent_cap: AgentCapability,
        target_tasks: List[TaskRequirement]
    ) -> Dict[str, float]:
        """Recommend skills for training
        
        Args:
            agent_cap: Agent capability
            target_tasks: Upcoming tasks
            
        Returns:
            Skill training recommendations
        """
        skill_gaps = {}
        
        for task in target_tasks:
            for skill_name, req_level in task.required_skills.items():
                importance = task.skill_importance.get(skill_name, 1.0)
                
                if skill_name in agent_cap.skills:
                    current_level = agent_cap.skills[skill_name].level.value
                    gap = max(0, req_level.value - current_level)
                else:
                    gap = req_level.value
                
                if skill_name not in skill_gaps:
                    skill_gaps[skill_name] = 0
                
                skill_gaps[skill_name] += gap * importance
        
        # Normalize and sort
        total_gap = sum(skill_gaps.values())
        if total_gap > 0:
            recommendations = {
                skill: gap / total_gap
                for skill, gap in skill_gaps.items()
            }
        else:
            recommendations = {}
        
        # Sort by priority
        sorted_recommendations = dict(
            sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_recommendations


class TaskRequirements:
    """Analyzes and manages task requirements"""
    
    def __init__(self):
        """Initialize requirement analyzer"""
        # Requirement prediction network
        self.requirement_net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 12)  # Skill requirements
        )
    
    def analyze_task(
        self,
        task_description: Dict[str, Any],
        historical_data: Optional[List[TaskRequirement]] = None
    ) -> TaskRequirement:
        """Analyze task to determine requirements
        
        Args:
            task_description: Task description
            historical_data: Historical task requirements
            
        Returns:
            Task requirements
        """
        # Extract features from description
        features = self._extract_task_features(task_description)
        
        # Predict requirements
        req_logits = self.requirement_net(features)
        req_probs = torch.sigmoid(req_logits)
        
        # Convert to skill requirements
        required_skills = {}
        skill_importance = {}
        
        skill_categories = [
            'navigation', 'perception', 'manipulation',
            'planning', 'communication', 'computation',
            'coordination', 'emergency_response', 'maintenance',
            'data_analysis', 'surveillance', 'cargo_handling'
        ]
        
        for i, skill in enumerate(skill_categories):
            if req_probs[i] > 0.3:
                # Determine required level based on probability
                if req_probs[i] > 0.8:
                    level = SkillLevel.ADVANCED
                elif req_probs[i] > 0.6:
                    level = SkillLevel.INTERMEDIATE
                else:
                    level = SkillLevel.BEGINNER
                
                required_skills[skill] = level
                skill_importance[skill] = req_probs[i].item()
        
        # Add task-specific requirements
        task_type = task_description.get('type', 'generic')
        specific_reqs = self._get_specific_requirements(task_type)
        required_skills.update(specific_reqs)
        
        # Create requirement object
        requirement = TaskRequirement(
            task_id=task_description.get('task_id', 'unknown'),
            required_skills=required_skills,
            preferred_skills={},  # Could be extended
            skill_importance=skill_importance,
            complexity=task_description.get('complexity', 0.5),
            flexibility=task_description.get('flexibility', 0.3)
        )
        
        return requirement
    
    def _extract_task_features(
        self,
        task_description: Dict[str, Any]
    ) -> torch.Tensor:
        """Extract features from task description
        
        Args:
            task_description: Task description
            
        Returns:
            Feature vector
        """
        features = []
        
        # Task type encoding
        task_types = [
            'surveillance', 'delivery', 'search_rescue',
            'mapping', 'inspection', 'patrol'
        ]
        
        task_type = task_description.get('type', 'generic')
        for t in task_types:
            features.append(1.0 if t == task_type else 0.0)
        
        # Task properties
        features.append(task_description.get('priority', 0.5))
        features.append(task_description.get('duration', 300) / 1000)
        features.append(task_description.get('complexity', 0.5))
        features.append(task_description.get('num_agents', 1) / 5)
        
        # Environmental factors
        features.append(task_description.get('outdoor', 1.0))
        features.append(task_description.get('hazardous', 0.0))
        features.append(task_description.get('precision_required', 0.5))
        
        # Resource requirements
        features.append(task_description.get('energy_intensive', 0.5))
        features.append(task_description.get('computation_intensive', 0.3))
        features.append(task_description.get('communication_intensive', 0.5))
        
        # Pad to 20
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features[:20])
    
    def _get_specific_requirements(
        self,
        task_type: str
    ) -> Dict[str, SkillLevel]:
        """Get task-specific requirements
        
        Args:
            task_type: Type of task
            
        Returns:
            Specific skill requirements
        """
        specific_reqs = {
            'surveillance': {
                'perception': SkillLevel.ADVANCED,
                'navigation': SkillLevel.INTERMEDIATE
            },
            'delivery': {
                'cargo_handling': SkillLevel.ADVANCED,
                'navigation': SkillLevel.INTERMEDIATE
            },
            'search_rescue': {
                'perception': SkillLevel.EXPERT,
                'emergency_response': SkillLevel.ADVANCED,
                'coordination': SkillLevel.ADVANCED
            },
            'mapping': {
                'perception': SkillLevel.ADVANCED,
                'data_analysis': SkillLevel.INTERMEDIATE,
                'navigation': SkillLevel.ADVANCED
            },
            'inspection': {
                'perception': SkillLevel.EXPERT,
                'data_analysis': SkillLevel.ADVANCED,
                'maintenance': SkillLevel.INTERMEDIATE
            }
        }
        
        return specific_reqs.get(task_type, {})


class SkillLearning:
    """Implements skill learning and improvement"""
    
    def __init__(self):
        """Initialize skill learning"""
        self.learning_net = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)  # Learning outcomes
        )
    
    def learn_from_task(
        self,
        agent_cap: AgentCapability,
        task_req: TaskRequirement,
        performance: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Learn from task execution
        
        Args:
            agent_cap: Agent capability
            task_req: Task requirements
            performance: Task performance
            
        Returns:
            Skill updates
        """
        skill_updates = {}
        
        # For each skill used in task
        for skill_name in task_req.required_skills:
            if skill_name not in agent_cap.skills:
                continue
            
            skill = agent_cap.skills[skill_name]
            
            # Compute learning features
            features = self._compute_learning_features(
                skill, task_req, performance
            )
            
            # Predict learning outcomes
            outcomes = self.learning_net(features)
            
            # Convert to updates
            updates = {
                'experience_gain': F.softplus(outcomes[0]).item() * 10,
                'success_rate': torch.sigmoid(outcomes[1]).item(),
                'confidence_boost': torch.sigmoid(outcomes[2]).item(),
                'technique_improvement': torch.sigmoid(outcomes[3]).item(),
                'timestamp': time.time()
            }
            
            # Adjust based on performance
            perf_factor = performance.get('overall', 0.5)
            updates['experience_gain'] *= perf_factor
            
            # Learning rate adjustment
            updates['experience_gain'] *= agent_cap.learning_rate
            
            skill_updates[skill_name] = updates
        
        return skill_updates
    
    def _compute_learning_features(
        self,
        skill: Skill,
        task_req: TaskRequirement,
        performance: Dict[str, float]
    ) -> torch.Tensor:
        """Compute features for learning
        
        Args:
            skill: Current skill
            task_req: Task requirements
            performance: Performance metrics
            
        Returns:
            Learning features
        """
        features = []
        
        # Current skill state
        features.append(skill.level.value / 5.0)
        features.append(skill.experience / 100.0)
        features.append(skill.success_rate)
        
        # Task challenge
        req_level = task_req.required_skills.get(skill.name, SkillLevel.NOVICE)
        challenge = req_level.value - skill.level.value
        features.append((challenge + 5) / 10.0)  # Normalized
        
        # Performance metrics
        features.append(performance.get('overall', 0.5))
        features.append(performance.get('efficiency', 0.5))
        features.append(performance.get('quality', 0.5))
        
        # Task properties
        features.append(task_req.complexity)
        features.append(task_req.flexibility)
        
        # Time since last use
        if skill.last_used:
            time_gap = (time.time() - skill.last_used) / 86400  # Days
            features.append(min(time_gap / 30, 1.0))  # Normalized to month
        else:
            features.append(1.0)
        
        # Pad to 15
        while len(features) < 15:
            features.append(0.0)
        
        return torch.tensor(features[:15])


class CapabilityEvolution:
    """Manages long-term capability evolution"""
    
    def __init__(self):
        """Initialize capability evolution"""
        self.evolution_net = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Evolution parameters
        )
    
    def evolve(
        self,
        agent_cap: AgentCapability,
        recent_performance: Dict[str, float]
    ) -> AgentCapability:
        """Evolve agent capabilities
        
        Args:
            agent_cap: Current capabilities
            recent_performance: Recent performance
            
        Returns:
            Evolved capabilities
        """
        # Extract evolution features
        features = self._extract_evolution_features(agent_cap, recent_performance)
        
        # Predict evolution parameters
        evolution_params = self.evolution_net(features)
        
        # Apply evolution
        # Adjust learning rate
        lr_adjustment = torch.sigmoid(evolution_params[0]).item()
        agent_cap.learning_rate = agent_cap.learning_rate * (0.9 + 0.2 * lr_adjustment)
        agent_cap.learning_rate = np.clip(agent_cap.learning_rate, 0.01, 0.5)
        
        # Adjust adaptability
        adapt_adjustment = torch.sigmoid(evolution_params[1]).item()
        agent_cap.adaptability = agent_cap.adaptability * (0.9 + 0.2 * adapt_adjustment)
        agent_cap.adaptability = np.clip(agent_cap.adaptability, 0.1, 0.9)
        
        # Specialization tendency
        spec_tendency = torch.sigmoid(evolution_params[2]).item()
        
        # Identify emerging specializations
        if spec_tendency > 0.7:
            top_skills = self._identify_top_skills(agent_cap)
            for skill_name in top_skills[:2]:  # Top 2 skills
                if skill_name not in agent_cap.specializations:
                    agent_cap.specializations.append(skill_name)
                    logger.info(f"Agent {agent_cap.agent_id} specialized in {skill_name}")
        
        return agent_cap
    
    def _extract_evolution_features(
        self,
        agent_cap: AgentCapability,
        recent_performance: Dict[str, float]
    ) -> torch.Tensor:
        """Extract features for evolution
        
        Args:
            agent_cap: Agent capabilities
            recent_performance: Recent performance
            
        Returns:
            Evolution features
        """
        features = []
        
        # Performance trajectory
        if len(agent_cap.performance_history) > 5:
            recent = agent_cap.performance_history[-5:]
            features.extend([
                np.mean(recent),
                np.std(recent),
                np.max(recent) - np.min(recent)
            ])
        else:
            features.extend([0.5, 0.1, 0.0])
        
        # Skill diversity
        skill_levels = [s.level.value for s in agent_cap.skills.values()]
        features.append(np.mean(skill_levels) / 5.0)
        features.append(np.std(skill_levels) / 2.0)
        
        # Specialization degree
        features.append(len(agent_cap.specializations) / 5.0)
        
        # Learning metrics
        features.append(agent_cap.learning_rate)
        features.append(agent_cap.adaptability)
        
        # Recent performance
        features.append(recent_performance.get('overall', 0.5))
        features.append(recent_performance.get('efficiency', 0.5))
        
        # Experience distribution
        total_exp = sum(s.experience for s in agent_cap.skills.values())
        if total_exp > 0:
            exp_concentration = max(s.experience for s in agent_cap.skills.values()) / total_exp
        else:
            exp_concentration = 0.0
        features.append(exp_concentration)
        
        # Success rates
        success_rates = [s.success_rate for s in agent_cap.skills.values()]
        features.append(np.mean(success_rates))
        features.append(np.std(success_rates))
        
        # Pad to 20
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features[:20])
    
    def _identify_top_skills(
        self,
        agent_cap: AgentCapability,
        top_n: int = 3
    ) -> List[str]:
        """Identify top skills for specialization
        
        Args:
            agent_cap: Agent capabilities
            top_n: Number of top skills
            
        Returns:
            Top skill names
        """
        # Score skills by level, experience, and success rate
        skill_scores = {}
        
        for skill_name, skill in agent_cap.skills.items():
            score = (
                skill.level.value * 10 +
                skill.experience * 0.1 +
                skill.success_rate * 5
            )
            skill_scores[skill_name] = score
        
        # Sort by score
        sorted_skills = sorted(
            skill_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [skill[0] for skill in sorted_skills[:top_n]]

import time