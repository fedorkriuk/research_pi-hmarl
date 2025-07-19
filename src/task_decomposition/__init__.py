"""Cooperative Task Decomposition and Assignment System

This module implements intelligent task decomposition and assignment
for multi-agent teams with real-world constraints.
"""

from .task_analyzer import (
    TaskAnalyzer, TaskComplexityEstimator, TaskDecomposer,
    TaskDependencyGraph, TaskPrimitives
)
from .assignment_optimizer import (
    AssignmentOptimizer, CapabilityMatcher, LoadBalancer,
    AuctionBasedAssignment, HungarianAssignment
)
from .cooperative_planner import (
    CooperativePlanner, TeamFormation, RoleAssignment,
    CoordinationProtocol, SynchronizationManager
)
from .task_monitor import (
    TaskMonitor, ProgressTracker, PerformanceEvaluator,
    TaskReallocation, FailureHandler
)
from .skill_matching import (
    SkillMatcher, AgentCapabilities, TaskRequirements,
    SkillLearning, CapabilityEvolution
)

__all__ = [
    # Task Analyzer
    'TaskAnalyzer', 'TaskComplexityEstimator', 'TaskDecomposer',
    'TaskDependencyGraph', 'TaskPrimitives',
    
    # Assignment Optimizer
    'AssignmentOptimizer', 'CapabilityMatcher', 'LoadBalancer',
    'AuctionBasedAssignment', 'HungarianAssignment',
    
    # Cooperative Planner
    'CooperativePlanner', 'TeamFormation', 'RoleAssignment',
    'CoordinationProtocol', 'SynchronizationManager',
    
    # Task Monitor
    'TaskMonitor', 'ProgressTracker', 'PerformanceEvaluator',
    'TaskReallocation', 'FailureHandler',
    
    # Skill Matching
    'SkillMatcher', 'AgentCapabilities', 'TaskRequirements',
    'SkillLearning', 'CapabilityEvolution'
]