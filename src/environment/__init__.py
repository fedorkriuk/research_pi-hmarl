"""Multi-Agent Environment for PI-HMARL

This module implements the multi-agent environment with real physics
integration for the PI-HMARL framework.
"""

from .base_env import MultiAgentEnvironment
from .agent_manager import AgentManager
from .spaces import ObservationSpace, ActionSpace
from .communication import CommunicationProtocol
from .state_manager import StateManager
from .reward_calculator import RewardCalculator
from .episode_manager import EpisodeManager
from .visualization import EnvironmentVisualizer

__all__ = [
    'MultiAgentEnvironment',
    'AgentManager',
    'ObservationSpace',
    'ActionSpace',
    'CommunicationProtocol',
    'StateManager',
    'RewardCalculator',
    'EpisodeManager',
    'EnvironmentVisualizer'
]