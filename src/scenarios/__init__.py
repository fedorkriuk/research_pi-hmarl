"""Advanced Scenarios and Extensions Module

This module provides implementations of advanced multi-agent scenarios
and extensions for the PI-HMARL system.
"""

from .search_rescue import (
    SearchRescueScenario, SearchAgent, RescueCoordinator,
    VictimModel, SearchPattern, RescueMission
)
from .swarm_exploration import (
    SwarmExplorationScenario, ExplorationAgent, SwarmCoordinator,
    MapBuilder, FrontierExploration, AreaCoverage
)
from .formation_control import (
    FormationControlScenario, FormationAgent, FormationController,
    FormationPattern, ObstacleAvoidance, DynamicFormation
)
from .cooperative_manipulation import (
    CooperativeManipulationScenario, ManipulationAgent,
    ObjectTransport, ForceCoordination, GraspPlanning
)
from .adversarial_scenarios import (
    AdversarialScenario, PursuitEvasion, TerritorialDefense,
    CompetitiveResourceGathering, StrategicPlanning
)

__all__ = [
    # Search and Rescue
    'SearchRescueScenario', 'SearchAgent', 'RescueCoordinator',
    'VictimModel', 'SearchPattern', 'RescueMission',
    
    # Swarm Exploration
    'SwarmExplorationScenario', 'ExplorationAgent', 'SwarmCoordinator',
    'MapBuilder', 'FrontierExploration', 'AreaCoverage',
    
    # Formation Control
    'FormationControlScenario', 'FormationAgent', 'FormationController',
    'FormationPattern', 'ObstacleAvoidance', 'DynamicFormation',
    
    # Cooperative Manipulation
    'CooperativeManipulationScenario', 'ManipulationAgent',
    'ObjectTransport', 'ForceCoordination', 'GraspPlanning',
    
    # Adversarial Scenarios
    'AdversarialScenario', 'PursuitEvasion', 'TerritorialDefense',
    'CompetitiveResourceGathering', 'StrategicPlanning'
]