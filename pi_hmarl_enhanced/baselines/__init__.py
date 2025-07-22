"""
Advanced baseline implementations for PI-HMARL comparison
"""

from .sebastian_physics_marl import SebastianPhysicsMARL
from .scalable_mappo_lagrangian import ScalableMappoLagrangian
from .macpo import MACPO
from .hc_marl import HierarchicalConsensusMARL

__all__ = [
    'SebastianPhysicsMARL',
    'ScalableMappoLagrangian', 
    'MACPO',
    'HierarchicalConsensusMARL'
]