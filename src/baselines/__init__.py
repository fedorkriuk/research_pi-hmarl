# Baseline algorithms for Q1 comparison
"""
Comprehensive baseline implementations for fair comparison
"""

from .ippo_baseline import IPPOBaseline
from .iql_baseline import IQLBaseline
from .physics_penalty_mappo import PhysicsPenaltyMAPPO

__all__ = [
    'IPPOBaseline',
    'IQLBaseline', 
    'PhysicsPenaltyMAPPO'
]