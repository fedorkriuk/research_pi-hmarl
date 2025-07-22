# Q1 Publication-Grade Experimental Framework for PI-HMARL
"""
Comprehensive experimental framework meeting Q1 journal standards
(JMLR, Nature Machine Intelligence, IEEE TPAMI)
"""

from .q1_statistical_analyzer import Q1StatisticalAnalyzer
from .comprehensive_baseline_suite import ComprehensiveBaselineSuite
from .theoretical_analyzer import TheoreticalAnalyzer
from .computational_profiler import Q1ComputationalProfiler
from .multi_domain_scenario_generator import MultiDomainScenarioGenerator
from .q1_experiment_orchestrator import Q1ExperimentOrchestrator

__all__ = [
    'Q1StatisticalAnalyzer',
    'ComprehensiveBaselineSuite',
    'TheoreticalAnalyzer',
    'Q1ComputationalProfiler',
    'MultiDomainScenarioGenerator',
    'Q1ExperimentOrchestrator'
]

# Q1 Journal Standards Configuration
Q1_STANDARDS = {
    'minimum_seeds': 30,
    'bootstrap_iterations': 50000,
    'permutation_iterations': 50000,
    'significance_level': 0.01,
    'effect_size_threshold': 1.2,  # Flag anything above this as potentially unrealistic
    'required_baselines': [
        'IPPO', 'IQL', 'QMIX', 'MADDPG', 'MAPPO',
        'Physics-MAPPO', 'SOTA-Physics-RL', 'HAD', 'HC-MARL',
        'Random', 'Centralized-Optimal', 'Human-Expert'
    ],
    'theoretical_requirements': [
        'convergence_proof',
        'sample_complexity_bounds',
        'regret_analysis',
        'stability_guarantees',
        'optimality_gap'
    ],
    'domains': ['aerial', 'ground', 'underwater', 'space'],
    'minimum_episode_length': 1000,
    'noise_levels': [0.05, 0.10, 0.15],  # 5%, 10%, 15% observation noise
    'failure_rates': [0.1, 0.2, 0.3],    # Up to 30% agent failure
}