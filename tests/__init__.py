"""Testing Framework & Validation Suite

This module provides comprehensive testing and validation tools
for the PI-HMARL system.
"""

from .test_framework import (
    TestFramework, TestConfig, TestResult,
    TestSuite, TestCase
)
from .unit_tests import (
    PhysicsTests, CommunicationTests, CoordinationTests,
    EnergyTests, TaskTests
)
from .integration_tests import (
    SystemIntegrationTests, ScenarioTests,
    PerformanceTests, StressTests
)
from .validation import (
    ModelValidator, PhysicsValidator, SafetyValidator,
    PerformanceValidator, BehaviorValidator
)
from .benchmarks import (
    BenchmarkSuite, PerformanceBenchmark,
    ScalabilityBenchmark, EfficiencyBenchmark
)
from .simulation_tests import (
    SimulationValidator, ScenarioGenerator,
    MonteCarloTester, EdgeCaseTester
)

__all__ = [
    # Framework
    'TestFramework', 'TestConfig', 'TestResult',
    'TestSuite', 'TestCase',
    
    # Unit Tests
    'PhysicsTests', 'CommunicationTests', 'CoordinationTests',
    'EnergyTests', 'TaskTests',
    
    # Integration Tests
    'SystemIntegrationTests', 'ScenarioTests',
    'PerformanceTests', 'StressTests',
    
    # Validation
    'ModelValidator', 'PhysicsValidator', 'SafetyValidator',
    'PerformanceValidator', 'BehaviorValidator',
    
    # Benchmarks
    'BenchmarkSuite', 'PerformanceBenchmark',
    'ScalabilityBenchmark', 'EfficiencyBenchmark',
    
    # Simulation Tests
    'SimulationValidator', 'ScenarioGenerator',
    'MonteCarloTester', 'EdgeCaseTester'
]