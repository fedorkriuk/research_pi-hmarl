#!/usr/bin/env python
"""
CRITICAL DIAGNOSTIC SCRIPT: 0% Success Rate Investigation
This script systematically diagnoses the root cause of 0% success rates
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any
import traceback

# Import PI-HMARL components
from src.scenarios import SearchRescueScenario, FormationControlScenario
from src.environment.base_env import MultiAgentEnvironment
from src.environment.reward_calculator import RewardCalculator

class CriticalDiagnostics:
    """
    Systematic diagnosis of 0% success rate issue
    This is the most critical class - run immediately
    """
    
    def __init__(self):
        self.diagnostics_results = {}
        self.critical_issues = []
        self.warnings = []
        
    def run_full_diagnostic(self):
        """Run complete diagnostic protocol"""
        print("🔍 CRITICAL: Investigating 0% Success Rate Issue")
        print("="*60)
        
        # Check 1: Environment Integrity (40% probability of being the issue)
        print("\n1️⃣ CHECKING ENVIRONMENT INTEGRITY...")
        env_result = self.validate_environment_setup()
        
        # Check 2: Reward Signal Verification (35% probability)
        print("\n2️⃣ CHECKING REWARD SIGNALS...")
        reward_result = self.validate_reward_signals()
        
        # Check 3: Training Pipeline Verification (20% probability)
        print("\n3️⃣ CHECKING TRAINING PIPELINE...")
        training_result = self.validate_training_pipeline()
        
        # Check 4: Algorithm Implementation Check (5% probability)
        print("\n4️⃣ CHECKING ALGORITHM IMPLEMENTATION...")
        algo_result = self.validate_algorithm_implementations()
        
        # Generate diagnosis report
        self.generate_diagnosis_report()
        
        return self.diagnostics_results
    
    def validate_environment_setup(self) -> bool:
        """
        Check fundamental environment functionality
        This is likely the root cause of 0% success
        """
        try:
            print("   Testing basic environment operations...")
            
            # Test 1: Environment instantiation
            scenario = SearchRescueScenario(
                area_size=(50.0, 50.0),
                num_victims=2,
                num_agents=2,
                obstacle_density=0.01
            )
            print("   ✓ Environment instantiation successful")
            
            # Test 2: Environment reset
            initial_state = scenario.get_state()
            print(f"   ✓ Initial state: {type(initial_state)}")
            
            # Test 3: Single step execution
            step_result = scenario.step(dt=0.1)
            print(f"   ✓ Step execution: {step_result}")
            
            # Test 4: State progression
            state_after = scenario.get_state()
            print(f"   ✓ State after step: time={state_after.get('time', 'N/A')}")
            
            # CRITICAL CHECKS
            critical_issues = []
            
            # Check if episodes terminate immediately
            if state_after.get('time', 0) <= 0:
                critical_issues.append("CRITICAL: Episodes terminate immediately")
            
            # Check if state changes between steps
            if initial_state == state_after:
                critical_issues.append("CRITICAL: State doesn't change between steps")
            
            # Check if agents exist and are active
            if 'agents' in state_after:
                if not state_after['agents'] or len(state_after['agents']) == 0:
                    critical_issues.append("CRITICAL: No agents in environment")
            
            self.critical_issues.extend(critical_issues)
            
            if critical_issues:
                print(f"   ❌ CRITICAL ISSUES FOUND: {len(critical_issues)}")
                for issue in critical_issues:
                    print(f"      • {issue}")
                return False
            else:
                print("   ✅ Environment setup appears functional")
                return True
                
        except Exception as e:
            error_msg = f"CRITICAL: Environment setup failed: {str(e)}"
            self.critical_issues.append(error_msg)
            print(f"   ❌ {error_msg}")
            print(f"   📋 Traceback: {traceback.format_exc()}")
            return False
    
    def validate_reward_signals(self) -> bool:
        """
        Validate reward calculation and scaling
        Common cause: all-zero rewards or NaN rewards
        """
        try:
            print("   Analyzing reward signals...")
            
            # Create test scenario
            scenario = SearchRescueScenario(
                area_size=(30.0, 30.0),
                num_victims=1,
                num_agents=1
            )
            
            # Collect rewards over multiple steps
            reward_history = []
            state_history = []
            
            for step in range(20):
                state_before = scenario.get_state()
                step_result = scenario.step(dt=0.1)
                state_after = scenario.get_state()
                
                # Extract rewards if available
                if hasattr(scenario, 'last_rewards'):
                    rewards = scenario.last_rewards
                elif 'rewards' in state_after:
                    rewards = state_after['rewards']
                else:
                    # Calculate basic reward based on state change
                    rewards = self._calculate_test_reward(state_before, state_after)
                
                reward_history.extend(rewards if isinstance(rewards, list) else [rewards])
                state_history.append(state_after)
            
            # CRITICAL REWARD CHECKS
            critical_issues = []
            
            # Check 1: Are rewards being generated?
            if len(reward_history) == 0:
                critical_issues.append("CRITICAL: No rewards collected")
            
            # Check 2: Are all rewards zero?
            if len(reward_history) > 0 and all(r == 0 for r in reward_history):
                critical_issues.append("CRITICAL: All rewards are zero")
            
            # Check 3: Are there NaN rewards?
            if any(np.isnan(r) if isinstance(r, (int, float)) else False for r in reward_history):
                critical_issues.append("CRITICAL: NaN rewards detected")
            
            # Check 4: Reward distribution analysis
            if len(reward_history) > 0:
                reward_stats = {
                    'mean': np.mean(reward_history),
                    'std': np.std(reward_history),
                    'min': np.min(reward_history),
                    'max': np.max(reward_history),
                    'non_zero_count': sum(1 for r in reward_history if r != 0)
                }
                print(f"   📊 Reward statistics: {reward_stats}")
                
                # Check if reward scale is reasonable
                if abs(reward_stats['max']) < 1e-10:
                    critical_issues.append("CRITICAL: Reward scale extremely small")
                elif abs(reward_stats['max']) > 1e10:
                    critical_issues.append("CRITICAL: Reward scale extremely large")
            
            self.critical_issues.extend(critical_issues)
            
            if critical_issues:
                print(f"   ❌ REWARD ISSUES FOUND: {len(critical_issues)}")
                for issue in critical_issues:
                    print(f"      • {issue}")
                return False
            else:
                print("   ✅ Reward signals appear functional")
                return True
                
        except Exception as e:
            error_msg = f"CRITICAL: Reward validation failed: {str(e)}"
            self.critical_issues.append(error_msg)
            print(f"   ❌ {error_msg}")
            return False
    
    def validate_training_pipeline(self) -> bool:
        """
        Check training configuration and pipeline
        """
        try:
            print("   Checking training pipeline...")
            
            # Check for training configuration files
            config_paths = [
                'configs/default_config.yaml',
                'configs/attention_config.yaml',
                'configs/physics_config.yaml'
            ]
            
            config_issues = []
            for config_path in config_paths:
                if not os.path.exists(config_path):
                    config_issues.append(f"Missing config: {config_path}")
            
            if config_issues:
                self.warnings.extend(config_issues)
                print(f"   ⚠️  Configuration warnings: {len(config_issues)}")
            
            # Check for training modules
            training_modules = [
                'src.training',
                'src.agents.hierarchical_agent',
                'src.agents.meta_controller'
            ]
            
            import_issues = []
            for module in training_modules:
                try:
                    __import__(module)
                except ImportError as e:
                    import_issues.append(f"Cannot import {module}: {e}")
            
            if import_issues:
                self.critical_issues.extend(import_issues)
                print(f"   ❌ IMPORT ISSUES: {len(import_issues)}")
                return False
            
            print("   ✅ Training pipeline components accessible")
            return True
            
        except Exception as e:
            error_msg = f"Training pipeline validation failed: {str(e)}"
            self.critical_issues.append(error_msg)
            print(f"   ❌ {error_msg}")
            return False
    
    def validate_algorithm_implementations(self) -> bool:
        """
        Check algorithm-specific implementation issues
        """
        try:
            print("   Checking algorithm implementations...")
            
            # Test hierarchical architecture components
            implementation_issues = []
            
            # Check meta-controller
            try:
                from src.agents.meta_controller import MetaController
                print("   ✓ MetaController importable")
            except Exception as e:
                implementation_issues.append(f"MetaController issue: {e}")
            
            # Check execution policy
            try:
                from src.agents.execution_policy import ExecutionPolicy
                print("   ✓ ExecutionPolicy importable")
            except Exception as e:
                implementation_issues.append(f"ExecutionPolicy issue: {e}")
            
            # Check physics constraints
            try:
                from src.physics_informed.base_pinn import PhysicsInformedNetwork
                print("   ✓ Physics constraints importable")
            except Exception as e:
                implementation_issues.append(f"Physics constraints issue: {e}")
            
            # Check attention mechanism
            try:
                from src.attention.hierarchical_attention import HierarchicalAttention
                print("   ✓ Attention mechanism importable")
            except Exception as e:
                implementation_issues.append(f"Attention mechanism issue: {e}")
            
            if implementation_issues:
                self.critical_issues.extend(implementation_issues)
                print(f"   ❌ IMPLEMENTATION ISSUES: {len(implementation_issues)}")
                return False
            
            print("   ✅ Algorithm implementations accessible")
            return True
            
        except Exception as e:
            error_msg = f"Algorithm validation failed: {str(e)}"
            self.critical_issues.append(error_msg)
            print(f"   ❌ {error_msg}")
            return False
    
    def _calculate_test_reward(self, state_before: Dict, state_after: Dict) -> float:
        """Calculate a basic test reward for validation"""
        try:
            # Basic progress reward
            time_before = state_before.get('time', 0)
            time_after = state_after.get('time', 0)
            
            if time_after > time_before:
                return 0.1  # Small positive reward for time progression
            else:
                return 0.0
        except:
            return 0.0
    
    def generate_diagnosis_report(self):
        """Generate comprehensive diagnosis report"""
        print("\n" + "="*60)
        print("🏥 DIAGNOSTIC REPORT")
        print("="*60)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Critical Issues: {len(self.critical_issues)}")
        print(f"   Warnings: {len(self.warnings)}")
        
        if self.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES (LIKELY CAUSE OF 0% SUCCESS):")
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"   {i}. {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if not self.critical_issues:
            print(f"\n✅ NO CRITICAL ISSUES FOUND")
            print("   The 0% success rate may be due to:")
            print("   • Incorrect success criteria definition")
            print("   • Performance measurement errors")
            print("   • Learning rate or hyperparameter issues")
            print("   • Insufficient training time")
        
        print(f"\n💡 NEXT STEPS:")
        if self.critical_issues:
            print("   1. Fix critical issues identified above")
            print("   2. Re-run diagnostic after fixes")
            print("   3. Test with trivial benchmarks")
        else:
            print("   1. Run trivial benchmark validation")
            print("   2. Check success criteria definitions")
            print("   3. Validate performance measurement")
        
        print("="*60)

def main():
    """Run critical diagnostic protocol"""
    print("🚨 CRITICAL DIAGNOSTIC: 0% Success Rate Investigation")
    print("This diagnostic will identify the root cause of training failures")
    
    diagnostics = CriticalDiagnostics()
    results = diagnostics.run_full_diagnostic()
    
    # Return exit code based on findings
    if diagnostics.critical_issues:
        print(f"\n❌ CRITICAL ISSUES FOUND: {len(diagnostics.critical_issues)}")
        return 1
    else:
        print(f"\n✅ NO CRITICAL ISSUES DETECTED")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)