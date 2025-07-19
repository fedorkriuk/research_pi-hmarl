"""Validation Tools for PI-HMARL System

This module provides comprehensive validation tools for model
correctness, physics accuracy, and safety verification.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation result structure"""
    validator_name: str
    passed: bool
    score: float
    metrics: Dict[str, Any]
    violations: List[str]
    recommendations: List[str]


class ModelValidator:
    """Validates model architecture and behavior"""
    
    def __init__(self):
        """Initialize model validator"""
        self.validation_tests = [
            self._validate_architecture,
            self._validate_gradients,
            self._validate_outputs,
            self._validate_stability,
            self._validate_convergence
        ]
    
    def validate_model(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int = 100
    ) -> ValidationResult:
        """Validate model comprehensively
        
        Args:
            model: Model to validate
            sample_input: Sample input tensor
            num_iterations: Number of test iterations
            
        Returns:
            Validation result
        """
        violations = []
        metrics = {}
        recommendations = []
        
        # Run validation tests
        for test in self.validation_tests:
            try:
                test_result = test(model, sample_input, num_iterations)
                metrics.update(test_result['metrics'])
                
                if not test_result['passed']:
                    violations.extend(test_result['violations'])
                    recommendations.extend(test_result['recommendations'])
                    
            except Exception as e:
                violations.append(f"Test {test.__name__} failed: {str(e)}")
        
        # Calculate overall score
        score = 1.0 - (len(violations) / (len(self.validation_tests) * 5))
        score = max(0.0, score)
        
        return ValidationResult(
            validator_name="ModelValidator",
            passed=len(violations) == 0,
            score=score,
            metrics=metrics,
            violations=violations,
            recommendations=recommendations
        )
    
    def _validate_architecture(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int
    ) -> Dict[str, Any]:
        """Validate model architecture"""
        violations = []
        recommendations = []
        
        # Check input/output dimensions
        try:
            output = model(sample_input)
            
            # Verify output shape
            if len(output.shape) < 2:
                violations.append("Output dimension too low")
                recommendations.append("Ensure output has batch dimension")
            
        except Exception as e:
            violations.append(f"Forward pass failed: {str(e)}")
            recommendations.append("Check model architecture and input compatibility")
        
        # Check parameter count
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if param_count == 0:
            violations.append("Model has no parameters")
        
        if trainable_count == 0:
            violations.append("Model has no trainable parameters")
        
        # Check for NaN/Inf parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                violations.append(f"NaN values in parameter {name}")
            if torch.isinf(param).any():
                violations.append(f"Inf values in parameter {name}")
        
        metrics = {
            'param_count': param_count,
            'trainable_count': trainable_count,
            'architecture_valid': len(violations) == 0
        }
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'recommendations': recommendations,
            'metrics': metrics
        }
    
    def _validate_gradients(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int
    ) -> Dict[str, Any]:
        """Validate gradient flow"""
        violations = []
        recommendations = []
        
        # Enable gradient computation
        model.train()
        
        # Track gradient statistics
        grad_norms = []
        vanishing_count = 0
        exploding_count = 0
        
        for _ in range(min(num_iterations, 10)):
            # Forward pass
            output = model(sample_input)
            loss = output.mean()  # Simple loss
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Check gradients
            total_norm = 0.0
            param_count = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.norm().item()
                    total_norm += param_norm ** 2
                    param_count += 1
                    
                    if param_norm < 1e-7:
                        vanishing_count += 1
                    elif param_norm > 1e3:
                        exploding_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)
        
        # Analyze gradient flow
        if grad_norms:
            avg_grad_norm = np.mean(grad_norms)
            
            if avg_grad_norm < 1e-6:
                violations.append("Vanishing gradients detected")
                recommendations.append("Check activation functions and initialization")
            
            if avg_grad_norm > 1e2:
                violations.append("Exploding gradients detected")
                recommendations.append("Consider gradient clipping or different initialization")
            
            if vanishing_count > param_count * 0.5:
                violations.append("Many parameters have vanishing gradients")
                recommendations.append("Review network architecture and depth")
        
        metrics = {
            'avg_gradient_norm': avg_grad_norm if grad_norms else 0.0,
            'vanishing_grad_ratio': vanishing_count / max(param_count * 10, 1),
            'exploding_grad_ratio': exploding_count / max(param_count * 10, 1)
        }
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'recommendations': recommendations,
            'metrics': metrics
        }
    
    def _validate_outputs(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int
    ) -> Dict[str, Any]:
        """Validate model outputs"""
        violations = []
        recommendations = []
        
        model.eval()
        outputs = []
        
        # Test with different inputs
        for i in range(min(num_iterations, 20)):
            # Vary input slightly
            noise = torch.randn_like(sample_input) * 0.1
            test_input = sample_input + noise
            
            with torch.no_grad():
                output = model(test_input)
                outputs.append(output)
            
            # Check for NaN/Inf
            if torch.isnan(output).any():
                violations.append(f"NaN in output at iteration {i}")
            if torch.isinf(output).any():
                violations.append(f"Inf in output at iteration {i}")
        
        if outputs:
            # Check output statistics
            all_outputs = torch.stack(outputs)
            output_mean = all_outputs.mean(dim=0)
            output_std = all_outputs.std(dim=0)
            
            # Check for dead neurons
            if (output_std < 1e-6).sum() > output_std.numel() * 0.5:
                violations.append("Many output neurons are dead (zero variance)")
                recommendations.append("Check activation functions and dropout")
            
            # Check output range
            if output_mean.abs().max() > 1e3:
                violations.append("Output values are very large")
                recommendations.append("Consider output normalization or scaling")
        
        metrics = {
            'output_mean': output_mean.mean().item() if outputs else 0.0,
            'output_std': output_std.mean().item() if outputs else 0.0,
            'outputs_valid': len(violations) == 0
        }
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'recommendations': recommendations,
            'metrics': metrics
        }
    
    def _validate_stability(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int
    ) -> Dict[str, Any]:
        """Validate model stability"""
        violations = []
        recommendations = []
        
        model.eval()
        
        # Test sensitivity to input perturbations
        base_output = model(sample_input)
        
        sensitivities = []
        for eps in [1e-3, 1e-2, 1e-1]:
            perturbed = sample_input + torch.randn_like(sample_input) * eps
            perturbed_output = model(perturbed)
            
            # Calculate sensitivity
            output_change = torch.norm(perturbed_output - base_output)
            input_change = torch.norm(perturbed - sample_input)
            
            if input_change > 0:
                sensitivity = (output_change / input_change).item()
                sensitivities.append(sensitivity)
                
                if sensitivity > 100:
                    violations.append(f"High sensitivity to perturbation (eps={eps})")
                    recommendations.append("Model may be unstable - check regularization")
        
        # Test temporal stability
        if hasattr(model, 'reset_hidden'):
            model.reset_hidden()
        
        temporal_outputs = []
        for _ in range(10):
            output = model(sample_input)
            temporal_outputs.append(output)
        
        if len(temporal_outputs) > 1:
            temporal_variance = torch.stack(temporal_outputs).var(dim=0).mean().item()
            if temporal_variance > 0.1:
                violations.append("High temporal variance in outputs")
                recommendations.append("Check recurrent connections and state handling")
        
        metrics = {
            'avg_sensitivity': np.mean(sensitivities) if sensitivities else 0.0,
            'max_sensitivity': np.max(sensitivities) if sensitivities else 0.0,
            'stable': len(violations) == 0
        }
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'recommendations': recommendations,
            'metrics': metrics
        }
    
    def _validate_convergence(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int
    ) -> Dict[str, Any]:
        """Validate model convergence properties"""
        violations = []
        recommendations = []
        
        model.train()
        
        # Simple convergence test
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        losses = []
        for _ in range(min(num_iterations, 50)):
            output = model(sample_input)
            loss = output.mean()  # Simple loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        if len(losses) > 10:
            # Check if loss is decreasing
            early_loss = np.mean(losses[:10])
            late_loss = np.mean(losses[-10:])
            
            if late_loss > early_loss:
                violations.append("Loss is increasing during training")
                recommendations.append("Check learning rate and loss function")
            
            # Check convergence rate
            if len(losses) > 20:
                loss_diff = np.diff(losses[-20:])
                if np.mean(np.abs(loss_diff)) < 1e-6:
                    # Converged
                    pass
                elif np.std(loss_diff) > np.mean(np.abs(loss_diff)):
                    violations.append("Loss is oscillating")
                    recommendations.append("Reduce learning rate or add momentum")
        
        metrics = {
            'initial_loss': losses[0] if losses else 0.0,
            'final_loss': losses[-1] if losses else 0.0,
            'converged': len(violations) == 0
        }
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'recommendations': recommendations,
            'metrics': metrics
        }


class PhysicsValidator:
    """Validates physics simulation accuracy"""
    
    def __init__(self, tolerance: float = 0.01):
        """Initialize physics validator
        
        Args:
            tolerance: Maximum acceptable error (fraction)
        """
        self.tolerance = tolerance
    
    def validate_physics(
        self,
        physics_engine: Any,
        test_cases: Optional[List[Dict]] = None
    ) -> ValidationResult:
        """Validate physics engine
        
        Args:
            physics_engine: Physics engine to validate
            test_cases: Optional test cases
            
        Returns:
            Validation result
        """
        if test_cases is None:
            test_cases = self._generate_test_cases()
        
        violations = []
        metrics = {}
        total_error = 0.0
        
        for test_case in test_cases:
            result = self._run_physics_test(physics_engine, test_case)
            
            metrics[test_case['name']] = result['error']
            total_error += result['error']
            
            if result['error'] > self.tolerance:
                violations.append(
                    f"{test_case['name']}: error {result['error']:.3f} > {self.tolerance}"
                )
        
        avg_error = total_error / len(test_cases) if test_cases else 0.0
        
        recommendations = []
        if violations:
            recommendations.append("Increase physics simulation timestep resolution")
            recommendations.append("Verify physical constants and parameters")
        
        return ValidationResult(
            validator_name="PhysicsValidator",
            passed=len(violations) == 0,
            score=max(0.0, 1.0 - avg_error / self.tolerance),
            metrics=metrics,
            violations=violations,
            recommendations=recommendations
        )
    
    def _generate_test_cases(self) -> List[Dict]:
        """Generate physics test cases"""
        return [
            {
                'name': 'free_fall',
                'description': 'Test gravity acceleration',
                'initial_state': {
                    'position': torch.tensor([0.0, 0.0, 100.0]),
                    'velocity': torch.tensor([0.0, 0.0, 0.0])
                },
                'duration': 2.0,
                'expected': {
                    'position_z': 100.0 - 0.5 * 9.81 * 4.0  # z = z0 - 0.5*g*t^2
                }
            },
            {
                'name': 'projectile_motion',
                'description': 'Test ballistic trajectory',
                'initial_state': {
                    'position': torch.tensor([0.0, 0.0, 0.0]),
                    'velocity': torch.tensor([10.0, 0.0, 10.0])
                },
                'duration': 1.0,
                'expected': {
                    'position_x': 10.0,  # x = v_x * t
                    'position_z': 10.0 - 0.5 * 9.81  # z = v_z * t - 0.5*g*t^2
                }
            },
            {
                'name': 'drag_force',
                'description': 'Test aerodynamic drag',
                'initial_state': {
                    'position': torch.tensor([0.0, 0.0, 50.0]),
                    'velocity': torch.tensor([20.0, 0.0, 0.0])
                },
                'duration': 2.0,
                'expected': {
                    'velocity_reduction': 0.1  # Expect ~10% velocity reduction
                }
            },
            {
                'name': 'hover_stability',
                'description': 'Test hover physics',
                'initial_state': {
                    'position': torch.tensor([0.0, 0.0, 50.0]),
                    'velocity': torch.tensor([0.0, 0.0, 0.0]),
                    'mass': 1.3,
                    'thrust': torch.tensor([0.0, 0.0, 1.3 * 9.81])
                },
                'duration': 5.0,
                'expected': {
                    'position_z': 50.0,  # Should maintain altitude
                    'velocity_z': 0.0
                }
            }
        ]
    
    def _run_physics_test(
        self,
        physics_engine: Any,
        test_case: Dict
    ) -> Dict[str, float]:
        """Run individual physics test
        
        Args:
            physics_engine: Physics engine
            test_case: Test case specification
            
        Returns:
            Test results with error metrics
        """
        # Setup initial state
        state = test_case['initial_state'].copy()
        
        # Run simulation
        timestep = 0.01
        steps = int(test_case['duration'] / timestep)
        
        for _ in range(steps):
            # Apply physics update
            if 'thrust' in state:
                new_state = physics_engine.step_drone_physics(
                    state['position'],
                    state['velocity'],
                    torch.zeros(3),  # orientation
                    state['thrust'],
                    torch.zeros(3),  # torque
                    state.get('mass', 1.3)
                )
            else:
                # Free motion
                new_state = physics_engine.step_physics(
                    state['position'],
                    state['velocity'],
                    timestep
                )
            
            state['position'] = new_state['position']
            state['velocity'] = new_state['velocity']
        
        # Calculate error
        error = 0.0
        count = 0
        
        for key, expected_value in test_case['expected'].items():
            if key == 'position_x':
                actual = state['position'][0].item()
                error += abs(actual - expected_value) / abs(expected_value + 1e-6)
                count += 1
            elif key == 'position_z':
                actual = state['position'][2].item()
                error += abs(actual - expected_value) / abs(expected_value + 1e-6)
                count += 1
            elif key == 'velocity_reduction':
                initial_speed = test_case['initial_state']['velocity'].norm().item()
                final_speed = state['velocity'].norm().item()
                actual_reduction = (initial_speed - final_speed) / initial_speed
                error += abs(actual_reduction - expected_value) / expected_value
                count += 1
        
        avg_error = error / count if count > 0 else 0.0
        
        return {'error': avg_error}


class SafetyValidator:
    """Validates safety constraints and requirements"""
    
    def __init__(self):
        """Initialize safety validator"""
        self.safety_constraints = {
            'min_altitude': 5.0,  # meters
            'max_altitude': 120.0,  # meters (FAA limit)
            'max_velocity': 20.0,  # m/s
            'min_separation': 5.0,  # meters
            'max_acceleration': 20.0,  # m/s^2
            'max_angular_velocity': 3.14,  # rad/s
            'min_battery': 0.2,  # 20% reserve
            'max_wind_speed': 15.0,  # m/s
            'geofence_margin': 10.0  # meters
        }
    
    def validate_safety(
        self,
        trajectory: torch.Tensor,
        agent_states: List[Dict[str, Any]],
        environment: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate safety constraints
        
        Args:
            trajectory: Planned trajectory
            agent_states: Current agent states
            environment: Environmental conditions
            
        Returns:
            Validation result
        """
        violations = []
        metrics = {}
        
        # Check altitude constraints
        altitude_violations = self._check_altitude_constraints(trajectory)
        violations.extend(altitude_violations)
        
        # Check velocity constraints
        velocity_violations = self._check_velocity_constraints(trajectory)
        violations.extend(velocity_violations)
        
        # Check separation constraints
        if len(agent_states) > 1:
            separation_violations = self._check_separation_constraints(agent_states)
            violations.extend(separation_violations)
        
        # Check battery constraints
        battery_violations = self._check_battery_constraints(agent_states)
        violations.extend(battery_violations)
        
        # Check environmental constraints
        if environment:
            env_violations = self._check_environmental_constraints(
                agent_states,
                environment
            )
            violations.extend(env_violations)
        
        # Calculate safety score
        total_checks = 5
        failed_checks = min(len(violations), total_checks)
        score = (total_checks - failed_checks) / total_checks
        
        recommendations = []
        if violations:
            recommendations.append("Adjust trajectory to respect safety constraints")
            recommendations.append("Implement safety margin in planning")
            recommendations.append("Consider emergency landing procedures")
        
        return ValidationResult(
            validator_name="SafetyValidator",
            passed=len(violations) == 0,
            score=score,
            metrics=metrics,
            violations=violations,
            recommendations=recommendations
        )
    
    def _check_altitude_constraints(self, trajectory: torch.Tensor) -> List[str]:
        """Check altitude safety constraints"""
        violations = []
        
        if len(trajectory.shape) >= 2 and trajectory.shape[1] >= 3:
            altitudes = trajectory[:, 2]
            
            min_alt = altitudes.min().item()
            max_alt = altitudes.max().item()
            
            if min_alt < self.safety_constraints['min_altitude']:
                violations.append(
                    f"Altitude below minimum: {min_alt:.1f}m < {self.safety_constraints['min_altitude']}m"
                )
            
            if max_alt > self.safety_constraints['max_altitude']:
                violations.append(
                    f"Altitude above maximum: {max_alt:.1f}m > {self.safety_constraints['max_altitude']}m"
                )
        
        return violations
    
    def _check_velocity_constraints(self, trajectory: torch.Tensor) -> List[str]:
        """Check velocity safety constraints"""
        violations = []
        
        if len(trajectory) > 1:
            # Calculate velocities from trajectory
            dt = 0.1  # Assumed timestep
            velocities = torch.diff(trajectory, dim=0) / dt
            
            speeds = torch.norm(velocities, dim=1)
            max_speed = speeds.max().item()
            
            if max_speed > self.safety_constraints['max_velocity']:
                violations.append(
                    f"Velocity exceeds maximum: {max_speed:.1f}m/s > {self.safety_constraints['max_velocity']}m/s"
                )
            
            # Check accelerations
            if len(velocities) > 1:
                accelerations = torch.diff(velocities, dim=0) / dt
                max_accel = torch.norm(accelerations, dim=1).max().item()
                
                if max_accel > self.safety_constraints['max_acceleration']:
                    violations.append(
                        f"Acceleration exceeds maximum: {max_accel:.1f}m/s² > {self.safety_constraints['max_acceleration']}m/s²"
                    )
        
        return violations
    
    def _check_separation_constraints(self, agent_states: List[Dict[str, Any]]) -> List[str]:
        """Check agent separation constraints"""
        violations = []
        
        positions = []
        for state in agent_states:
            if 'position' in state:
                positions.append(state['position'])
        
        if len(positions) > 1:
            positions_tensor = torch.stack(positions)
            
            # Calculate pairwise distances
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    distance = torch.norm(positions_tensor[i] - positions_tensor[j]).item()
                    
                    if distance < self.safety_constraints['min_separation']:
                        violations.append(
                            f"Agents {i} and {j} too close: {distance:.1f}m < {self.safety_constraints['min_separation']}m"
                        )
        
        return violations
    
    def _check_battery_constraints(self, agent_states: List[Dict[str, Any]]) -> List[str]:
        """Check battery safety constraints"""
        violations = []
        
        for i, state in enumerate(agent_states):
            if 'battery_soc' in state:
                soc = state['battery_soc']
                
                if soc < self.safety_constraints['min_battery']:
                    violations.append(
                        f"Agent {i} battery critical: {soc:.1%} < {self.safety_constraints['min_battery']:.0%}"
                    )
        
        return violations
    
    def _check_environmental_constraints(
        self,
        agent_states: List[Dict[str, Any]],
        environment: Dict[str, Any]
    ) -> List[str]:
        """Check environmental safety constraints"""
        violations = []
        
        # Check wind speed
        if 'wind_speed' in environment:
            wind_speed = environment['wind_speed']
            
            if wind_speed > self.safety_constraints['max_wind_speed']:
                violations.append(
                    f"Wind speed exceeds safe limit: {wind_speed:.1f}m/s > {self.safety_constraints['max_wind_speed']}m/s"
                )
        
        # Check geofence
        if 'geofence' in environment:
            geofence = environment['geofence']
            
            for i, state in enumerate(agent_states):
                if 'position' in state:
                    pos = state['position']
                    
                    # Simple rectangular geofence check
                    if (pos[0] < geofence[0] + self.safety_constraints['geofence_margin'] or
                        pos[0] > geofence[2] - self.safety_constraints['geofence_margin'] or
                        pos[1] < geofence[1] + self.safety_constraints['geofence_margin'] or
                        pos[1] > geofence[3] - self.safety_constraints['geofence_margin']):
                        
                        violations.append(f"Agent {i} near geofence boundary")
        
        return violations


class PerformanceValidator:
    """Validates system performance metrics"""
    
    def __init__(self):
        """Initialize performance validator"""
        self.performance_targets = {
            'inference_time': 0.033,  # 30 FPS
            'planning_time': 0.1,  # 100ms
            'communication_latency': 0.05,  # 50ms
            'energy_efficiency': 0.8,  # 80%
            'task_success_rate': 0.9,  # 90%
            'coordination_efficiency': 0.85  # 85%
        }
    
    def validate_performance(
        self,
        performance_data: Dict[str, List[float]],
        duration: float
    ) -> ValidationResult:
        """Validate performance metrics
        
        Args:
            performance_data: Performance measurements
            duration: Test duration
            
        Returns:
            Validation result
        """
        violations = []
        metrics = {}
        
        # Analyze each metric
        for metric_name, measurements in performance_data.items():
            if not measurements:
                continue
            
            # Calculate statistics
            mean_val = np.mean(measurements)
            std_val = np.std(measurements)
            p95_val = np.percentile(measurements, 95)
            
            metrics[f"{metric_name}_mean"] = mean_val
            metrics[f"{metric_name}_std"] = std_val
            metrics[f"{metric_name}_p95"] = p95_val
            
            # Check against targets
            if metric_name in self.performance_targets:
                target = self.performance_targets[metric_name]
                
                if metric_name.endswith('_time') or metric_name.endswith('_latency'):
                    # Lower is better
                    if p95_val > target:
                        violations.append(
                            f"{metric_name} P95 ({p95_val:.3f}s) exceeds target ({target}s)"
                        )
                else:
                    # Higher is better
                    if mean_val < target:
                        violations.append(
                            f"{metric_name} mean ({mean_val:.3f}) below target ({target})"
                        )
        
        # Calculate overall score
        score = 1.0
        for violation in violations:
            score *= 0.9  # 10% penalty per violation
        
        recommendations = []
        if violations:
            recommendations.append("Optimize computational bottlenecks")
            recommendations.append("Consider hardware acceleration")
            recommendations.append("Profile and optimize critical paths")
        
        return ValidationResult(
            validator_name="PerformanceValidator",
            passed=len(violations) == 0,
            score=score,
            metrics=metrics,
            violations=violations,
            recommendations=recommendations
        )


class BehaviorValidator:
    """Validates agent behavior and decision making"""
    
    def __init__(self):
        """Initialize behavior validator"""
        self.behavior_criteria = {
            'rationality': 0.8,
            'consistency': 0.9,
            'adaptability': 0.7,
            'cooperation': 0.85
        }
    
    def validate_behavior(
        self,
        agent_traces: List[Dict[str, Any]],
        scenario_type: str
    ) -> ValidationResult:
        """Validate agent behavior
        
        Args:
            agent_traces: Agent behavior traces
            scenario_type: Type of scenario
            
        Returns:
            Validation result
        """
        violations = []
        metrics = {}
        
        # Analyze rationality
        rationality_score = self._analyze_rationality(agent_traces)
        metrics['rationality'] = rationality_score
        
        if rationality_score < self.behavior_criteria['rationality']:
            violations.append(f"Low rationality score: {rationality_score:.2f}")
        
        # Analyze consistency
        consistency_score = self._analyze_consistency(agent_traces)
        metrics['consistency'] = consistency_score
        
        if consistency_score < self.behavior_criteria['consistency']:
            violations.append(f"Low consistency score: {consistency_score:.2f}")
        
        # Analyze adaptability
        adaptability_score = self._analyze_adaptability(agent_traces)
        metrics['adaptability'] = adaptability_score
        
        if adaptability_score < self.behavior_criteria['adaptability']:
            violations.append(f"Low adaptability score: {adaptability_score:.2f}")
        
        # Analyze cooperation
        if len(agent_traces) > 1:
            cooperation_score = self._analyze_cooperation(agent_traces)
            metrics['cooperation'] = cooperation_score
            
            if cooperation_score < self.behavior_criteria['cooperation']:
                violations.append(f"Low cooperation score: {cooperation_score:.2f}")
        
        # Calculate overall score
        behavior_scores = [metrics[k] for k in self.behavior_criteria.keys() if k in metrics]
        overall_score = np.mean(behavior_scores) if behavior_scores else 0.0
        
        recommendations = []
        if violations:
            recommendations.append("Review reward function design")
            recommendations.append("Increase training diversity")
            recommendations.append("Add behavioral regularization")
        
        return ValidationResult(
            validator_name="BehaviorValidator",
            passed=len(violations) == 0,
            score=overall_score,
            metrics=metrics,
            violations=violations,
            recommendations=recommendations
        )
    
    def _analyze_rationality(self, agent_traces: List[Dict[str, Any]]) -> float:
        """Analyze decision rationality"""
        rational_decisions = 0
        total_decisions = 0
        
        for trace in agent_traces:
            for decision in trace.get('decisions', []):
                total_decisions += 1
                
                # Check if decision maximizes expected reward
                if decision.get('expected_reward', 0) >= max(
                    option.get('expected_reward', 0)
                    for option in decision.get('alternatives', [])
                ):
                    rational_decisions += 1
        
        return rational_decisions / total_decisions if total_decisions > 0 else 0.0
    
    def _analyze_consistency(self, agent_traces: List[Dict[str, Any]]) -> float:
        """Analyze behavioral consistency"""
        consistency_scores = []
        
        for trace in agent_traces:
            states = trace.get('states', [])
            actions = trace.get('actions', [])
            
            if len(states) > 1 and len(actions) > 1:
                # Check consistency of similar states leading to similar actions
                state_action_map = {}
                
                for state, action in zip(states, actions):
                    state_key = self._discretize_state(state)
                    
                    if state_key not in state_action_map:
                        state_action_map[state_key] = []
                    
                    state_action_map[state_key].append(action)
                
                # Calculate action consistency for each state
                for state_key, action_list in state_action_map.items():
                    if len(action_list) > 1:
                        # Calculate variance in actions
                        action_variance = np.var(action_list, axis=0).mean()
                        consistency = 1.0 / (1.0 + action_variance)
                        consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _analyze_adaptability(self, agent_traces: List[Dict[str, Any]]) -> float:
        """Analyze adaptability to changing conditions"""
        adaptability_scores = []
        
        for trace in agent_traces:
            # Look for environment changes and response
            env_changes = trace.get('environment_changes', [])
            responses = trace.get('adaptation_responses', [])
            
            if env_changes and responses:
                # Calculate response quality
                for change, response in zip(env_changes, responses):
                    if response.get('adapted', False):
                        response_time = response.get('response_time', float('inf'))
                        # Faster response = higher score
                        score = 1.0 / (1.0 + response_time / 10.0)
                        adaptability_scores.append(score)
                    else:
                        adaptability_scores.append(0.0)
        
        return np.mean(adaptability_scores) if adaptability_scores else 0.5
    
    def _analyze_cooperation(self, agent_traces: List[Dict[str, Any]]) -> float:
        """Analyze multi-agent cooperation"""
        cooperation_events = 0
        total_opportunities = 0
        
        # Look for cooperation opportunities and outcomes
        for i in range(len(agent_traces)):
            for j in range(i + 1, len(agent_traces)):
                trace_i = agent_traces[i]
                trace_j = agent_traces[j]
                
                # Find overlapping time periods
                interactions = self._find_interactions(trace_i, trace_j)
                
                for interaction in interactions:
                    total_opportunities += 1
                    
                    if interaction.get('cooperative_outcome', False):
                        cooperation_events += 1
        
        return cooperation_events / total_opportunities if total_opportunities > 0 else 1.0
    
    def _discretize_state(self, state: Any) -> str:
        """Discretize continuous state for consistency analysis"""
        # Simple discretization - in practice would be more sophisticated
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        
        if isinstance(state, np.ndarray):
            # Round to nearest 0.1
            discretized = np.round(state, 1)
            return str(discretized)
        
        return str(state)
    
    def _find_interactions(
        self,
        trace1: Dict[str, Any],
        trace2: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find interaction points between two agent traces"""
        interactions = []
        
        # Simplified - look for proximity events
        positions1 = trace1.get('positions', [])
        positions2 = trace2.get('positions', [])
        
        min_len = min(len(positions1), len(positions2))
        
        for t in range(min_len):
            if isinstance(positions1[t], (list, np.ndarray)) and isinstance(positions2[t], (list, np.ndarray)):
                distance = np.linalg.norm(np.array(positions1[t]) - np.array(positions2[t]))
                
                if distance < 50.0:  # Within 50m
                    interactions.append({
                        'time': t,
                        'distance': distance,
                        'cooperative_outcome': trace1.get('rewards', [0])[t] > 0 and trace2.get('rewards', [0])[t] > 0
                    })
        
        return interactions