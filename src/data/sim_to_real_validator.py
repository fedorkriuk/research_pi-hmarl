"""Sim-to-Real Validation Framework

This module provides validation tools to verify that models trained on
synthetic data with real parameters transfer effectively to real-world
scenarios.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for sim-to-real validation metrics"""
    # Performance metrics
    task_success_rate: float
    performance_gap: float  # Real vs synthetic performance difference
    
    # Physics compliance
    constraint_violation_rate: float
    physics_realism_score: float
    
    # Energy metrics
    energy_prediction_error: float
    endurance_prediction_error: float
    
    # Control metrics
    control_smoothness: float
    response_accuracy: float
    
    # Safety metrics
    safety_violation_count: int
    min_separation_maintained: float
    
    # Transfer metrics
    adaptation_iterations: int
    domain_shift_magnitude: float
    
    # Statistical significance
    confidence_interval: Tuple[float, float]
    p_value: float


class SimToRealValidator:
    """Validates sim-to-real transfer for PI-HMARL models"""
    
    def __init__(
        self,
        synthetic_model_path: Optional[Path] = None,
        real_data_path: Optional[Path] = None
    ):
        """Initialize the sim-to-real validator
        
        Args:
            synthetic_model_path: Path to model trained on synthetic data
            real_data_path: Path to real validation data
        """
        self.synthetic_model_path = synthetic_model_path
        self.real_data_path = real_data_path
        
        # Validation thresholds
        self.performance_threshold = 0.9  # 90% of synthetic performance
        self.physics_compliance_threshold = 0.95  # 95% physics compliance
        self.safety_threshold = 0.99  # 99% safety compliance
        
        # Domain adaptation parameters
        self.adaptation_steps = 100
        self.adaptation_lr = 0.001
        
        logger.info("Initialized SimToRealValidator")
    
    def validate_transfer(
        self,
        synthetic_results: Dict[str, np.ndarray],
        real_results: Dict[str, np.ndarray],
        scenario_type: str = "general"
    ) -> ValidationMetrics:
        """Validate sim-to-real transfer performance
        
        Args:
            synthetic_results: Results from synthetic data
            real_results: Results from real validation data
            scenario_type: Type of scenario being validated
            
        Returns:
            Validation metrics
        """
        # Task performance
        task_success_rate = self._compute_task_success(real_results, scenario_type)
        synthetic_success = self._compute_task_success(synthetic_results, scenario_type)
        performance_gap = abs(synthetic_success - task_success_rate)
        
        # Physics compliance
        constraint_violations = self._compute_constraint_violations(real_results)
        constraint_violation_rate = np.mean(constraint_violations)
        physics_realism = self._compute_physics_realism(real_results)
        
        # Energy metrics
        energy_error = self._compute_energy_prediction_error(
            synthetic_results, real_results
        )
        endurance_error = self._compute_endurance_error(
            synthetic_results, real_results
        )
        
        # Control metrics
        control_smoothness = self._compute_control_smoothness(real_results)
        response_accuracy = self._compute_response_accuracy(
            synthetic_results, real_results
        )
        
        # Safety metrics
        safety_violations = self._count_safety_violations(real_results)
        min_separation = self._compute_min_separation(real_results)
        
        # Transfer metrics
        adaptation_iters = self._estimate_adaptation_iterations(
            synthetic_results, real_results
        )
        domain_shift = self._compute_domain_shift(
            synthetic_results, real_results
        )
        
        # Statistical analysis
        confidence_interval, p_value = self._compute_statistics(
            synthetic_results, real_results
        )
        
        return ValidationMetrics(
            task_success_rate=task_success_rate,
            performance_gap=performance_gap,
            constraint_violation_rate=constraint_violation_rate,
            physics_realism_score=physics_realism,
            energy_prediction_error=energy_error,
            endurance_prediction_error=endurance_error,
            control_smoothness=control_smoothness,
            response_accuracy=response_accuracy,
            safety_violation_count=safety_violations,
            min_separation_maintained=min_separation,
            adaptation_iterations=adaptation_iters,
            domain_shift_magnitude=domain_shift,
            confidence_interval=confidence_interval,
            p_value=p_value
        )
    
    def _compute_task_success(
        self,
        results: Dict[str, np.ndarray],
        scenario_type: str
    ) -> float:
        """Compute task success rate
        
        Args:
            results: Experiment results
            scenario_type: Type of scenario
            
        Returns:
            Success rate
        """
        if scenario_type == "search_rescue":
            # Success: target found within time limit
            if "target_found" in results:
                return np.mean(results["target_found"])
            
        elif scenario_type == "formation":
            # Success: formation maintained
            if "formation_error" in results:
                # Success if error below threshold
                return np.mean(results["formation_error"] < 2.0)  # 2m threshold
                
        elif scenario_type == "delivery":
            # Success: package delivered
            if "delivery_success" in results:
                return np.mean(results["delivery_success"])
        
        # Default success computation
        if "mission_success" in results:
            return np.mean(results["mission_success"])
        
        return 0.0
    
    def _compute_constraint_violations(
        self,
        results: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute physics constraint violations
        
        Args:
            results: Experiment results
            
        Returns:
            Array of violation flags
        """
        violations = []
        
        # Energy constraints
        if "battery_soc" in results:
            energy_violations = results["battery_soc"] < 0.1  # Below 10%
            violations.append(energy_violations)
        
        # Collision constraints
        if "min_separation" in results:
            collision_violations = results["min_separation"] < 2.0  # Below 2m
            violations.append(collision_violations)
        
        # Velocity constraints
        if "velocity" in results:
            speeds = np.linalg.norm(results["velocity"], axis=-1)
            velocity_violations = speeds > 20.0  # Above max speed
            violations.append(velocity_violations)
        
        if violations:
            return np.logical_or.reduce(violations)
        
        return np.array([False])
    
    def _compute_physics_realism(
        self,
        results: Dict[str, np.ndarray]
    ) -> float:
        """Compute physics realism score
        
        Args:
            results: Experiment results
            
        Returns:
            Realism score (0-1)
        """
        scores = []
        
        # Energy consumption realism
        if "power_consumption" in results:
            power = results["power_consumption"]
            # Check if power consumption is realistic (50-500W range)
            realistic_power = np.logical_and(power >= 50, power <= 500)
            scores.append(np.mean(realistic_power))
        
        # Acceleration realism
        if "acceleration" in results:
            accel_mag = np.linalg.norm(results["acceleration"], axis=-1)
            # Check if accelerations are realistic (< 3g)
            realistic_accel = accel_mag < 30.0  # 3g
            scores.append(np.mean(realistic_accel))
        
        # Trajectory smoothness
        if "positions" in results:
            # Compute jerk (derivative of acceleration)
            if len(results["positions"]) > 3:
                jerk = np.diff(results["positions"], n=3, axis=0)
                jerk_mag = np.linalg.norm(jerk, axis=-1)
                # Low jerk indicates smooth, realistic motion
                smooth_motion = jerk_mag < 10.0
                scores.append(np.mean(smooth_motion))
        
        return np.mean(scores) if scores else 1.0
    
    def _compute_energy_prediction_error(
        self,
        synthetic_results: Dict[str, np.ndarray],
        real_results: Dict[str, np.ndarray]
    ) -> float:
        """Compute energy prediction error
        
        Args:
            synthetic_results: Synthetic experiment results
            real_results: Real experiment results
            
        Returns:
            Relative error in energy prediction
        """
        if "power_consumption" not in synthetic_results or \
           "power_consumption" not in real_results:
            return 0.0
        
        synthetic_energy = np.sum(synthetic_results["power_consumption"]) * 0.01  # Wh
        real_energy = np.sum(real_results["power_consumption"]) * 0.01  # Wh
        
        if real_energy > 0:
            return abs(synthetic_energy - real_energy) / real_energy
        
        return 0.0
    
    def _compute_endurance_error(
        self,
        synthetic_results: Dict[str, np.ndarray],
        real_results: Dict[str, np.ndarray]
    ) -> float:
        """Compute endurance prediction error
        
        Args:
            synthetic_results: Synthetic experiment results
            real_results: Real experiment results
            
        Returns:
            Error in flight time prediction (minutes)
        """
        if "battery_soc" not in synthetic_results or \
           "battery_soc" not in real_results:
            return 0.0
        
        # Find time to 10% battery
        synthetic_time = self._find_battery_depletion_time(
            synthetic_results["battery_soc"]
        )
        real_time = self._find_battery_depletion_time(
            real_results["battery_soc"]
        )
        
        return abs(synthetic_time - real_time) / 60.0  # Convert to minutes
    
    def _find_battery_depletion_time(self, battery_soc: np.ndarray) -> float:
        """Find time when battery reaches 10%
        
        Args:
            battery_soc: Battery state of charge array
            
        Returns:
            Time in seconds
        """
        if len(battery_soc.shape) > 1:
            # Multiple agents - use average
            battery_soc = np.mean(battery_soc, axis=1)
        
        # Find first time below 10%
        below_threshold = battery_soc < 0.1
        if np.any(below_threshold):
            return np.argmax(below_threshold) * 0.1  # Assuming 10Hz data
        
        # If never depleted, return max time
        return len(battery_soc) * 0.1
    
    def _compute_control_smoothness(
        self,
        results: Dict[str, np.ndarray]
    ) -> float:
        """Compute control smoothness metric
        
        Args:
            results: Experiment results
            
        Returns:
            Smoothness score (0-1)
        """
        if "control_inputs" not in results:
            return 1.0
        
        controls = results["control_inputs"]
        
        # Compute control derivatives
        control_diff = np.diff(controls, axis=0)
        control_smoothness = 1.0 / (1.0 + np.mean(np.abs(control_diff)))
        
        return control_smoothness
    
    def _compute_response_accuracy(
        self,
        synthetic_results: Dict[str, np.ndarray],
        real_results: Dict[str, np.ndarray]
    ) -> float:
        """Compute control response accuracy
        
        Args:
            synthetic_results: Synthetic experiment results
            real_results: Real experiment results
            
        Returns:
            Response accuracy (0-1)
        """
        if "positions" not in synthetic_results or \
           "positions" not in real_results:
            return 1.0
        
        # Compare trajectory following accuracy
        # Align trajectories by resampling if needed
        min_len = min(len(synthetic_results["positions"]), 
                     len(real_results["positions"]))
        
        synthetic_pos = synthetic_results["positions"][:min_len]
        real_pos = real_results["positions"][:min_len]
        
        # Compute position errors
        position_errors = np.linalg.norm(synthetic_pos - real_pos, axis=-1)
        
        # Convert to accuracy (inverse of normalized error)
        max_error = 5.0  # 5m maximum expected error
        accuracy = 1.0 - np.clip(np.mean(position_errors) / max_error, 0, 1)
        
        return accuracy
    
    def _count_safety_violations(
        self,
        results: Dict[str, np.ndarray]
    ) -> int:
        """Count safety violations
        
        Args:
            results: Experiment results
            
        Returns:
            Number of safety violations
        """
        violations = 0
        
        # Collision violations
        if "min_separation" in results:
            violations += np.sum(results["min_separation"] < 1.0)  # Critical: < 1m
        
        # Battery critical violations
        if "battery_soc" in results:
            violations += np.sum(results["battery_soc"] < 0.05)  # Critical: < 5%
        
        # Altitude violations
        if "positions" in results:
            altitudes = results["positions"][..., 2]
            violations += np.sum(altitudes < 0.5)  # Too low: < 0.5m
            violations += np.sum(altitudes > 120)  # Too high: > 120m
        
        return int(violations)
    
    def _compute_min_separation(
        self,
        results: Dict[str, np.ndarray]
    ) -> float:
        """Compute minimum separation distance maintained
        
        Args:
            results: Experiment results
            
        Returns:
            Minimum separation in meters
        """
        if "min_separation" in results:
            return np.min(results["min_separation"])
        
        if "positions" in results:
            positions = results["positions"]
            min_sep = float('inf')
            
            # Compute pairwise distances
            num_steps, num_agents = positions.shape[:2]
            for t in range(num_steps):
                for i in range(num_agents):
                    for j in range(i+1, num_agents):
                        dist = np.linalg.norm(
                            positions[t, i] - positions[t, j]
                        )
                        min_sep = min(min_sep, dist)
            
            return min_sep
        
        return float('inf')
    
    def _estimate_adaptation_iterations(
        self,
        synthetic_results: Dict[str, np.ndarray],
        real_results: Dict[str, np.ndarray]
    ) -> int:
        """Estimate iterations needed for domain adaptation
        
        Args:
            synthetic_results: Synthetic experiment results
            real_results: Real experiment results
            
        Returns:
            Estimated adaptation iterations
        """
        # Estimate based on performance gap
        synthetic_perf = self._compute_task_success(synthetic_results, "general")
        real_perf = self._compute_task_success(real_results, "general")
        
        perf_gap = abs(synthetic_perf - real_perf)
        
        # Heuristic: more iterations for larger gaps
        if perf_gap < 0.05:
            return 0  # No adaptation needed
        elif perf_gap < 0.1:
            return 50
        elif perf_gap < 0.2:
            return 100
        else:
            return 200
    
    def _compute_domain_shift(
        self,
        synthetic_results: Dict[str, np.ndarray],
        real_results: Dict[str, np.ndarray]
    ) -> float:
        """Compute domain shift magnitude
        
        Args:
            synthetic_results: Synthetic experiment results  
            real_results: Real experiment results
            
        Returns:
            Domain shift magnitude
        """
        shifts = []
        
        # Compare distributions of key variables
        for key in ["power_consumption", "velocity", "acceleration"]:
            if key in synthetic_results and key in real_results:
                synthetic_data = synthetic_results[key].flatten()
                real_data = real_results[key].flatten()
                
                # Compute KL divergence between distributions
                # Use histogram approximation
                min_val = min(synthetic_data.min(), real_data.min())
                max_val = max(synthetic_data.max(), real_data.max())
                
                bins = np.linspace(min_val, max_val, 50)
                
                synthetic_hist, _ = np.histogram(synthetic_data, bins=bins)
                real_hist, _ = np.histogram(real_data, bins=bins)
                
                # Normalize
                synthetic_hist = synthetic_hist / (synthetic_hist.sum() + 1e-10)
                real_hist = real_hist / (real_hist.sum() + 1e-10)
                
                # KL divergence
                kl_div = np.sum(
                    synthetic_hist * np.log(
                        (synthetic_hist + 1e-10) / (real_hist + 1e-10)
                    )
                )
                
                shifts.append(kl_div)
        
        return np.mean(shifts) if shifts else 0.0
    
    def _compute_statistics(
        self,
        synthetic_results: Dict[str, np.ndarray],
        real_results: Dict[str, np.ndarray]
    ) -> Tuple[Tuple[float, float], float]:
        """Compute statistical significance
        
        Args:
            synthetic_results: Synthetic experiment results
            real_results: Real experiment results
            
        Returns:
            Confidence interval and p-value
        """
        # Compare success rates
        synthetic_success = self._compute_task_success(synthetic_results, "general")
        real_success = self._compute_task_success(real_results, "general")
        
        # Compute confidence interval for real performance
        n_real = len(real_results.get("mission_success", [1]))
        se = np.sqrt(real_success * (1 - real_success) / n_real)
        ci_lower = real_success - 1.96 * se
        ci_upper = real_success + 1.96 * se
        
        # Compute p-value for difference
        # Using normal approximation
        diff = abs(synthetic_success - real_success)
        se_diff = np.sqrt(
            synthetic_success * (1 - synthetic_success) / n_real +
            real_success * (1 - real_success) / n_real
        )
        
        if se_diff > 0:
            z_score = diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(z_score))
        else:
            p_value = 1.0
        
        return (ci_lower, ci_upper), p_value
    
    def visualize_validation(
        self,
        validation_metrics: ValidationMetrics,
        output_path: Optional[Path] = None
    ):
        """Visualize validation results
        
        Args:
            validation_metrics: Computed validation metrics
            output_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Performance comparison
        ax = axes[0]
        metrics = ['Task Success', 'Physics Realism', 'Control Smooth', 'Response Acc']
        values = [
            validation_metrics.task_success_rate,
            validation_metrics.physics_realism_score,
            validation_metrics.control_smoothness,
            validation_metrics.response_accuracy
        ]
        bars = ax.bar(metrics, values)
        ax.axhline(y=0.9, color='r', linestyle='--', label='Target')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        ax.legend()
        
        # Constraint violations
        ax = axes[1]
        ax.bar(['Constraint\nViolations', 'Safety\nViolations'],
               [validation_metrics.constraint_violation_rate,
                validation_metrics.safety_violation_count / 100])  # Normalize
        ax.set_ylabel('Rate / Normalized Count')
        ax.set_title('Violation Metrics')
        
        # Energy metrics
        ax = axes[2]
        ax.bar(['Energy\nError', 'Endurance\nError'],
               [validation_metrics.energy_prediction_error,
                validation_metrics.endurance_prediction_error / 10])  # Normalize
        ax.set_ylabel('Relative Error')
        ax.set_title('Energy Prediction')
        
        # Transfer metrics
        ax = axes[3]
        ax.bar(['Domain\nShift', 'Adaptation\nIters'],
               [validation_metrics.domain_shift_magnitude,
                validation_metrics.adaptation_iterations / 100])  # Normalize
        ax.set_ylabel('Magnitude / Normalized Count')
        ax.set_title('Transfer Metrics')
        
        # Statistical significance
        ax = axes[4]
        ci_low, ci_high = validation_metrics.confidence_interval
        ax.bar(['CI Lower', 'CI Upper', 'p-value'],
               [ci_low, ci_high, validation_metrics.p_value])
        ax.set_ylabel('Value')
        ax.set_title('Statistical Analysis')
        
        # Overall validation status
        ax = axes[5]
        validation_passed = (
            validation_metrics.task_success_rate > 0.9 and
            validation_metrics.constraint_violation_rate < 0.05 and
            validation_metrics.safety_violation_count == 0
        )
        
        status_color = 'green' if validation_passed else 'red'
        status_text = 'PASSED' if validation_passed else 'FAILED'
        
        ax.text(0.5, 0.5, status_text, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=24,
                color=status_color,
                weight='bold')
        ax.set_title('Validation Status')
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
    
    def generate_validation_report(
        self,
        validation_metrics: ValidationMetrics
    ) -> str:
        """Generate detailed validation report
        
        Args:
            validation_metrics: Computed validation metrics
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("PI-HMARL Sim-to-Real Validation Report")
        report.append("=" * 60)
        
        # Performance section
        report.append("\nPerformance Metrics:")
        report.append("-" * 40)
        report.append(f"  Task Success Rate: {validation_metrics.task_success_rate:.2%}")
        report.append(f"  Performance Gap: {validation_metrics.performance_gap:.2%}")
        report.append(f"  Physics Realism: {validation_metrics.physics_realism_score:.2%}")
        report.append(f"  Control Smoothness: {validation_metrics.control_smoothness:.2f}")
        report.append(f"  Response Accuracy: {validation_metrics.response_accuracy:.2%}")
        
        # Safety section
        report.append("\nSafety Metrics:")
        report.append("-" * 40)
        report.append(f"  Constraint Violations: {validation_metrics.constraint_violation_rate:.2%}")
        report.append(f"  Safety Violations: {validation_metrics.safety_violation_count}")
        report.append(f"  Min Separation: {validation_metrics.min_separation_maintained:.2f} m")
        
        # Energy section
        report.append("\nEnergy Metrics:")
        report.append("-" * 40)
        report.append(f"  Energy Prediction Error: {validation_metrics.energy_prediction_error:.2%}")
        report.append(f"  Endurance Error: {validation_metrics.endurance_prediction_error:.1f} min")
        
        # Transfer section
        report.append("\nTransfer Metrics:")
        report.append("-" * 40)
        report.append(f"  Domain Shift: {validation_metrics.domain_shift_magnitude:.3f}")
        report.append(f"  Adaptation Iterations: {validation_metrics.adaptation_iterations}")
        
        # Statistical section
        report.append("\nStatistical Analysis:")
        report.append("-" * 40)
        ci_low, ci_high = validation_metrics.confidence_interval
        report.append(f"  95% Confidence Interval: [{ci_low:.3f}, {ci_high:.3f}]")
        report.append(f"  p-value: {validation_metrics.p_value:.4f}")
        
        # Validation decision
        report.append("\nValidation Decision:")
        report.append("-" * 40)
        
        passed_criteria = []
        failed_criteria = []
        
        if validation_metrics.task_success_rate >= 0.9:
            passed_criteria.append("✓ Task performance meets threshold")
        else:
            failed_criteria.append("✗ Task performance below threshold")
            
        if validation_metrics.constraint_violation_rate < 0.05:
            passed_criteria.append("✓ Physics compliance acceptable")
        else:
            failed_criteria.append("✗ Physics violations too frequent")
            
        if validation_metrics.safety_violation_count == 0:
            passed_criteria.append("✓ No safety violations")
        else:
            failed_criteria.append("✗ Safety violations detected")
            
        if validation_metrics.p_value > 0.05:
            passed_criteria.append("✓ No significant performance difference")
        else:
            failed_criteria.append("✗ Significant performance gap")
        
        for criterion in passed_criteria:
            report.append(f"  {criterion}")
        for criterion in failed_criteria:
            report.append(f"  {criterion}")
        
        # Overall result
        validation_passed = len(failed_criteria) == 0
        report.append("\n" + "=" * 60)
        if validation_passed:
            report.append("VALIDATION RESULT: PASSED")
            report.append("Model is ready for real-world deployment")
        else:
            report.append("VALIDATION RESULT: FAILED")
            report.append("Additional training or adaptation required")
        report.append("=" * 60)
        
        return "\n".join(report)


# Convenience function
def create_sim_to_real_validator(
    synthetic_model_path: Optional[Path] = None,
    real_data_path: Optional[Path] = None
) -> SimToRealValidator:
    """Create sim-to-real validator
    
    Args:
        synthetic_model_path: Path to synthetic model
        real_data_path: Path to real validation data
        
    Returns:
        SimToRealValidator instance
    """
    return SimToRealValidator(synthetic_model_path, real_data_path)