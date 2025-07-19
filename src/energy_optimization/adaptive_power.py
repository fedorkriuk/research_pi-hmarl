"""Adaptive Power Management for Dynamic Performance Scaling

This module implements adaptive power management strategies that dynamically
adjust performance based on energy availability and thermal constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """Power mode definitions"""
    ULTRA_LOW = "ultra_low"      # Emergency mode - 30% power
    LOW = "low"                  # Power saving - 50% power  
    NORMAL = "normal"            # Standard - 75% power
    HIGH = "high"                # Performance - 100% power
    BOOST = "boost"              # Temporary boost - 120% power


class AdaptivePowerManager:
    """Manages adaptive power modes and performance scaling"""
    
    def __init__(
        self,
        base_power: float = 80.0,  # W - nominal power
        max_power: float = 200.0,   # W - maximum power
        thermal_limit: float = 60.0,  # Celsius
        adaptation_rate: float = 0.1
    ):
        """Initialize adaptive power manager
        
        Args:
            base_power: Base power consumption
            max_power: Maximum power limit
            thermal_limit: Temperature limit
            adaptation_rate: How fast to adapt
        """
        self.base_power = base_power
        self.max_power = max_power
        self.thermal_limit = thermal_limit
        self.adaptation_rate = adaptation_rate
        
        # Power mode settings
        self.power_modes = {
            PowerMode.ULTRA_LOW: 0.3,
            PowerMode.LOW: 0.5,
            PowerMode.NORMAL: 0.75,
            PowerMode.HIGH: 1.0,
            PowerMode.BOOST: 1.2
        }
        
        # Current mode
        self.current_mode = PowerMode.NORMAL
        
        # Performance scaler
        self.performance_scaler = PerformanceScaler(
            base_power=base_power,
            max_power=max_power
        )
        
        # Thermal throttling
        self.thermal_throttle = ThermalThrottling(
            thermal_limit=thermal_limit
        )
        
        # Dynamic voltage/frequency scaling
        self.dvfs = DynamicVoltageScaling()
        
        # Mode transition network
        self.mode_predictor = PowerModePredictor()
        
        # History tracking
        self.power_history = []
        self.mode_history = []
        
        logger.info(f"Initialized AdaptivePowerManager with {base_power}W base power")
    
    def update(
        self,
        energy_state: Dict[str, torch.Tensor],
        task_requirements: Dict[str, Any],
        dt: float = 0.1
    ) -> Dict[str, Any]:
        """Update power management
        
        Args:
            energy_state: Current energy state
            task_requirements: Task power requirements
            dt: Time step
            
        Returns:
            Power management decisions
        """
        # Get current constraints
        soc = energy_state['soc']
        temperature = energy_state.get('temperature', 25.0)
        current_power = energy_state.get('power', self.base_power)
        
        # Predict optimal mode
        predicted_mode = self.mode_predictor.predict_mode(
            energy_state, task_requirements
        )
        
        # Apply thermal constraints
        thermal_limited_mode = self.thermal_throttle.apply_thermal_limit(
            predicted_mode, temperature
        )
        
        # Apply SOC constraints
        soc_limited_mode = self._apply_soc_constraints(
            thermal_limited_mode, soc
        )
        
        # Smooth mode transitions
        target_mode = soc_limited_mode
        if target_mode != self.current_mode:
            self._transition_mode(target_mode, dt)
        
        # Get power scaling factor
        power_factor = self.power_modes[self.current_mode]
        
        # Apply performance scaling
        scaled_performance = self.performance_scaler.scale_performance(
            task_requirements, power_factor
        )
        
        # Apply DVFS
        dvfs_settings = self.dvfs.optimize_settings(
            power_factor, temperature
        )
        
        # Compute actual power
        actual_power = self.base_power * power_factor
        actual_power = min(actual_power, self.max_power)
        
        # Update history
        self.power_history.append(actual_power)
        self.mode_history.append(self.current_mode)
        
        return {
            'mode': self.current_mode,
            'power_factor': power_factor,
            'actual_power': actual_power,
            'performance_scaling': scaled_performance,
            'dvfs_settings': dvfs_settings,
            'thermal_throttle_active': temperature > self.thermal_limit * 0.9,
            'soc_limited': soc < 0.3,
            'predicted_mode': predicted_mode,
            'mode_change_allowed': self._can_change_mode()
        }
    
    def _apply_soc_constraints(
        self,
        mode: PowerMode,
        soc: torch.Tensor
    ) -> PowerMode:
        """Apply SOC-based mode constraints
        
        Args:
            mode: Desired mode
            soc: State of charge
            
        Returns:
            Constrained mode
        """
        if soc < 0.15:  # Critical
            return PowerMode.ULTRA_LOW
        elif soc < 0.25:  # Low
            return min(mode, PowerMode.LOW)
        elif soc < 0.4:  # Moderate
            return min(mode, PowerMode.NORMAL)
        else:
            return mode  # No constraint
    
    def _transition_mode(self, target_mode: PowerMode, dt: float):
        """Smoothly transition between modes
        
        Args:
            target_mode: Target power mode
            dt: Time step
        """
        # Get numeric values for interpolation
        current_value = self.power_modes[self.current_mode]
        target_value = self.power_modes[target_mode]
        
        # Smooth transition
        if abs(target_value - current_value) > 0.1:
            # Gradual change
            change = self.adaptation_rate * dt
            if target_value > current_value:
                new_value = min(current_value + change, target_value)
            else:
                new_value = max(current_value - change, target_value)
            
            # Find closest mode
            min_diff = float('inf')
            for mode, value in self.power_modes.items():
                diff = abs(value - new_value)
                if diff < min_diff:
                    min_diff = diff
                    self.current_mode = mode
        else:
            # Direct transition for small changes
            self.current_mode = target_mode
    
    def _can_change_mode(self) -> bool:
        """Check if mode change is allowed
        
        Returns:
            Whether mode change is allowed
        """
        # Prevent rapid mode changes
        if len(self.mode_history) < 10:
            return True
        
        # Check recent mode changes
        recent_modes = self.mode_history[-10:]
        unique_modes = len(set(recent_modes))
        
        # Allow change if stable
        return unique_modes <= 2
    
    def request_boost(
        self,
        duration: float,
        energy_state: Dict[str, torch.Tensor]
    ) -> bool:
        """Request temporary boost mode
        
        Args:
            duration: Requested boost duration
            energy_state: Current energy state
            
        Returns:
            Whether boost was granted
        """
        soc = energy_state['soc']
        temperature = energy_state.get('temperature', 25.0)
        
        # Check if boost is safe
        if (soc > 0.5 and 
            temperature < self.thermal_limit * 0.8 and
            self.current_mode != PowerMode.BOOST):
            
            # Grant boost
            self.current_mode = PowerMode.BOOST
            logger.info(f"Boost mode granted for {duration}s")
            return True
        
        return False


class PerformanceScaler(nn.Module):
    """Scales performance based on available power"""
    
    def __init__(
        self,
        base_power: float = 80.0,
        max_power: float = 200.0
    ):
        """Initialize performance scaler
        
        Args:
            base_power: Base power
            max_power: Maximum power
        """
        super().__init__()
        
        self.base_power = base_power
        self.max_power = max_power
        
        # Performance scaling network
        self.scaling_net = nn.Sequential(
            nn.Linear(5, 32),  # power_factor + task features
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # scaling factors
        )
        
        # Component scalers
        self.component_scalers = {
            'compute': ComputeScaler(),
            'sensors': SensorScaler(),
            'communication': CommScaler(),
            'actuators': ActuatorScaler()
        }
    
    def scale_performance(
        self,
        task_requirements: Dict[str, Any],
        power_factor: float
    ) -> Dict[str, float]:
        """Scale performance based on power
        
        Args:
            task_requirements: Task requirements
            power_factor: Power scaling factor
            
        Returns:
            Performance scaling factors
        """
        # Extract task features
        task_features = torch.tensor([
            task_requirements.get('compute_intensity', 0.5),
            task_requirements.get('sensor_usage', 0.5),
            task_requirements.get('comm_bandwidth', 0.5),
            task_requirements.get('actuation_level', 0.5)
        ])
        
        # Neural scaling
        features = torch.cat([
            torch.tensor([power_factor]),
            task_features
        ])
        
        scale_factors = torch.sigmoid(self.scaling_net(features))
        
        # Apply component-specific scaling
        scaled = {}
        component_names = ['compute', 'sensors', 'communication', 'actuators']
        
        for i, name in enumerate(component_names):
            base_scale = scale_factors[i].item()
            
            # Component-specific adjustments
            if name in self.component_scalers:
                adjusted_scale = self.component_scalers[name].adjust_scale(
                    base_scale, power_factor, task_requirements
                )
            else:
                adjusted_scale = base_scale * power_factor
            
            scaled[name] = adjusted_scale
        
        # Add aggregate metrics
        scaled['overall'] = np.mean(list(scaled.values()))
        scaled['min_component'] = min(scaled.values())
        scaled['bottleneck'] = min(scaled.items(), key=lambda x: x[1])[0]
        
        return scaled


class ThermalThrottling:
    """Manages thermal throttling"""
    
    def __init__(
        self,
        thermal_limit: float = 60.0,
        throttle_start: float = 50.0
    ):
        """Initialize thermal throttling
        
        Args:
            thermal_limit: Maximum temperature
            throttle_start: Temperature to start throttling
        """
        self.thermal_limit = thermal_limit
        self.throttle_start = throttle_start
        
        # Throttling curve
        self.throttle_curve = self._create_throttle_curve()
    
    def apply_thermal_limit(
        self,
        mode: PowerMode,
        temperature: torch.Tensor
    ) -> PowerMode:
        """Apply thermal throttling to mode
        
        Args:
            mode: Desired mode
            temperature: Current temperature
            
        Returns:
            Thermally limited mode
        """
        if temperature < self.throttle_start:
            return mode
        
        # Calculate throttle factor
        throttle_factor = self._get_throttle_factor(temperature)
        
        # Map to restricted modes
        if throttle_factor < 0.3:
            return PowerMode.ULTRA_LOW
        elif throttle_factor < 0.5:
            return PowerMode.LOW
        elif throttle_factor < 0.75:
            return min(mode, PowerMode.NORMAL)
        else:
            return min(mode, PowerMode.HIGH)  # No boost when hot
    
    def _get_throttle_factor(self, temperature: torch.Tensor) -> float:
        """Get throttling factor from temperature
        
        Args:
            temperature: Current temperature
            
        Returns:
            Throttle factor (0-1)
        """
        if temperature < self.throttle_start:
            return 1.0
        elif temperature > self.thermal_limit:
            return 0.0
        else:
            # Linear throttling
            range_temp = self.thermal_limit - self.throttle_start
            excess_temp = temperature - self.throttle_start
            return 1.0 - (excess_temp / range_temp).item()
    
    def _create_throttle_curve(self) -> Dict[float, float]:
        """Create thermal throttle curve
        
        Returns:
            Temperature to throttle factor mapping
        """
        curve = {}
        
        temps = np.linspace(20, 80, 61)
        for temp in temps:
            if temp < self.throttle_start:
                curve[temp] = 1.0
            elif temp > self.thermal_limit:
                curve[temp] = 0.0
            else:
                # Smooth curve
                x = (temp - self.throttle_start) / (self.thermal_limit - self.throttle_start)
                curve[temp] = 1.0 - x**2  # Quadratic throttling
        
        return curve
    
    def get_cooling_requirement(
        self,
        temperature: torch.Tensor,
        current_power: torch.Tensor
    ) -> torch.Tensor:
        """Calculate cooling power requirement
        
        Args:
            temperature: Current temperature
            current_power: Current power consumption
            
        Returns:
            Required cooling power
        """
        if temperature < self.throttle_start:
            return torch.tensor(0.0)
        
        # Cooling needed to maintain throttle_start temperature
        temp_excess = temperature - self.throttle_start
        
        # Simple model: cooling power proportional to temperature excess
        cooling_power = temp_excess * 2.0  # W/°C
        
        return cooling_power


class DynamicVoltageScaling:
    """Dynamic voltage and frequency scaling"""
    
    def __init__(self):
        """Initialize DVFS"""
        # Voltage-frequency pairs (normalized)
        self.vf_pairs = [
            (0.7, 0.5),   # Low power
            (0.85, 0.75), # Normal
            (1.0, 1.0),   # High performance
            (1.1, 1.2)    # Boost (if allowed)
        ]
        
        # Power model: P = C * V^2 * f
        self.capacitance = 1.0  # Normalized
    
    def optimize_settings(
        self,
        power_factor: float,
        temperature: torch.Tensor
    ) -> Dict[str, float]:
        """Optimize voltage and frequency
        
        Args:
            power_factor: Desired power factor
            temperature: Current temperature
            
        Returns:
            DVFS settings
        """
        # Find best V-F pair for power factor
        best_pair = None
        min_error = float('inf')
        
        for voltage, frequency in self.vf_pairs:
            # Calculate power for this pair
            power = self.capacitance * voltage**2 * frequency
            error = abs(power - power_factor)
            
            if error < min_error:
                min_error = error
                best_pair = (voltage, frequency)
        
        voltage, frequency = best_pair
        
        # Temperature derating
        if temperature > 50:
            temp_factor = 1.0 - (temperature - 50) / 30.0
            temp_factor = max(0.7, temp_factor)
            voltage *= temp_factor
            frequency *= temp_factor
        
        return {
            'voltage': voltage,
            'frequency': frequency,
            'power': self.capacitance * voltage**2 * frequency,
            'efficiency': self._compute_efficiency(voltage, frequency)
        }
    
    def _compute_efficiency(self, voltage: float, frequency: float) -> float:
        """Compute energy efficiency
        
        Args:
            voltage: Voltage setting
            frequency: Frequency setting
            
        Returns:
            Efficiency (0-1)
        """
        # Efficiency peaks at moderate settings
        optimal_v = 0.85
        optimal_f = 0.75
        
        v_eff = 1.0 - abs(voltage - optimal_v) / optimal_v
        f_eff = 1.0 - abs(frequency - optimal_f) / optimal_f
        
        return v_eff * f_eff


class PowerModePredictor(nn.Module):
    """Predicts optimal power mode"""
    
    def __init__(self, hidden_dim: int = 64):
        """Initialize mode predictor
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.mode_net = nn.Sequential(
            nn.Linear(10, hidden_dim),  # Energy + task features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # One per power mode
        )
    
    def predict_mode(
        self,
        energy_state: Dict[str, torch.Tensor],
        task_requirements: Dict[str, Any]
    ) -> PowerMode:
        """Predict optimal power mode
        
        Args:
            energy_state: Energy state
            task_requirements: Task requirements
            
        Returns:
            Predicted power mode
        """
        # Extract features
        features = torch.tensor([
            energy_state['soc'],
            energy_state.get('temperature', 25.0) / 100.0,
            energy_state.get('power', 80.0) / 200.0,
            task_requirements.get('compute_intensity', 0.5),
            task_requirements.get('sensor_usage', 0.5),
            task_requirements.get('comm_bandwidth', 0.5),
            task_requirements.get('actuation_level', 0.5),
            task_requirements.get('priority', 0.5),
            task_requirements.get('deadline_pressure', 0.5),
            task_requirements.get('quality_requirement', 0.5)
        ])
        
        # Predict mode scores
        logits = self.mode_net(features)
        mode_idx = torch.argmax(logits).item()
        
        # Map to power mode
        modes = list(PowerMode)
        return modes[mode_idx]


# Component-specific scalers

class ComputeScaler:
    """Scales compute performance"""
    
    def adjust_scale(
        self,
        base_scale: float,
        power_factor: float,
        requirements: Dict[str, Any]
    ) -> float:
        """Adjust compute scaling
        
        Args:
            base_scale: Base scale factor
            power_factor: Power factor
            requirements: Task requirements
            
        Returns:
            Adjusted scale
        """
        intensity = requirements.get('compute_intensity', 0.5)
        
        # Compute scales roughly linearly with power
        return base_scale * power_factor * (0.8 + 0.2 * intensity)


class SensorScaler:
    """Scales sensor performance"""
    
    def adjust_scale(
        self,
        base_scale: float,
        power_factor: float,
        requirements: Dict[str, Any]
    ) -> float:
        """Adjust sensor scaling
        
        Args:
            base_scale: Base scale factor
            power_factor: Power factor
            requirements: Task requirements
            
        Returns:
            Adjusted scale
        """
        usage = requirements.get('sensor_usage', 0.5)
        
        # Sensors can often be duty cycled
        if power_factor < 0.5:
            # Aggressive duty cycling
            return base_scale * 0.3 * (1 + usage)
        else:
            return base_scale * (0.7 + 0.3 * power_factor)


class CommScaler:
    """Scales communication performance"""
    
    def adjust_scale(
        self,
        base_scale: float,
        power_factor: float,
        requirements: Dict[str, Any]
    ) -> float:
        """Adjust communication scaling
        
        Args:
            base_scale: Base scale factor
            power_factor: Power factor
            requirements: Task requirements
            
        Returns:
            Adjusted scale
        """
        bandwidth = requirements.get('comm_bandwidth', 0.5)
        
        # Communication has high power cost
        if power_factor < 0.5:
            # Reduce transmission power and rate
            return base_scale * power_factor * 0.5
        else:
            return base_scale * (0.5 + 0.5 * power_factor ** 2)


class ActuatorScaler:
    """Scales actuator performance"""
    
    def adjust_scale(
        self,
        base_scale: float,
        power_factor: float,
        requirements: Dict[str, Any]
    ) -> float:
        """Adjust actuator scaling
        
        Args:
            base_scale: Base scale factor
            power_factor: Power factor
            requirements: Task requirements
            
        Returns:
            Adjusted scale
        """
        actuation = requirements.get('actuation_level', 0.5)
        
        # Actuators need minimum power to function
        min_power = 0.4
        
        if power_factor < min_power:
            return 0.0  # Cannot operate
        else:
            # Scale with available power above minimum
            excess = (power_factor - min_power) / (1.0 - min_power)
            return base_scale * excess * (0.7 + 0.3 * actuation)