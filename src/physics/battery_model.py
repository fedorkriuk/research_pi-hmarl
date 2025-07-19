"""Battery Model with Real Discharge Curves

This module implements realistic battery models based on actual
battery specifications and discharge characteristics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatteryParameters:
    """Real battery parameters based on actual models"""
    # Basic specifications
    name: str
    nominal_voltage: float  # V
    capacity: float  # Ah
    energy_capacity: float  # Wh
    max_discharge_rate: float  # C-rate
    internal_resistance: float  # Ohms
    
    # Discharge curve parameters (voltage vs SOC)
    soc_points: List[float]  # State of charge points (0-1)
    voltage_points: List[float]  # Corresponding voltages
    
    # Temperature effects
    temp_coefficient: float  # Capacity change per degree C
    optimal_temp_range: Tuple[float, float]  # Celsius
    
    # Aging parameters
    cycle_life: int  # Number of cycles to 80% capacity
    calendar_life_years: float  # Years to 80% capacity
    
    @classmethod
    def samsung_18650(cls) -> "BatteryParameters":
        """Samsung 18650 battery parameters (common in drones)"""
        return cls(
            name="Samsung 18650-30Q",
            nominal_voltage=3.6,
            capacity=3.0,  # 3000mAh
            energy_capacity=10.8,  # 3.6V * 3.0Ah
            max_discharge_rate=15.0,  # 15C
            internal_resistance=0.02,
            # Discharge curve (simplified)
            soc_points=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            voltage_points=[3.0, 3.3, 3.45, 3.55, 3.6, 3.65, 3.7, 3.75, 3.85, 4.0, 4.2],
            temp_coefficient=-0.005,  # -0.5% per degree below 25C
            optimal_temp_range=(15.0, 35.0),
            cycle_life=500,
            calendar_life_years=5.0
        )
    
    @classmethod
    def dji_intelligent_battery(cls) -> "BatteryParameters":
        """DJI Intelligent Flight Battery parameters"""
        return cls(
            name="DJI Intelligent Battery",
            nominal_voltage=15.4,  # 4S LiPo
            capacity=5.0,  # 5000mAh
            energy_capacity=77.0,  # 15.4V * 5.0Ah
            max_discharge_rate=10.0,  # 10C
            internal_resistance=0.08,
            # More accurate discharge curve
            soc_points=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            voltage_points=[12.0, 13.2, 14.0, 14.6, 15.0, 15.2, 15.4, 15.6, 15.8, 16.0, 16.4, 16.6, 16.8],
            temp_coefficient=-0.008,
            optimal_temp_range=(10.0, 40.0),
            cycle_life=300,
            calendar_life_years=3.0
        )


class BatteryModel:
    """Realistic battery model with discharge curves and thermal effects"""
    
    def __init__(
        self,
        parameters: BatteryParameters,
        initial_soc: float = 1.0,
        initial_temperature: float = 25.0,
        initial_cycles: int = 0
    ):
        """Initialize battery model
        
        Args:
            parameters: Battery parameters
            initial_soc: Initial state of charge (0-1)
            initial_temperature: Initial temperature (Celsius)
            initial_cycles: Initial cycle count
        """
        self.params = parameters
        
        # State variables
        self.soc = initial_soc
        self.temperature = initial_temperature
        self.cycles = initial_cycles
        self.age_factor = 1.0  # Capacity degradation factor
        
        # Operating variables
        self.voltage = self._soc_to_voltage(initial_soc)
        self.current = 0.0
        self.power = 0.0
        
        # History tracking
        self.energy_delivered = 0.0
        self.charge_throughput = 0.0
        self.time_since_full = 0.0
        
        # Safety limits
        self.min_voltage = self.params.voltage_points[0]
        self.max_voltage = self.params.voltage_points[-1]
        self.max_current = self.params.capacity * self.params.max_discharge_rate
        
        logger.info(f"Initialized BatteryModel: {parameters.name}")
    
    def update(self, power_demand: float, dt: float, ambient_temp: float = 25.0):
        """Update battery state
        
        Args:
            power_demand: Power demand (W)
            dt: Time step (s)
            ambient_temp: Ambient temperature (C)
        """
        # Update temperature (simplified thermal model)
        self._update_temperature(power_demand, ambient_temp, dt)
        
        # Calculate current from power demand
        self.current = power_demand / self.voltage
        
        # Apply current limits
        if abs(self.current) > self.max_current:
            self.current = np.sign(self.current) * self.max_current
            power_demand = self.current * self.voltage
        
        # Calculate actual power (accounting for losses)
        power_loss = self.current**2 * self.params.internal_resistance
        self.power = power_demand + power_loss
        
        # Update SOC
        capacity_effective = self._get_effective_capacity()
        charge_delta = self.current * dt / 3600.0  # Ah
        soc_delta = charge_delta / capacity_effective
        
        self.soc = np.clip(self.soc - soc_delta, 0.0, 1.0)
        
        # Update voltage based on new SOC
        self.voltage = self._soc_to_voltage(self.soc)
        
        # Update statistics
        self.energy_delivered += power_demand * dt / 3600.0  # Wh
        self.charge_throughput += abs(charge_delta)
        
        if self.soc < 0.99:
            self.time_since_full += dt
        else:
            self.time_since_full = 0.0
        
        # Update aging
        self._update_aging(dt)
    
    def _soc_to_voltage(self, soc: float) -> float:
        """Convert SOC to voltage using discharge curve"""
        # Interpolate voltage from SOC
        voltage = np.interp(soc, self.params.soc_points, self.params.voltage_points)
        
        # Apply aging effect
        voltage *= (0.95 + 0.05 * self.age_factor)
        
        return voltage
    
    def _get_effective_capacity(self) -> float:
        """Get effective capacity considering temperature and aging"""
        # Base capacity
        capacity = self.params.capacity
        
        # Temperature effect
        temp_delta = self.temperature - 25.0
        temp_factor = 1.0 + self.params.temp_coefficient * temp_delta
        
        # Ensure reasonable bounds
        temp_factor = np.clip(temp_factor, 0.5, 1.2)
        
        # Apply aging
        capacity *= self.age_factor * temp_factor
        
        return capacity
    
    def _update_temperature(self, power: float, ambient_temp: float, dt: float):
        """Update battery temperature"""
        # Simple thermal model
        # Heat generation from I²R losses
        heat_generation = self.current**2 * self.params.internal_resistance
        
        # Heat dissipation (proportional to temperature difference)
        thermal_resistance = 5.0  # K/W (simplified)
        heat_dissipation = (self.temperature - ambient_temp) / thermal_resistance
        
        # Temperature change
        specific_heat = 900.0  # J/(kg·K) for lithium battery
        mass = self.params.energy_capacity / 100.0  # Rough estimate (kg)
        
        temp_change = (heat_generation - heat_dissipation) * dt / (mass * specific_heat)
        self.temperature += temp_change
        
        # Clamp to reasonable range
        self.temperature = np.clip(self.temperature, -20, 60)
    
    def _update_aging(self, dt: float):
        """Update battery aging"""
        # Cycle aging (simplified)
        # Count partial cycles
        cycle_increment = abs(self.current) * dt / (3600.0 * self.params.capacity * 2)
        self.cycles += cycle_increment
        
        # Calendar aging
        calendar_age_factor = 1.0 - (dt / (365 * 24 * 3600)) / self.params.calendar_life_years * 0.2
        
        # Cycle aging factor
        cycle_age_factor = 1.0 - (self.cycles / self.params.cycle_life) * 0.2
        
        # Combined aging
        self.age_factor = max(0.6, min(calendar_age_factor, cycle_age_factor))
    
    def get_state(self) -> Dict[str, float]:
        """Get battery state"""
        return {
            "soc": self.soc,
            "voltage": self.voltage,
            "current": self.current,
            "power": self.power,
            "temperature": self.temperature,
            "capacity": self._get_effective_capacity(),
            "energy_remaining": self.soc * self.params.energy_capacity * self.age_factor,
            "age_factor": self.age_factor,
            "cycles": self.cycles,
            "time_since_full": self.time_since_full
        }
    
    def get_health_metrics(self) -> Dict[str, float]:
        """Get battery health metrics"""
        return {
            "state_of_health": self.age_factor,
            "remaining_cycles": max(0, self.params.cycle_life - self.cycles),
            "temperature_status": self._get_temperature_status(),
            "power_capability": self._get_power_capability(),
            "internal_resistance": self.params.internal_resistance * (2 - self.age_factor)
        }
    
    def _get_temperature_status(self) -> float:
        """Get temperature status (0=critical, 1=optimal)"""
        opt_min, opt_max = self.params.optimal_temp_range
        
        if opt_min <= self.temperature <= opt_max:
            return 1.0
        elif self.temperature < opt_min:
            return max(0, (self.temperature + 20) / (opt_min + 20))
        else:
            return max(0, (60 - self.temperature) / (60 - opt_max))
    
    def _get_power_capability(self) -> float:
        """Get current power capability (W)"""
        max_current = self.params.capacity * self.params.max_discharge_rate
        max_current *= self._get_temperature_status()  # Derate for temperature
        max_current *= self.age_factor  # Derate for aging
        
        return max_current * self.voltage
    
    def estimate_flight_time(self, average_power: float) -> float:
        """Estimate remaining flight time
        
        Args:
            average_power: Average power consumption (W)
            
        Returns:
            Estimated flight time (seconds)
        """
        if average_power <= 0:
            return float('inf')
        
        # Energy remaining
        energy_remaining = self.soc * self.params.energy_capacity * self.age_factor
        
        # Account for temperature
        temp_factor = 1.0 + self.params.temp_coefficient * (self.temperature - 25.0)
        energy_remaining *= temp_factor
        
        # Account for efficiency losses
        efficiency = 0.9  # 90% efficiency
        
        # Calculate time
        time_hours = energy_remaining * efficiency / average_power
        
        return time_hours * 3600  # Convert to seconds
    
    def check_safety_limits(self) -> Dict[str, bool]:
        """Check battery safety limits"""
        return {
            "voltage_low": self.voltage <= self.min_voltage * 1.1,
            "voltage_critical": self.voltage <= self.min_voltage,
            "current_high": abs(self.current) >= self.max_current * 0.9,
            "temperature_high": self.temperature >= 45.0,
            "temperature_critical": self.temperature >= 55.0,
            "soc_low": self.soc <= 0.2,
            "soc_critical": self.soc <= 0.1
        }