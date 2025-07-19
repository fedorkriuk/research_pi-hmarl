"""Battery Model with Real Electrochemical Physics

This module implements realistic battery models including electrochemical
dynamics, thermal effects, and degradation based on real battery data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BatteryModel(nn.Module):
    """Base class for battery models"""
    
    def __init__(
        self,
        capacity: float = 5.0,  # Ah - Samsung 18650 typical capacity
        voltage_nominal: float = 3.7,  # V - Li-ion nominal voltage
        voltage_min: float = 2.5,  # V - Cutoff voltage
        voltage_max: float = 4.2,  # V - Max charge voltage
        internal_resistance: float = 0.05,  # Ohms - typical for 18650
        efficiency: float = 0.95
    ):
        """Initialize battery model
        
        Args:
            capacity: Battery capacity in Ah
            voltage_nominal: Nominal voltage in V
            voltage_min: Minimum cutoff voltage
            voltage_max: Maximum charge voltage
            internal_resistance: Internal resistance in Ohms
            efficiency: Charge/discharge efficiency
        """
        super().__init__()
        
        self.capacity = capacity
        self.voltage_nominal = voltage_nominal
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.internal_resistance = internal_resistance
        self.efficiency = efficiency
        
        # Energy capacity in Wh
        self.energy_capacity = capacity * voltage_nominal
        
        # Register parameters
        self.register_buffer('capacity_tensor', torch.tensor(capacity))
        self.register_buffer('voltage_nominal_tensor', torch.tensor(voltage_nominal))
        
        logger.info(f"Initialized BatteryModel with {self.energy_capacity:.1f} Wh capacity")
    
    def get_voltage(self, soc: torch.Tensor) -> torch.Tensor:
        """Get voltage based on state of charge
        
        Args:
            soc: State of charge (0-1)
            
        Returns:
            Voltage
        """
        # Simple linear model - override in subclasses for more accuracy
        voltage_range = self.voltage_max - self.voltage_min
        return self.voltage_min + soc * voltage_range
    
    def get_power_loss(
        self,
        current: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate power loss due to internal resistance
        
        Args:
            current: Current in Amperes
            temperature: Temperature in Celsius
            
        Returns:
            Power loss in Watts
        """
        # Adjust resistance for temperature if provided
        if temperature is not None:
            # Resistance increases with temperature
            temp_factor = 1.0 + 0.02 * (temperature - 25.0) / 10.0
            resistance = self.internal_resistance * temp_factor
        else:
            resistance = self.internal_resistance
        
        # P = I²R
        power_loss = current ** 2 * resistance
        
        return power_loss


class ElectrochemicalBattery(BatteryModel):
    """Detailed electrochemical battery model"""
    
    def __init__(
        self,
        capacity: float = 5.0,
        voltage_nominal: float = 3.7,
        voltage_min: float = 2.5,
        voltage_max: float = 4.2,
        internal_resistance: float = 0.05,
        efficiency: float = 0.95,
        chemistry: str = "li_ion"
    ):
        """Initialize electrochemical battery model
        
        Args:
            capacity: Battery capacity in Ah
            voltage_nominal: Nominal voltage
            voltage_min: Minimum voltage
            voltage_max: Maximum voltage
            internal_resistance: Internal resistance
            efficiency: Efficiency
            chemistry: Battery chemistry type
        """
        super().__init__(
            capacity, voltage_nominal, voltage_min, voltage_max,
            internal_resistance, efficiency
        )
        
        self.chemistry = chemistry
        
        # Chemistry-specific parameters
        if chemistry == "li_ion":
            # Li-ion specific parameters
            self.ocv_params = {
                'k0': 3.0,  # Base voltage
                'k1': 0.7,  # Linear coefficient
                'k2': 0.3,  # Quadratic coefficient
                'k3': -0.2,  # Cubic coefficient
                'k4': 0.1   # Quartic coefficient
            }
            
            # Arrhenius parameters for temperature dependence
            self.activation_energy = 20000  # J/mol
            self.gas_constant = 8.314  # J/(mol·K)
            
        # Neural network for complex dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(5, 32),  # [soc, current, temp, cycles, time]
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [voltage, resistance, capacity_fade]
        )
    
    def get_open_circuit_voltage(
        self,
        soc: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate open circuit voltage using electrochemical model
        
        Args:
            soc: State of charge
            temperature: Temperature in Celsius
            
        Returns:
            Open circuit voltage
        """
        # Polynomial OCV model for Li-ion
        ocv = (
            self.ocv_params['k0'] +
            self.ocv_params['k1'] * soc +
            self.ocv_params['k2'] * soc**2 +
            self.ocv_params['k3'] * soc**3 +
            self.ocv_params['k4'] * soc**4
        )
        
        # Temperature adjustment using Nernst equation
        if temperature is not None:
            temp_kelvin = temperature + 273.15
            temp_factor = temp_kelvin / 298.15  # Normalized to 25°C
            ocv = ocv * temp_factor
        
        # Clamp to valid range
        ocv = torch.clamp(ocv, self.voltage_min, self.voltage_max)
        
        return ocv
    
    def get_voltage(
        self,
        soc: torch.Tensor,
        current: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get terminal voltage considering current and temperature
        
        Args:
            soc: State of charge
            current: Current (positive for discharge)
            temperature: Temperature
            
        Returns:
            Terminal voltage
        """
        # Open circuit voltage
        ocv = self.get_open_circuit_voltage(soc, temperature)
        
        # Voltage drop due to internal resistance
        voltage_drop = current * self.internal_resistance
        
        # Terminal voltage
        terminal_voltage = ocv - voltage_drop
        
        return terminal_voltage
    
    def compute_capacity_fade(
        self,
        cycles: torch.Tensor,
        temperature: torch.Tensor,
        c_rate: torch.Tensor,
        soc_avg: torch.Tensor
    ) -> torch.Tensor:
        """Compute capacity fade using empirical model
        
        Args:
            cycles: Number of charge/discharge cycles
            temperature: Average temperature
            c_rate: Average C-rate
            soc_avg: Average SOC
            
        Returns:
            Capacity retention factor (0-1)
        """
        # Calendar aging
        calendar_fade = 0.02 * torch.sqrt(cycles / 1000.0)
        
        # Cycle aging - worse at high C-rates
        cycle_fade = 0.001 * cycles * (1.0 + 0.5 * c_rate)
        
        # Temperature stress - accelerated at high temps
        temp_stress = torch.where(
            temperature > 25.0,
            0.001 * (temperature - 25.0) * cycles / 1000.0,
            torch.zeros_like(temperature)
        )
        
        # SOC stress - worse at high SOC
        soc_stress = 0.001 * soc_avg * cycles / 1000.0
        
        # Total fade
        total_fade = calendar_fade + cycle_fade + temp_stress + soc_stress
        
        # Capacity retention
        capacity_retention = 1.0 - torch.clamp(total_fade, 0.0, 0.3)
        
        return capacity_retention


class SimplifiedBattery(BatteryModel):
    """Simplified battery model for fast computation"""
    
    def __init__(
        self,
        capacity: float = 5.0,
        voltage_nominal: float = 3.7,
        voltage_min: float = 2.5,
        voltage_max: float = 4.2,
        internal_resistance: float = 0.05,
        efficiency: float = 0.95
    ):
        """Initialize simplified battery model"""
        super().__init__(
            capacity, voltage_nominal, voltage_min, voltage_max,
            internal_resistance, efficiency
        )
        
        # Simple linear discharge curve
        self.discharge_rate = nn.Parameter(torch.tensor(1.0))
    
    def update_soc(
        self,
        soc: torch.Tensor,
        power: torch.Tensor,
        dt: float = 0.1
    ) -> torch.Tensor:
        """Update state of charge based on power consumption
        
        Args:
            soc: Current state of charge
            power: Power consumption in Watts
            dt: Time step in seconds
            
        Returns:
            New state of charge
        """
        # Convert power to current
        voltage = self.get_voltage(soc)
        current = power / voltage
        
        # Account for efficiency
        if current > 0:  # Discharging
            effective_current = current / self.efficiency
        else:  # Charging
            effective_current = current * self.efficiency
        
        # Update SOC
        # SOC_new = SOC_old - (I * dt) / (Capacity * 3600)
        soc_change = (effective_current * dt) / (self.capacity * 3600.0)
        new_soc = soc - soc_change
        
        # Clamp to valid range
        new_soc = torch.clamp(new_soc, 0.0, 1.0)
        
        return new_soc


class BatteryDegradation(nn.Module):
    """Models battery degradation over time"""
    
    def __init__(
        self,
        initial_capacity: float = 5.0,
        degradation_rate: float = 0.0002,  # Per cycle
        temperature_factor: float = 2.0,  # Arrhenius factor
        soc_stress_factor: float = 0.5
    ):
        """Initialize battery degradation model
        
        Args:
            initial_capacity: Initial capacity in Ah
            degradation_rate: Base degradation per cycle
            temperature_factor: Temperature acceleration factor
            soc_stress_factor: SOC stress factor
        """
        super().__init__()
        
        self.initial_capacity = initial_capacity
        self.degradation_rate = degradation_rate
        self.temperature_factor = temperature_factor
        self.soc_stress_factor = soc_stress_factor
        
        # State tracking
        self.register_buffer('cycles', torch.tensor(0.0))
        self.register_buffer('capacity_retention', torch.tensor(1.0))
    
    def update_degradation(
        self,
        temperature: torch.Tensor,
        soc: torch.Tensor,
        c_rate: torch.Tensor,
        cycle_fraction: float = 0.001
    ) -> Dict[str, torch.Tensor]:
        """Update degradation state
        
        Args:
            temperature: Temperature in Celsius
            soc: State of charge
            c_rate: Charge/discharge rate
            cycle_fraction: Fraction of cycle completed
            
        Returns:
            Degradation metrics
        """
        # Temperature stress (Arrhenius)
        temp_kelvin = temperature + 273.15
        temp_ref = 298.15  # 25°C reference
        temp_stress = torch.exp(
            self.temperature_factor * (1/temp_ref - 1/temp_kelvin)
        )
        
        # SOC stress (higher at extremes)
        soc_stress = 1.0 + self.soc_stress_factor * (
            (soc - 0.5) ** 2
        )
        
        # C-rate stress
        c_rate_stress = 1.0 + 0.3 * c_rate
        
        # Total stress factor
        total_stress = temp_stress * soc_stress * c_rate_stress
        
        # Degradation for this fraction of cycle
        degradation = self.degradation_rate * cycle_fraction * total_stress
        
        # Update capacity retention
        self.capacity_retention = self.capacity_retention * (1.0 - degradation)
        self.cycles = self.cycles + cycle_fraction
        
        # Current capacity
        current_capacity = self.initial_capacity * self.capacity_retention
        
        return {
            'capacity_retention': self.capacity_retention,
            'current_capacity': current_capacity,
            'cycles': self.cycles,
            'degradation_rate': degradation,
            'stress_factor': total_stress
        }


class ThermalModel(nn.Module):
    """Battery thermal model"""
    
    def __init__(
        self,
        thermal_mass: float = 50.0,  # J/K
        thermal_resistance: float = 10.0,  # K/W
        ambient_temp: float = 25.0  # Celsius
    ):
        """Initialize thermal model
        
        Args:
            thermal_mass: Thermal mass in J/K
            thermal_resistance: Thermal resistance to ambient
            ambient_temp: Ambient temperature
        """
        super().__init__()
        
        self.thermal_mass = thermal_mass
        self.thermal_resistance = thermal_resistance
        self.ambient_temp = ambient_temp
        
        # Temperature state
        self.register_buffer('temperature', torch.tensor(ambient_temp))
    
    def update_temperature(
        self,
        power_loss: torch.Tensor,
        dt: float = 0.1,
        cooling_power: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update battery temperature
        
        Args:
            power_loss: Power dissipated as heat
            dt: Time step
            cooling_power: Active cooling power
            
        Returns:
            New temperature
        """
        # Heat generation
        heat_gen = power_loss
        
        # Heat dissipation to ambient
        temp_diff = self.temperature - self.ambient_temp
        heat_diss= temp_diff / self.thermal_resistance
        
        # Active cooling if provided
        if cooling_power is not None:
            heat_diss= heat_diss+ cooling_power
        
        # Net heat flow
        net_heat = heat_gen - heat_diss
        
        # Temperature change
        temp_change = (net_heat * dt) / self.thermal_mass
        
        # Update temperature
        self.temperature = self.temperature + temp_change
        
        return self.temperature
    
    def get_safe_operating_power(
        self,
        max_temp: float = 60.0
    ) -> torch.Tensor:
        """Get maximum safe power to avoid overheating
        
        Args:
            max_temp: Maximum safe temperature
            
        Returns:
            Maximum safe power
        """
        # Available temperature margin
        temp_margin = max_temp - self.temperature
        
        # Maximum power that can be dissipated
        max_power = temp_margin / self.thermal_resistance
        
        # Safety factor
        safe_power = 0.8 * max_power
        
        return torch.clamp(safe_power, min=0.0)


class StateOfCharge(nn.Module):
    """Advanced state of charge estimation"""
    
    def __init__(
        self,
        method: str = "coulomb_counting",
        voltage_lookup: Optional[Dict[float, float]] = None
    ):
        """Initialize SOC estimator
        
        Args:
            method: Estimation method
            voltage_lookup: Voltage-SOC lookup table
        """
        super().__init__()
        
        self.method = method
        
        if voltage_lookup is None:
            # Default Li-ion voltage-SOC curve
            self.voltage_lookup = {
                2.5: 0.0,
                3.0: 0.05,
                3.3: 0.10,
                3.5: 0.20,
                3.6: 0.30,
                3.7: 0.50,
                3.8: 0.70,
                3.9: 0.85,
                4.0: 0.95,
                4.2: 1.0
            }
        else:
            self.voltage_lookup = voltage_lookup
        
        # Kalman filter parameters for advanced estimation
        self.register_buffer('P', torch.eye(2) * 0.01)  # Error covariance
        self.register_buffer('Q', torch.eye(2) * 0.001)  # Process noise
        self.register_buffer('R', torch.tensor(0.01))  # Measurement noise
    
    def estimate_from_voltage(
        self,
        voltage: torch.Tensor,
        current: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Estimate SOC from voltage measurement
        
        Args:
            voltage: Measured voltage
            current: Current (for compensation)
            
        Returns:
            Estimated SOC
        """
        # Compensate for voltage drop if current provided
        if current is not None:
            # Assume 50mOhm internal resistance
            ocv = voltage + current * 0.05
        else:
            ocv = voltage
        
        # Interpolate from lookup table
        voltages = torch.tensor(list(self.voltage_lookup.keys()))
        socs = torch.tensor(list(self.voltage_lookup.values()))
        
        # Find interpolation indices
        idx = torch.searchsorted(voltages, ocv.item())
        
        if idx == 0:
            soc = socs[0]
        elif idx >= len(voltages):
            soc = socs[-1]
        else:
            # Linear interpolation
            v1, v2 = voltages[idx-1], voltages[idx]
            s1, s2 = socs[idx-1], socs[idx]
            
            soc = s1 + (ocv - v1) * (s2 - s1) / (v2 - v1)
        
        return soc
    
    def kalman_update(
        self,
        soc_pred: torch.Tensor,
        voltage_meas: torch.Tensor,
        current: torch.Tensor,
        dt: float = 0.1
    ) -> torch.Tensor:
        """Kalman filter update for SOC estimation
        
        Args:
            soc_pred: Predicted SOC from coulomb counting
            voltage_meas: Measured voltage
            current: Current
            dt: Time step
            
        Returns:
            Updated SOC estimate
        """
        # State: [SOC, voltage]
        x = torch.tensor([soc_pred, voltage_meas])
        
        # Measurement model: voltage from SOC
        soc_from_voltage = self.estimate_from_voltage(voltage_meas, current)
        
        # Innovation
        y = soc_from_voltage - soc_pred
        
        # Kalman gain
        S = self.P[0, 0] + self.R
        K = self.P[0, 0] / S
        
        # Update
        soc_updated = soc_pred + K * y
        
        # Update error covariance
        self.P[0, 0] = (1 - K) * self.P[0, 0]
        
        return soc_updated