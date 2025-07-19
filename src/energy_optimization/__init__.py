"""Energy-Aware Optimization Module

This module implements energy-aware optimization algorithms with real
battery physics and collaborative energy management.
"""

from .battery_model import (
    BatteryModel, ElectrochemicalBattery, SimplifiedBattery,
    BatteryDegradation, ThermalModel, StateOfCharge
)
from .energy_optimizer import (
    EnergyAwareOptimizer, EnergyObjective, TaskEnergyTradeoff,
    EnergyConstraint, PowerAllocation
)
from .collaborative_energy import (
    CollaborativeEnergyManager, EnergySharing, ChargingScheduler,
    TeamEnergyOptimizer, EnergyBalancer
)
from .return_to_base import (
    ReturnToBasePlanner, EnergyConstrainedPath, SafeReturnPolicy,
    ChargingStationSelector, EmergencyReturn
)
from .adaptive_power import (
    AdaptivePowerManager, PowerMode, PerformanceScaler,
    ThermalThrottling, DynamicVoltageScaling
)

__all__ = [
    # Battery Model
    'BatteryModel', 'ElectrochemicalBattery', 'SimplifiedBattery',
    'BatteryDegradation', 'ThermalModel', 'StateOfCharge',
    
    # Energy Optimizer
    'EnergyAwareOptimizer', 'EnergyObjective', 'TaskEnergyTradeoff',
    'EnergyConstraint', 'PowerAllocation',
    
    # Collaborative Energy
    'CollaborativeEnergyManager', 'EnergySharing', 'ChargingScheduler',
    'TeamEnergyOptimizer', 'EnergyBalancer',
    
    # Return to Base
    'ReturnToBasePlanner', 'EnergyConstrainedPath', 'SafeReturnPolicy',
    'ChargingStationSelector', 'EmergencyReturn',
    
    # Adaptive Power
    'AdaptivePowerManager', 'PowerMode', 'PerformanceScaler',
    'ThermalThrottling', 'DynamicVoltageScaling'
]