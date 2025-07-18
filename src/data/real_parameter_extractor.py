"""Real Parameter Extractor for PI-HMARL

This module extracts real-world physics parameters from manufacturer
specifications, research papers, and validated test data to enable
physics-accurate synthetic data generation.
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class DroneSpecifications:
    """Container for drone specifications"""
    name: str
    manufacturer: str
    mass: float  # kg
    max_speed: float  # m/s
    max_ascent_speed: float  # m/s
    max_descent_speed: float  # m/s
    max_altitude: float  # m
    battery_capacity: float  # mAh
    voltage: float  # V
    flight_time: float  # minutes
    dimensions: Dict[str, float]  # length, width, height in meters
    propeller_diameter: float  # meters
    motor_kv: float  # RPM/V
    max_payload: float  # kg
    operating_temperature: Tuple[float, float]  # min, max in Celsius
    
    # Aerodynamic parameters
    drag_coefficient: float
    frontal_area: float  # m^2
    lift_coefficient: float
    
    # Performance parameters
    motor_efficiency: float  # 0-1
    propeller_efficiency: float  # 0-1
    esc_efficiency: float  # 0-1
    
    # Communication
    control_frequency: float  # GHz
    video_frequency: float  # GHz
    max_transmission_range: float  # meters


@dataclass
class BatterySpecifications:
    """Container for battery specifications and discharge curves"""
    name: str
    manufacturer: str
    chemistry: str  # Li-ion, LiPo, etc.
    nominal_voltage: float  # V
    capacity: float  # mAh
    max_discharge_rate: float  # C rating
    internal_resistance: float  # mOhm
    weight: float  # grams
    
    # Discharge curves at different C rates
    discharge_curves: Dict[float, List[Tuple[float, float]]]  # C-rate -> [(time, voltage)]
    
    # Temperature effects
    temp_capacity_factor: Dict[float, float]  # temperature -> capacity factor
    temp_resistance_factor: Dict[float, float]  # temperature -> resistance factor
    
    # Aging parameters
    cycle_life: int  # number of cycles to 80% capacity
    calendar_life_days: int  # days to 80% capacity at storage


@dataclass
class CommunicationSpecifications:
    """Container for communication specifications"""
    protocol: str  # WiFi, 5G, LoRa, etc.
    frequency_band: float  # GHz
    bandwidth: float  # MHz
    max_data_rate: float  # Mbps
    
    # Latency characteristics (ms)
    latency_mean: float
    latency_std: float
    latency_min: float
    latency_max: float
    
    # Packet loss characteristics
    packet_loss_rate: float  # 0-1
    packet_loss_burst_prob: float  # probability of burst losses
    
    # Range and power
    max_range: float  # meters
    transmit_power: float  # dBm
    receiver_sensitivity: float  # dBm
    
    # Environmental effects
    rain_attenuation: float  # dB/km
    fog_attenuation: float  # dB/km


class RealParameterExtractor:
    """Extracts and manages real-world parameters for synthetic data generation"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the parameter extractor
        
        Args:
            data_dir: Directory containing real parameter data files
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "real_parameters"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parameter storage
        self.drone_specs = {}
        self.battery_specs = {}
        self.communication_specs = {}
        self.weather_data = {}
        
        # Load built-in specifications
        self._load_builtin_specs()
        
        # Load any custom specifications from files
        self._load_custom_specs()
        
        logger.info(f"Initialized RealParameterExtractor with {len(self.drone_specs)} drone specs")
    
    def _load_builtin_specs(self):
        """Load built-in real-world specifications"""
        
        # DJI Mavic 3 specifications
        self.drone_specs["dji_mavic_3"] = DroneSpecifications(
            name="Mavic 3",
            manufacturer="DJI",
            mass=0.895,  # kg
            max_speed=19.0,  # m/s (68.4 km/h)
            max_ascent_speed=8.0,  # m/s
            max_descent_speed=6.0,  # m/s
            max_altitude=6000.0,  # m
            battery_capacity=5000.0,  # mAh
            voltage=17.6,  # V (4S LiPo)
            flight_time=46.0,  # minutes
            dimensions={"length": 0.2831, "width": 0.2420, "height": 0.1071},  # m
            propeller_diameter=0.239,  # m (9.4 inches)
            motor_kv=920.0,  # RPM/V (estimated)
            max_payload=0.250,  # kg
            operating_temperature=(-10.0, 40.0),  # Celsius
            drag_coefficient=0.47,  # typical for quadcopter
            frontal_area=0.06,  # m^2 (estimated)
            lift_coefficient=1.2,  # typical for drone propellers
            motor_efficiency=0.85,
            propeller_efficiency=0.80,
            esc_efficiency=0.95,
            control_frequency=2.4,  # GHz
            video_frequency=5.8,  # GHz
            max_transmission_range=15000.0  # m
        )
        
        # Parrot ANAFI specifications
        self.drone_specs["parrot_anafi"] = DroneSpecifications(
            name="ANAFI",
            manufacturer="Parrot",
            mass=0.320,  # kg
            max_speed=16.0,  # m/s (57.6 km/h)
            max_ascent_speed=4.0,  # m/s
            max_descent_speed=3.0,  # m/s
            max_altitude=4500.0,  # m
            battery_capacity=2700.0,  # mAh
            voltage=7.6,  # V (2S LiPo)
            flight_time=25.0,  # minutes
            dimensions={"length": 0.175, "width": 0.240, "height": 0.065},  # m
            propeller_diameter=0.150,  # m
            motor_kv=1400.0,  # RPM/V (estimated)
            max_payload=0.100,  # kg
            operating_temperature=(-10.0, 40.0),  # Celsius
            drag_coefficient=0.45,
            frontal_area=0.04,  # m^2
            lift_coefficient=1.1,
            motor_efficiency=0.82,
            propeller_efficiency=0.78,
            esc_efficiency=0.94,
            control_frequency=2.4,  # GHz
            video_frequency=5.0,  # GHz
            max_transmission_range=4000.0  # m
        )
        
        # Samsung INR18650-25R battery specifications
        self.battery_specs["samsung_25r"] = BatterySpecifications(
            name="INR18650-25R",
            manufacturer="Samsung",
            chemistry="Li-ion",
            nominal_voltage=3.6,  # V
            capacity=2500.0,  # mAh
            max_discharge_rate=20.0,  # C
            internal_resistance=13.0,  # mOhm
            weight=45.0,  # grams
            discharge_curves={
                0.2: [(0, 4.2), (1200, 3.7), (2400, 3.4), (2500, 3.0)],  # 0.2C discharge
                1.0: [(0, 4.2), (240, 3.7), (480, 3.4), (500, 3.0)],     # 1C discharge
                10.0: [(0, 4.2), (24, 3.5), (48, 3.2), (50, 2.5)],       # 10C discharge
            },
            temp_capacity_factor={
                -20: 0.70, -10: 0.85, 0: 0.92, 10: 0.96, 
                20: 1.00, 30: 1.00, 40: 0.98, 50: 0.95
            },
            temp_resistance_factor={
                -20: 3.0, -10: 2.0, 0: 1.5, 10: 1.2,
                20: 1.0, 30: 1.0, 40: 1.1, 50: 1.3
            },
            cycle_life=250,  # cycles to 80% capacity
            calendar_life_days=730  # 2 years
        )
        
        # WiFi 802.11ac specifications
        self.communication_specs["wifi_ac"] = CommunicationSpecifications(
            protocol="WiFi 802.11ac",
            frequency_band=5.0,  # GHz
            bandwidth=80.0,  # MHz
            max_data_rate=866.7,  # Mbps
            latency_mean=3.0,  # ms
            latency_std=1.5,  # ms
            latency_min=1.0,  # ms
            latency_max=20.0,  # ms
            packet_loss_rate=0.001,  # 0.1%
            packet_loss_burst_prob=0.05,
            max_range=150.0,  # meters (outdoor)
            transmit_power=20.0,  # dBm
            receiver_sensitivity=-82.0,  # dBm
            rain_attenuation=0.04,  # dB/km at 5GHz
            fog_attenuation=0.2  # dB/km
        )
        
        # 5G NR specifications
        self.communication_specs["5g_nr"] = CommunicationSpecifications(
            protocol="5G NR",
            frequency_band=3.5,  # GHz (mid-band)
            bandwidth=100.0,  # MHz
            max_data_rate=2000.0,  # Mbps
            latency_mean=1.0,  # ms
            latency_std=0.5,  # ms
            latency_min=0.5,  # ms
            latency_max=5.0,  # ms
            packet_loss_rate=0.0001,  # 0.01%
            packet_loss_burst_prob=0.01,
            max_range=1000.0,  # meters
            transmit_power=23.0,  # dBm
            receiver_sensitivity=-90.0,  # dBm
            rain_attenuation=0.02,  # dB/km at 3.5GHz
            fog_attenuation=0.1  # dB/km
        )
        
        logger.info("Loaded built-in specifications")
    
    def _load_custom_specs(self):
        """Load custom specifications from data files"""
        # Load drone specs
        drone_specs_dir = self.data_dir / "drone_specs"
        if drone_specs_dir.exists():
            for spec_file in drone_specs_dir.glob("*.json"):
                try:
                    with open(spec_file, 'r') as f:
                        spec_data = json.load(f)
                        drone_spec = DroneSpecifications(**spec_data)
                        key = f"{drone_spec.manufacturer.lower()}_{drone_spec.name.lower().replace(' ', '_')}"
                        self.drone_specs[key] = drone_spec
                        logger.info(f"Loaded drone spec: {key}")
                except Exception as e:
                    logger.error(f"Error loading drone spec from {spec_file}: {e}")
        
        # Load battery specs
        battery_specs_dir = self.data_dir / "battery_data"
        if battery_specs_dir.exists():
            for spec_file in battery_specs_dir.glob("*.json"):
                try:
                    with open(spec_file, 'r') as f:
                        spec_data = json.load(f)
                        battery_spec = BatterySpecifications(**spec_data)
                        key = f"{battery_spec.manufacturer.lower()}_{battery_spec.name.lower().replace(' ', '_')}"
                        self.battery_specs[key] = battery_spec
                        logger.info(f"Loaded battery spec: {key}")
                except Exception as e:
                    logger.error(f"Error loading battery spec from {spec_file}: {e}")
    
    def get_drone_specs(self, drone_name: str) -> Optional[DroneSpecifications]:
        """Get drone specifications by name
        
        Args:
            drone_name: Name/key of the drone
            
        Returns:
            DroneSpecifications or None if not found
        """
        return self.drone_specs.get(drone_name)
    
    def get_battery_specs(self, battery_name: str) -> Optional[BatterySpecifications]:
        """Get battery specifications by name
        
        Args:
            battery_name: Name/key of the battery
            
        Returns:
            BatterySpecifications or None if not found
        """
        return self.battery_specs.get(battery_name)
    
    def get_communication_specs(self, protocol: str) -> Optional[CommunicationSpecifications]:
        """Get communication specifications by protocol
        
        Args:
            protocol: Communication protocol name
            
        Returns:
            CommunicationSpecifications or None if not found
        """
        return self.communication_specs.get(protocol)
    
    def get_battery_voltage(self, battery_name: str, soc: float, c_rate: float = 1.0) -> float:
        """Get battery voltage at given state of charge and discharge rate
        
        Args:
            battery_name: Battery name/key
            soc: State of charge (0-1)
            c_rate: Discharge rate in C
            
        Returns:
            Voltage in V
        """
        battery = self.get_battery_specs(battery_name)
        if not battery:
            return 0.0
        
        # Find closest C-rate curve
        available_rates = sorted(battery.discharge_curves.keys())
        closest_rate = min(available_rates, key=lambda x: abs(x - c_rate))
        
        # Get discharge curve
        curve = battery.discharge_curves[closest_rate]
        
        # Interpolate voltage based on capacity used
        capacity_used = (1 - soc) * battery.capacity
        
        # Find surrounding points
        for i, (time, voltage) in enumerate(curve):
            if i > 0:
                prev_time, prev_voltage = curve[i-1]
                if prev_time <= capacity_used <= time:
                    # Linear interpolation
                    ratio = (capacity_used - prev_time) / (time - prev_time)
                    return prev_voltage + ratio * (voltage - prev_voltage)
        
        # If beyond curve, return minimum voltage
        return curve[-1][1]
    
    def get_weather_conditions(self, scenario: str = "nominal") -> Dict[str, Any]:
        """Get weather conditions for a scenario
        
        Args:
            scenario: Weather scenario (nominal, windy, rainy, extreme)
            
        Returns:
            Dictionary of weather parameters
        """
        scenarios = {
            "nominal": {
                "wind_speed": 2.0,  # m/s
                "wind_direction": 0.0,  # degrees
                "temperature": 20.0,  # Celsius
                "pressure": 101325.0,  # Pa
                "humidity": 50.0,  # %
                "visibility": 10000.0,  # m
                "precipitation": 0.0  # mm/h
            },
            "windy": {
                "wind_speed": 10.0,  # m/s
                "wind_direction": 45.0,  # degrees
                "temperature": 15.0,  # Celsius
                "pressure": 101000.0,  # Pa
                "humidity": 60.0,  # %
                "visibility": 8000.0,  # m
                "precipitation": 0.0  # mm/h
            },
            "rainy": {
                "wind_speed": 5.0,  # m/s
                "wind_direction": 90.0,  # degrees
                "temperature": 12.0,  # Celsius
                "pressure": 100500.0,  # Pa
                "humidity": 90.0,  # %
                "visibility": 2000.0,  # m
                "precipitation": 10.0  # mm/h
            },
            "extreme": {
                "wind_speed": 20.0,  # m/s
                "wind_direction": 180.0,  # degrees
                "temperature": 5.0,  # Celsius
                "pressure": 99000.0,  # Pa
                "humidity": 95.0,  # %
                "visibility": 500.0,  # m
                "precipitation": 25.0  # mm/h
            }
        }
        
        return scenarios.get(scenario, scenarios["nominal"])
    
    def save_specs_to_file(self, output_dir: Optional[Path] = None):
        """Save all specifications to JSON files
        
        Args:
            output_dir: Directory to save files (uses data_dir if None)
        """
        if output_dir is None:
            output_dir = self.data_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save drone specs
        drone_output = output_dir / "all_drone_specs.json"
        drone_data = {k: asdict(v) for k, v in self.drone_specs.items()}
        with open(drone_output, 'w') as f:
            json.dump(drone_data, f, indent=2)
        
        # Save battery specs
        battery_output = output_dir / "all_battery_specs.json"
        battery_data = {k: asdict(v) for k, v in self.battery_specs.items()}
        with open(battery_output, 'w') as f:
            json.dump(battery_data, f, indent=2)
        
        # Save communication specs
        comm_output = output_dir / "all_communication_specs.json"
        comm_data = {k: asdict(v) for k, v in self.communication_specs.items()}
        with open(comm_output, 'w') as f:
            json.dump(comm_data, f, indent=2)
        
        logger.info(f"Saved all specifications to {output_dir}")
    
    def generate_parameter_report(self) -> str:
        """Generate a summary report of all loaded parameters
        
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("PI-HMARL Real Parameter Summary")
        report.append("=" * 60)
        
        # Drone specifications
        report.append(f"\nDrone Specifications ({len(self.drone_specs)} loaded):")
        report.append("-" * 40)
        for name, spec in self.drone_specs.items():
            report.append(f"  {name}:")
            report.append(f"    - Mass: {spec.mass} kg")
            report.append(f"    - Max Speed: {spec.max_speed} m/s")
            report.append(f"    - Battery: {spec.battery_capacity} mAh")
            report.append(f"    - Flight Time: {spec.flight_time} min")
        
        # Battery specifications
        report.append(f"\nBattery Specifications ({len(self.battery_specs)} loaded):")
        report.append("-" * 40)
        for name, spec in self.battery_specs.items():
            report.append(f"  {name}:")
            report.append(f"    - Chemistry: {spec.chemistry}")
            report.append(f"    - Capacity: {spec.capacity} mAh")
            report.append(f"    - Voltage: {spec.nominal_voltage} V")
            report.append(f"    - Max Discharge: {spec.max_discharge_rate}C")
        
        # Communication specifications
        report.append(f"\nCommunication Specifications ({len(self.communication_specs)} loaded):")
        report.append("-" * 40)
        for name, spec in self.communication_specs.items():
            report.append(f"  {name}:")
            report.append(f"    - Frequency: {spec.frequency_band} GHz")
            report.append(f"    - Data Rate: {spec.max_data_rate} Mbps")
            report.append(f"    - Latency: {spec.latency_mean}±{spec.latency_std} ms")
            report.append(f"    - Range: {spec.max_range} m")
        
        return "\n".join(report)


# Convenience function
def create_parameter_extractor(data_dir: Optional[Path] = None) -> RealParameterExtractor:
    """Create and return a RealParameterExtractor instance
    
    Args:
        data_dir: Optional data directory path
        
    Returns:
        RealParameterExtractor instance
    """
    return RealParameterExtractor(data_dir)