"""Environmental Factors Model

This module simulates environmental conditions that affect drone flight,
including wind, weather, and terrain effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WeatherConditions:
    """Weather condition parameters"""
    temperature: float = 20.0  # Celsius
    pressure: float = 101325.0  # Pa (sea level)
    humidity: float = 0.5  # Relative humidity (0-1)
    visibility: float = 10000.0  # meters
    precipitation_rate: float = 0.0  # mm/hour
    cloud_ceiling: float = 1000.0  # meters


class WindField:
    """3D wind field model with turbulence"""
    
    def __init__(
        self,
        base_velocity: np.ndarray = None,
        turbulence_intensity: float = 0.1,
        gust_frequency: float = 0.1,
        shear_coefficient: float = 0.2
    ):
        """Initialize wind field
        
        Args:
            base_velocity: Base wind velocity [vx, vy, vz] m/s
            turbulence_intensity: Turbulence intensity (0-1)
            gust_frequency: Frequency of wind gusts (Hz)
            shear_coefficient: Wind shear coefficient
        """
        self.base_velocity = base_velocity if base_velocity is not None else np.zeros(3)
        self.turbulence_intensity = turbulence_intensity
        self.gust_frequency = gust_frequency
        self.shear_coefficient = shear_coefficient
        
        # Turbulence state
        self.turbulence_state = np.zeros(3)
        self.gust_timer = 0.0
        self.current_gust = np.zeros(3)
        
        # Dryden turbulence model parameters
        self.turbulence_length_scale = 100.0  # meters
        self.turbulence_bandwidth = 1.0  # rad/s
    
    def get_wind_at_position(
        self,
        position: np.ndarray,
        time: float
    ) -> np.ndarray:
        """Get wind velocity at specific position and time
        
        Args:
            position: Position [x, y, z]
            time: Current time
            
        Returns:
            Wind velocity vector
        """
        # Base wind
        wind = self.base_velocity.copy()
        
        # Height-dependent wind shear (power law)
        if position[2] > 0:
            shear_factor = (position[2] / 10.0) ** self.shear_coefficient
            wind[:2] *= shear_factor
        
        # Add turbulence
        turbulence = self._calculate_turbulence(position, time)
        wind += turbulence
        
        # Add gusts
        gust = self._calculate_gust(time)
        wind += gust
        
        return wind
    
    def _calculate_turbulence(self, position: np.ndarray, time: float) -> np.ndarray:
        """Calculate turbulence using simplified Dryden model"""
        # Spatial variation
        spatial_factor = np.sin(position[0] / 50) * np.cos(position[1] / 50)
        
        # Temporal variation (colored noise)
        dt = 0.01  # Assumed timestep
        tau = self.turbulence_length_scale / np.linalg.norm(self.base_velocity + 0.1)
        alpha = dt / tau
        
        # Update turbulence state (first-order filter)
        noise = np.random.randn(3) * self.turbulence_intensity
        self.turbulence_state = (1 - alpha) * self.turbulence_state + alpha * noise
        
        # Scale by intensity and add spatial variation
        turbulence = self.turbulence_state * (1 + 0.5 * spatial_factor)
        
        # Scale based on base wind speed
        wind_speed = np.linalg.norm(self.base_velocity)
        turbulence *= np.sqrt(wind_speed + 1.0)
        
        return turbulence
    
    def _calculate_gust(self, time: float) -> np.ndarray:
        """Calculate wind gusts"""
        # Check if new gust should start
        if time - self.gust_timer > 1.0 / self.gust_frequency:
            if np.random.random() < 0.1:  # 10% chance per check
                # Generate new gust
                gust_magnitude = np.random.uniform(5, 15)  # m/s
                gust_direction = np.random.randn(3)
                gust_direction /= np.linalg.norm(gust_direction)
                
                self.current_gust = gust_magnitude * gust_direction
                self.gust_timer = time
        
        # Decay gust
        gust_decay_time = 3.0  # seconds
        elapsed = time - self.gust_timer
        if elapsed < gust_decay_time:
            decay_factor = np.exp(-elapsed / gust_decay_time)
            return self.current_gust * decay_factor
        else:
            return np.zeros(3)


class EnvironmentalFactors:
    """Comprehensive environmental factors affecting flight"""
    
    def __init__(
        self,
        enable_wind: bool = True,
        enable_weather: bool = True,
        enable_terrain: bool = True
    ):
        """Initialize environmental factors
        
        Args:
            enable_wind: Enable wind effects
            enable_weather: Enable weather effects
            enable_terrain: Enable terrain effects
        """
        self.enable_wind = enable_wind
        self.enable_weather = enable_weather
        self.enable_terrain = enable_terrain
        
        # Wind field
        self.wind_field = WindField()
        
        # Weather conditions
        self.weather = WeatherConditions()
        
        # Terrain
        self.terrain_height_map: Optional[Callable] = None
        self.ground_effect_height = 5.0  # meters
        
        # Time tracking
        self.time = 0.0
        
        logger.info("Initialized EnvironmentalFactors")
    
    def update(self, dt: float):
        """Update environmental conditions
        
        Args:
            dt: Time step
        """
        self.time += dt
        
        # Could add dynamic weather changes here
        # For now, weather remains constant unless explicitly changed
    
    def set_wind_conditions(
        self,
        base_velocity: np.ndarray,
        turbulence_intensity: float = 0.1,
        gust_frequency: float = 0.1
    ):
        """Set wind conditions
        
        Args:
            base_velocity: Base wind velocity
            turbulence_intensity: Turbulence intensity
            gust_frequency: Gust frequency
        """
        self.wind_field.base_velocity = base_velocity.copy()
        self.wind_field.turbulence_intensity = turbulence_intensity
        self.wind_field.gust_frequency = gust_frequency
        
        logger.info(f"Set wind: base={base_velocity}, turbulence={turbulence_intensity}")
    
    def set_weather_conditions(self, weather: WeatherConditions):
        """Set weather conditions
        
        Args:
            weather: Weather conditions
        """
        self.weather = weather
        logger.info(f"Set weather: T={weather.temperature}C, P={weather.pressure}Pa")
    
    def set_terrain_function(self, height_function: Callable[[float, float], float]):
        """Set terrain height function
        
        Args:
            height_function: Function that returns height given (x, y)
        """
        self.terrain_height_map = height_function
    
    def get_air_density(self, altitude: float) -> float:
        """Calculate air density at altitude
        
        Args:
            altitude: Altitude above sea level (m)
            
        Returns:
            Air density (kg/m³)
        """
        if not self.enable_weather:
            return 1.225  # Standard sea level density
        
        # International Standard Atmosphere model
        T0 = 288.15  # Sea level temperature (K)
        L = 0.0065   # Temperature lapse rate (K/m)
        R = 287.05   # Gas constant for air (J/(kg·K))
        g = 9.81     # Gravity (m/s²)
        
        # Temperature at altitude
        T = self.weather.temperature + 273.15 - L * altitude
        
        # Pressure at altitude
        P = self.weather.pressure * (T / T0) ** (g / (R * L))
        
        # Density from ideal gas law
        density = P / (R * T)
        
        # Humidity correction (simplified)
        density *= (1 - 0.0035 * self.weather.humidity)
        
        return density
    
    def get_wind_velocity(self, position: np.ndarray) -> np.ndarray:
        """Get wind velocity at position
        
        Args:
            position: Position [x, y, z]
            
        Returns:
            Wind velocity vector
        """
        if not self.enable_wind:
            return np.zeros(3)
        
        return self.wind_field.get_wind_at_position(position, self.time)
    
    def get_terrain_height(self, x: float, y: float) -> float:
        """Get terrain height at position
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Terrain height
        """
        if not self.enable_terrain or self.terrain_height_map is None:
            return 0.0
        
        return self.terrain_height_map(x, y)
    
    def calculate_environmental_forces(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vehicle_params: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """Calculate all environmental forces on vehicle
        
        Args:
            position: Vehicle position
            velocity: Vehicle velocity
            vehicle_params: Vehicle parameters (mass, area, etc.)
            
        Returns:
            Dictionary of force vectors
        """
        forces = {}
        
        # Wind forces
        if self.enable_wind:
            wind_velocity = self.get_wind_velocity(position)
            relative_velocity = velocity - wind_velocity
            
            # Drag force
            air_density = self.get_air_density(position[2])
            drag_coefficient = vehicle_params.get("drag_coefficient", 0.5)
            reference_area = vehicle_params.get("reference_area", 0.1)
            
            v_mag = np.linalg.norm(relative_velocity)
            if v_mag > 0:
                drag_force = -0.5 * air_density * drag_coefficient * \
                           reference_area * v_mag * relative_velocity
            else:
                drag_force = np.zeros(3)
            
            forces["wind_drag"] = drag_force
            forces["wind_velocity"] = wind_velocity
        
        # Ground effect
        if self.enable_terrain:
            ground_height = self.get_terrain_height(position[0], position[1])
            height_above_ground = position[2] - ground_height
            
            if 0 < height_above_ground < self.ground_effect_height:
                # Increased lift near ground
                effect_strength = 1 - height_above_ground / self.ground_effect_height
                ground_effect_force = np.array([0, 0, effect_strength * 10.0])
                forces["ground_effect"] = ground_effect_force
        
        # Weather effects
        if self.enable_weather:
            # Rain/precipitation drag
            if self.weather.precipitation_rate > 0:
                rain_drag_factor = 0.001 * self.weather.precipitation_rate
                rain_drag = -rain_drag_factor * velocity
                forces["rain_drag"] = rain_drag
            
            # Reduced visibility effects (for sensor modeling)
            forces["visibility_factor"] = self.weather.visibility / 10000.0
        
        return forces
    
    def check_flight_conditions(self, position: np.ndarray) -> Dict[str, bool]:
        """Check if flight conditions are safe
        
        Args:
            position: Vehicle position
            
        Returns:
            Dictionary of condition checks
        """
        conditions = {}
        
        # Wind speed check
        wind = self.get_wind_velocity(position)
        wind_speed = np.linalg.norm(wind)
        conditions["wind_safe"] = wind_speed < 15.0  # 15 m/s limit
        conditions["wind_warning"] = 10.0 < wind_speed < 15.0
        
        # Visibility check
        conditions["visibility_safe"] = self.weather.visibility > 1000.0
        conditions["visibility_warning"] = 1000.0 < self.weather.visibility < 3000.0
        
        # Ceiling check
        conditions["ceiling_safe"] = position[2] < self.weather.cloud_ceiling - 50.0
        
        # Precipitation check
        conditions["precipitation_safe"] = self.weather.precipitation_rate < 10.0  # mm/h
        
        # Temperature check
        conditions["temperature_safe"] = -10.0 < self.weather.temperature < 40.0
        
        # Overall safety
        conditions["overall_safe"] = all([
            conditions["wind_safe"],
            conditions["visibility_safe"],
            conditions["precipitation_safe"],
            conditions["temperature_safe"]
        ])
        
        return conditions
    
    def get_info(self) -> Dict[str, Any]:
        """Get environmental information
        
        Returns:
            Dictionary of environmental parameters
        """
        return {
            "time": self.time,
            "wind": {
                "base_velocity": self.wind_field.base_velocity.tolist(),
                "turbulence_intensity": self.wind_field.turbulence_intensity,
                "gust_frequency": self.wind_field.gust_frequency
            },
            "weather": {
                "temperature": self.weather.temperature,
                "pressure": self.weather.pressure,
                "humidity": self.weather.humidity,
                "visibility": self.weather.visibility,
                "precipitation_rate": self.weather.precipitation_rate,
                "cloud_ceiling": self.weather.cloud_ceiling
            },
            "features_enabled": {
                "wind": self.enable_wind,
                "weather": self.enable_weather,
                "terrain": self.enable_terrain
            }
        }