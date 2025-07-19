"""Sensor Hardware Interface

This module provides interfaces for various sensors including cameras,
LiDAR, IMU, GPS, and sensor fusion capabilities.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
import threading
from queue import Queue
import cv2

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors"""
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    GPS = "gps"
    ULTRASONIC = "ultrasonic"
    INFRARED = "infrared"
    ENCODER = "encoder"


@dataclass
class SensorReading:
    """Generic sensor reading"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Any
    metadata: Dict[str, Any]
    quality: float  # 0-1 quality indicator


@dataclass
class CameraData:
    """Camera sensor data"""
    image: np.ndarray  # Image array
    resolution: Tuple[int, int]
    fps: float
    encoding: str  # e.g., 'rgb8', 'bgr8', 'mono8'
    intrinsics: Optional[np.ndarray] = None  # Camera matrix
    distortion: Optional[np.ndarray] = None  # Distortion coefficients


@dataclass
class LiDARData:
    """LiDAR sensor data"""
    points: np.ndarray  # Point cloud [N, 3] or [N, 4] with intensity
    ranges: np.ndarray  # Range measurements
    angles: np.ndarray  # Angle for each measurement
    intensities: Optional[np.ndarray] = None
    timestamp: float


@dataclass
class IMUData:
    """IMU sensor data"""
    acceleration: np.ndarray  # [ax, ay, az] m/s^2
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    orientation: Optional[np.ndarray] = None  # Quaternion [w, x, y, z]
    temperature: Optional[float] = None


@dataclass
class GPSData:
    """GPS sensor data"""
    latitude: float
    longitude: float
    altitude: float
    accuracy: float  # meters
    satellites: int
    fix_type: str  # 'none', '2d', '3d', 'dgps', 'rtk'


class SensorInterface(ABC):
    """Abstract base class for sensor interfaces"""
    
    def __init__(self, sensor_id: str, sensor_type: SensorType):
        """Initialize sensor interface
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.is_connected = False
        self.is_streaming = False
        self._callbacks = {}
        self._data_queue = Queue(maxsize=100)
        
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to sensor
        
        Args:
            config: Sensor configuration
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from sensor"""
        pass
    
    @abstractmethod
    async def start_streaming(self) -> bool:
        """Start data streaming
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def stop_streaming(self):
        """Stop data streaming"""
        pass
    
    @abstractmethod
    async def get_reading(self) -> SensorReading:
        """Get single sensor reading
        
        Returns:
            Sensor reading
        """
        pass
    
    @abstractmethod
    async def calibrate(self) -> bool:
        """Calibrate sensor
        
        Returns:
            Success status
        """
        pass
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback
        
        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger event callbacks
        
        Args:
            event: Event name
            data: Event data
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")


class CameraInterface(SensorInterface):
    """Interface for camera sensors"""
    
    def __init__(self, sensor_id: str):
        """Initialize camera interface
        
        Args:
            sensor_id: Camera identifier
        """
        super().__init__(sensor_id, SensorType.CAMERA)
        self.capture = None
        self.stream_thread = None
        self.resolution = (640, 480)
        self.fps = 30
        
        logger.info(f"Initialized CameraInterface for {sensor_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to camera
        
        Args:
            config: Camera configuration
            
        Returns:
            Success status
        """
        try:
            # Get camera source
            source = config.get('source', 0)  # Default to first camera
            self.resolution = config.get('resolution', (640, 480))
            self.fps = config.get('fps', 30)
            
            # Initialize OpenCV capture
            self.capture = cv2.VideoCapture(source)
            
            if not self.capture.isOpened():
                raise Exception("Failed to open camera")
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_connected = True
            logger.info(f"Connected to camera {self.sensor_id}")
            return True
            
        except Exception as e:
            logger.error(f"Camera connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from camera"""
        if self.is_streaming:
            await self.stop_streaming()
        
        if self.capture:
            self.capture.release()
        
        self.is_connected = False
        logger.info("Disconnected from camera")
    
    async def start_streaming(self) -> bool:
        """Start camera streaming
        
        Returns:
            Success status
        """
        if not self.is_connected or self.is_streaming:
            return False
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(
            target=self._streaming_loop,
            daemon=True
        )
        self.stream_thread.start()
        
        logger.info("Started camera streaming")
        return True
    
    async def stop_streaming(self):
        """Stop camera streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)
        logger.info("Stopped camera streaming")
    
    async def get_reading(self) -> SensorReading:
        """Get camera frame
        
        Returns:
            Camera reading
        """
        if not self.is_connected:
            raise Exception("Camera not connected")
        
        ret, frame = self.capture.read()
        
        if not ret:
            raise Exception("Failed to capture frame")
        
        camera_data = CameraData(
            image=frame,
            resolution=self.resolution,
            fps=self.fps,
            encoding='bgr8'
        )
        
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            data=camera_data,
            metadata={'frame_number': int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))},
            quality=1.0
        )
    
    async def calibrate(self) -> bool:
        """Calibrate camera
        
        Returns:
            Success status
        """
        # Implement camera calibration using checkerboard
        logger.info("Camera calibration not implemented")
        return True
    
    def _streaming_loop(self):
        """Camera streaming loop"""
        while self.is_streaming:
            try:
                ret, frame = self.capture.read()
                
                if ret:
                    camera_data = CameraData(
                        image=frame,
                        resolution=self.resolution,
                        fps=self.fps,
                        encoding='bgr8'
                    )
                    
                    reading = SensorReading(
                        sensor_id=self.sensor_id,
                        sensor_type=self.sensor_type,
                        timestamp=time.time(),
                        data=camera_data,
                        metadata={'frame_number': int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))},
                        quality=1.0
                    )
                    
                    # Add to queue
                    if not self._data_queue.full():
                        self._data_queue.put(reading)
                    
                    # Trigger callbacks
                    self._trigger_callbacks('frame', reading)
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
    
    def set_exposure(self, exposure: float):
        """Set camera exposure
        
        Args:
            exposure: Exposure value
        """
        if self.capture:
            self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
    
    def set_gain(self, gain: float):
        """Set camera gain
        
        Args:
            gain: Gain value
        """
        if self.capture:
            self.capture.set(cv2.CAP_PROP_GAIN, gain)


class LiDARInterface(SensorInterface):
    """Interface for LiDAR sensors"""
    
    def __init__(self, sensor_id: str):
        """Initialize LiDAR interface
        
        Args:
            sensor_id: LiDAR identifier
        """
        super().__init__(sensor_id, SensorType.LIDAR)
        self.scan_rate = 10  # Hz
        self.angle_min = -np.pi
        self.angle_max = np.pi
        self.angle_increment = np.pi / 180  # 1 degree
        self.range_min = 0.1
        self.range_max = 30.0
        
        logger.info(f"Initialized LiDARInterface for {sensor_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to LiDAR
        
        Args:
            config: LiDAR configuration
            
        Returns:
            Success status
        """
        try:
            # In real implementation, would connect to LiDAR
            # e.g., via serial port or ethernet
            
            self.scan_rate = config.get('scan_rate', 10)
            self.angle_min = config.get('angle_min', -np.pi)
            self.angle_max = config.get('angle_max', np.pi)
            
            self.is_connected = True
            logger.info(f"Connected to LiDAR {self.sensor_id}")
            return True
            
        except Exception as e:
            logger.error(f"LiDAR connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from LiDAR"""
        if self.is_streaming:
            await self.stop_streaming()
        
        self.is_connected = False
        logger.info("Disconnected from LiDAR")
    
    async def start_streaming(self) -> bool:
        """Start LiDAR streaming"""
        if not self.is_connected:
            return False
        
        self.is_streaming = True
        # Start streaming thread
        threading.Thread(target=self._scan_loop, daemon=True).start()
        
        logger.info("Started LiDAR streaming")
        return True
    
    async def stop_streaming(self):
        """Stop LiDAR streaming"""
        self.is_streaming = False
        logger.info("Stopped LiDAR streaming")
    
    async def get_reading(self) -> SensorReading:
        """Get LiDAR scan
        
        Returns:
            LiDAR reading
        """
        if not self.is_connected:
            raise Exception("LiDAR not connected")
        
        # Simulate LiDAR scan
        num_points = int((self.angle_max - self.angle_min) / self.angle_increment)
        angles = np.linspace(self.angle_min, self.angle_max, num_points)
        
        # Generate simulated ranges (would be real data)
        ranges = np.random.uniform(self.range_min, self.range_max, num_points)
        
        # Convert to point cloud
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])
        
        lidar_data = LiDARData(
            points=points,
            ranges=ranges,
            angles=angles,
            timestamp=time.time()
        )
        
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            data=lidar_data,
            metadata={'scan_time': 1.0 / self.scan_rate},
            quality=0.95
        )
    
    async def calibrate(self) -> bool:
        """Calibrate LiDAR"""
        logger.info("LiDAR calibration performed")
        return True
    
    def _scan_loop(self):
        """LiDAR scanning loop"""
        while self.is_streaming:
            try:
                # Get scan
                reading = asyncio.run(self.get_reading())
                
                # Add to queue
                if not self._data_queue.full():
                    self._data_queue.put(reading)
                
                # Trigger callbacks
                self._trigger_callbacks('scan', reading)
                
                # Control scan rate
                time.sleep(1.0 / self.scan_rate)
                
            except Exception as e:
                logger.error(f"Scan error: {e}")


class IMUInterface(SensorInterface):
    """Interface for IMU sensors"""
    
    def __init__(self, sensor_id: str):
        """Initialize IMU interface
        
        Args:
            sensor_id: IMU identifier
        """
        super().__init__(sensor_id, SensorType.IMU)
        self.sample_rate = 100  # Hz
        
        logger.info(f"Initialized IMUInterface for {sensor_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to IMU
        
        Args:
            config: IMU configuration
            
        Returns:
            Success status
        """
        try:
            # In real implementation, would connect via I2C/SPI
            self.sample_rate = config.get('sample_rate', 100)
            
            self.is_connected = True
            logger.info(f"Connected to IMU {self.sensor_id}")
            return True
            
        except Exception as e:
            logger.error(f"IMU connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from IMU"""
        self.is_connected = False
        logger.info("Disconnected from IMU")
    
    async def start_streaming(self) -> bool:
        """Start IMU streaming"""
        if not self.is_connected:
            return False
        
        self.is_streaming = True
        threading.Thread(target=self._imu_loop, daemon=True).start()
        
        return True
    
    async def stop_streaming(self):
        """Stop IMU streaming"""
        self.is_streaming = False
    
    async def get_reading(self) -> SensorReading:
        """Get IMU reading
        
        Returns:
            IMU reading
        """
        if not self.is_connected:
            raise Exception("IMU not connected")
        
        # Simulate IMU data
        imu_data = IMUData(
            acceleration=np.array([0.1, 0.0, 9.81]),
            angular_velocity=np.array([0.0, 0.0, 0.1]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            temperature=25.0
        )
        
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            data=imu_data,
            metadata={'sample_rate': self.sample_rate},
            quality=0.98
        )
    
    async def calibrate(self) -> bool:
        """Calibrate IMU (gyro and accel bias)"""
        logger.info("Performing IMU calibration...")
        
        # Collect samples while stationary
        samples = []
        for _ in range(1000):
            reading = await self.get_reading()
            samples.append(reading.data)
            await asyncio.sleep(0.01)
        
        # Calculate biases
        # In real implementation, would store and apply these
        
        logger.info("IMU calibration complete")
        return True
    
    def _imu_loop(self):
        """IMU data loop"""
        while self.is_streaming:
            try:
                reading = asyncio.run(self.get_reading())
                
                if not self._data_queue.full():
                    self._data_queue.put(reading)
                
                self._trigger_callbacks('imu_data', reading)
                
                time.sleep(1.0 / self.sample_rate)
                
            except Exception as e:
                logger.error(f"IMU error: {e}")


class GPSInterface(SensorInterface):
    """Interface for GPS sensors"""
    
    def __init__(self, sensor_id: str):
        """Initialize GPS interface
        
        Args:
            sensor_id: GPS identifier
        """
        super().__init__(sensor_id, SensorType.GPS)
        self.update_rate = 10  # Hz
        
        logger.info(f"Initialized GPSInterface for {sensor_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to GPS
        
        Args:
            config: GPS configuration
            
        Returns:
            Success status
        """
        try:
            # In real implementation, would connect via serial/USB
            self.update_rate = config.get('update_rate', 10)
            
            self.is_connected = True
            logger.info(f"Connected to GPS {self.sensor_id}")
            return True
            
        except Exception as e:
            logger.error(f"GPS connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from GPS"""
        self.is_connected = False
        logger.info("Disconnected from GPS")
    
    async def start_streaming(self) -> bool:
        """Start GPS streaming"""
        if not self.is_connected:
            return False
        
        self.is_streaming = True
        return True
    
    async def stop_streaming(self):
        """Stop GPS streaming"""
        self.is_streaming = False
    
    async def get_reading(self) -> SensorReading:
        """Get GPS reading
        
        Returns:
            GPS reading
        """
        if not self.is_connected:
            raise Exception("GPS not connected")
        
        # Simulate GPS data
        gps_data = GPSData(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=50.0,
            accuracy=2.5,
            satellites=12,
            fix_type='3d'
        )
        
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            data=gps_data,
            metadata={'hdop': 1.2, 'vdop': 1.5},
            quality=0.9
        )
    
    async def calibrate(self) -> bool:
        """Calibrate GPS (not typically needed)"""
        return True
    
    def get_distance_to(
        self,
        target_lat: float,
        target_lon: float
    ) -> float:
        """Calculate distance to target coordinates
        
        Args:
            target_lat: Target latitude
            target_lon: Target longitude
            
        Returns:
            Distance in meters
        """
        # Haversine formula
        R = 6371000  # Earth radius in meters
        
        # Get current position
        current = asyncio.run(self.get_reading()).data
        
        lat1, lon1 = np.radians([current.latitude, current.longitude])
        lat2, lon2 = np.radians([target_lat, target_lon])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c


class SensorFusion:
    """Sensor fusion for combining multiple sensor inputs"""
    
    def __init__(self):
        """Initialize sensor fusion"""
        self.sensors: Dict[str, SensorInterface] = {}
        self.fusion_thread = None
        self.is_running = False
        
        # Fusion parameters
        self.imu_weight = 0.8
        self.gps_weight = 0.2
        
        # State estimate
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion
        
        logger.info("Initialized SensorFusion")
    
    def add_sensor(self, sensor: SensorInterface):
        """Add sensor to fusion
        
        Args:
            sensor: Sensor interface
        """
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Added sensor {sensor.sensor_id} to fusion")
    
    def start_fusion(self):
        """Start sensor fusion"""
        self.is_running = True
        self.fusion_thread = threading.Thread(
            target=self._fusion_loop,
            daemon=True
        )
        self.fusion_thread.start()
        logger.info("Started sensor fusion")
    
    def stop_fusion(self):
        """Stop sensor fusion"""
        self.is_running = False
        if self.fusion_thread:
            self.fusion_thread.join()
        logger.info("Stopped sensor fusion")
    
    def _fusion_loop(self):
        """Main fusion loop"""
        while self.is_running:
            try:
                # Collect sensor data
                sensor_data = {}
                
                for sensor_id, sensor in self.sensors.items():
                    if sensor.is_connected:
                        try:
                            reading = asyncio.run(sensor.get_reading())
                            sensor_data[sensor_id] = reading
                        except Exception as e:
                            logger.error(f"Failed to get reading from {sensor_id}: {e}")
                
                # Perform fusion
                self._update_state(sensor_data)
                
                # Sleep for fusion rate
                time.sleep(0.01)  # 100Hz fusion
                
            except Exception as e:
                logger.error(f"Fusion error: {e}")
    
    def _update_state(self, sensor_data: Dict[str, SensorReading]):
        """Update fused state estimate
        
        Args:
            sensor_data: Dictionary of sensor readings
        """
        # Simple fusion example - would use EKF/UKF in practice
        
        # Update from IMU
        for sensor_id, reading in sensor_data.items():
            if reading.sensor_type == SensorType.IMU:
                imu_data = reading.data
                
                # Integrate acceleration for velocity
                dt = 0.01  # Fusion timestep
                self.velocity += imu_data.acceleration * dt
                
                # Update orientation
                if imu_data.orientation is not None:
                    self.orientation = imu_data.orientation
        
        # Update from GPS
        for sensor_id, reading in sensor_data.items():
            if reading.sensor_type == SensorType.GPS:
                gps_data = reading.data
                
                # Convert GPS to local coordinates
                # Simplified - would use proper coordinate transformation
                gps_position = np.array([
                    gps_data.longitude * 111320,  # Rough meters per degree
                    gps_data.latitude * 111320,
                    gps_data.altitude
                ])
                
                # Weighted update
                self.position = (
                    self.imu_weight * self.position +
                    self.gps_weight * gps_position
                )
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current fused state
        
        Returns:
            State dictionary
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy()
        }


# Example usage
async def example_sensor_usage():
    """Example of sensor usage"""
    
    # Create sensors
    camera = CameraInterface("camera_front")
    lidar = LiDARInterface("lidar_360")
    imu = IMUInterface("imu_main")
    gps = GPSInterface("gps_rtk")
    
    # Connect sensors
    await camera.connect({'source': 0, 'resolution': (1280, 720)})
    await lidar.connect({'scan_rate': 20})
    await imu.connect({'sample_rate': 200})
    await gps.connect({'update_rate': 10})
    
    # Create sensor fusion
    fusion = SensorFusion()
    fusion.add_sensor(camera)
    fusion.add_sensor(lidar)
    fusion.add_sensor(imu)
    fusion.add_sensor(gps)
    
    # Start fusion
    fusion.start_fusion()
    
    # Start streaming
    await camera.start_streaming()
    await lidar.start_streaming()
    
    # Get some readings
    for _ in range(10):
        state = fusion.get_state()
        logger.info(f"Fused state: {state}")
        await asyncio.sleep(1.0)
    
    # Cleanup
    fusion.stop_fusion()
    await camera.disconnect()
    await lidar.disconnect()
    await imu.disconnect()
    await gps.disconnect()