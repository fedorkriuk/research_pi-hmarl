"""Hardware Communication Interface

This module provides communication interfaces for hardware integration
including MAVLink, ROS bridge, serial, and network protocols.
"""

import asyncio
import serial
import socket
import struct
import json
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
import threading
from queue import Queue, Empty
import numpy as np

logger = logging.getLogger(__name__)


class CommunicationType(Enum):
    """Types of communication protocols"""
    SERIAL = "serial"
    NETWORK = "network"
    MAVLINK = "mavlink"
    ROS = "ros"
    CAN = "can"
    I2C = "i2c"
    SPI = "spi"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class HardwareMessage:
    """Generic hardware message"""
    message_id: str
    message_type: str
    payload: Dict[str, Any]
    priority: MessagePriority
    timestamp: float
    source: str
    destination: Optional[str] = None
    requires_ack: bool = False


class HardwareCommunication(ABC):
    """Abstract base class for hardware communication"""
    
    def __init__(self, interface_id: str, comm_type: CommunicationType):
        """Initialize communication interface
        
        Args:
            interface_id: Unique interface identifier
            comm_type: Type of communication
        """
        self.interface_id = interface_id
        self.comm_type = comm_type
        self.is_connected = False
        
        # Message handling
        self.rx_queue = Queue(maxsize=1000)
        self.tx_queue = Queue(maxsize=1000)
        self.message_handlers = {}
        
        # Threading
        self.rx_thread = None
        self.tx_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'bytes_sent': 0,
            'bytes_received': 0
        }
        
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to hardware interface
        
        Args:
            config: Connection configuration
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from hardware interface"""
        pass
    
    @abstractmethod
    async def send_raw(self, data: bytes) -> bool:
        """Send raw data
        
        Args:
            data: Raw bytes to send
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def receive_raw(self) -> Optional[bytes]:
        """Receive raw data
        
        Returns:
            Received bytes or None
        """
        pass
    
    def send_message(self, message: HardwareMessage) -> bool:
        """Send hardware message
        
        Args:
            message: Message to send
            
        Returns:
            Success status
        """
        if not self.is_connected:
            return False
        
        try:
            self.tx_queue.put(message, timeout=1.0)
            return True
        except:
            return False
    
    def receive_message(self, timeout: Optional[float] = None) -> Optional[HardwareMessage]:
        """Receive hardware message
        
        Args:
            timeout: Receive timeout
            
        Returns:
            Received message or None
        """
        try:
            return self.rx_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def start_communication(self):
        """Start communication threads"""
        self.running = True
        
        self.rx_thread = threading.Thread(
            target=self._rx_loop,
            daemon=True
        )
        self.tx_thread = threading.Thread(
            target=self._tx_loop,
            daemon=True
        )
        
        self.rx_thread.start()
        self.tx_thread.start()
        
        logger.info(f"Started communication for {self.interface_id}")
    
    def stop_communication(self):
        """Stop communication threads"""
        self.running = False
        
        if self.rx_thread:
            self.rx_thread.join(timeout=1.0)
        if self.tx_thread:
            self.tx_thread.join(timeout=1.0)
        
        logger.info(f"Stopped communication for {self.interface_id}")
    
    def _rx_loop(self):
        """Receive loop"""
        asyncio.run(self._async_rx_loop())
    
    async def _async_rx_loop(self):
        """Async receive loop"""
        while self.running:
            try:
                # Receive raw data
                raw_data = await self.receive_raw()
                
                if raw_data:
                    # Parse message
                    message = self._parse_message(raw_data)
                    
                    if message:
                        # Add to queue
                        if not self.rx_queue.full():
                            self.rx_queue.put(message)
                        
                        # Handle message
                        self._handle_message(message)
                        
                        # Update stats
                        self.stats['messages_received'] += 1
                        self.stats['bytes_received'] += len(raw_data)
                
                await asyncio.sleep(0.001)  # Small delay
                
            except Exception as e:
                logger.error(f"RX error: {e}")
                self.stats['errors'] += 1
    
    def _tx_loop(self):
        """Transmit loop"""
        asyncio.run(self._async_tx_loop())
    
    async def _async_tx_loop(self):
        """Async transmit loop"""
        while self.running:
            try:
                # Get message from queue
                message = self.tx_queue.get(timeout=0.1)
                
                # Serialize message
                raw_data = self._serialize_message(message)
                
                if raw_data:
                    # Send raw data
                    success = await self.send_raw(raw_data)
                    
                    if success:
                        self.stats['messages_sent'] += 1
                        self.stats['bytes_sent'] += len(raw_data)
                    else:
                        self.stats['errors'] += 1
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"TX error: {e}")
                self.stats['errors'] += 1
    
    def _parse_message(self, raw_data: bytes) -> Optional[HardwareMessage]:
        """Parse raw data into message
        
        Args:
            raw_data: Raw bytes
            
        Returns:
            Parsed message or None
        """
        # Default implementation - override in subclasses
        try:
            # Simple JSON parsing
            data = json.loads(raw_data.decode())
            
            return HardwareMessage(
                message_id=data.get('id', ''),
                message_type=data.get('type', ''),
                payload=data.get('payload', {}),
                priority=MessagePriority(data.get('priority', 1)),
                timestamp=data.get('timestamp', time.time()),
                source=data.get('source', ''),
                destination=data.get('destination'),
                requires_ack=data.get('requires_ack', False)
            )
        except:
            return None
    
    def _serialize_message(self, message: HardwareMessage) -> Optional[bytes]:
        """Serialize message to raw data
        
        Args:
            message: Message to serialize
            
        Returns:
            Serialized bytes or None
        """
        # Default implementation - override in subclasses
        try:
            data = {
                'id': message.message_id,
                'type': message.message_type,
                'payload': message.payload,
                'priority': message.priority.value,
                'timestamp': message.timestamp,
                'source': message.source,
                'destination': message.destination,
                'requires_ack': message.requires_ack
            }
            
            return json.dumps(data).encode()
        except:
            return None
    
    def _handle_message(self, message: HardwareMessage):
        """Handle received message
        
        Args:
            message: Received message
        """
        # Call registered handlers
        handlers = self.message_handlers.get(message.message_type, [])
        handlers.extend(self.message_handlers.get('*', []))  # Wildcard handlers
        
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()


class SerialInterface(HardwareCommunication):
    """Serial communication interface"""
    
    def __init__(self, interface_id: str):
        """Initialize serial interface
        
        Args:
            interface_id: Interface identifier
        """
        super().__init__(interface_id, CommunicationType.SERIAL)
        self.serial_port = None
        self.port_name = None
        self.baudrate = 115200
        
        logger.info(f"Initialized SerialInterface {interface_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to serial port
        
        Args:
            config: Serial configuration
            
        Returns:
            Success status
        """
        try:
            self.port_name = config.get('port', '/dev/ttyUSB0')
            self.baudrate = config.get('baudrate', 115200)
            
            self.serial_port = serial.Serial(
                port=self.port_name,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            
            self.is_connected = True
            self.start_communication()
            
            logger.info(f"Connected to serial port {self.port_name} at {self.baudrate} baud")
            return True
            
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from serial port"""
        self.stop_communication()
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self.is_connected = False
        logger.info("Disconnected from serial port")
    
    async def send_raw(self, data: bytes) -> bool:
        """Send raw data over serial
        
        Args:
            data: Data to send
            
        Returns:
            Success status
        """
        if not self.serial_port or not self.serial_port.is_open:
            return False
        
        try:
            self.serial_port.write(data)
            self.serial_port.flush()
            return True
        except Exception as e:
            logger.error(f"Serial send error: {e}")
            return False
    
    async def receive_raw(self) -> Optional[bytes]:
        """Receive raw data from serial
        
        Returns:
            Received data or None
        """
        if not self.serial_port or not self.serial_port.is_open:
            return None
        
        try:
            # Read available data
            if self.serial_port.in_waiting > 0:
                data = self.serial_port.read(self.serial_port.in_waiting)
                return data
            return None
        except Exception as e:
            logger.error(f"Serial receive error: {e}")
            return None
    
    def set_baudrate(self, baudrate: int):
        """Change baudrate
        
        Args:
            baudrate: New baudrate
        """
        if self.serial_port:
            self.serial_port.baudrate = baudrate
            self.baudrate = baudrate


class NetworkInterface(HardwareCommunication):
    """Network communication interface (TCP/UDP)"""
    
    def __init__(self, interface_id: str, protocol: str = "tcp"):
        """Initialize network interface
        
        Args:
            interface_id: Interface identifier
            protocol: Network protocol (tcp/udp)
        """
        super().__init__(interface_id, CommunicationType.NETWORK)
        self.protocol = protocol
        self.socket = None
        self.host = None
        self.port = None
        self.is_server = False
        self.client_socket = None
        
        logger.info(f"Initialized NetworkInterface {interface_id} ({protocol})")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect network interface
        
        Args:
            config: Network configuration
            
        Returns:
            Success status
        """
        try:
            self.host = config.get('host', 'localhost')
            self.port = config.get('port', 5000)
            self.is_server = config.get('is_server', False)
            
            if self.protocol == "tcp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                if self.is_server:
                    self.socket.bind((self.host, self.port))
                    self.socket.listen(1)
                    self.socket.settimeout(0.1)
                    logger.info(f"TCP server listening on {self.host}:{self.port}")
                else:
                    self.socket.connect((self.host, self.port))
                    logger.info(f"Connected to TCP server {self.host}:{self.port}")
                    
            else:  # UDP
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                
                if self.is_server:
                    self.socket.bind((self.host, self.port))
                    logger.info(f"UDP server bound to {self.host}:{self.port}")
            
            self.socket.settimeout(0.1)
            self.is_connected = True
            self.start_communication()
            
            return True
            
        except Exception as e:
            logger.error(f"Network connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect network interface"""
        self.stop_communication()
        
        if self.client_socket:
            self.client_socket.close()
        
        if self.socket:
            self.socket.close()
        
        self.is_connected = False
        logger.info("Disconnected from network")
    
    async def send_raw(self, data: bytes) -> bool:
        """Send raw data over network
        
        Args:
            data: Data to send
            
        Returns:
            Success status
        """
        if not self.socket:
            return False
        
        try:
            if self.protocol == "tcp":
                if self.is_server and self.client_socket:
                    self.client_socket.send(data)
                elif not self.is_server:
                    self.socket.send(data)
            else:  # UDP
                if self.is_server:
                    # Need destination address for UDP
                    pass
                else:
                    self.socket.sendto(data, (self.host, self.port))
            
            return True
            
        except Exception as e:
            logger.error(f"Network send error: {e}")
            return False
    
    async def receive_raw(self) -> Optional[bytes]:
        """Receive raw data from network
        
        Returns:
            Received data or None
        """
        if not self.socket:
            return None
        
        try:
            if self.protocol == "tcp":
                if self.is_server:
                    # Accept connections
                    if not self.client_socket:
                        try:
                            self.client_socket, addr = self.socket.accept()
                            self.client_socket.settimeout(0.1)
                            logger.info(f"Client connected from {addr}")
                        except socket.timeout:
                            return None
                    
                    # Receive from client
                    if self.client_socket:
                        try:
                            data = self.client_socket.recv(4096)
                            if data:
                                return data
                        except socket.timeout:
                            pass
                else:
                    # Receive as client
                    data = self.socket.recv(4096)
                    if data:
                        return data
            
            else:  # UDP
                data, addr = self.socket.recvfrom(4096)
                if data:
                    return data
            
            return None
            
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Network receive error: {e}")
            return None


class MAVLinkProtocol(HardwareCommunication):
    """MAVLink protocol implementation"""
    
    def __init__(self, interface_id: str, system_id: int = 1, component_id: int = 1):
        """Initialize MAVLink protocol
        
        Args:
            interface_id: Interface identifier
            system_id: MAVLink system ID
            component_id: MAVLink component ID
        """
        super().__init__(interface_id, CommunicationType.MAVLINK)
        self.system_id = system_id
        self.component_id = component_id
        self.sequence = 0
        self.transport = None  # Underlying transport (serial/network)
        
        logger.info(f"Initialized MAVLinkProtocol {interface_id}")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect MAVLink interface
        
        Args:
            config: Connection configuration
            
        Returns:
            Success status
        """
        try:
            # Create underlying transport
            transport_type = config.get('transport', 'serial')
            
            if transport_type == 'serial':
                self.transport = SerialInterface(f"{self.interface_id}_transport")
            else:
                self.transport = NetworkInterface(f"{self.interface_id}_transport", "udp")
            
            # Connect transport
            if not await self.transport.connect(config):
                return False
            
            self.is_connected = True
            self.start_communication()
            
            # Send heartbeat
            await self.send_heartbeat()
            
            logger.info("MAVLink protocol connected")
            return True
            
        except Exception as e:
            logger.error(f"MAVLink connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect MAVLink interface"""
        self.stop_communication()
        
        if self.transport:
            await self.transport.disconnect()
        
        self.is_connected = False
        logger.info("MAVLink protocol disconnected")
    
    async def send_raw(self, data: bytes) -> bool:
        """Send raw MAVLink data
        
        Args:
            data: MAVLink packet
            
        Returns:
            Success status
        """
        if self.transport:
            return await self.transport.send_raw(data)
        return False
    
    async def receive_raw(self) -> Optional[bytes]:
        """Receive raw MAVLink data
        
        Returns:
            MAVLink packet or None
        """
        if self.transport:
            return await self.transport.receive_raw()
        return None
    
    async def send_heartbeat(self):
        """Send MAVLink heartbeat"""
        # In real implementation, would use pymavlink
        heartbeat_msg = HardwareMessage(
            message_id=f"hb_{self.sequence}",
            message_type="HEARTBEAT",
            payload={
                'type': 1,  # MAV_TYPE_GENERIC
                'autopilot': 0,  # MAV_AUTOPILOT_GENERIC
                'base_mode': 0,
                'custom_mode': 0,
                'system_status': 0
            },
            priority=MessagePriority.HIGH,
            timestamp=time.time(),
            source=f"{self.system_id}:{self.component_id}"
        )
        
        self.send_message(heartbeat_msg)
        self.sequence += 1
    
    def _parse_message(self, raw_data: bytes) -> Optional[HardwareMessage]:
        """Parse MAVLink message
        
        Args:
            raw_data: Raw MAVLink packet
            
        Returns:
            Parsed message or None
        """
        # In real implementation, would use MAVLink parser
        # This is simplified
        try:
            # Check for MAVLink v1 start byte
            if raw_data[0] == 0xFE:
                # Parse header
                payload_len = raw_data[1]
                seq = raw_data[2]
                sys_id = raw_data[3]
                comp_id = raw_data[4]
                msg_id = raw_data[5]
                
                # Extract payload
                payload_data = raw_data[6:6+payload_len]
                
                return HardwareMessage(
                    message_id=f"mav_{seq}",
                    message_type=f"MAVLINK_{msg_id}",
                    payload={'raw': payload_data.hex()},
                    priority=MessagePriority.NORMAL,
                    timestamp=time.time(),
                    source=f"{sys_id}:{comp_id}"
                )
        except:
            pass
        
        return None


class ROSBridge(HardwareCommunication):
    """ROS bridge for hardware communication"""
    
    def __init__(self, interface_id: str, ros_version: int = 1):
        """Initialize ROS bridge
        
        Args:
            interface_id: Interface identifier
            ros_version: ROS version (1 or 2)
        """
        super().__init__(interface_id, CommunicationType.ROS)
        self.ros_version = ros_version
        self.node_name = f"pi_hmarl_{interface_id}"
        self.publishers = {}
        self.subscribers = {}
        
        logger.info(f"Initialized ROSBridge {interface_id} (ROS{ros_version})")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect ROS bridge
        
        Args:
            config: ROS configuration
            
        Returns:
            Success status
        """
        try:
            if self.ros_version == 1:
                # import rospy
                # rospy.init_node(self.node_name)
                pass
            else:
                # import rclpy
                # rclpy.init()
                pass
            
            self.is_connected = True
            self.start_communication()
            
            logger.info(f"ROS{self.ros_version} bridge connected")
            return True
            
        except Exception as e:
            logger.error(f"ROS bridge connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect ROS bridge"""
        self.stop_communication()
        
        if self.ros_version == 1:
            # rospy.signal_shutdown("Disconnecting")
            pass
        else:
            # rclpy.shutdown()
            pass
        
        self.is_connected = False
        logger.info("ROS bridge disconnected")
    
    async def send_raw(self, data: bytes) -> bool:
        """Not used for ROS bridge"""
        return False
    
    async def receive_raw(self) -> Optional[bytes]:
        """Not used for ROS bridge"""
        return None
    
    def create_publisher(self, topic: str, msg_type: str):
        """Create ROS publisher
        
        Args:
            topic: Topic name
            msg_type: Message type
        """
        # In real implementation, would create actual ROS publisher
        self.publishers[topic] = msg_type
        logger.info(f"Created publisher for {topic}")
    
    def create_subscriber(self, topic: str, msg_type: str, callback: Callable):
        """Create ROS subscriber
        
        Args:
            topic: Topic name
            msg_type: Message type
            callback: Message callback
        """
        # In real implementation, would create actual ROS subscriber
        self.subscribers[topic] = {
            'type': msg_type,
            'callback': callback
        }
        logger.info(f"Created subscriber for {topic}")
    
    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish ROS message
        
        Args:
            topic: Topic name
            message: Message data
        """
        if topic in self.publishers:
            # Convert to HardwareMessage for internal handling
            hw_msg = HardwareMessage(
                message_id=f"ros_{time.time()}",
                message_type=f"ROS_{topic}",
                payload=message,
                priority=MessagePriority.NORMAL,
                timestamp=time.time(),
                source=self.node_name
            )
            
            self.send_message(hw_msg)


# Protocol factory
def create_communication_interface(
    interface_id: str,
    protocol: str,
    config: Dict[str, Any]
) -> HardwareCommunication:
    """Create communication interface
    
    Args:
        interface_id: Interface identifier
        protocol: Protocol type
        config: Configuration
        
    Returns:
        Communication interface
    """
    if protocol == "serial":
        interface = SerialInterface(interface_id)
    elif protocol == "network":
        interface = NetworkInterface(interface_id, config.get('network_protocol', 'tcp'))
    elif protocol == "mavlink":
        interface = MAVLinkProtocol(interface_id)
    elif protocol == "ros":
        interface = ROSBridge(interface_id, config.get('ros_version', 1))
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    
    return interface


# Example usage
async def example_communication():
    """Example of hardware communication"""
    
    # Create serial interface
    serial_comm = SerialInterface("arduino_serial")
    
    # Connect
    if await serial_comm.connect({'port': '/dev/ttyUSB0', 'baudrate': 115200}):
        
        # Register message handler
        def handle_sensor_data(message: HardwareMessage):
            logger.info(f"Received sensor data: {message.payload}")
        
        serial_comm.register_handler("SENSOR_DATA", handle_sensor_data)
        
        # Send command
        command = HardwareMessage(
            message_id="cmd_001",
            message_type="MOTOR_COMMAND",
            payload={'motor': 1, 'speed': 50},
            priority=MessagePriority.NORMAL,
            timestamp=time.time(),
            source="controller"
        )
        
        serial_comm.send_message(command)
        
        # Wait for responses
        await asyncio.sleep(5)
        
        # Get statistics
        stats = serial_comm.get_statistics()
        logger.info(f"Communication stats: {stats}")
        
        # Disconnect
        await serial_comm.disconnect()
    
    # Create MAVLink interface
    mavlink = MAVLinkProtocol("drone_mavlink")
    
    if await mavlink.connect({'transport': 'udp', 'host': '127.0.0.1', 'port': 14550}):
        # Send heartbeats
        for _ in range(5):
            await mavlink.send_heartbeat()
            await asyncio.sleep(1)
        
        await mavlink.disconnect()