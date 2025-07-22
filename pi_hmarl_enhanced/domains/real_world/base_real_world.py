"""
Base class for real-world domain integration
Supports seamless simulation-to-hardware deployment
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import time
import threading
from queue import Queue

@dataclass
class HardwareConfig:
    """Configuration for hardware interfaces"""
    robot_type: str
    communication_protocol: str
    control_frequency: float = 10.0  # Hz
    safety_timeout: float = 0.5  # seconds
    emergency_stop_enabled: bool = True
    data_logging_enabled: bool = True
    
@dataclass
class RealWorldMetrics:
    """Metrics specific to real-world deployment"""
    task_completion_time: float
    collision_count: int
    energy_consumption: float
    communication_latency: float
    hardware_failures: int
    safety_violations: int
    coordination_quality: float

class RealWorldDomain(ABC):
    """
    Base class for domains that support both simulation and real hardware
    
    Key Features:
    - Unified interface for sim and real deployment
    - Safety monitoring and emergency stop
    - Real-time data collection and logging
    - Hardware abstraction layer
    """
    
    def __init__(self, 
                 sim_mode: bool = True,
                 hardware_interface: Optional[Any] = None,
                 config: Optional[HardwareConfig] = None):
        """
        Initialize real-world domain
        
        Args:
            sim_mode: If True, run in simulation. If False, connect to hardware
            hardware_interface: Interface to actual robots (e.g., ROS2 node)
            config: Hardware configuration parameters
        """
        self.sim_mode = sim_mode
        self.hardware_interface = hardware_interface
        self.config = config or HardwareConfig(
            robot_type="generic",
            communication_protocol="ROS2"
        )
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Safety systems
        self.emergency_stop = False
        self.safety_monitor_thread = None
        self.command_queue = Queue()
        self.state_queue = Queue()
        
        # Metrics tracking
        self.metrics = RealWorldMetrics(
            task_completion_time=0.0,
            collision_count=0,
            energy_consumption=0.0,
            communication_latency=0.0,
            hardware_failures=0,
            safety_violations=0,
            coordination_quality=0.0
        )
        
        # Data collection
        self.collected_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'physics_violations': [],
            'hardware_status': []
        }
        
        # Initialize domain-specific components
        self._initialize_domain()
        
        # Start safety monitoring if in hardware mode
        if not sim_mode and self.config.emergency_stop_enabled:
            self._start_safety_monitor()
    
    @abstractmethod
    def _initialize_domain(self):
        """Initialize domain-specific components"""
        pass
    
    @abstractmethod
    def get_physics_constraints(self) -> Dict[str, Any]:
        """Return domain-specific physics constraints"""
        pass
    
    @abstractmethod
    def validate_action_physics(self, action: np.ndarray, agent_id: int) -> Tuple[bool, Optional[str]]:
        """
        Validate if action satisfies physics constraints
        
        Returns:
            (is_valid, violation_message)
        """
        pass
    
    def deploy_to_hardware(self) -> bool:
        """
        Deploy trained policy to actual hardware
        
        Returns:
            Success status
        """
        if self.sim_mode:
            self.logger.warning("Cannot deploy to hardware while in simulation mode")
            return False
            
        if self.hardware_interface is None:
            self.logger.error("No hardware interface configured")
            return False
        
        try:
            self.logger.info("Starting hardware deployment...")
            
            # Perform pre-deployment checks
            if not self._hardware_safety_check():
                self.logger.error("Hardware safety check failed")
                return False
            
            # Initialize hardware connections
            self.hardware_interface.initialize()
            
            # Start control loop
            self._start_hardware_control_loop()
            
            self.logger.info("Successfully deployed to hardware")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware deployment failed: {e}")
            self.emergency_stop = True
            return False
    
    def collect_real_world_data(self, duration: float = 60.0) -> Dict[str, List]:
        """
        Collect data from actual hardware execution
        
        Args:
            duration: Collection duration in seconds
            
        Returns:
            Dictionary of collected data
        """
        if self.sim_mode:
            self.logger.warning("Collecting simulated data, not real-world data")
        
        start_time = time.time()
        self.logger.info(f"Starting data collection for {duration} seconds")
        
        while time.time() - start_time < duration:
            # Collect current state
            state = self._get_current_state()
            
            # Get action from policy
            action = self._compute_action(state)
            
            # Validate physics
            is_valid, violation = self.validate_action_physics(action, agent_id=0)
            
            if not is_valid:
                self.logger.warning(f"Physics violation: {violation}")
                self.collected_data['physics_violations'].append({
                    'time': time.time() - start_time,
                    'violation': violation,
                    'action': action
                })
                # Apply safety action instead
                action = self._get_safe_action(state)
            
            # Execute action
            if not self.sim_mode:
                self._send_hardware_command(action)
            
            # Record data
            self.collected_data['states'].append(state)
            self.collected_data['actions'].append(action)
            
            # Update metrics
            self._update_metrics()
            
            # Control loop frequency
            time.sleep(1.0 / self.config.control_frequency)
        
        self.logger.info("Data collection complete")
        return self.collected_data
    
    def _hardware_safety_check(self) -> bool:
        """Perform pre-deployment safety checks"""
        checks = {
            'communication': self._check_communication(),
            'sensors': self._check_sensors(),
            'actuators': self._check_actuators(),
            'emergency_stop': self._check_emergency_stop(),
            'workspace_clear': self._check_workspace()
        }
        
        for check_name, status in checks.items():
            if not status:
                self.logger.error(f"Safety check failed: {check_name}")
                return False
        
        self.logger.info("All safety checks passed")
        return True
    
    def _start_safety_monitor(self):
        """Start background thread for safety monitoring"""
        def monitor():
            while not self.emergency_stop:
                # Check for collisions
                if self._detect_collision():
                    self.logger.error("Collision detected! Emergency stop activated")
                    self.emergency_stop = True
                    self._execute_emergency_stop()
                
                # Check communication health
                if not self._check_communication_health():
                    self.logger.error("Communication failure! Emergency stop activated")
                    self.emergency_stop = True
                    self._execute_emergency_stop()
                
                # Check hardware status
                if self._detect_hardware_failure():
                    self.logger.error("Hardware failure detected!")
                    self.metrics.hardware_failures += 1
                
                time.sleep(0.1)  # 10Hz monitoring
        
        self.safety_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.safety_monitor_thread.start()
    
    def _execute_emergency_stop(self):
        """Execute emergency stop procedure"""
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        
        if not self.sim_mode and self.hardware_interface:
            # Send stop commands to all robots
            self.hardware_interface.emergency_stop_all()
            
            # Log incident
            incident_data = {
                'timestamp': time.time(),
                'metrics': self.metrics,
                'last_states': self.collected_data['states'][-10:],
                'last_actions': self.collected_data['actions'][-10:]
            }
            
            self._save_incident_report(incident_data)
    
    @abstractmethod
    def _get_current_state(self) -> np.ndarray:
        """Get current state from simulation or hardware"""
        pass
    
    @abstractmethod
    def _compute_action(self, state: np.ndarray) -> np.ndarray:
        """Compute action given current state"""
        pass
    
    @abstractmethod
    def _get_safe_action(self, state: np.ndarray) -> np.ndarray:
        """Get safe fallback action"""
        pass
    
    @abstractmethod
    def _send_hardware_command(self, action: np.ndarray):
        """Send command to hardware"""
        pass
    
    def _update_metrics(self):
        """Update real-world performance metrics"""
        # Update based on current execution
        pass
    
    # Abstract safety check methods
    @abstractmethod
    def _check_communication(self) -> bool:
        pass
    
    @abstractmethod
    def _check_sensors(self) -> bool:
        pass
    
    @abstractmethod
    def _check_actuators(self) -> bool:
        pass
    
    @abstractmethod
    def _check_emergency_stop(self) -> bool:
        pass
    
    @abstractmethod
    def _check_workspace(self) -> bool:
        pass
    
    @abstractmethod
    def _detect_collision(self) -> bool:
        pass
    
    @abstractmethod
    def _check_communication_health(self) -> bool:
        pass
    
    @abstractmethod
    def _detect_hardware_failure(self) -> bool:
        pass
    
    def _save_incident_report(self, data: Dict):
        """Save incident report for safety analysis"""
        import json
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"incident_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Incident report saved to {filename}")