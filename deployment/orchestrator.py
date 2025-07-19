"""System Orchestration Module

This module handles system-wide orchestration, service management,
health monitoring, and auto-scaling for the PI-HMARL system.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import psutil
import aiohttp
import subprocess
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service states"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RESTARTING = "restarting"


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_check: Optional[Dict[str, Any]] = None
    restart_policy: str = "on-failure"
    max_restarts: int = 3
    cpu_limit: Optional[float] = None
    memory_limit: Optional[int] = None  # MB
    replicas: int = 1
    
    # Scaling configuration
    scaling: Optional[Dict[str, Any]] = None


@dataclass
class ServiceStatus:
    """Service runtime status"""
    name: str
    state: ServiceState
    health: HealthStatus
    pid: Optional[int] = None
    start_time: Optional[float] = None
    restart_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ServiceManager:
    """Manages individual services"""
    
    def __init__(self):
        """Initialize service manager"""
        self.services: Dict[str, ServiceConfig] = {}
        self.statuses: Dict[str, ServiceStatus] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self._lock = asyncio.Lock()
        
        logger.info("Initialized ServiceManager")
    
    async def register_service(self, config: ServiceConfig):
        """Register a service
        
        Args:
            config: Service configuration
        """
        async with self._lock:
            self.services[config.name] = config
            self.statuses[config.name] = ServiceStatus(
                name=config.name,
                state=ServiceState.STOPPED,
                health=HealthStatus.UNKNOWN
            )
            
            logger.info(f"Registered service: {config.name}")
    
    async def start_service(self, name: str) -> bool:
        """Start a service
        
        Args:
            name: Service name
            
        Returns:
            Success status
        """
        if name not in self.services:
            logger.error(f"Unknown service: {name}")
            return False
        
        async with self._lock:
            status = self.statuses[name]
            
            if status.state == ServiceState.RUNNING:
                logger.warning(f"Service already running: {name}")
                return True
            
            # Check dependencies
            config = self.services[name]
            for dep in config.dependencies:
                if dep not in self.statuses:
                    logger.error(f"Missing dependency {dep} for {name}")
                    return False
                
                if self.statuses[dep].state != ServiceState.RUNNING:
                    logger.error(f"Dependency {dep} not running for {name}")
                    return False
            
            # Start service
            status.state = ServiceState.STARTING
            
            try:
                # Build command
                cmd = [config.command] + config.args
                
                # Create environment
                env = os.environ.copy()
                env.update(config.env)
                
                # Start process
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                self.processes[name] = process
                
                # Update status
                status.state = ServiceState.RUNNING
                status.pid = process.pid
                status.start_time = time.time()
                
                logger.info(f"Started service {name} (PID: {process.pid})")
                
                return True
                
            except Exception as e:
                status.state = ServiceState.FAILED
                status.last_error = str(e)
                logger.error(f"Failed to start {name}: {e}")
                return False
    
    async def stop_service(self, name: str, timeout: float = 10.0) -> bool:
        """Stop a service
        
        Args:
            name: Service name
            timeout: Stop timeout
            
        Returns:
            Success status
        """
        if name not in self.services:
            logger.error(f"Unknown service: {name}")
            return False
        
        async with self._lock:
            status = self.statuses[name]
            
            if status.state == ServiceState.STOPPED:
                return True
            
            status.state = ServiceState.STOPPING
            
            if name in self.processes:
                process = self.processes[name]
                
                try:
                    # Graceful shutdown
                    process.terminate()
                    
                    # Wait for termination
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        if process.poll() is not None:
                            break
                        await asyncio.sleep(0.1)
                    
                    # Force kill if needed
                    if process.poll() is None:
                        process.kill()
                        await asyncio.sleep(0.5)
                    
                    del self.processes[name]
                    
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
                    return False
            
            status.state = ServiceState.STOPPED
            status.pid = None
            
            logger.info(f"Stopped service: {name}")
            return True
    
    async def restart_service(self, name: str) -> bool:
        """Restart a service
        
        Args:
            name: Service name
            
        Returns:
            Success status
        """
        logger.info(f"Restarting service: {name}")
        
        if await self.stop_service(name):
            await asyncio.sleep(1.0)  # Brief pause
            
            async with self._lock:
                self.statuses[name].restart_count += 1
            
            return await self.start_service(name)
        
        return False
    
    async def get_status(self, name: str) -> Optional[ServiceStatus]:
        """Get service status
        
        Args:
            name: Service name
            
        Returns:
            Service status
        """
        return self.statuses.get(name)
    
    async def get_all_statuses(self) -> Dict[str, ServiceStatus]:
        """Get all service statuses
        
        Returns:
            All service statuses
        """
        return self.statuses.copy()


class HealthMonitor:
    """Monitors system and service health"""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health monitor
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: Dict[str, List[Dict[str, Any]]] = {}
        self._running = False
        
        logger.info(f"Initialized HealthMonitor (interval: {check_interval}s)")
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """Register health check
        
        Args:
            name: Check name
            check_func: Check function
            config: Check configuration
        """
        self.health_checks[name] = {
            'func': check_func,
            'config': config or {}
        }
        self.health_history[name] = []
        
        logger.info(f"Registered health check: {name}")
    
    async def start(self):
        """Start health monitoring"""
        self._running = True
        
        logger.info("Starting health monitoring")
        
        while self._running:
            await self._run_health_checks()
            await asyncio.sleep(self.check_interval)
    
    async def stop(self):
        """Stop health monitoring"""
        self._running = False
        logger.info("Stopped health monitoring")
    
    async def _run_health_checks(self):
        """Run all health checks"""
        for name, check in self.health_checks.items():
            try:
                # Run check
                result = await self._run_single_check(name, check)
                
                # Store result
                self.health_history[name].append({
                    'timestamp': time.time(),
                    'status': result['status'],
                    'details': result.get('details', {})
                })
                
                # Keep history limited
                if len(self.health_history[name]) > 100:
                    self.health_history[name] = self.health_history[name][-100:]
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                
                self.health_history[name].append({
                    'timestamp': time.time(),
                    'status': HealthStatus.UNHEALTHY,
                    'error': str(e)
                })
    
    async def _run_single_check(
        self,
        name: str,
        check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run single health check
        
        Args:
            name: Check name
            check: Check specification
            
        Returns:
            Check result
        """
        func = check['func']
        config = check['config']
        
        # Run check function
        if asyncio.iscoroutinefunction(func):
            result = await func(**config)
        else:
            result = func(**config)
        
        return result
    
    def get_health_status(self, name: str) -> HealthStatus:
        """Get current health status
        
        Args:
            name: Check name
            
        Returns:
            Health status
        """
        if name not in self.health_history:
            return HealthStatus.UNKNOWN
        
        history = self.health_history[name]
        if not history:
            return HealthStatus.UNKNOWN
        
        return history[-1].get('status', HealthStatus.UNKNOWN)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary
        
        Returns:
            Health summary
        """
        summary = {
            'overall': HealthStatus.HEALTHY,
            'checks': {},
            'timestamp': time.time()
        }
        
        for name in self.health_checks:
            status = self.get_health_status(name)
            summary['checks'][name] = status.value
            
            # Update overall status
            if status == HealthStatus.UNHEALTHY:
                summary['overall'] = HealthStatus.UNHEALTHY
            elif status == HealthStatus.DEGRADED and summary['overall'] == HealthStatus.HEALTHY:
                summary['overall'] = HealthStatus.DEGRADED
        
        return summary


class AutoScaler:
    """Auto-scaling manager"""
    
    def __init__(
        self,
        service_manager: ServiceManager,
        metrics_provider: Optional[Callable] = None
    ):
        """Initialize auto-scaler
        
        Args:
            service_manager: Service manager
            metrics_provider: Function to get metrics
        """
        self.service_manager = service_manager
        self.metrics_provider = metrics_provider or self._default_metrics
        self.scaling_policies: Dict[str, Dict[str, Any]] = {}
        self._running = False
        
        logger.info("Initialized AutoScaler")
    
    def register_scaling_policy(
        self,
        service_name: str,
        policy: Dict[str, Any]
    ):
        """Register scaling policy
        
        Args:
            service_name: Service name
            policy: Scaling policy
        """
        self.scaling_policies[service_name] = policy
        logger.info(f"Registered scaling policy for {service_name}")
    
    async def start(self, check_interval: float = 60.0):
        """Start auto-scaling
        
        Args:
            check_interval: Scaling check interval
        """
        self._running = True
        
        logger.info("Starting auto-scaler")
        
        while self._running:
            await self._check_scaling()
            await asyncio.sleep(check_interval)
    
    async def stop(self):
        """Stop auto-scaling"""
        self._running = False
        logger.info("Stopped auto-scaler")
    
    async def _check_scaling(self):
        """Check and apply scaling decisions"""
        for service_name, policy in self.scaling_policies.items():
            try:
                # Get current metrics
                metrics = await self.metrics_provider(service_name)
                
                # Make scaling decision
                decision = self._make_scaling_decision(
                    service_name,
                    policy,
                    metrics
                )
                
                # Apply decision
                if decision != 0:
                    await self._apply_scaling(service_name, decision)
                    
            except Exception as e:
                logger.error(f"Scaling check failed for {service_name}: {e}")
    
    def _make_scaling_decision(
        self,
        service_name: str,
        policy: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> int:
        """Make scaling decision
        
        Args:
            service_name: Service name
            policy: Scaling policy
            metrics: Current metrics
            
        Returns:
            Scaling change (+/- replicas)
        """
        # Get current replicas
        config = self.service_manager.services.get(service_name)
        if not config:
            return 0
        
        current_replicas = config.replicas
        
        # Check metrics against thresholds
        scale_up = False
        scale_down = False
        
        # CPU-based scaling
        if 'cpu_threshold' in policy:
            cpu_usage = metrics.get('cpu_usage', 0)
            
            if cpu_usage > policy['cpu_threshold']['scale_up']:
                scale_up = True
            elif cpu_usage < policy['cpu_threshold']['scale_down']:
                scale_down = True
        
        # Memory-based scaling
        if 'memory_threshold' in policy:
            memory_usage = metrics.get('memory_usage', 0)
            
            if memory_usage > policy['memory_threshold']['scale_up']:
                scale_up = True
            elif memory_usage < policy['memory_threshold']['scale_down']:
                scale_down = True
        
        # Queue-based scaling
        if 'queue_threshold' in policy:
            queue_length = metrics.get('queue_length', 0)
            
            if queue_length > policy['queue_threshold']['scale_up']:
                scale_up = True
            elif queue_length < policy['queue_threshold']['scale_down']:
                scale_down = True
        
        # Apply limits
        min_replicas = policy.get('min_replicas', 1)
        max_replicas = policy.get('max_replicas', 10)
        
        if scale_up and current_replicas < max_replicas:
            return 1
        elif scale_down and current_replicas > min_replicas:
            return -1
        
        return 0
    
    async def _apply_scaling(self, service_name: str, change: int):
        """Apply scaling change
        
        Args:
            service_name: Service name
            change: Replica change
        """
        config = self.service_manager.services.get(service_name)
        if not config:
            return
        
        new_replicas = config.replicas + change
        
        logger.info(
            f"Scaling {service_name}: {config.replicas} -> {new_replicas}"
        )
        
        config.replicas = new_replicas
        
        # TODO: Implement actual replica management
        # This would involve starting/stopping service instances
    
    async def _default_metrics(self, service_name: str) -> Dict[str, Any]:
        """Default metrics provider
        
        Args:
            service_name: Service name
            
        Returns:
            Service metrics
        """
        # Get process metrics
        status = await self.service_manager.get_status(service_name)
        
        if status and status.pid:
            try:
                process = psutil.Process(status.pid)
                
                return {
                    'cpu_usage': process.cpu_percent(interval=1.0),
                    'memory_usage': process.memory_percent(),
                    'num_threads': process.num_threads()
                }
            except:
                pass
        
        return {}


class SystemOrchestrator:
    """Main system orchestrator"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize system orchestrator
        
        Args:
            config_file: Configuration file path
        """
        self.config_file = config_file
        self.service_manager = ServiceManager()
        self.health_monitor = HealthMonitor()
        self.auto_scaler = AutoScaler(self.service_manager)
        
        # Load configuration
        if config_file and config_file.exists():
            self._load_configuration()
        
        logger.info("Initialized SystemOrchestrator")
    
    def _load_configuration(self):
        """Load orchestration configuration"""
        if not self.config_file:
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Register services
            for service_config in config.get('services', []):
                asyncio.create_task(
                    self.service_manager.register_service(
                        ServiceConfig(**service_config)
                    )
                )
            
            # Register health checks
            for check in config.get('health_checks', []):
                # Create check function based on type
                check_func = self._create_health_check(check)
                self.health_monitor.register_health_check(
                    check['name'],
                    check_func,
                    check.get('config')
                )
            
            # Register scaling policies
            for policy in config.get('scaling_policies', []):
                self.auto_scaler.register_scaling_policy(
                    policy['service'],
                    policy['policy']
                )
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _create_health_check(self, check_spec: Dict[str, Any]) -> Callable:
        """Create health check function
        
        Args:
            check_spec: Check specification
            
        Returns:
            Health check function
        """
        check_type = check_spec['type']
        
        if check_type == 'http':
            async def http_check(url: str, timeout: float = 5.0) -> Dict[str, Any]:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=timeout) as response:
                            if response.status == 200:
                                return {'status': HealthStatus.HEALTHY}
                            else:
                                return {
                                    'status': HealthStatus.UNHEALTHY,
                                    'details': {'status_code': response.status}
                                }
                except Exception as e:
                    return {
                        'status': HealthStatus.UNHEALTHY,
                        'details': {'error': str(e)}
                    }
            
            return http_check
        
        elif check_type == 'process':
            def process_check(pid: int) -> Dict[str, Any]:
                try:
                    process = psutil.Process(pid)
                    if process.is_running():
                        cpu = process.cpu_percent(interval=0.1)
                        memory = process.memory_percent()
                        
                        # Determine health based on resource usage
                        if cpu > 90 or memory > 90:
                            status = HealthStatus.DEGRADED
                        else:
                            status = HealthStatus.HEALTHY
                        
                        return {
                            'status': status,
                            'details': {
                                'cpu_percent': cpu,
                                'memory_percent': memory
                            }
                        }
                    else:
                        return {'status': HealthStatus.UNHEALTHY}
                except:
                    return {'status': HealthStatus.UNHEALTHY}
            
            return process_check
        
        else:
            # Default check
            def default_check() -> Dict[str, Any]:
                return {'status': HealthStatus.UNKNOWN}
            
            return default_check
    
    async def start(self):
        """Start orchestration system"""
        logger.info("Starting system orchestration")
        
        # Start monitoring
        asyncio.create_task(self.health_monitor.start())
        
        # Start auto-scaling
        asyncio.create_task(self.auto_scaler.start())
        
        # Start services in dependency order
        await self._start_services()
    
    async def stop(self):
        """Stop orchestration system"""
        logger.info("Stopping system orchestration")
        
        # Stop auto-scaling
        await self.auto_scaler.stop()
        
        # Stop monitoring
        await self.health_monitor.stop()
        
        # Stop all services
        await self._stop_all_services()
    
    async def _start_services(self):
        """Start services in dependency order"""
        # Get all services
        services = list(self.service_manager.services.keys())
        
        # Start services
        started: Set[str] = set()
        
        while len(started) < len(services):
            made_progress = False
            
            for service_name in services:
                if service_name in started:
                    continue
                
                # Check if dependencies are met
                config = self.service_manager.services[service_name]
                deps_met = all(dep in started for dep in config.dependencies)
                
                if deps_met:
                    success = await self.service_manager.start_service(service_name)
                    if success:
                        started.add(service_name)
                        made_progress = True
            
            if not made_progress:
                logger.error("Circular dependency detected or failed to start services")
                break
    
    async def _stop_all_services(self):
        """Stop all services in reverse dependency order"""
        # Get all running services
        statuses = await self.service_manager.get_all_statuses()
        running = [
            name for name, status in statuses.items()
            if status.state == ServiceState.RUNNING
        ]
        
        # Stop in reverse order
        stopped: Set[str] = set()
        
        while len(stopped) < len(running):
            made_progress = False
            
            for service_name in running:
                if service_name in stopped:
                    continue
                
                # Check if dependents are stopped
                dependents_stopped = True
                for other_name, other_config in self.service_manager.services.items():
                    if service_name in other_config.dependencies:
                        if other_name not in stopped and other_name in running:
                            dependents_stopped = False
                            break
                
                if dependents_stopped:
                    await self.service_manager.stop_service(service_name)
                    stopped.add(service_name)
                    made_progress = True
            
            if not made_progress:
                # Force stop remaining
                for service_name in running:
                    if service_name not in stopped:
                        await self.service_manager.stop_service(service_name)
                        stopped.add(service_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status
        
        Returns:
            System status
        """
        return {
            'services': {
                name: {
                    'state': status.state.value,
                    'health': status.health.value,
                    'uptime': time.time() - status.start_time
                              if status.start_time else 0,
                    'restart_count': status.restart_count
                }
                for name, status in self.service_manager.statuses.items()
            },
            'health': self.health_monitor.get_health_summary(),
            'timestamp': time.time()
        }


# Example orchestration configuration
EXAMPLE_CONFIG = {
    "services": [
        {
            "name": "coordinator",
            "command": "python",
            "args": ["-m", "src.coordinator"],
            "env": {"LOG_LEVEL": "INFO"},
            "dependencies": [],
            "health_check": {
                "type": "http",
                "url": "http://localhost:8080/health"
            },
            "restart_policy": "always",
            "max_restarts": 5
        },
        {
            "name": "agent_manager",
            "command": "python",
            "args": ["-m", "src.agent_manager"],
            "dependencies": ["coordinator"],
            "scaling": {
                "min_replicas": 1,
                "max_replicas": 5
            }
        },
        {
            "name": "dashboard",
            "command": "python",
            "args": ["-m", "src.visualization.dashboard"],
            "dependencies": ["coordinator", "agent_manager"],
            "health_check": {
                "type": "http",
                "url": "http://localhost:5000/health"
            }
        }
    ],
    "health_checks": [
        {
            "name": "system_resources",
            "type": "custom",
            "interval": 30
        }
    ],
    "scaling_policies": [
        {
            "service": "agent_manager",
            "policy": {
                "min_replicas": 1,
                "max_replicas": 5,
                "cpu_threshold": {
                    "scale_up": 70,
                    "scale_down": 30
                },
                "memory_threshold": {
                    "scale_up": 80,
                    "scale_down": 40
                }
            }
        }
    ]
}