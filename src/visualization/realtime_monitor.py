"""Real-Time System Monitoring

This module implements real-time monitoring capabilities for agents,
tasks, and system health.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import queue
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = "idle"
    ACTIVE = "active"
    CHARGING = "charging"
    RETURNING = "returning"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentState:
    """Complete agent state"""
    agent_id: int
    position: torch.Tensor
    velocity: torch.Tensor
    orientation: torch.Tensor
    energy: float  # SOC percentage
    temperature: float
    status: AgentStatus
    current_task: Optional[str]
    health_metrics: Dict[str, float]
    last_update: float
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class TaskState:
    """Complete task state"""
    task_id: str
    task_type: str
    status: str
    progress: float
    assigned_agents: List[int]
    start_time: float
    estimated_completion: float
    actual_completion: Optional[float]
    energy_consumed: float
    subtasks: List[str]
    errors: List[str]


@dataclass
class SystemAlert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    category: str
    message: str
    source: str
    timestamp: float
    data: Dict[str, Any]
    acknowledged: bool = False


class RealtimeMonitor:
    """Main real-time monitoring system"""
    
    def __init__(
        self,
        update_rate: float = 10.0,  # Hz
        alert_threshold: Dict[str, float] = None
    ):
        """Initialize real-time monitor
        
        Args:
            update_rate: Update frequency in Hz
            alert_threshold: Alert thresholds
        """
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        
        # Default thresholds
        self.alert_thresholds = alert_threshold or {
            'low_energy': 0.2,  # 20% SOC
            'high_temperature': 55.0,  # Celsius
            'communication_timeout': 5.0,  # seconds
            'task_delay': 300.0,  # seconds
            'position_error': 10.0  # meters
        }
        
        # Components
        self.agent_tracker = AgentTracker()
        self.task_tracker = TaskTracker()
        self.health_monitor = SystemHealthMonitor()
        self.alert_manager = AlertManager()
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.update_queue = queue.Queue()
        
        # Subscribers
        self.subscribers = defaultdict(list)
        
        logger.info(f"RealtimeMonitor initialized at {update_rate} Hz")
    
    def start(self):
        """Start monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("Real-time monitoring stopped")
    
    def update_agent(self, agent_id: int, state_data: Dict[str, Any]):
        """Update agent state
        
        Args:
            agent_id: Agent ID
            state_data: State data
        """
        self.update_queue.put(('agent', agent_id, state_data))
    
    def update_task(self, task_id: str, task_data: Dict[str, Any]):
        """Update task state
        
        Args:
            task_id: Task ID
            task_data: Task data
        """
        self.update_queue.put(('task', task_id, task_data))
    
    def subscribe(self, event_type: str, callback: callable):
        """Subscribe to monitoring events
        
        Args:
            event_type: Event type to subscribe to
            callback: Callback function
        """
        self.subscribers[event_type].append(callback)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        last_update = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Process updates
            self._process_updates()
            
            # Run monitors
            if current_time - last_update >= self.update_interval:
                self._run_monitors()
                last_update = current_time
            
            # Small sleep to prevent busy waiting
            time.sleep(0.001)
    
    def _process_updates(self):
        """Process queued updates"""
        try:
            while not self.update_queue.empty():
                update_type, entity_id, data = self.update_queue.get_nowait()
                
                if update_type == 'agent':
                    self.agent_tracker.update_agent(entity_id, data)
                elif update_type == 'task':
                    self.task_tracker.update_task(entity_id, data)
                
        except queue.Empty:
            pass
    
    def _run_monitors(self):
        """Run all monitoring checks"""
        current_time = time.time()
        
        # Check agent health
        agent_alerts = self.agent_tracker.check_health(
            self.alert_thresholds, current_time
        )
        
        # Check task progress
        task_alerts = self.task_tracker.check_progress(
            self.alert_thresholds, current_time
        )
        
        # System health check
        system_alerts = self.health_monitor.check_system(
            self.agent_tracker.agents,
            self.task_tracker.tasks
        )
        
        # Process alerts
        all_alerts = agent_alerts + task_alerts + system_alerts
        for alert in all_alerts:
            self.alert_manager.add_alert(alert)
            self._notify_subscribers('alert', alert)
        
        # Notify subscribers of updates
        self._notify_subscribers('update', {
            'agents': self.agent_tracker.get_summary(),
            'tasks': self.task_tracker.get_summary(),
            'health': self.health_monitor.get_status(),
            'timestamp': current_time
        })
    
    def _notify_subscribers(self, event_type: str, data: Any):
        """Notify subscribers of event
        
        Args:
            event_type: Event type
            data: Event data
        """
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current monitoring state
        
        Returns:
            Current state
        """
        return {
            'agents': self.agent_tracker.get_all_states(),
            'tasks': self.task_tracker.get_all_states(),
            'health': self.health_monitor.get_status(),
            'alerts': self.alert_manager.get_active_alerts(),
            'timestamp': time.time()
        }


class AgentTracker:
    """Tracks agent states and health"""
    
    def __init__(self):
        """Initialize agent tracker"""
        self.agents: Dict[int, AgentState] = {}
        self.communication_times: Dict[int, float] = {}
        self.performance_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
    
    def update_agent(self, agent_id: int, data: Dict[str, Any]):
        """Update agent state
        
        Args:
            agent_id: Agent ID
            data: Agent data
        """
        current_time = time.time()
        
        # Create or update agent state
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                position=torch.zeros(3),
                velocity=torch.zeros(3),
                orientation=torch.zeros(3),
                energy=1.0,
                temperature=25.0,
                status=AgentStatus.IDLE,
                current_task=None,
                health_metrics={},
                last_update=current_time
            )
        
        agent = self.agents[agent_id]
        
        # Update fields
        if 'position' in data:
            agent.position = torch.tensor(data['position'])
            agent.trajectory.append(agent.position.clone())
        
        if 'velocity' in data:
            agent.velocity = torch.tensor(data['velocity'])
        
        if 'orientation' in data:
            agent.orientation = torch.tensor(data['orientation'])
        
        if 'soc' in data:
            agent.energy = data['soc']
        
        if 'temperature' in data:
            agent.temperature = data['temperature']
        
        if 'status' in data:
            agent.status = AgentStatus(data['status'])
        
        if 'current_task' in data:
            agent.current_task = data['current_task']
        
        if 'health_metrics' in data:
            agent.health_metrics.update(data['health_metrics'])
        
        # Update timestamps
        agent.last_update = current_time
        self.communication_times[agent_id] = current_time
        
        # Track performance
        self.performance_history[agent_id].append({
            'energy': agent.energy,
            'temperature': agent.temperature,
            'speed': torch.norm(agent.velocity).item(),
            'timestamp': current_time
        })
    
    def check_health(
        self,
        thresholds: Dict[str, float],
        current_time: float
    ) -> List[SystemAlert]:
        """Check agent health
        
        Args:
            thresholds: Alert thresholds
            current_time: Current time
            
        Returns:
            List of alerts
        """
        alerts = []
        
        for agent_id, agent in self.agents.items():
            # Communication timeout
            if current_time - agent.last_update > thresholds['communication_timeout']:
                alerts.append(SystemAlert(
                    alert_id=f"comm_timeout_{agent_id}_{int(current_time)}",
                    level=AlertLevel.WARNING,
                    category="communication",
                    message=f"Agent {agent_id} communication timeout",
                    source=f"agent_{agent_id}",
                    timestamp=current_time,
                    data={'agent_id': agent_id, 'last_update': agent.last_update}
                ))
            
            # Low energy
            if agent.energy < thresholds['low_energy']:
                alerts.append(SystemAlert(
                    alert_id=f"low_energy_{agent_id}_{int(current_time)}",
                    level=AlertLevel.WARNING if agent.energy > 0.1 else AlertLevel.CRITICAL,
                    category="energy",
                    message=f"Agent {agent_id} low energy: {agent.energy:.1%}",
                    source=f"agent_{agent_id}",
                    timestamp=current_time,
                    data={'agent_id': agent_id, 'energy': agent.energy}
                ))
            
            # High temperature
            if agent.temperature > thresholds['high_temperature']:
                alerts.append(SystemAlert(
                    alert_id=f"high_temp_{agent_id}_{int(current_time)}",
                    level=AlertLevel.WARNING,
                    category="thermal",
                    message=f"Agent {agent_id} high temperature: {agent.temperature:.1f}°C",
                    source=f"agent_{agent_id}",
                    timestamp=current_time,
                    data={'agent_id': agent_id, 'temperature': agent.temperature}
                ))
            
            # Position deviation (if assigned to task)
            if agent.current_task and len(agent.trajectory) > 10:
                # Simple deviation check
                recent_positions = list(agent.trajectory)[-10:]
                center = torch.stack(recent_positions).mean(dim=0)
                max_dev = max(
                    torch.norm(pos - center).item()
                    for pos in recent_positions
                )
                
                if max_dev > thresholds['position_error']:
                    alerts.append(SystemAlert(
                        alert_id=f"pos_error_{agent_id}_{int(current_time)}",
                        level=AlertLevel.INFO,
                        category="navigation",
                        message=f"Agent {agent_id} position deviation: {max_dev:.1f}m",
                        source=f"agent_{agent_id}",
                        timestamp=current_time,
                        data={'agent_id': agent_id, 'deviation': max_dev}
                    ))
        
        return alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get agent summary
        
        Returns:
            Summary statistics
        """
        if not self.agents:
            return {
                'total_agents': 0,
                'active_agents': 0,
                'avg_energy': 0,
                'avg_temperature': 0
            }
        
        active_count = sum(
            1 for a in self.agents.values()
            if a.status == AgentStatus.ACTIVE
        )
        
        avg_energy = np.mean([a.energy for a in self.agents.values()])
        avg_temp = np.mean([a.temperature for a in self.agents.values()])
        
        return {
            'total_agents': len(self.agents),
            'active_agents': active_count,
            'avg_energy': avg_energy,
            'avg_temperature': avg_temp,
            'status_distribution': self._get_status_distribution()
        }
    
    def _get_status_distribution(self) -> Dict[str, int]:
        """Get agent status distribution
        
        Returns:
            Status counts
        """
        distribution = defaultdict(int)
        for agent in self.agents.values():
            distribution[agent.status.value] += 1
        return dict(distribution)
    
    def get_all_states(self) -> Dict[int, Dict[str, Any]]:
        """Get all agent states
        
        Returns:
            Agent states
        """
        return {
            agent_id: {
                'position': agent.position.tolist(),
                'velocity': agent.velocity.tolist(),
                'energy': agent.energy,
                'temperature': agent.temperature,
                'status': agent.status.value,
                'current_task': agent.current_task,
                'health_metrics': agent.health_metrics,
                'last_update': agent.last_update
            }
            for agent_id, agent in self.agents.items()
        }


class TaskTracker:
    """Tracks task execution and progress"""
    
    def __init__(self):
        """Initialize task tracker"""
        self.tasks: Dict[str, TaskState] = {}
        self.completion_history: deque = deque(maxlen=100)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def update_task(self, task_id: str, data: Dict[str, Any]):
        """Update task state
        
        Args:
            task_id: Task ID
            data: Task data
        """
        current_time = time.time()
        
        # Create or update task state
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskState(
                task_id=task_id,
                task_type=data.get('type', 'unknown'),
                status='pending',
                progress=0.0,
                assigned_agents=[],
                start_time=current_time,
                estimated_completion=current_time + 300,
                actual_completion=None,
                energy_consumed=0.0,
                subtasks=[],
                errors=[]
            )
        
        task = self.tasks[task_id]
        
        # Update fields
        for field in ['task_type', 'status', 'progress', 'energy_consumed']:
            if field in data:
                setattr(task, field, data[field])
        
        if 'assigned_agents' in data:
            task.assigned_agents = data['assigned_agents']
        
        if 'subtasks' in data:
            task.subtasks = data['subtasks']
        
        if 'errors' in data:
            task.errors.extend(data['errors'])
        
        if 'estimated_completion' in data:
            task.estimated_completion = data['estimated_completion']
        
        # Check completion
        if task.status == 'completed' and task.actual_completion is None:
            task.actual_completion = current_time
            self._record_completion(task)
    
    def check_progress(
        self,
        thresholds: Dict[str, float],
        current_time: float
    ) -> List[SystemAlert]:
        """Check task progress
        
        Args:
            thresholds: Alert thresholds
            current_time: Current time
            
        Returns:
            List of alerts
        """
        alerts = []
        
        for task_id, task in self.tasks.items():
            if task.status not in ['active', 'in_progress']:
                continue
            
            # Task delay
            if current_time > task.estimated_completion + thresholds['task_delay']:
                alerts.append(SystemAlert(
                    alert_id=f"task_delay_{task_id}_{int(current_time)}",
                    level=AlertLevel.WARNING,
                    category="task",
                    message=f"Task {task_id} delayed by {current_time - task.estimated_completion:.0f}s",
                    source=f"task_{task_id}",
                    timestamp=current_time,
                    data={
                        'task_id': task_id,
                        'delay': current_time - task.estimated_completion
                    }
                ))
            
            # Stalled progress
            if task.progress > 0 and task.progress < 100:
                # Check if progress hasn't changed (simplified)
                if len(self.performance_metrics.get(task_id, [])) > 5:
                    recent_progress = self.performance_metrics[task_id][-5:]
                    if all(p == recent_progress[0] for p in recent_progress):
                        alerts.append(SystemAlert(
                            alert_id=f"task_stalled_{task_id}_{int(current_time)}",
                            level=AlertLevel.INFO,
                            category="task",
                            message=f"Task {task_id} progress stalled at {task.progress:.1f}%",
                            source=f"task_{task_id}",
                            timestamp=current_time,
                            data={'task_id': task_id, 'progress': task.progress}
                        ))
            
            # High energy consumption
            expected_energy = (task.progress / 100.0) * 20.0  # Expected 20 Wh total
            if task.energy_consumed > expected_energy * 1.5:
                alerts.append(SystemAlert(
                    alert_id=f"high_energy_{task_id}_{int(current_time)}",
                    level=AlertLevel.INFO,
                    category="energy",
                    message=f"Task {task_id} high energy consumption: {task.energy_consumed:.1f} Wh",
                    source=f"task_{task_id}",
                    timestamp=current_time,
                    data={
                        'task_id': task_id,
                        'energy_consumed': task.energy_consumed,
                        'expected': expected_energy
                    }
                ))
            
            # Errors
            if task.errors:
                alerts.append(SystemAlert(
                    alert_id=f"task_errors_{task_id}_{int(current_time)}",
                    level=AlertLevel.WARNING,
                    category="task",
                    message=f"Task {task_id} has {len(task.errors)} errors",
                    source=f"task_{task_id}",
                    timestamp=current_time,
                    data={'task_id': task_id, 'errors': task.errors}
                ))
            
            # Track progress
            self.performance_metrics[task_id].append(task.progress)
        
        return alerts
    
    def _record_completion(self, task: TaskState):
        """Record task completion
        
        Args:
            task: Completed task
        """
        duration = task.actual_completion - task.start_time
        delay = task.actual_completion - task.estimated_completion
        
        completion_data = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'duration': duration,
            'delay': delay,
            'energy_consumed': task.energy_consumed,
            'num_agents': len(task.assigned_agents),
            'had_errors': len(task.errors) > 0,
            'timestamp': task.actual_completion
        }
        
        self.completion_history.append(completion_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get task summary
        
        Returns:
            Summary statistics
        """
        if not self.tasks:
            return {
                'total_tasks': 0,
                'active_tasks': 0,
                'completion_rate': 0,
                'avg_progress': 0
            }
        
        status_counts = defaultdict(int)
        progress_values = []
        
        for task in self.tasks.values():
            status_counts[task.status] += 1
            if task.status in ['active', 'in_progress']:
                progress_values.append(task.progress)
        
        completion_rate = (
            status_counts.get('completed', 0) / len(self.tasks)
            if self.tasks else 0
        )
        
        return {
            'total_tasks': len(self.tasks),
            'active_tasks': status_counts.get('active', 0) + status_counts.get('in_progress', 0),
            'completion_rate': completion_rate,
            'avg_progress': np.mean(progress_values) if progress_values else 0,
            'status_distribution': dict(status_counts)
        }
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all task states
        
        Returns:
            Task states
        """
        return {
            task_id: {
                'task_type': task.task_type,
                'status': task.status,
                'progress': task.progress,
                'assigned_agents': task.assigned_agents,
                'energy_consumed': task.energy_consumed,
                'start_time': task.start_time,
                'estimated_completion': task.estimated_completion,
                'errors': len(task.errors)
            }
            for task_id, task in self.tasks.items()
        }


class SystemHealthMonitor:
    """Monitors overall system health"""
    
    def __init__(self):
        """Initialize system health monitor"""
        self.health_metrics = {
            'system_uptime': 0,
            'total_tasks_completed': 0,
            'total_energy_consumed': 0,
            'average_efficiency': 0,
            'error_rate': 0,
            'communication_health': 1.0
        }
        
        self.start_time = time.time()
        self.error_counts = deque(maxlen=1000)
    
    def check_system(
        self,
        agents: Dict[int, AgentState],
        tasks: Dict[str, TaskState]
    ) -> List[SystemAlert]:
        """Check system health
        
        Args:
            agents: Agent states
            tasks: Task states
            
        Returns:
            System alerts
        """
        alerts = []
        current_time = time.time()
        
        # Update metrics
        self.health_metrics['system_uptime'] = current_time - self.start_time
        
        # Communication health
        if agents:
            recent_comms = sum(
                1 for a in agents.values()
                if current_time - a.last_update < 5.0
            )
            self.health_metrics['communication_health'] = recent_comms / len(agents)
            
            if self.health_metrics['communication_health'] < 0.8:
                alerts.append(SystemAlert(
                    alert_id=f"comm_degraded_{int(current_time)}",
                    level=AlertLevel.WARNING,
                    category="system",
                    message="System communication degraded",
                    source="system",
                    timestamp=current_time,
                    data={'health': self.health_metrics['communication_health']}
                ))
        
        # Task completion rate
        if tasks:
            completed = sum(1 for t in tasks.values() if t.status == 'completed')
            self.health_metrics['total_tasks_completed'] = completed
            
            # Calculate efficiency
            total_energy = sum(t.energy_consumed for t in tasks.values())
            self.health_metrics['total_energy_consumed'] = total_energy
            
            if completed > 0:
                self.health_metrics['average_efficiency'] = completed / (total_energy + 1)
        
        # Error rate
        recent_errors = sum(
            len(t.errors) for t in tasks.values()
            if t.start_time > current_time - 300  # Last 5 minutes
        )
        self.error_counts.append(recent_errors)
        
        if len(self.error_counts) > 10:
            self.health_metrics['error_rate'] = np.mean(list(self.error_counts)[-10:])
            
            if self.health_metrics['error_rate'] > 5:
                alerts.append(SystemAlert(
                    alert_id=f"high_error_rate_{int(current_time)}",
                    level=AlertLevel.WARNING,
                    category="system",
                    message=f"High system error rate: {self.health_metrics['error_rate']:.1f} errors/period",
                    source="system",
                    timestamp=current_time,
                    data={'error_rate': self.health_metrics['error_rate']}
                ))
        
        # Resource constraints
        if agents:
            low_energy_agents = sum(1 for a in agents.values() if a.energy < 0.3)
            if low_energy_agents > len(agents) * 0.5:
                alerts.append(SystemAlert(
                    alert_id=f"low_fleet_energy_{int(current_time)}",
                    level=AlertLevel.WARNING,
                    category="energy",
                    message="Over 50% of fleet has low energy",
                    source="system",
                    timestamp=current_time,
                    data={'low_energy_count': low_energy_agents}
                ))
        
        return alerts
    
    def get_status(self) -> Dict[str, Any]:
        """Get system health status
        
        Returns:
            Health status
        """
        return {
            'metrics': self.health_metrics.copy(),
            'status': self._determine_status(),
            'uptime_hours': self.health_metrics['system_uptime'] / 3600,
            'timestamp': time.time()
        }
    
    def _determine_status(self) -> str:
        """Determine overall system status
        
        Returns:
            Status string
        """
        if self.health_metrics['communication_health'] < 0.5:
            return 'critical'
        elif self.health_metrics['error_rate'] > 10:
            return 'degraded'
        elif self.health_metrics['communication_health'] < 0.8:
            return 'warning'
        else:
            return 'healthy'


class AlertManager:
    """Manages system alerts"""
    
    def __init__(self, max_alerts: int = 1000):
        """Initialize alert manager
        
        Args:
            max_alerts: Maximum alerts to store
        """
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_counts: Dict[str, int] = defaultdict(int)
        
        # Alert suppression
        self.suppression_rules: Dict[str, float] = {
            'comm_timeout': 60.0,  # Suppress repeated comm timeouts for 60s
            'low_energy': 120.0,   # Suppress low energy alerts for 2 min
            'task_stalled': 300.0  # Suppress stalled task alerts for 5 min
        }
        self.last_alert_times: Dict[str, float] = {}
    
    def add_alert(self, alert: SystemAlert):
        """Add new alert
        
        Args:
            alert: Alert to add
        """
        # Check suppression
        alert_type = alert.alert_id.split('_')[0]
        if alert_type in self.suppression_rules:
            last_time = self.last_alert_times.get(alert_type, 0)
            if alert.timestamp - last_time < self.suppression_rules[alert_type]:
                return  # Suppress alert
            self.last_alert_times[alert_type] = alert.timestamp
        
        # Add alert
        self.alerts.append(alert)
        self.active_alerts[alert.alert_id] = alert
        self.alert_counts[alert.category] += 1
        
        # Auto-acknowledge old alerts
        self._cleanup_old_alerts(alert.timestamp)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert
        
        Args:
            alert_id: Alert ID
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            del self.active_alerts[alert_id]
    
    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        category: Optional[str] = None
    ) -> List[SystemAlert]:
        """Get active alerts
        
        Args:
            level: Filter by level
            category: Filter by category
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary
        
        Returns:
            Alert summary
        """
        level_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            level_counts[alert.level.value] += 1
        
        return {
            'total_active': len(self.active_alerts),
            'level_distribution': dict(level_counts),
            'category_counts': dict(self.alert_counts),
            'oldest_active': min(
                (a.timestamp for a in self.active_alerts.values()),
                default=None
            )
        }
    
    def _cleanup_old_alerts(self, current_time: float):
        """Clean up old alerts
        
        Args:
            current_time: Current time
        """
        # Auto-acknowledge alerts older than 1 hour
        old_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if current_time - alert.timestamp > 3600
        ]
        
        for alert_id in old_alerts:
            self.acknowledge_alert(alert_id)