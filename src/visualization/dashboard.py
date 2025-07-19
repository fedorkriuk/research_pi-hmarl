"""Main Dashboard Implementation

This module implements the main dashboard interface for real-time
monitoring and control of the multi-agent system.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import asyncio
import json
import time
import logging
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from collections import deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    host: str = "127.0.0.1"
    port: int = 5000
    update_interval: float = 0.1  # seconds
    history_length: int = 1000
    map_bounds: Tuple[float, float, float, float] = (-1000, -1000, 1000, 1000)
    enable_3d: bool = True
    theme: str = "dark"
    auth_required: bool = False


class Dashboard:
    """Main dashboard for multi-agent system monitoring"""
    
    def __init__(
        self,
        config: DashboardConfig,
        system_interface: Any = None
    ):
        """Initialize dashboard
        
        Args:
            config: Dashboard configuration
            system_interface: Interface to multi-agent system
        """
        self.config = config
        self.system_interface = system_interface
        
        # Flask setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'pi-hmarl-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Components
        self.metrics_collector = MetricsCollector(
            history_length=config.history_length
        )
        self.event_logger = EventLogger()
        
        # Data storage
        self.agent_data = {}
        self.task_data = {}
        self.system_metrics = {}
        self.alerts = deque(maxlen=100)
        
        # Update thread
        self.running = False
        self.update_thread = None
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
        logger.info(f"Dashboard initialized on {config.host}:{config.port}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html', config=self.config)
        
        @self.app.route('/api/agents')
        def get_agents():
            """Get agent data"""
            return jsonify(self.agent_data)
        
        @self.app.route('/api/tasks')
        def get_tasks():
            """Get task data"""
            return jsonify(self.task_data)
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get system metrics"""
            return jsonify(self.system_metrics)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get recent alerts"""
            return jsonify(list(self.alerts))
        
        @self.app.route('/api/control/<action>', methods=['POST'])
        def control_action(action):
            """Handle control actions"""
            if not self.system_interface:
                return jsonify({'error': 'No system interface'}), 404
            
            data = request.json
            result = self._handle_control_action(action, data)
            return jsonify(result)
    
    def _setup_socketio(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'data': 'Connected to dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_update_request(data):
            """Handle update request"""
            update_type = data.get('type', 'all')
            update_data = self._get_update_data(update_type)
            emit('update_response', update_data)
    
    def start(self):
        """Start dashboard server"""
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start Flask-SocketIO server
        self.socketio.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            debug=False
        )
    
    def stop(self):
        """Stop dashboard server"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                # Collect data from system
                if self.system_interface:
                    self._collect_system_data()
                
                # Update metrics
                self.metrics_collector.update(
                    self.agent_data,
                    self.task_data,
                    self.system_metrics
                )
                
                # Broadcast updates
                self._broadcast_updates()
                
                # Sleep
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Update loop error: {e}")
    
    def _collect_system_data(self):
        """Collect data from system interface"""
        if not self.system_interface:
            return
        
        # Get agent states
        agent_states = self.system_interface.get_agent_states()
        for agent_id, state in agent_states.items():
            self.agent_data[agent_id] = {
                'id': agent_id,
                'position': state.get('position', [0, 0, 0]).tolist(),
                'velocity': state.get('velocity', [0, 0, 0]).tolist(),
                'energy': state.get('soc', 0.5) * 100,
                'status': state.get('status', 'idle'),
                'task': state.get('current_task', None),
                'health': state.get('health', 1.0)
            }
        
        # Get task states
        task_states = self.system_interface.get_task_states()
        for task_id, state in task_states.items():
            self.task_data[task_id] = {
                'id': task_id,
                'type': state.get('type', 'unknown'),
                'status': state.get('status', 'pending'),
                'progress': state.get('progress', 0),
                'assigned_agents': state.get('assigned_agents', []),
                'priority': state.get('priority', 0.5),
                'deadline': state.get('deadline', None)
            }
        
        # Get system metrics
        self.system_metrics = {
            'total_agents': len(agent_states),
            'active_agents': sum(
                1 for a in agent_states.values()
                if a.get('status') == 'active'
            ),
            'total_tasks': len(task_states),
            'completed_tasks': sum(
                1 for t in task_states.values()
                if t.get('status') == 'completed'
            ),
            'system_efficiency': self._calculate_efficiency(),
            'total_energy': sum(
                a.get('soc', 0.5) * 18.5
                for a in agent_states.values()
            ),
            'timestamp': time.time()
        }
    
    def _broadcast_updates(self):
        """Broadcast updates to all connected clients"""
        update_data = {
            'agents': self.agent_data,
            'tasks': self.task_data,
            'metrics': self.system_metrics,
            'timestamp': time.time()
        }
        
        self.socketio.emit('system_update', update_data, broadcast=True)
    
    def _get_update_data(self, update_type: str) -> Dict[str, Any]:
        """Get update data by type
        
        Args:
            update_type: Type of update requested
            
        Returns:
            Update data
        """
        if update_type == 'agents':
            return {'agents': self.agent_data}
        elif update_type == 'tasks':
            return {'tasks': self.task_data}
        elif update_type == 'metrics':
            return {'metrics': self.system_metrics}
        elif update_type == 'history':
            return self.metrics_collector.get_history()
        else:
            return {
                'agents': self.agent_data,
                'tasks': self.task_data,
                'metrics': self.system_metrics
            }
    
    def _handle_control_action(
        self,
        action: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle control action
        
        Args:
            action: Action type
            data: Action data
            
        Returns:
            Action result
        """
        try:
            if action == 'pause_agent':
                agent_id = data.get('agent_id')
                result = self.system_interface.pause_agent(agent_id)
                
            elif action == 'resume_agent':
                agent_id = data.get('agent_id')
                result = self.system_interface.resume_agent(agent_id)
                
            elif action == 'abort_task':
                task_id = data.get('task_id')
                result = self.system_interface.abort_task(task_id)
                
            elif action == 'reassign_task':
                task_id = data.get('task_id')
                new_agents = data.get('agents', [])
                result = self.system_interface.reassign_task(task_id, new_agents)
                
            elif action == 'emergency_stop':
                result = self.system_interface.emergency_stop()
                
            else:
                result = {'error': f'Unknown action: {action}'}
            
            # Log action
            self.event_logger.log_event({
                'type': 'control_action',
                'action': action,
                'data': data,
                'result': result,
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Control action error: {e}")
            return {'error': str(e)}
    
    def _calculate_efficiency(self) -> float:
        """Calculate system efficiency
        
        Returns:
            System efficiency (0-1)
        """
        if not self.task_data:
            return 1.0
        
        completed = sum(
            1 for t in self.task_data.values()
            if t['status'] == 'completed'
        )
        
        total = len(self.task_data)
        
        return completed / total if total > 0 else 0.0
    
    def add_alert(self, alert: Dict[str, Any]):
        """Add alert to dashboard
        
        Args:
            alert: Alert data
        """
        alert['timestamp'] = time.time()
        self.alerts.append(alert)
        
        # Broadcast alert
        self.socketio.emit('alert', alert, broadcast=True)


class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, history_length: int = 1000):
        """Initialize metrics collector
        
        Args:
            history_length: Maximum history length
        """
        self.history_length = history_length
        
        # Metric histories
        self.histories = {
            'agent_count': deque(maxlen=history_length),
            'active_agents': deque(maxlen=history_length),
            'task_count': deque(maxlen=history_length),
            'completed_tasks': deque(maxlen=history_length),
            'total_energy': deque(maxlen=history_length),
            'avg_agent_energy': deque(maxlen=history_length),
            'system_efficiency': deque(maxlen=history_length),
            'timestamps': deque(maxlen=history_length)
        }
        
        # Statistics
        self.stats = {}
    
    def update(
        self,
        agent_data: Dict[int, Dict[str, Any]],
        task_data: Dict[str, Dict[str, Any]],
        system_metrics: Dict[str, Any]
    ):
        """Update metrics
        
        Args:
            agent_data: Agent data
            task_data: Task data
            system_metrics: System metrics
        """
        timestamp = time.time()
        
        # Collect metrics
        metrics = {
            'agent_count': len(agent_data),
            'active_agents': sum(
                1 for a in agent_data.values()
                if a.get('status') == 'active'
            ),
            'task_count': len(task_data),
            'completed_tasks': sum(
                1 for t in task_data.values()
                if t.get('status') == 'completed'
            ),
            'total_energy': system_metrics.get('total_energy', 0),
            'avg_agent_energy': np.mean([
                a.get('energy', 0) for a in agent_data.values()
            ]) if agent_data else 0,
            'system_efficiency': system_metrics.get('system_efficiency', 0)
        }
        
        # Add to history
        for key, value in metrics.items():
            self.histories[key].append(value)
        self.histories['timestamps'].append(timestamp)
        
        # Update statistics
        self._update_statistics()
    
    def _update_statistics(self):
        """Update metric statistics"""
        for key in self.histories:
            if key == 'timestamps':
                continue
            
            if self.histories[key]:
                values = list(self.histories[key])
                self.stats[key] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': self._calculate_trend(values)
                }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from values
        
        Args:
            values: Metric values
            
        Returns:
            Trend (-1 to 1)
        """
        if len(values) < 10:
            return 0.0
        
        # Simple linear regression on last 10 values
        recent = values[-10:]
        x = np.arange(len(recent))
        
        # Normalize
        if np.std(recent) > 0:
            recent_norm = (recent - np.mean(recent)) / np.std(recent)
            slope = np.polyfit(x, recent_norm, 1)[0]
            return np.clip(slope, -1, 1)
        
        return 0.0
    
    def get_history(self, metric: Optional[str] = None) -> Dict[str, List[float]]:
        """Get metric history
        
        Args:
            metric: Specific metric or None for all
            
        Returns:
            Metric history
        """
        if metric:
            return {
                metric: list(self.histories.get(metric, [])),
                'timestamps': list(self.histories['timestamps'])
            }
        else:
            return {
                key: list(values)
                for key, values in self.histories.items()
            }
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get metric statistics
        
        Returns:
            Metric statistics
        """
        return self.stats.copy()


class EventLogger:
    """Logs system events"""
    
    def __init__(self, max_events: int = 10000):
        """Initialize event logger
        
        Args:
            max_events: Maximum events to store
        """
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        
        # Event categories
        self.categories = {
            'system': [],
            'agent': [],
            'task': [],
            'alert': [],
            'control': [],
            'error': []
        }
    
    def log_event(self, event: Dict[str, Any]):
        """Log an event
        
        Args:
            event: Event data
        """
        # Add timestamp if not present
        if 'timestamp' not in event:
            event['timestamp'] = time.time()
        
        # Add to main log
        self.events.append(event)
        
        # Categorize
        category = event.get('category', 'system')
        if category in self.categories:
            self.categories[category].append(event)
            
            # Limit category size
            if len(self.categories[category]) > self.max_events // 10:
                self.categories[category].pop(0)
    
    def get_events(
        self,
        category: Optional[str] = None,
        since: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get events
        
        Args:
            category: Event category filter
            since: Timestamp filter
            limit: Maximum events to return
            
        Returns:
            List of events
        """
        # Get base events
        if category and category in self.categories:
            events = self.categories[category]
        else:
            events = list(self.events)
        
        # Apply timestamp filter
        if since:
            events = [e for e in events if e.get('timestamp', 0) >= since]
        
        # Apply limit
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get event summary
        
        Returns:
            Event summary statistics
        """
        total_events = len(self.events)
        
        # Category counts
        category_counts = {
            cat: len(events)
            for cat, events in self.categories.items()
        }
        
        # Recent event rate
        recent_events = [
            e for e in self.events
            if e.get('timestamp', 0) > time.time() - 60
        ]
        event_rate = len(recent_events)  # Events per minute
        
        # Error count
        error_count = len([
            e for e in self.events
            if e.get('level') == 'error' or e.get('category') == 'error'
        ])
        
        return {
            'total_events': total_events,
            'category_counts': category_counts,
            'event_rate': event_rate,
            'error_count': error_count,
            'oldest_event': self.events[0].get('timestamp') if self.events else None,
            'newest_event': self.events[-1].get('timestamp') if self.events else None
        }


class DashboardServer:
    """Standalone dashboard server"""
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        system_interface: Optional[Any] = None
    ):
        """Initialize dashboard server
        
        Args:
            config: Dashboard configuration
            system_interface: System interface
        """
        self.config = config or DashboardConfig()
        self.dashboard = Dashboard(self.config, system_interface)
    
    def start(self):
        """Start dashboard server"""
        logger.info("Starting dashboard server...")
        self.dashboard.start()
    
    def stop(self):
        """Stop dashboard server"""
        logger.info("Stopping dashboard server...")
        self.dashboard.stop()
    
    def add_alert(self, alert: Dict[str, Any]):
        """Add alert to dashboard
        
        Args:
            alert: Alert data
        """
        self.dashboard.add_alert(alert)