"""Monitoring Module for PI-HMARL System

This module provides comprehensive monitoring, logging, and alerting
capabilities for the deployed system.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import aiohttp
from pathlib import Path
import psutil
import numpy as np
from collections import deque, defaultdict
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    severity: AlertSeverity
    condition: str
    message: str
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self, namespace: str = "pi_hmarl"):
        """Initialize metrics collector
        
        Args:
            namespace: Metrics namespace
        """
        self.namespace = namespace
        self.metrics: Dict[str, Any] = {}
        self._metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        logger.info(f"Initialized MetricsCollector with namespace: {namespace}")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # System metrics
        self.metrics['cpu_usage'] = Gauge(
            f'{self.namespace}_cpu_usage_percent',
            'CPU usage percentage',
            ['node']
        )
        
        self.metrics['memory_usage'] = Gauge(
            f'{self.namespace}_memory_usage_percent',
            'Memory usage percentage',
            ['node']
        )
        
        self.metrics['disk_usage'] = Gauge(
            f'{self.namespace}_disk_usage_percent',
            'Disk usage percentage',
            ['node', 'mount']
        )
        
        # Agent metrics
        self.metrics['agent_count'] = Gauge(
            f'{self.namespace}_agent_count',
            'Number of active agents'
        )
        
        self.metrics['agent_energy'] = Gauge(
            f'{self.namespace}_agent_energy_percent',
            'Agent energy level',
            ['agent_id']
        )
        
        self.metrics['agent_task_completion'] = Counter(
            f'{self.namespace}_agent_task_completions_total',
            'Total task completions',
            ['agent_id', 'task_type']
        )
        
        # Communication metrics
        self.metrics['messages_sent'] = Counter(
            f'{self.namespace}_messages_sent_total',
            'Total messages sent',
            ['source', 'destination', 'type']
        )
        
        self.metrics['message_latency'] = Histogram(
            f'{self.namespace}_message_latency_seconds',
            'Message delivery latency',
            ['type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        # Task metrics
        self.metrics['task_duration'] = Histogram(
            f'{self.namespace}_task_duration_seconds',
            'Task execution duration',
            ['task_type'],
            buckets=[1, 5, 10, 30, 60, 300, 600]
        )
        
        self.metrics['task_queue_size'] = Gauge(
            f'{self.namespace}_task_queue_size',
            'Current task queue size',
            ['priority']
        )
        
        # Error metrics
        self.metrics['errors'] = Counter(
            f'{self.namespace}_errors_total',
            'Total errors',
            ['component', 'error_type']
        )
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value
        
        Args:
            metric_name: Metric name
            value: Metric value
            labels: Metric labels
        """
        if metric_name not in self.metrics:
            logger.warning(f"Unknown metric: {metric_name}")
            return
        
        metric = self.metrics[metric_name]
        
        try:
            if isinstance(metric, Counter):
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
            
            elif isinstance(metric, Gauge):
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            
            elif isinstance(metric, Histogram):
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
            
            elif isinstance(metric, Summary):
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
            
            # Store in buffer for analysis
            key = f"{metric_name}:{json.dumps(labels or {}, sort_keys=True)}"
            self._metric_buffers[key].append({
                'timestamp': time.time(),
                'value': value
            })
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")
    
    async def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('cpu_usage', cpu_percent, {'node': 'main'})
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric('memory_usage', memory.percent, {'node': 'main'})
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.record_metric(
                        'disk_usage',
                        usage.percent,
                        {'node': 'main', 'mount': partition.mountpoint}
                    )
                except:
                    pass
            
            # Network stats
            net_io = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'process_count': process_count,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def get_metric_statistics(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        window_minutes: int = 5
    ) -> Dict[str, float]:
        """Get metric statistics over time window
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            window_minutes: Time window in minutes
            
        Returns:
            Metric statistics
        """
        key = f"{metric_name}:{json.dumps(labels or {}, sort_keys=True)}"
        buffer = self._metric_buffers.get(key, deque())
        
        if not buffer:
            return {}
        
        # Filter by time window
        cutoff_time = time.time() - (window_minutes * 60)
        values = [
            item['value'] for item in buffer
            if item['timestamp'] > cutoff_time
        ]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format
        
        Returns:
            Prometheus-formatted metrics
        """
        return prometheus_client.generate_latest()


class LogAggregator:
    """Aggregates and manages logs from multiple sources"""
    
    def __init__(
        self,
        log_dir: Path = Path("logs"),
        max_size_mb: int = 1000,
        retention_days: int = 30
    ):
        """Initialize log aggregator
        
        Args:
            log_dir: Log directory
            max_size_mb: Maximum log size in MB
            retention_days: Log retention in days
        """
        self.log_dir = log_dir
        self.max_size_mb = max_size_mb
        self.retention_days = retention_days
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log buffers
        self.log_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Setup handlers
        self._setup_handlers()
        
        logger.info(f"Initialized LogAggregator at {log_dir}")
    
    def _setup_handlers(self):
        """Setup log handlers"""
        # File handler
        log_file = self.log_dir / f"pi_hmarl_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    def log_event(
        self,
        source: str,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an event
        
        Args:
            source: Log source
            level: Log level
            message: Log message
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': time.time(),
            'source': source,
            'level': level,
            'message': message,
            'metadata': metadata or {}
        }
        
        # Add to buffer
        self.log_buffers[source].append(log_entry)
        
        # Write to appropriate logger
        source_logger = logging.getLogger(source)
        
        if level == 'DEBUG':
            source_logger.debug(message, extra=metadata)
        elif level == 'INFO':
            source_logger.info(message, extra=metadata)
        elif level == 'WARNING':
            source_logger.warning(message, extra=metadata)
        elif level == 'ERROR':
            source_logger.error(message, extra=metadata)
        elif level == 'CRITICAL':
            source_logger.critical(message, extra=metadata)
    
    async def rotate_logs(self):
        """Rotate old logs"""
        try:
            current_time = datetime.now()
            
            for log_file in self.log_dir.glob("*.log"):
                # Check file age
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                age_days = (current_time - file_time).days
                
                if age_days > self.retention_days:
                    logger.info(f"Removing old log: {log_file}")
                    log_file.unlink()
                
                # Check file size
                size_mb = log_file.stat().st_size / (1024 * 1024)
                if size_mb > self.max_size_mb:
                    # Archive and compress
                    archive_name = log_file.with_suffix('.log.gz')
                    
                    import gzip
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(archive_name, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    log_file.unlink()
                    logger.info(f"Archived large log: {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to rotate logs: {e}")
    
    def search_logs(
        self,
        query: str,
        source: Optional[str] = None,
        level: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs
        
        Args:
            query: Search query
            source: Filter by source
            level: Filter by level
            start_time: Start timestamp
            end_time: End timestamp
            limit: Result limit
            
        Returns:
            Matching log entries
        """
        results = []
        
        # Search in buffers
        for buffer_source, buffer in self.log_buffers.items():
            if source and buffer_source != source:
                continue
            
            for entry in buffer:
                # Time filter
                if start_time and entry['timestamp'] < start_time:
                    continue
                if end_time and entry['timestamp'] > end_time:
                    continue
                
                # Level filter
                if level and entry['level'] != level:
                    continue
                
                # Query filter
                if query and query.lower() not in entry['message'].lower():
                    continue
                
                results.append(entry)
                
                if len(results) >= limit:
                    break
        
        # Sort by timestamp
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return results[:limit]


class AlertingSystem:
    """Manages alerts and notifications"""
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None
    ):
        """Initialize alerting system
        
        Args:
            webhook_url: Webhook URL for notifications
            email_config: Email configuration
        """
        self.webhook_url = webhook_url
        self.email_config = email_config
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Alert rules
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Notification channels
        self.notification_channels: Dict[str, Callable] = {
            'webhook': self._send_webhook,
            'email': self._send_email,
            'log': self._log_alert
        }
        
        logger.info("Initialized AlertingSystem")
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable,
        severity: AlertSeverity,
        message_template: str,
        channels: List[str],
        cooldown_minutes: int = 5
    ):
        """Add alert rule
        
        Args:
            name: Rule name
            condition: Condition function
            severity: Alert severity
            message_template: Message template
            channels: Notification channels
            cooldown_minutes: Cooldown period
        """
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'message_template': message_template,
            'channels': channels,
            'cooldown_minutes': cooldown_minutes,
            'last_triggered': 0
        })
        
        logger.info(f"Added alert rule: {name}")
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check alert conditions
        
        Args:
            metrics: Current metrics
        """
        current_time = time.time()
        
        for rule in self.alert_rules:
            # Check cooldown
            if current_time - rule['last_triggered'] < rule['cooldown_minutes'] * 60:
                continue
            
            # Check condition
            try:
                if rule['condition'](metrics):
                    # Create alert
                    alert = Alert(
                        id=f"{rule['name']}_{int(current_time)}",
                        name=rule['name'],
                        severity=rule['severity'],
                        condition=str(rule['condition']),
                        message=rule['message_template'].format(**metrics),
                        timestamp=current_time
                    )
                    
                    # Trigger alert
                    await self._trigger_alert(alert, rule['channels'])
                    
                    # Update last triggered
                    rule['last_triggered'] = current_time
                    
            except Exception as e:
                logger.error(f"Failed to check alert rule {rule['name']}: {e}")
    
    async def _trigger_alert(self, alert: Alert, channels: List[str]):
        """Trigger alert notifications
        
        Args:
            alert: Alert to trigger
            channels: Notification channels
        """
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        for channel in channels:
            if channel in self.notification_channels:
                try:
                    await self.notification_channels[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
    
    async def _send_webhook(self, alert: Alert):
        """Send webhook notification
        
        Args:
            alert: Alert to send
        """
        if not self.webhook_url:
            return
        
        payload = {
            'id': alert.id,
            'name': alert.name,
            'severity': alert.severity.value,
            'message': alert.message,
            'timestamp': alert.timestamp,
            'labels': alert.labels
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                json=payload,
                timeout=10
            ) as response:
                if response.status != 200:
                    logger.error(f"Webhook failed: {response.status}")
    
    async def _send_email(self, alert: Alert):
        """Send email notification
        
        Args:
            alert: Alert to send
        """
        if not self.email_config:
            return
        
        # Email implementation would go here
        logger.info(f"Email alert: {alert.name} - {alert.message}")
    
    async def _log_alert(self, alert: Alert):
        """Log alert
        
        Args:
            alert: Alert to log
        """
        logger.warning(
            f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.message}"
        )
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert
        
        Args:
            alert_id: Alert ID
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            
            del self.active_alerts[alert_id]
            
            logger.info(f"Resolved alert: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts
        
        Returns:
            List of active alerts
        """
        return list(self.active_alerts.values())


class DashboardIntegration:
    """Integration with monitoring dashboards"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        log_aggregator: LogAggregator,
        alerting_system: AlertingSystem
    ):
        """Initialize dashboard integration
        
        Args:
            metrics_collector: Metrics collector
            log_aggregator: Log aggregator
            alerting_system: Alerting system
        """
        self.metrics_collector = metrics_collector
        self.log_aggregator = log_aggregator
        self.alerting_system = alerting_system
        
        # Dashboard endpoints
        self.endpoints: Dict[str, Callable] = {
            '/metrics': self._get_metrics,
            '/logs': self._get_logs,
            '/alerts': self._get_alerts,
            '/health': self._get_health,
            '/system': self._get_system_info
        }
        
        logger.info("Initialized DashboardIntegration")
    
    async def _get_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics data
        
        Args:
            params: Request parameters
            
        Returns:
            Metrics data
        """
        metric_name = params.get('metric')
        window = params.get('window', 5)
        
        if metric_name:
            # Get specific metric
            stats = self.metrics_collector.get_metric_statistics(
                metric_name,
                params.get('labels'),
                window
            )
            
            return {
                'metric': metric_name,
                'statistics': stats,
                'timestamp': time.time()
            }
        else:
            # Get all metrics
            metrics = {}
            
            for name in self.metrics_collector.metrics:
                stats = self.metrics_collector.get_metric_statistics(
                    name,
                    window_minutes=window
                )
                if stats:
                    metrics[name] = stats
            
            return {
                'metrics': metrics,
                'timestamp': time.time()
            }
    
    async def _get_logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get logs
        
        Args:
            params: Request parameters
            
        Returns:
            Log data
        """
        logs = self.log_aggregator.search_logs(
            query=params.get('query', ''),
            source=params.get('source'),
            level=params.get('level'),
            start_time=params.get('start_time'),
            end_time=params.get('end_time'),
            limit=params.get('limit', 100)
        )
        
        return {
            'logs': logs,
            'count': len(logs),
            'timestamp': time.time()
        }
    
    async def _get_alerts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get alerts
        
        Args:
            params: Request parameters
            
        Returns:
            Alert data
        """
        active_alerts = self.alerting_system.get_active_alerts()
        
        # Get alert history
        history = list(self.alerting_system.alert_history)
        
        return {
            'active': [
                {
                    'id': alert.id,
                    'name': alert.name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in active_alerts
            ],
            'history': [
                {
                    'id': alert.id,
                    'name': alert.name,
                    'severity': alert.severity.value,
                    'resolved': alert.resolved,
                    'timestamp': alert.timestamp
                }
                for alert in history[-50:]  # Last 50 alerts
            ],
            'timestamp': time.time()
        }
    
    async def _get_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system health
        
        Args:
            params: Request parameters
            
        Returns:
            Health data
        """
        # Collect system metrics
        system_metrics = await self.metrics_collector.collect_system_metrics()
        
        # Determine health status
        health_status = 'healthy'
        issues = []
        
        if system_metrics.get('cpu_percent', 0) > 80:
            health_status = 'degraded'
            issues.append('High CPU usage')
        
        if system_metrics.get('memory_percent', 0) > 85:
            health_status = 'degraded'
            issues.append('High memory usage')
        
        if len(self.alerting_system.get_active_alerts()) > 0:
            health_status = 'unhealthy'
            issues.append('Active alerts')
        
        return {
            'status': health_status,
            'issues': issues,
            'metrics': system_metrics,
            'timestamp': time.time()
        }
    
    async def _get_system_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system information
        
        Args:
            params: Request parameters
            
        Returns:
            System information
        """
        return {
            'system': {
                'platform': psutil.sys.platform,
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'boot_time': psutil.boot_time()
            },
            'process': {
                'pid': os.getpid(),
                'memory_percent': psutil.Process().memory_percent(),
                'cpu_percent': psutil.Process().cpu_percent(),
                'num_threads': psutil.Process().num_threads()
            },
            'timestamp': time.time()
        }
    
    async def handle_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle dashboard request
        
        Args:
            endpoint: Request endpoint
            params: Request parameters
            
        Returns:
            Response data
        """
        if endpoint in self.endpoints:
            return await self.endpoints[endpoint](params)
        else:
            return {
                'error': 'Unknown endpoint',
                'available_endpoints': list(self.endpoints.keys())
            }


# Example alert rules
def create_default_alert_rules() -> List[Dict[str, Any]]:
    """Create default alert rules
    
    Returns:
        List of alert rules
    """
    rules = [
        {
            'name': 'high_cpu_usage',
            'condition': lambda m: m.get('cpu_percent', 0) > 85,
            'severity': AlertSeverity.WARNING,
            'message_template': 'High CPU usage: {cpu_percent:.1f}%',
            'channels': ['log', 'webhook']
        },
        {
            'name': 'critical_memory_usage',
            'condition': lambda m: m.get('memory_percent', 0) > 90,
            'severity': AlertSeverity.CRITICAL,
            'message_template': 'Critical memory usage: {memory_percent:.1f}%',
            'channels': ['log', 'webhook', 'email']
        },
        {
            'name': 'agent_failure',
            'condition': lambda m: m.get('failed_agents', 0) > 0,
            'severity': AlertSeverity.ERROR,
            'message_template': 'Agent failures detected: {failed_agents} agents',
            'channels': ['log', 'webhook']
        },
        {
            'name': 'task_queue_overflow',
            'condition': lambda m: m.get('task_queue_size', 0) > 100,
            'severity': AlertSeverity.WARNING,
            'message_template': 'Task queue overflow: {task_queue_size} pending tasks',
            'channels': ['log']
        },
        {
            'name': 'communication_degraded',
            'condition': lambda m: m.get('packet_loss', 0) > 5,
            'severity': AlertSeverity.WARNING,
            'message_template': 'Communication degraded: {packet_loss:.1f}% packet loss',
            'channels': ['log', 'webhook']
        }
    ]
    
    return rules