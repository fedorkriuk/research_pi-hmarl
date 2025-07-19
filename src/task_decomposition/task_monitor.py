"""Task Monitoring and Progress Tracking

This module implements real-time task monitoring, progress tracking,
and dynamic reallocation for multi-agent systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Task progress information"""
    task_id: str
    status: TaskStatus
    progress_percentage: float
    start_time: Optional[float]
    end_time: Optional[float]
    assigned_agents: List[int]
    energy_consumed: float
    errors: List[str]
    metrics: Dict[str, float]


class TaskMonitor:
    """Monitors task execution and progress"""
    
    def __init__(
        self,
        num_agents: int,
        monitoring_interval: float = 1.0,
        failure_threshold: float = 0.8
    ):
        """Initialize task monitor
        
        Args:
            num_agents: Number of agents
            monitoring_interval: Monitoring update interval
            failure_threshold: Threshold for failure detection
        """
        self.num_agents = num_agents
        self.monitoring_interval = monitoring_interval
        self.failure_threshold = failure_threshold
        
        # Components
        self.progress_tracker = ProgressTracker()
        self.performance_evaluator = PerformanceEvaluator()
        self.task_reallocator = TaskReallocation(num_agents)
        self.failure_handler = FailureHandler()
        
        # Monitoring state
        self.task_registry = {}
        self.agent_assignments = {i: [] for i in range(num_agents)}
        self.monitoring_active = False
        
        logger.info(f"Initialized TaskMonitor with {monitoring_interval}s interval")
    
    def register_task(
        self,
        task_id: str,
        task_info: Dict[str, Any],
        assigned_agents: List[int]
    ):
        """Register task for monitoring
        
        Args:
            task_id: Task ID
            task_info: Task information
            assigned_agents: Assigned agents
        """
        self.task_registry[task_id] = TaskProgress(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress_percentage=0.0,
            start_time=None,
            end_time=None,
            assigned_agents=assigned_agents,
            energy_consumed=0.0,
            errors=[],
            metrics={}
        )
        
        # Update agent assignments
        for agent_id in assigned_agents:
            if agent_id in self.agent_assignments:
                self.agent_assignments[agent_id].append(task_id)
    
    def update_progress(
        self,
        task_id: str,
        agent_updates: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update task progress
        
        Args:
            task_id: Task ID
            agent_updates: Progress updates from agents
            
        Returns:
            Monitoring decision
        """
        if task_id not in self.task_registry:
            return {'error': 'Task not registered'}
        
        task_progress = self.task_registry[task_id]
        
        # Update progress
        progress_info = self.progress_tracker.update(
            task_progress, agent_updates
        )
        
        # Evaluate performance
        performance = self.performance_evaluator.evaluate(
            task_progress, progress_info
        )
        
        # Check for issues
        monitoring_decision = {
            'task_id': task_id,
            'status': task_progress.status,
            'progress': task_progress.progress_percentage,
            'performance': performance,
            'actions': []
        }
        
        # Handle failures
        if performance['failure_risk'] > self.failure_threshold:
            failure_response = self.failure_handler.handle_failure_risk(
                task_progress, performance
            )
            monitoring_decision['actions'].extend(failure_response['actions'])
        
        # Check for reallocation need
        if performance['efficiency'] < 0.5:
            reallocation = self.task_reallocator.evaluate_reallocation(
                task_progress, performance, self.agent_assignments
            )
            if reallocation['recommended']:
                monitoring_decision['actions'].append({
                    'type': 'reallocate',
                    'details': reallocation
                })
        
        return monitoring_decision
    
    def get_team_status(self) -> Dict[str, Any]:
        """Get overall team status
        
        Returns:
            Team status summary
        """
        active_tasks = [
            task for task in self.task_registry.values()
            if task.status == TaskStatus.IN_PROGRESS
        ]
        
        completed_tasks = [
            task for task in self.task_registry.values()
            if task.status == TaskStatus.COMPLETED
        ]
        
        failed_tasks = [
            task for task in self.task_registry.values()
            if task.status == TaskStatus.FAILED
        ]
        
        # Compute metrics
        total_progress = np.mean([
            task.progress_percentage for task in active_tasks
        ]) if active_tasks else 0.0
        
        completion_rate = (
            len(completed_tasks) / len(self.task_registry)
            if self.task_registry else 0.0
        )
        
        # Agent utilization
        agent_utilization = {}
        for agent_id, tasks in self.agent_assignments.items():
            active = sum(
                1 for task_id in tasks
                if self.task_registry.get(task_id, None) and
                self.task_registry[task_id].status == TaskStatus.IN_PROGRESS
            )
            agent_utilization[agent_id] = active
        
        return {
            'active_tasks': len(active_tasks),
            'completed_tasks': len(completed_tasks),
            'failed_tasks': len(failed_tasks),
            'total_progress': total_progress,
            'completion_rate': completion_rate,
            'agent_utilization': agent_utilization,
            'total_energy_consumed': sum(
                task.energy_consumed for task in self.task_registry.values()
            )
        }


class ProgressTracker:
    """Tracks detailed task progress"""
    
    def __init__(self):
        """Initialize progress tracker"""
        self.progress_history = {}
        
        # Progress estimation network
        self.progress_net = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # [progress, confidence, time_remaining]
        )
    
    def update(
        self,
        task_progress: TaskProgress,
        agent_updates: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update task progress
        
        Args:
            task_progress: Current task progress
            agent_updates: Updates from agents
            
        Returns:
            Progress information
        """
        # Start task if pending
        if task_progress.status == TaskStatus.PENDING and agent_updates:
            task_progress.status = TaskStatus.IN_PROGRESS
            task_progress.start_time = time.time()
        
        # Aggregate agent progress
        if agent_updates:
            progress_values = []
            energy_values = []
            
            for agent_id, update in agent_updates.items():
                if 'progress' in update:
                    progress_values.append(update['progress'])
                if 'energy_consumed' in update:
                    energy_values.append(update['energy_consumed'])
            
            # Update progress (average for now)
            if progress_values:
                task_progress.progress_percentage = np.mean(progress_values)
            
            # Update energy
            if energy_values:
                task_progress.energy_consumed = sum(energy_values)
            
            # Check completion
            if task_progress.progress_percentage >= 100.0:
                task_progress.status = TaskStatus.COMPLETED
                task_progress.end_time = time.time()
        
        # Estimate remaining time
        features = self._extract_progress_features(task_progress, agent_updates)
        estimates = self.progress_net(features)
        
        progress_info = {
            'current_progress': task_progress.progress_percentage,
            'estimated_progress': torch.sigmoid(estimates[0]).item() * 100,
            'confidence': torch.sigmoid(estimates[1]).item(),
            'time_remaining': F.softplus(estimates[2]).item() * 300,  # seconds
            'energy_rate': task_progress.energy_consumed / max(
                time.time() - task_progress.start_time, 1.0
            ) if task_progress.start_time else 0.0
        }
        
        # Store history
        if task_progress.task_id not in self.progress_history:
            self.progress_history[task_progress.task_id] = []
        
        self.progress_history[task_progress.task_id].append({
            'timestamp': time.time(),
            'progress': task_progress.progress_percentage,
            'status': task_progress.status
        })
        
        return progress_info
    
    def _extract_progress_features(
        self,
        task_progress: TaskProgress,
        agent_updates: Dict[int, Dict[str, Any]]
    ) -> torch.Tensor:
        """Extract features for progress estimation
        
        Args:
            task_progress: Task progress
            agent_updates: Agent updates
            
        Returns:
            Feature vector
        """
        features = []
        
        # Current progress
        features.append(task_progress.progress_percentage / 100.0)
        
        # Time elapsed
        if task_progress.start_time:
            elapsed = (time.time() - task_progress.start_time) / 3600.0  # hours
            features.append(min(elapsed, 1.0))
        else:
            features.append(0.0)
        
        # Number of active agents
        features.append(len(agent_updates) / len(task_progress.assigned_agents))
        
        # Energy consumption rate
        features.append(min(task_progress.energy_consumed / 50.0, 1.0))
        
        # Error count
        features.append(len(task_progress.errors) / 10.0)
        
        # Agent-specific features
        avg_agent_progress = np.mean([
            update.get('progress', 0) for update in agent_updates.values()
        ]) if agent_updates else 0.0
        features.append(avg_agent_progress / 100.0)
        
        # Progress variance (indicates synchronization)
        if len(agent_updates) > 1:
            progress_values = [
                update.get('progress', 0) for update in agent_updates.values()
            ]
            progress_var = np.var(progress_values) / 100.0
            features.append(min(progress_var, 1.0))
        else:
            features.append(0.0)
        
        # Pad to 15 features
        while len(features) < 15:
            features.append(0.0)
        
        return torch.tensor(features[:15])


class PerformanceEvaluator:
    """Evaluates task performance"""
    
    def __init__(self):
        """Initialize performance evaluator"""
        self.performance_net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # Performance metrics
        )
        
        self.metric_names = [
            'efficiency', 'quality', 'timeliness',
            'resource_usage', 'failure_risk'
        ]
    
    def evaluate(
        self,
        task_progress: TaskProgress,
        progress_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate task performance
        
        Args:
            task_progress: Task progress
            progress_info: Progress information
            
        Returns:
            Performance metrics
        """
        # Extract features
        features = self._extract_performance_features(
            task_progress, progress_info
        )
        
        # Neural performance evaluation
        metrics = torch.sigmoid(self.performance_net(features))
        
        performance = {
            name: metrics[i].item()
            for i, name in enumerate(self.metric_names)
        }
        
        # Add derived metrics
        performance['overall'] = np.mean(list(performance.values()))
        
        # Specific evaluations
        performance.update(self._evaluate_specifics(task_progress, progress_info))
        
        return performance
    
    def _extract_performance_features(
        self,
        task_progress: TaskProgress,
        progress_info: Dict[str, Any]
    ) -> torch.Tensor:
        """Extract performance features
        
        Args:
            task_progress: Task progress
            progress_info: Progress info
            
        Returns:
            Feature vector
        """
        features = []
        
        # Progress features
        features.append(task_progress.progress_percentage / 100.0)
        features.append(progress_info.get('confidence', 0.5))
        
        # Time features
        if task_progress.start_time:
            elapsed = time.time() - task_progress.start_time
            # Assume 10 minute nominal duration
            time_ratio = elapsed / 600.0
            features.append(min(time_ratio, 2.0) / 2.0)
        else:
            features.append(0.0)
        
        # Resource features
        features.append(min(task_progress.energy_consumed / 20.0, 1.0))
        features.append(len(task_progress.assigned_agents) / 5.0)
        
        # Quality indicators
        features.append(1.0 - len(task_progress.errors) / 10.0)
        
        # Progress rate
        if task_progress.start_time and task_progress.progress_percentage > 0:
            elapsed = time.time() - task_progress.start_time
            progress_rate = task_progress.progress_percentage / (elapsed / 60.0)
            features.append(min(progress_rate / 20.0, 1.0))  # % per minute
        else:
            features.append(0.0)
        
        # Energy efficiency
        if task_progress.energy_consumed > 0:
            energy_eff = task_progress.progress_percentage / task_progress.energy_consumed
            features.append(min(energy_eff / 10.0, 1.0))
        else:
            features.append(1.0)
        
        # Status encoding
        status_encoding = [0.0] * 6
        status_idx = list(TaskStatus).index(task_progress.status)
        status_encoding[status_idx] = 1.0
        features.extend(status_encoding)
        
        # Pad to 20
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features[:20])
    
    def _evaluate_specifics(
        self,
        task_progress: TaskProgress,
        progress_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate specific performance aspects
        
        Args:
            task_progress: Task progress
            progress_info: Progress info
            
        Returns:
            Specific metrics
        """
        specifics = {}
        
        # Progress consistency
        if task_progress.task_id in getattr(self, '_progress_history', {}):
            history = self._progress_history[task_progress.task_id]
            if len(history) > 2:
                progress_values = [h['progress'] for h in history[-5:]]
                progress_gradient = np.gradient(progress_values).mean()
                specifics['progress_consistency'] = min(progress_gradient / 10.0, 1.0)
        
        # Deadline adherence
        if progress_info.get('time_remaining', 0) > 0:
            remaining_progress = 100.0 - task_progress.progress_percentage
            required_rate = remaining_progress / (progress_info['time_remaining'] / 60.0)
            
            current_rate = 10.0  # Assumed nominal rate
            specifics['deadline_feasibility'] = min(current_rate / required_rate, 1.0)
        
        return specifics


class TaskReallocation:
    """Handles dynamic task reallocation"""
    
    def __init__(self, num_agents: int):
        """Initialize task reallocation
        
        Args:
            num_agents: Number of agents
        """
        self.num_agents = num_agents
        
        # Reallocation decision network
        self.reallocation_net = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_agents + 1)  # Agent scores + no_realloc
        )
    
    def evaluate_reallocation(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float],
        agent_assignments: Dict[int, List[str]]
    ) -> Dict[str, Any]:
        """Evaluate need for task reallocation
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            agent_assignments: Current assignments
            
        Returns:
            Reallocation recommendation
        """
        # Extract features
        features = self._extract_reallocation_features(
            task_progress, performance, agent_assignments
        )
        
        # Get reallocation scores
        scores = self.reallocation_net(features)
        
        # Softmax for probabilities
        probs = F.softmax(scores, dim=-1)
        
        # Decision
        no_realloc_prob = probs[-1].item()
        
        if no_realloc_prob > 0.7:
            return {
                'recommended': False,
                'reason': 'Current allocation is adequate'
            }
        
        # Find best alternative agents
        agent_scores = probs[:-1]
        current_agents = set(task_progress.assigned_agents)
        
        alternatives = []
        for agent_id in range(self.num_agents):
            if agent_id not in current_agents:
                score = agent_scores[agent_id].item()
                if score > 0.3:
                    alternatives.append((agent_id, score))
        
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        if not alternatives:
            return {
                'recommended': False,
                'reason': 'No suitable alternatives available'
            }
        
        # Build reallocation plan
        reallocation = {
            'recommended': True,
            'reason': self._determine_reason(performance),
            'remove_agents': self._identify_underperformers(
                task_progress, performance
            ),
            'add_agents': [a[0] for a in alternatives[:2]],  # Top 2
            'priority': self._determine_priority(performance),
            'expected_improvement': alternatives[0][1] - no_realloc_prob
        }
        
        return reallocation
    
    def _extract_reallocation_features(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float],
        agent_assignments: Dict[int, List[str]]
    ) -> torch.Tensor:
        """Extract features for reallocation decision
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            agent_assignments: Assignments
            
        Returns:
            Feature vector
        """
        features = []
        
        # Task features
        features.append(task_progress.progress_percentage / 100.0)
        features.append(len(task_progress.assigned_agents) / 5.0)
        features.append(len(task_progress.errors) / 10.0)
        
        # Performance features
        for metric in ['efficiency', 'quality', 'timeliness', 'failure_risk']:
            features.append(performance.get(metric, 0.5))
        
        # Agent load features
        agent_loads = []
        for agent_id in range(self.num_agents):
            load = len(agent_assignments.get(agent_id, []))
            agent_loads.append(load)
        
        features.extend([
            np.mean(agent_loads) / 5.0,
            np.std(agent_loads) / 2.0,
            np.max(agent_loads) / 10.0
        ])
        
        # Current agent features
        for agent_id in task_progress.assigned_agents[:3]:  # First 3
            if agent_id < self.num_agents:
                load = len(agent_assignments.get(agent_id, []))
                features.append(load / 5.0)
        
        # Pad if needed
        while len(features) < 14:
            features.append(0.0)
        
        # Agent availability (binary for each agent)
        for agent_id in range(self.num_agents):
            load = len(agent_assignments.get(agent_id, []))
            features.append(1.0 if load < 3 else 0.0)
        
        # Pad to 25
        while len(features) < 25:
            features.append(0.0)
        
        return torch.tensor(features[:25])
    
    def _determine_reason(self, performance: Dict[str, float]) -> str:
        """Determine reallocation reason
        
        Args:
            performance: Performance metrics
            
        Returns:
            Reason string
        """
        if performance.get('failure_risk', 0) > 0.7:
            return 'High failure risk'
        elif performance.get('efficiency', 1) < 0.3:
            return 'Low efficiency'
        elif performance.get('timeliness', 1) < 0.4:
            return 'Behind schedule'
        else:
            return 'Performance optimization'
    
    def _identify_underperformers(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> List[int]:
        """Identify underperforming agents
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Agents to remove
        """
        # For now, simple heuristic
        if len(task_progress.assigned_agents) > 2:
            # Remove one agent if team is large and underperforming
            if performance.get('efficiency', 1) < 0.5:
                return [task_progress.assigned_agents[-1]]  # Last agent
        
        return []
    
    def _determine_priority(self, performance: Dict[str, float]) -> str:
        """Determine reallocation priority
        
        Args:
            performance: Performance metrics
            
        Returns:
            Priority level
        """
        if performance.get('failure_risk', 0) > 0.8:
            return 'critical'
        elif performance.get('overall', 1) < 0.3:
            return 'high'
        elif performance.get('overall', 1) < 0.5:
            return 'medium'
        else:
            return 'low'


class FailureHandler:
    """Handles task failures and recovery"""
    
    def __init__(self):
        """Initialize failure handler"""
        self.failure_patterns = {
            'timeout': self._handle_timeout,
            'resource_exhaustion': self._handle_resource_exhaustion,
            'coordination_failure': self._handle_coordination_failure,
            'environmental': self._handle_environmental,
            'technical': self._handle_technical
        }
    
    def handle_failure_risk(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Handle high failure risk
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Failure response
        """
        # Identify failure type
        failure_type = self._identify_failure_type(task_progress, performance)
        
        # Get appropriate handler
        handler = self.failure_patterns.get(
            failure_type,
            self._handle_generic
        )
        
        # Generate response
        response = handler(task_progress, performance)
        
        # Log failure risk
        logger.warning(
            f"High failure risk for task {task_progress.task_id}: "
            f"{failure_type} (risk={performance.get('failure_risk', 0):.2f})"
        )
        
        return response
    
    def _identify_failure_type(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> str:
        """Identify type of failure
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Failure type
        """
        # Check various failure indicators
        if performance.get('timeliness', 1) < 0.2:
            return 'timeout'
        elif task_progress.energy_consumed > 30:
            return 'resource_exhaustion'
        elif len(task_progress.errors) > 5:
            return 'technical'
        elif len(task_progress.assigned_agents) > 3 and performance.get('efficiency', 1) < 0.3:
            return 'coordination_failure'
        else:
            return 'generic'
    
    def _handle_timeout(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Handle timeout failure
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Response actions
        """
        return {
            'failure_type': 'timeout',
            'actions': [
                {
                    'type': 'extend_deadline',
                    'extension': 300.0,  # 5 minutes
                    'justification': 'Progress made but behind schedule'
                },
                {
                    'type': 'simplify_task',
                    'reduce_requirements': True,
                    'accept_partial': True
                }
            ],
            'recovery_strategy': 'graceful_degradation'
        }
    
    def _handle_resource_exhaustion(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Handle resource exhaustion
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Response actions
        """
        return {
            'failure_type': 'resource_exhaustion',
            'actions': [
                {
                    'type': 'reduce_resource_usage',
                    'power_reduction': 0.7,
                    'quality_reduction': 0.8
                },
                {
                    'type': 'request_support',
                    'support_type': 'energy_sharing',
                    'urgency': 'high'
                }
            ],
            'recovery_strategy': 'resource_conservation'
        }
    
    def _handle_coordination_failure(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Handle coordination failure
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Response actions
        """
        return {
            'failure_type': 'coordination_failure',
            'actions': [
                {
                    'type': 'restructure_team',
                    'reduce_team_size': True,
                    'assign_coordinator': True
                },
                {
                    'type': 'simplify_coordination',
                    'protocol': 'centralized',
                    'reduce_sync_points': True
                }
            ],
            'recovery_strategy': 'reorganization'
        }
    
    def _handle_environmental(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Handle environmental failure
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Response actions
        """
        return {
            'failure_type': 'environmental',
            'actions': [
                {
                    'type': 'wait_for_conditions',
                    'timeout': 600.0,
                    'monitor_interval': 60.0
                },
                {
                    'type': 'adapt_strategy',
                    'use_alternative_approach': True,
                    'increase_robustness': True
                }
            ],
            'recovery_strategy': 'environmental_adaptation'
        }
    
    def _handle_technical(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Handle technical failure
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Response actions
        """
        return {
            'failure_type': 'technical',
            'actions': [
                {
                    'type': 'debug_and_retry',
                    'collect_diagnostics': True,
                    'retry_with_fixes': True
                },
                {
                    'type': 'fallback_method',
                    'use_simpler_algorithm': True,
                    'reduce_precision': True
                }
            ],
            'recovery_strategy': 'technical_workaround'
        }
    
    def _handle_generic(
        self,
        task_progress: TaskProgress,
        performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Handle generic failure
        
        Args:
            task_progress: Task progress
            performance: Performance metrics
            
        Returns:
            Response actions
        """
        return {
            'failure_type': 'generic',
            'actions': [
                {
                    'type': 'increase_monitoring',
                    'frequency': 2.0,
                    'collect_detailed_metrics': True
                },
                {
                    'type': 'prepare_abort',
                    'save_partial_results': True,
                    'notify_operator': True
                }
            ],
            'recovery_strategy': 'monitored_continuation'
        }