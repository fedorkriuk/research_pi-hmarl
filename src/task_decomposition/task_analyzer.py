"""Task Analysis and Decomposition

This module implements task analysis, complexity estimation, and
intelligent decomposition into subtasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type definitions"""
    SURVEILLANCE = "surveillance"
    SEARCH_RESCUE = "search_rescue"
    DELIVERY = "delivery"
    MAPPING = "mapping"
    INSPECTION = "inspection"
    PATROL = "patrol"
    FORMATION = "formation"
    TRACKING = "tracking"


@dataclass
class Task:
    """Task representation"""
    task_id: str
    task_type: TaskType
    priority: float
    deadline: Optional[float]
    location: Optional[torch.Tensor]
    area: Optional[torch.Tensor]
    requirements: Dict[str, Any]
    constraints: Dict[str, Any]
    reward: float
    metadata: Dict[str, Any] = None


@dataclass
class Subtask:
    """Subtask representation"""
    subtask_id: str
    parent_task_id: str
    primitive_type: str
    estimated_duration: float
    required_agents: int
    required_capabilities: List[str]
    dependencies: List[str]
    location: Optional[torch.Tensor]
    parameters: Dict[str, Any]


class TaskAnalyzer:
    """Analyzes tasks and determines decomposition strategy"""
    
    def __init__(
        self,
        num_agents: int,
        capability_model: Optional[Any] = None
    ):
        """Initialize task analyzer
        
        Args:
            num_agents: Number of agents
            capability_model: Model of agent capabilities
        """
        self.num_agents = num_agents
        self.capability_model = capability_model
        
        # Task complexity estimator
        self.complexity_estimator = TaskComplexityEstimator()
        
        # Task decomposer
        self.decomposer = TaskDecomposer()
        
        # Dependency analyzer
        self.dependency_analyzer = TaskDependencyGraph()
        
        # Task primitives
        self.primitives = TaskPrimitives()
        
        logger.info(f"Initialized TaskAnalyzer for {num_agents} agents")
    
    def analyze_task(
        self,
        task: Task,
        team_state: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze task and determine decomposition
        
        Args:
            task: Task to analyze
            team_state: Current team state
            
        Returns:
            Task analysis results
        """
        # Estimate complexity
        complexity = self.complexity_estimator.estimate(task, team_state)
        
        # Check if decomposition needed
        needs_decomposition = self._needs_decomposition(
            task, complexity, team_state
        )
        
        analysis = {
            'task_id': task.task_id,
            'complexity': complexity,
            'needs_decomposition': needs_decomposition,
            'estimated_agents': self._estimate_required_agents(task, complexity),
            'estimated_time': self._estimate_completion_time(task, complexity),
            'feasibility': self._assess_feasibility(task, team_state)
        }
        
        if needs_decomposition:
            # Decompose into subtasks
            subtasks = self.decomposer.decompose(task, complexity)
            
            # Build dependency graph
            dependency_graph = self.dependency_analyzer.build_graph(subtasks)
            
            analysis['subtasks'] = subtasks
            analysis['dependency_graph'] = dependency_graph
            analysis['critical_path'] = self.dependency_analyzer.find_critical_path(
                dependency_graph
            )
        
        return analysis
    
    def _needs_decomposition(
        self,
        task: Task,
        complexity: Dict[str, float],
        team_state: Dict[str, torch.Tensor]
    ) -> bool:
        """Determine if task needs decomposition
        
        Args:
            task: Task
            complexity: Complexity metrics
            team_state: Team state
            
        Returns:
            Whether decomposition is needed
        """
        # Decompose if too complex for single agent
        if complexity['overall'] > 0.8:
            return True
        
        # Decompose if requires multiple capabilities
        required_capabilities = task.requirements.get('capabilities', [])
        if len(required_capabilities) > 3:
            return True
        
        # Decompose if covers large area
        if task.area is not None:
            area_size = torch.prod(task.area[2:] - task.area[:2])
            if area_size > 10000:  # m²
                return True
        
        # Decompose if time-critical and can parallelize
        if task.deadline is not None:
            estimated_time = self._estimate_completion_time(task, complexity)
            if estimated_time > task.deadline * 0.8:
                return True
        
        return False
    
    def _estimate_required_agents(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> int:
        """Estimate number of agents needed
        
        Args:
            task: Task
            complexity: Complexity metrics
            
        Returns:
            Estimated number of agents
        """
        base_agents = 1
        
        # Scale with complexity
        complexity_factor = complexity['overall']
        agents_from_complexity = int(1 + complexity_factor * 3)
        
        # Consider area coverage
        if task.area is not None:
            area_size = torch.prod(task.area[2:] - task.area[:2])
            agents_from_area = int(area_size / 5000)  # One agent per 5000 m²
            base_agents = max(base_agents, agents_from_area)
        
        # Consider required capabilities
        num_capabilities = len(task.requirements.get('capabilities', []))
        if num_capabilities > 2:
            base_agents = max(base_agents, num_capabilities // 2)
        
        return min(
            max(base_agents, agents_from_complexity),
            self.num_agents
        )
    
    def _estimate_completion_time(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> float:
        """Estimate task completion time
        
        Args:
            task: Task
            complexity: Complexity metrics
            
        Returns:
            Estimated time in seconds
        """
        # Base time by task type
        base_times = {
            TaskType.SURVEILLANCE: 300,
            TaskType.SEARCH_RESCUE: 600,
            TaskType.DELIVERY: 200,
            TaskType.MAPPING: 900,
            TaskType.INSPECTION: 400,
            TaskType.PATROL: 1200,
            TaskType.FORMATION: 100,
            TaskType.TRACKING: 500
        }
        
        base_time = base_times.get(task.task_type, 300)
        
        # Scale by complexity
        time = base_time * (1 + complexity['overall'])
        
        # Scale by area if applicable
        if task.area is not None:
            area_size = torch.prod(task.area[2:] - task.area[:2])
            area_factor = torch.sqrt(area_size / 1000).item()
            time *= area_factor
        
        return time
    
    def _assess_feasibility(
        self,
        task: Task,
        team_state: Dict[str, torch.Tensor]
    ) -> Dict[str, bool]:
        """Assess task feasibility
        
        Args:
            task: Task
            team_state: Team state
            
        Returns:
            Feasibility assessment
        """
        feasibility = {
            'energy': True,
            'capabilities': True,
            'time': True,
            'location': True,
            'overall': True
        }
        
        # Check energy feasibility
        required_energy = self._estimate_energy_requirement(task)
        available_energy = team_state.get('total_energy', 100.0)
        feasibility['energy'] = available_energy > required_energy * 1.2
        
        # Check capability match
        if self.capability_model:
            required_caps = set(task.requirements.get('capabilities', []))
            team_caps = self.capability_model.get_team_capabilities()
            feasibility['capabilities'] = required_caps.issubset(team_caps)
        
        # Check time feasibility
        if task.deadline:
            estimated_time = self._estimate_completion_time(task, {'overall': 0.5})
            feasibility['time'] = estimated_time < task.deadline
        
        # Overall feasibility
        feasibility['overall'] = all(
            v for k, v in feasibility.items() if k != 'overall'
        )
        
        return feasibility
    
    def _estimate_energy_requirement(self, task: Task) -> float:
        """Estimate energy required for task
        
        Args:
            task: Task
            
        Returns:
            Energy requirement in Wh
        """
        # Base energy by task type
        base_energy = {
            TaskType.SURVEILLANCE: 10.0,
            TaskType.SEARCH_RESCUE: 25.0,
            TaskType.DELIVERY: 15.0,
            TaskType.MAPPING: 30.0,
            TaskType.INSPECTION: 12.0,
            TaskType.PATROL: 40.0,
            TaskType.FORMATION: 5.0,
            TaskType.TRACKING: 20.0
        }
        
        energy = base_energy.get(task.task_type, 15.0)
        
        # Scale by area
        if task.area is not None:
            area_size = torch.prod(task.area[2:] - task.area[:2])
            energy *= (area_size / 1000) ** 0.5
        
        return energy


class TaskComplexityEstimator(nn.Module):
    """Estimates task complexity using neural networks"""
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64
    ):
        """Initialize complexity estimator
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Complexity prediction network
        self.complexity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # Different complexity aspects
        )
        
        # Aspect names
        self.aspects = [
            'spatial', 'temporal', 'coordination',
            'capability', 'uncertainty'
        ]
    
    def estimate(
        self,
        task: Task,
        team_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """Estimate task complexity
        
        Args:
            task: Task to analyze
            team_state: Current team state
            
        Returns:
            Complexity metrics
        """
        # Extract features
        features = self._extract_features(task, team_state)
        
        # Neural complexity estimation
        complexities = torch.sigmoid(self.complexity_net(features))
        
        # Create complexity dict
        complexity = {
            aspect: complexities[i].item()
            for i, aspect in enumerate(self.aspects)
        }
        
        # Overall complexity
        complexity['overall'] = complexities.mean().item()
        
        # Add heuristic adjustments
        complexity = self._apply_heuristics(complexity, task)
        
        return complexity
    
    def _extract_features(
        self,
        task: Task,
        team_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Extract features from task
        
        Args:
            task: Task
            team_state: Team state
            
        Returns:
            Feature vector
        """
        features = []
        
        # Task type encoding (8 dims)
        task_type_encoding = torch.zeros(8)
        task_type_idx = list(TaskType).index(task.task_type)
        task_type_encoding[task_type_idx] = 1.0
        features.extend(task_type_encoding.tolist())
        
        # Priority and deadline
        features.append(task.priority)
        features.append(1.0 if task.deadline else 0.0)
        
        # Spatial features
        if task.area is not None:
            area_size = torch.prod(task.area[2:] - task.area[:2])
            features.append(area_size.item() / 10000)  # Normalized
        else:
            features.append(0.0)
        
        # Requirements
        num_capabilities = len(task.requirements.get('capabilities', []))
        features.append(num_capabilities / 5.0)  # Normalized
        
        # Constraints
        num_constraints = len(task.constraints)
        features.append(num_constraints / 5.0)
        
        # Team state features
        if team_state:
            features.append(team_state.get('num_available', 0) / 10)
            features.append(team_state.get('avg_energy', 0.5))
        else:
            features.extend([0.5, 0.5])
        
        # Pad to input_dim
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features[:20])
    
    def _apply_heuristics(
        self,
        complexity: Dict[str, float],
        task: Task
    ) -> Dict[str, float]:
        """Apply heuristic adjustments to complexity
        
        Args:
            complexity: Initial complexity
            task: Task
            
        Returns:
            Adjusted complexity
        """
        # Increase spatial complexity for large areas
        if task.area is not None:
            area_size = torch.prod(task.area[2:] - task.area[:2])
            if area_size > 50000:
                complexity['spatial'] = min(1.0, complexity['spatial'] * 1.5)
        
        # Increase temporal complexity for tight deadlines
        if task.deadline and task.deadline < 300:  # 5 minutes
            complexity['temporal'] = min(1.0, complexity['temporal'] * 1.3)
        
        # Increase coordination complexity for multi-capability tasks
        num_caps = len(task.requirements.get('capabilities', []))
        if num_caps > 3:
            complexity['coordination'] = min(1.0, complexity['coordination'] * 1.2)
        
        # Recalculate overall
        complexity['overall'] = np.mean([
            v for k, v in complexity.items() if k != 'overall'
        ])
        
        return complexity


class TaskDecomposer:
    """Decomposes complex tasks into subtasks"""
    
    def __init__(self):
        """Initialize task decomposer"""
        self.decomposition_strategies = {
            TaskType.SURVEILLANCE: self._decompose_surveillance,
            TaskType.SEARCH_RESCUE: self._decompose_search_rescue,
            TaskType.DELIVERY: self._decompose_delivery,
            TaskType.MAPPING: self._decompose_mapping,
            TaskType.INSPECTION: self._decompose_inspection,
            TaskType.PATROL: self._decompose_patrol,
            TaskType.FORMATION: self._decompose_formation,
            TaskType.TRACKING: self._decompose_tracking
        }
    
    def decompose(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose task into subtasks
        
        Args:
            task: Task to decompose
            complexity: Task complexity
            
        Returns:
            List of subtasks
        """
        # Get decomposition strategy
        strategy = self.decomposition_strategies.get(
            task.task_type,
            self._decompose_generic
        )
        
        # Apply strategy
        subtasks = strategy(task, complexity)
        
        # Validate decomposition
        subtasks = self._validate_decomposition(subtasks, task)
        
        return subtasks
    
    def _decompose_surveillance(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose surveillance task
        
        Args:
            task: Surveillance task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        subtasks = []
        
        if task.area is None:
            # Point surveillance
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_observe",
                parent_task_id=task.task_id,
                primitive_type="observe_point",
                estimated_duration=300,
                required_agents=1,
                required_capabilities=["camera", "hover"],
                dependencies=[],
                location=task.location,
                parameters={'duration': 300, 'altitude': 50}
            ))
        else:
            # Area surveillance - divide into sectors
            sectors = self._divide_area(task.area, complexity)
            
            for i, sector in enumerate(sectors):
                subtasks.append(Subtask(
                    subtask_id=f"{task.task_id}_sector_{i}",
                    parent_task_id=task.task_id,
                    primitive_type="scan_area",
                    estimated_duration=180,
                    required_agents=1,
                    required_capabilities=["camera", "navigation"],
                    dependencies=[],
                    location=sector,
                    parameters={
                        'scan_pattern': 'lawnmower',
                        'overlap': 0.2,
                        'altitude': 100
                    }
                ))
        
        return subtasks
    
    def _decompose_search_rescue(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose search and rescue task
        
        Args:
            task: Search and rescue task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        subtasks = []
        
        # Initial search phase
        search_areas = self._divide_area(task.area, complexity)
        
        for i, area in enumerate(search_areas):
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_search_{i}",
                parent_task_id=task.task_id,
                primitive_type="search_area",
                estimated_duration=300,
                required_agents=1,
                required_capabilities=["thermal_camera", "navigation"],
                dependencies=[],
                location=area,
                parameters={
                    'search_pattern': 'expanding_square',
                    'detection_threshold': 0.8
                }
            ))
        
        # Coordination subtask
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_coordinate",
            parent_task_id=task.task_id,
            primitive_type="coordinate_search",
            estimated_duration=600,
            required_agents=1,
            required_capabilities=["communication", "planning"],
            dependencies=[],
            location=None,
            parameters={'update_interval': 30}
        ))
        
        # Rescue phase (conditional)
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_rescue",
            parent_task_id=task.task_id,
            primitive_type="execute_rescue",
            estimated_duration=300,
            required_agents=2,
            required_capabilities=["cargo", "precision_landing"],
            dependencies=[s.subtask_id for s in subtasks if 'search' in s.subtask_id],
            location=None,  # Will be determined by search
            parameters={'rescue_equipment': 'first_aid'}
        ))
        
        return subtasks
    
    def _decompose_delivery(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose delivery task
        
        Args:
            task: Delivery task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        subtasks = []
        
        # Pickup phase
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_pickup",
            parent_task_id=task.task_id,
            primitive_type="pickup_cargo",
            estimated_duration=60,
            required_agents=1,
            required_capabilities=["cargo", "precision_landing"],
            dependencies=[],
            location=task.requirements.get('pickup_location'),
            parameters={'cargo_weight': task.requirements.get('weight', 1.0)}
        ))
        
        # Transit phase
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_transit",
            parent_task_id=task.task_id,
            primitive_type="navigate_to",
            estimated_duration=180,
            required_agents=1,
            required_capabilities=["navigation", "cargo"],
            dependencies=[f"{task.task_id}_pickup"],
            location=task.location,
            parameters={'flight_mode': 'efficient', 'altitude': 120}
        ))
        
        # Delivery phase
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_deliver",
            parent_task_id=task.task_id,
            primitive_type="deliver_cargo",
            estimated_duration=60,
            required_agents=1,
            required_capabilities=["cargo", "precision_landing"],
            dependencies=[f"{task.task_id}_transit"],
            location=task.location,
            parameters={'delivery_method': 'landing'}
        ))
        
        return subtasks
    
    def _decompose_mapping(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose mapping task
        
        Args:
            task: Mapping task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        subtasks = []
        
        # Divide into mapping sectors
        sectors = self._divide_area(task.area, complexity, overlap=0.3)
        
        for i, sector in enumerate(sectors):
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_map_{i}",
                parent_task_id=task.task_id,
                primitive_type="map_area",
                estimated_duration=400,
                required_agents=1,
                required_capabilities=["lidar", "camera", "navigation"],
                dependencies=[],
                location=sector,
                parameters={
                    'resolution': 0.1,  # meters
                    'scan_pattern': 'lawnmower',
                    'altitude': 150
                }
            ))
        
        # Data fusion subtask
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_fusion",
            parent_task_id=task.task_id,
            primitive_type="fuse_map_data",
            estimated_duration=120,
            required_agents=1,
            required_capabilities=["computation", "communication"],
            dependencies=[s.subtask_id for s in subtasks if 'map' in s.subtask_id],
            location=None,
            parameters={'fusion_algorithm': 'icp', 'quality_threshold': 0.9}
        ))
        
        return subtasks
    
    def _decompose_inspection(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose inspection task
        
        Args:
            task: Inspection task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        subtasks = []
        
        # Approach phase
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_approach",
            parent_task_id=task.task_id,
            primitive_type="approach_target",
            estimated_duration=120,
            required_agents=1,
            required_capabilities=["navigation", "obstacle_avoidance"],
            dependencies=[],
            location=task.location,
            parameters={'approach_distance': 10, 'speed': 'slow'}
        ))
        
        # Detailed inspection
        inspection_points = task.requirements.get('inspection_points', 8)
        
        for i in range(inspection_points):
            angle = (i / inspection_points) * 2 * np.pi
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_inspect_{i}",
                parent_task_id=task.task_id,
                primitive_type="inspect_point",
                estimated_duration=60,
                required_agents=1,
                required_capabilities=["camera", "hover", "stabilization"],
                dependencies=[f"{task.task_id}_approach"],
                location=None,  # Computed from angle
                parameters={
                    'angle': angle,
                    'distance': 5,
                    'capture_mode': 'high_res'
                }
            ))
        
        # Analysis phase
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_analyze",
            parent_task_id=task.task_id,
            primitive_type="analyze_inspection",
            estimated_duration=180,
            required_agents=1,
            required_capabilities=["computation", "anomaly_detection"],
            dependencies=[s.subtask_id for s in subtasks if 'inspect' in s.subtask_id],
            location=None,
            parameters={'analysis_type': 'structural_integrity'}
        ))
        
        return subtasks
    
    def _decompose_patrol(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose patrol task
        
        Args:
            task: Patrol task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        subtasks = []
        
        # Generate patrol waypoints
        waypoints = self._generate_patrol_waypoints(task.area)
        
        for i, waypoint in enumerate(waypoints):
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_waypoint_{i}",
                parent_task_id=task.task_id,
                primitive_type="patrol_waypoint",
                estimated_duration=180,
                required_agents=1,
                required_capabilities=["navigation", "camera"],
                dependencies=[f"{task.task_id}_waypoint_{i-1}"] if i > 0 else [],
                location=waypoint,
                parameters={
                    'loiter_time': 30,
                    'scan_radius': 100,
                    'altitude': 100
                }
            ))
        
        # Make it cyclic
        if len(subtasks) > 0:
            subtasks[-1].parameters['next_waypoint'] = 0
        
        return subtasks
    
    def _decompose_formation(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose formation task
        
        Args:
            task: Formation task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        subtasks = []
        
        formation_type = task.requirements.get('formation_type', 'line')
        num_agents = task.requirements.get('num_agents', 3)
        
        # Assembly phase
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_assemble",
            parent_task_id=task.task_id,
            primitive_type="assemble_formation",
            estimated_duration=60,
            required_agents=num_agents,
            required_capabilities=["coordination", "communication"],
            dependencies=[],
            location=task.location,
            parameters={
                'formation_type': formation_type,
                'spacing': 10
            }
        ))
        
        # Maintain formation
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_maintain",
            parent_task_id=task.task_id,
            primitive_type="maintain_formation",
            estimated_duration=300,
            required_agents=num_agents,
            required_capabilities=["coordination", "navigation"],
            dependencies=[f"{task.task_id}_assemble"],
            location=None,
            parameters={
                'tolerance': 2.0,
                'update_rate': 10
            }
        ))
        
        return subtasks
    
    def _decompose_tracking(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Decompose tracking task
        
        Args:
            task: Tracking task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        subtasks = []
        
        # Detection phase
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_detect",
            parent_task_id=task.task_id,
            primitive_type="detect_target",
            estimated_duration=120,
            required_agents=1,
            required_capabilities=["detection", "camera"],
            dependencies=[],
            location=task.location,
            parameters={
                'target_type': task.requirements.get('target_type', 'vehicle'),
                'detection_range': 500
            }
        ))
        
        # Tracking phase
        subtasks.append(Subtask(
            subtask_id=f"{task.task_id}_track",
            parent_task_id=task.task_id,
            primitive_type="track_target",
            estimated_duration=600,
            required_agents=2,
            required_capabilities=["tracking", "navigation", "communication"],
            dependencies=[f"{task.task_id}_detect"],
            location=None,
            parameters={
                'tracking_mode': 'continuous',
                'handoff_enabled': True,
                'max_speed': 20
            }
        ))
        
        return subtasks
    
    def _decompose_generic(
        self,
        task: Task,
        complexity: Dict[str, float]
    ) -> List[Subtask]:
        """Generic task decomposition
        
        Args:
            task: Task
            complexity: Complexity metrics
            
        Returns:
            Subtasks
        """
        # Simple decomposition based on area
        if task.area is not None:
            sectors = self._divide_area(task.area, complexity)
            
            subtasks = []
            for i, sector in enumerate(sectors):
                subtasks.append(Subtask(
                    subtask_id=f"{task.task_id}_sector_{i}",
                    parent_task_id=task.task_id,
                    primitive_type="generic_area_task",
                    estimated_duration=300,
                    required_agents=1,
                    required_capabilities=task.requirements.get('capabilities', []),
                    dependencies=[],
                    location=sector,
                    parameters={}
                ))
            
            return subtasks
        else:
            # Single subtask
            return [Subtask(
                subtask_id=f"{task.task_id}_main",
                parent_task_id=task.task_id,
                primitive_type="generic_point_task",
                estimated_duration=300,
                required_agents=1,
                required_capabilities=task.requirements.get('capabilities', []),
                dependencies=[],
                location=task.location,
                parameters={}
            )]
    
    def _divide_area(
        self,
        area: torch.Tensor,
        complexity: Dict[str, float],
        overlap: float = 0.1
    ) -> List[torch.Tensor]:
        """Divide area into sectors
        
        Args:
            area: Area bounds [x_min, y_min, x_max, y_max]
            complexity: Complexity metrics
            overlap: Overlap between sectors
            
        Returns:
            List of sector bounds
        """
        # Determine grid size based on complexity
        if complexity['spatial'] > 0.7:
            grid_size = 3
        elif complexity['spatial'] > 0.4:
            grid_size = 2
        else:
            grid_size = 1
        
        x_min, y_min, x_max, y_max = area
        width = x_max - x_min
        height = y_max - y_min
        
        sectors = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate sector bounds with overlap
                sector_width = width / grid_size * (1 + overlap)
                sector_height = height / grid_size * (1 + overlap)
                
                sector_x_min = x_min + i * width / grid_size - overlap * sector_width / 2
                sector_y_min = y_min + j * height / grid_size - overlap * sector_height / 2
                sector_x_max = sector_x_min + sector_width
                sector_y_max = sector_y_min + sector_height
                
                # Clip to original bounds
                sector_x_min = max(sector_x_min, x_min)
                sector_y_min = max(sector_y_min, y_min)
                sector_x_max = min(sector_x_max, x_max)
                sector_y_max = min(sector_y_max, y_max)
                
                sectors.append(torch.tensor([
                    sector_x_min, sector_y_min,
                    sector_x_max, sector_y_max
                ]))
        
        return sectors
    
    def _generate_patrol_waypoints(
        self,
        area: torch.Tensor,
        num_waypoints: int = 4
    ) -> List[torch.Tensor]:
        """Generate patrol waypoints
        
        Args:
            area: Patrol area
            num_waypoints: Number of waypoints
            
        Returns:
            List of waypoint positions
        """
        x_min, y_min, x_max, y_max = area
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        radius_x = (x_max - x_min) / 2 * 0.8
        radius_y = (y_max - y_min) / 2 * 0.8
        
        waypoints = []
        
        for i in range(num_waypoints):
            angle = (i / num_waypoints) * 2 * np.pi
            x = center_x + radius_x * np.cos(angle)
            y = center_y + radius_y * np.sin(angle)
            waypoints.append(torch.tensor([x, y]))
        
        return waypoints
    
    def _validate_decomposition(
        self,
        subtasks: List[Subtask],
        parent_task: Task
    ) -> List[Subtask]:
        """Validate and fix decomposition
        
        Args:
            subtasks: Proposed subtasks
            parent_task: Parent task
            
        Returns:
            Validated subtasks
        """
        # Ensure all subtasks have unique IDs
        seen_ids = set()
        for subtask in subtasks:
            if subtask.subtask_id in seen_ids:
                subtask.subtask_id += f"_{np.random.randint(1000)}"
            seen_ids.add(subtask.subtask_id)
        
        # Validate dependencies exist
        valid_ids = {s.subtask_id for s in subtasks}
        for subtask in subtasks:
            subtask.dependencies = [
                dep for dep in subtask.dependencies if dep in valid_ids
            ]
        
        # Ensure at least one subtask has no dependencies
        if subtasks and all(s.dependencies for s in subtasks):
            subtasks[0].dependencies = []
        
        return subtasks


class TaskDependencyGraph:
    """Manages task dependencies and scheduling"""
    
    def __init__(self):
        """Initialize dependency graph"""
        pass
    
    def build_graph(self, subtasks: List[Subtask]) -> nx.DiGraph:
        """Build dependency graph from subtasks
        
        Args:
            subtasks: List of subtasks
            
        Returns:
            Dependency graph
        """
        graph = nx.DiGraph()
        
        # Add nodes
        for subtask in subtasks:
            graph.add_node(
                subtask.subtask_id,
                subtask=subtask,
                duration=subtask.estimated_duration,
                agents=subtask.required_agents
            )
        
        # Add edges
        for subtask in subtasks:
            for dep in subtask.dependencies:
                if dep in graph:
                    graph.add_edge(dep, subtask.subtask_id)
        
        return graph
    
    def find_critical_path(self, graph: nx.DiGraph) -> List[str]:
        """Find critical path in dependency graph
        
        Args:
            graph: Dependency graph
            
        Returns:
            List of subtask IDs on critical path
        """
        if not graph or len(graph) == 0:
            return []
        
        # Find sources and sinks
        sources = [n for n in graph if graph.in_degree(n) == 0]
        sinks = [n for n in graph if graph.out_degree(n) == 0]
        
        if not sources or not sinks:
            return []
        
        # Add super source and sink
        graph.add_node('_source')
        graph.add_node('_sink')
        
        for source in sources:
            graph.add_edge('_source', source, weight=0)
        
        for sink in sinks:
            graph.add_edge(sink, '_sink', weight=0)
        
        # Add weights (negative durations for longest path)
        for node in graph.nodes():
            if node not in ['_source', '_sink']:
                duration = graph.nodes[node].get('duration', 0)
                for successor in graph.successors(node):
                    graph[node][successor]['weight'] = -duration
        
        # Find longest path (critical path)
        try:
            # Use Bellman-Ford for longest path
            distances = nx.single_source_bellman_ford_path_length(
                graph, '_source', weight='weight'
            )
            
            # Reconstruct path
            path = nx.single_source_bellman_ford_path(
                graph, '_source', weight='weight'
            )['_sink']
            
            # Remove dummy nodes
            critical_path = [n for n in path if n not in ['_source', '_sink']]
            
        except nx.NetworkXException:
            # Fallback to topological sort
            critical_path = list(nx.topological_sort(graph))
            critical_path = [n for n in critical_path if n not in ['_source', '_sink']]
        
        # Remove temporary nodes
        graph.remove_node('_source')
        graph.remove_node('_sink')
        
        return critical_path
    
    def get_parallel_groups(self, graph: nx.DiGraph) -> List[List[str]]:
        """Get groups of subtasks that can run in parallel
        
        Args:
            graph: Dependency graph
            
        Returns:
            List of parallel groups
        """
        if not graph:
            return []
        
        # Topological generations
        generations = list(nx.topological_generations(graph))
        
        return generations
    
    def estimate_completion_time(
        self,
        graph: nx.DiGraph,
        available_agents: int
    ) -> float:
        """Estimate total completion time
        
        Args:
            graph: Dependency graph
            available_agents: Number of available agents
            
        Returns:
            Estimated completion time
        """
        if not graph:
            return 0.0
        
        # Get parallel groups
        parallel_groups = self.get_parallel_groups(graph)
        
        total_time = 0.0
        
        for group in parallel_groups:
            # Get tasks in this group
            group_tasks = []
            for task_id in group:
                subtask = graph.nodes[task_id]['subtask']
                group_tasks.append({
                    'duration': subtask.estimated_duration,
                    'agents': subtask.required_agents
                })
            
            # Schedule within group
            group_time = self._schedule_parallel_tasks(
                group_tasks, available_agents
            )
            
            total_time += group_time
        
        return total_time
    
    def _schedule_parallel_tasks(
        self,
        tasks: List[Dict[str, float]],
        available_agents: int
    ) -> float:
        """Schedule parallel tasks with agent constraints
        
        Args:
            tasks: List of tasks with duration and agent requirements
            available_agents: Available agents
            
        Returns:
            Completion time
        """
        if not tasks:
            return 0.0
        
        # Sort by duration (longest first)
        tasks = sorted(tasks, key=lambda x: x['duration'], reverse=True)
        
        # Agent availability timeline
        agent_timeline = [0.0] * available_agents
        
        completion_time = 0.0
        
        for task in tasks:
            required_agents = task['agents']
            duration = task['duration']
            
            if required_agents > available_agents:
                # Task cannot be performed
                continue
            
            # Find earliest time when required agents are available
            agent_timeline_sorted = sorted(agent_timeline)
            earliest_start = agent_timeline_sorted[required_agents - 1]
            
            # Assign agents
            assigned = 0
            for i in range(available_agents):
                if agent_timeline[i] <= earliest_start and assigned < required_agents:
                    agent_timeline[i] = earliest_start + duration
                    assigned += 1
            
            completion_time = max(completion_time, earliest_start + duration)
        
        return completion_time


class TaskPrimitives:
    """Defines primitive task operations"""
    
    def __init__(self):
        """Initialize task primitives"""
        self.primitives = {
            # Movement primitives
            'navigate_to': {
                'capabilities': ['navigation'],
                'parameters': ['target_position', 'speed', 'altitude']
            },
            'hover': {
                'capabilities': ['hover'],
                'parameters': ['position', 'duration', 'altitude']
            },
            'orbit': {
                'capabilities': ['navigation'],
                'parameters': ['center', 'radius', 'speed', 'duration']
            },
            
            # Sensing primitives
            'scan_area': {
                'capabilities': ['camera', 'navigation'],
                'parameters': ['area', 'pattern', 'overlap', 'altitude']
            },
            'observe_point': {
                'capabilities': ['camera', 'hover'],
                'parameters': ['position', 'duration', 'zoom_level']
            },
            'detect_target': {
                'capabilities': ['detection'],
                'parameters': ['target_type', 'detection_range', 'confidence']
            },
            
            # Interaction primitives
            'pickup_cargo': {
                'capabilities': ['cargo', 'precision_landing'],
                'parameters': ['position', 'cargo_weight', 'pickup_method']
            },
            'deliver_cargo': {
                'capabilities': ['cargo', 'precision_landing'],
                'parameters': ['position', 'delivery_method']
            },
            
            # Coordination primitives
            'coordinate_search': {
                'capabilities': ['communication', 'planning'],
                'parameters': ['team_size', 'update_interval']
            },
            'maintain_formation': {
                'capabilities': ['coordination', 'navigation'],
                'parameters': ['formation_type', 'spacing', 'tolerance']
            },
            
            # Analysis primitives
            'analyze_data': {
                'capabilities': ['computation'],
                'parameters': ['data_type', 'analysis_method']
            },
            'fuse_map_data': {
                'capabilities': ['computation', 'communication'],
                'parameters': ['fusion_algorithm', 'quality_threshold']
            }
        }
    
    def get_primitive_info(self, primitive_type: str) -> Dict[str, Any]:
        """Get information about a primitive
        
        Args:
            primitive_type: Type of primitive
            
        Returns:
            Primitive information
        """
        return self.primitives.get(primitive_type, {})
    
    def validate_primitive(
        self,
        primitive_type: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """Validate primitive parameters
        
        Args:
            primitive_type: Type of primitive
            parameters: Provided parameters
            
        Returns:
            Whether parameters are valid
        """
        if primitive_type not in self.primitives:
            return False
        
        required_params = self.primitives[primitive_type]['parameters']
        
        # Check all required parameters are provided
        for param in required_params:
            if param not in parameters:
                return False
        
        return True