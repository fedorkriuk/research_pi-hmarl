"""Cooperative Planning and Coordination

This module implements cooperative planning strategies for multi-agent
teams including team formation, role assignment, and coordination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class TeamRole(Enum):
    """Team role definitions"""
    LEADER = "leader"
    SCOUT = "scout"
    WORKER = "worker"
    SUPPORT = "support"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class CooperativePlanner:
    """Plans and coordinates multi-agent cooperation"""
    
    def __init__(
        self,
        num_agents: int,
        communication_range: float = 1000.0,  # meters
        planning_horizon: int = 10
    ):
        """Initialize cooperative planner
        
        Args:
            num_agents: Number of agents
            communication_range: Communication range
            planning_horizon: Planning steps ahead
        """
        self.num_agents = num_agents
        self.communication_range = communication_range
        self.planning_horizon = planning_horizon
        
        # Components
        self.team_formation = TeamFormation(num_agents)
        self.role_assignment = RoleAssignment()
        self.coordination_protocol = CoordinationProtocol(communication_range)
        self.sync_manager = SynchronizationManager(num_agents)
        
        # Planning state
        self.current_teams = {}
        self.agent_roles = {}
        
        logger.info(f"Initialized CooperativePlanner for {num_agents} agents")
    
    def plan_cooperation(
        self,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]],
        assignment: Dict[int, List[str]]
    ) -> Dict[str, Any]:
        """Plan cooperative execution
        
        Args:
            tasks: Tasks to execute
            agent_states: Current agent states
            assignment: Task assignments
            
        Returns:
            Cooperation plan
        """
        # Form teams based on assignments
        teams = self.team_formation.form_teams(
            assignment, tasks, agent_states
        )
        
        # Assign roles within teams
        team_roles = {}
        for team_id, team_agents in teams.items():
            roles = self.role_assignment.assign_roles(
                team_agents, tasks, agent_states
            )
            team_roles[team_id] = roles
        
        # Create coordination plan
        coordination_plan = self.coordination_protocol.create_plan(
            teams, team_roles, tasks, agent_states
        )
        
        # Synchronization points
        sync_points = self.sync_manager.identify_sync_points(
            teams, tasks, coordination_plan
        )
        
        # Build complete plan
        cooperation_plan = {
            'teams': teams,
            'roles': team_roles,
            'coordination': coordination_plan,
            'sync_points': sync_points,
            'communication_schedule': self._create_comm_schedule(teams, agent_states),
            'contingency_plans': self._create_contingency_plans(teams, tasks)
        }
        
        # Update internal state
        self.current_teams = teams
        self.agent_roles = self._flatten_roles(team_roles)
        
        return cooperation_plan
    
    def _create_comm_schedule(
        self,
        teams: Dict[str, List[int]],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Create communication schedule
        
        Args:
            teams: Team assignments
            agent_states: Agent states
            
        Returns:
            Communication schedule
        """
        schedule = {
            'regular_updates': {},
            'critical_events': {},
            'bandwidth_allocation': {}
        }
        
        for team_id, team_agents in teams.items():
            # Regular update frequency based on team size
            update_interval = 10.0 * (1 + len(team_agents) / 10)
            
            schedule['regular_updates'][team_id] = {
                'interval': update_interval,
                'participants': team_agents,
                'priority': 'normal'
            }
            
            # Critical event handling
            schedule['critical_events'][team_id] = {
                'triggers': ['emergency', 'task_failure', 'agent_failure'],
                'broadcast': True,
                'priority': 'high'
            }
            
            # Bandwidth allocation
            total_bandwidth = 100.0  # Mbps
            per_agent_bandwidth = total_bandwidth / len(team_agents)
            
            for agent in team_agents:
                schedule['bandwidth_allocation'][agent] = per_agent_bandwidth
        
        return schedule
    
    def _create_contingency_plans(
        self,
        teams: Dict[str, List[int]],
        tasks: List[Any]
    ) -> Dict[str, Any]:
        """Create contingency plans for failures
        
        Args:
            teams: Team assignments
            tasks: Tasks
            
        Returns:
            Contingency plans
        """
        contingencies = {}
        
        for team_id, team_agents in teams.items():
            contingencies[team_id] = {
                'agent_failure': self._agent_failure_plan(team_agents, tasks),
                'communication_failure': self._comm_failure_plan(team_agents),
                'task_failure': self._task_failure_plan(tasks),
                'energy_critical': self._energy_critical_plan(team_agents)
            }
        
        return contingencies
    
    def _agent_failure_plan(
        self,
        team_agents: List[int],
        tasks: List[Any]
    ) -> Dict[str, Any]:
        """Plan for agent failure
        
        Args:
            team_agents: Agents in team
            tasks: Team tasks
            
        Returns:
            Agent failure plan
        """
        return {
            'detection_method': 'heartbeat_timeout',
            'timeout_threshold': 30.0,  # seconds
            'response': {
                'redistribute_tasks': True,
                'request_replacement': len(team_agents) < 3,
                'continue_degraded': len(team_agents) >= 3
            }
        }
    
    def _comm_failure_plan(self, team_agents: List[int]) -> Dict[str, Any]:
        """Plan for communication failure
        
        Args:
            team_agents: Agents in team
            
        Returns:
            Communication failure plan
        """
        return {
            'fallback_protocol': 'predetermined_rendezvous',
            'rendezvous_interval': 300.0,  # seconds
            'autonomous_operation': True,
            'max_autonomous_duration': 600.0
        }
    
    def _task_failure_plan(self, tasks: List[Any]) -> Dict[str, Any]:
        """Plan for task failure
        
        Args:
            tasks: Tasks
            
        Returns:
            Task failure plan
        """
        return {
            'retry_attempts': 2,
            'alternative_approaches': ['change_parameters', 'request_assistance'],
            'escalation_threshold': 3,
            'abort_criteria': {'time_exceeded': 2.0, 'energy_depleted': 0.1}
        }
    
    def _energy_critical_plan(self, team_agents: List[int]) -> Dict[str, Any]:
        """Plan for energy critical situations
        
        Args:
            team_agents: Agents in team
            
        Returns:
            Energy critical plan
        """
        return {
            'threshold': 0.2,  # 20% SOC
            'response': {
                'immediate': 'reduce_power_consumption',
                'handoff_tasks': True,
                'return_to_base': True,
                'priority': 'emergency'
            }
        }
    
    def _flatten_roles(
        self,
        team_roles: Dict[str, Dict[int, TeamRole]]
    ) -> Dict[int, TeamRole]:
        """Flatten team roles to agent roles
        
        Args:
            team_roles: Roles by team
            
        Returns:
            Roles by agent
        """
        agent_roles = {}
        
        for team_id, roles in team_roles.items():
            for agent_id, role in roles.items():
                agent_roles[agent_id] = role
        
        return agent_roles


class TeamFormation:
    """Forms teams based on task requirements"""
    
    def __init__(self, num_agents: int):
        """Initialize team formation
        
        Args:
            num_agents: Number of agents
        """
        self.num_agents = num_agents
        
        # Team formation network
        self.formation_net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_agents * num_agents)  # Affinity matrix
        )
    
    def form_teams(
        self,
        assignment: Dict[int, List[str]],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, List[int]]:
        """Form teams from assignment
        
        Args:
            assignment: Task assignments
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Team assignments
        """
        # Group agents working on related tasks
        task_groups = self._identify_task_groups(tasks)
        
        # Build affinity matrix
        affinity = self._compute_affinity_matrix(
            assignment, task_groups, agent_states
        )
        
        # Cluster agents into teams
        teams = self._cluster_agents(affinity, assignment)
        
        # Validate teams
        teams = self._validate_teams(teams, agent_states)
        
        return teams
    
    def _identify_task_groups(self, tasks: List[Any]) -> Dict[str, Set[str]]:
        """Identify related task groups
        
        Args:
            tasks: All tasks
            
        Returns:
            Task groups
        """
        # Build task dependency graph
        task_graph = nx.Graph()
        
        for task in tasks:
            task_id = getattr(task, 'subtask_id', str(task))
            task_graph.add_node(task_id)
            
            # Add edges for dependencies
            deps = getattr(task, 'dependencies', [])
            for dep in deps:
                if task_graph.has_node(dep):
                    task_graph.add_edge(task_id, dep)
            
            # Add edges for spatial proximity
            if hasattr(task, 'location') and task.location is not None:
                for other_task in tasks:
                    if other_task != task:
                        other_id = getattr(other_task, 'subtask_id', str(other_task))
                        if hasattr(other_task, 'location') and other_task.location is not None:
                            dist = torch.norm(task.location - other_task.location)
                            if dist < 100:  # Within 100m
                                task_graph.add_edge(task_id, other_id)
        
        # Find connected components
        components = list(nx.connected_components(task_graph))
        
        # Create groups
        groups = {}
        for i, component in enumerate(components):
            group_id = f"group_{i}"
            groups[group_id] = set(component)
        
        return groups
    
    def _compute_affinity_matrix(
        self,
        assignment: Dict[int, List[str]],
        task_groups: Dict[str, Set[str]],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute agent affinity matrix
        
        Args:
            assignment: Task assignments
            task_groups: Task groupings
            agent_states: Agent states
            
        Returns:
            Affinity matrix
        """
        affinity = torch.zeros(self.num_agents, self.num_agents)
        
        # Affinity based on shared task groups
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if i in assignment and j in assignment:
                    # Check task overlap
                    tasks_i = set(assignment[i])
                    tasks_j = set(assignment[j])
                    
                    # Same group affinity
                    for group_id, group_tasks in task_groups.items():
                        if tasks_i & group_tasks and tasks_j & group_tasks:
                            affinity[i, j] += 1.0
                            affinity[j, i] += 1.0
        
        # Neural affinity adjustment
        features = self._extract_team_features(assignment, agent_states)
        neural_affinity = self.formation_net(features).view(
            self.num_agents, self.num_agents
        )
        
        # Combine affinities
        affinity = 0.7 * affinity + 0.3 * torch.sigmoid(neural_affinity)
        
        return affinity
    
    def _extract_team_features(
        self,
        assignment: Dict[int, List[str]],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Extract features for team formation
        
        Args:
            assignment: Assignments
            agent_states: Agent states
            
        Returns:
            Feature vector
        """
        features = []
        
        # Assignment density
        avg_tasks = np.mean([len(tasks) for tasks in assignment.values()])
        features.append(avg_tasks / 5.0)
        
        # Energy distribution
        energies = [
            agent_states[i].get('soc', 0.5)
            for i in agent_states
        ]
        features.extend([
            np.mean(energies),
            np.std(energies),
            np.min(energies)
        ])
        
        # Capability diversity
        all_caps = set()
        for state in agent_states.values():
            all_caps.update(state.get('capabilities', []))
        features.append(len(all_caps) / 20.0)
        
        # Spatial distribution
        positions = [
            agent_states[i].get('position', torch.zeros(2))
            for i in agent_states
        ]
        if positions:
            pos_tensor = torch.stack(positions)
            centroid = pos_tensor.mean(dim=0)
            spread = (pos_tensor - centroid).norm(dim=1).mean()
            features.append(spread.item() / 1000.0)
        else:
            features.append(0.0)
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features[:20])
    
    def _cluster_agents(
        self,
        affinity: torch.Tensor,
        assignment: Dict[int, List[str]]
    ) -> Dict[str, List[int]]:
        """Cluster agents into teams based on affinity
        
        Args:
            affinity: Affinity matrix
            assignment: Task assignments
            
        Returns:
            Team clusters
        """
        # Start with agents that have tasks
        active_agents = [i for i, tasks in assignment.items() if tasks]
        
        if not active_agents:
            return {}
        
        teams = {}
        assigned = set()
        team_id = 0
        
        # Greedy clustering
        while len(assigned) < len(active_agents):
            # Find unassigned agent with most tasks
            remaining = [a for a in active_agents if a not in assigned]
            if not remaining:
                break
            
            seed = max(remaining, key=lambda a: len(assignment.get(a, [])))
            
            # Build team around seed
            team = [seed]
            assigned.add(seed)
            
            # Add agents with high affinity
            candidates = [
                (a, affinity[seed, a].item())
                for a in remaining
                if a != seed and a not in assigned
            ]
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Add up to 4 more agents (team size 5)
            for agent, aff in candidates[:4]:
                if aff > 0.5:  # Affinity threshold
                    team.append(agent)
                    assigned.add(agent)
            
            teams[f"team_{team_id}"] = team
            team_id += 1
        
        return teams
    
    def _validate_teams(
        self,
        teams: Dict[str, List[int]],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, List[int]]:
        """Validate and adjust teams
        
        Args:
            teams: Proposed teams
            agent_states: Agent states
            
        Returns:
            Validated teams
        """
        validated = {}
        
        for team_id, team_agents in teams.items():
            # Check team has required capabilities
            team_caps = set()
            for agent in team_agents:
                if agent in agent_states:
                    team_caps.update(agent_states[agent].get('capabilities', []))
            
            # Minimum capabilities for a functional team
            required_caps = {'navigation', 'communication'}
            
            if required_caps.issubset(team_caps) and len(team_agents) > 0:
                validated[team_id] = team_agents
        
        return validated


class RoleAssignment:
    """Assigns roles within teams"""
    
    def __init__(self):
        """Initialize role assignment"""
        # Role suitability network
        self.role_net = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, len(TeamRole))
        )
        
        # Role requirements
        self.role_requirements = {
            TeamRole.LEADER: ['communication', 'planning', 'coordination'],
            TeamRole.SCOUT: ['navigation', 'detection', 'long_range'],
            TeamRole.WORKER: ['cargo', 'manipulation', 'precision_landing'],
            TeamRole.SUPPORT: ['communication', 'computation', 'sensor_fusion'],
            TeamRole.COORDINATOR: ['communication', 'planning', 'computation'],
            TeamRole.SPECIALIST: []  # Task-specific
        }
    
    def assign_roles(
        self,
        team_agents: List[int],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, TeamRole]:
        """Assign roles to team members
        
        Args:
            team_agents: Agents in team
            tasks: Team tasks
            agent_states: Agent states
            
        Returns:
            Role assignments
        """
        if not team_agents:
            return {}
        
        # Compute role suitability scores
        suitability = {}
        
        for agent_id in team_agents:
            if agent_id in agent_states:
                scores = self._compute_role_scores(
                    agent_id, agent_states[agent_id], tasks
                )
                suitability[agent_id] = scores
        
        # Assign roles to maximize team effectiveness
        assignments = self._optimize_role_assignment(
            suitability, team_agents, tasks
        )
        
        return assignments
    
    def _compute_role_scores(
        self,
        agent_id: int,
        agent_state: Dict[str, torch.Tensor],
        tasks: List[Any]
    ) -> Dict[TeamRole, float]:
        """Compute role suitability scores
        
        Args:
            agent_id: Agent ID
            agent_state: Agent state
            tasks: Tasks
            
        Returns:
            Role scores
        """
        # Extract features
        features = []
        
        # Agent capabilities
        agent_caps = set(agent_state.get('capabilities', []))
        
        # Energy level
        features.append(agent_state.get('soc', 0.5))
        
        # Experience (simplified)
        features.append(agent_state.get('experience', 0.5))
        
        # Current workload
        features.append(agent_state.get('workload', 0.0))
        
        # Capability match scores for each role
        for role in TeamRole:
            if role in self.role_requirements:
                required = set(self.role_requirements[role])
                if required:
                    match = len(agent_caps & required) / len(required)
                else:
                    match = 0.5
            else:
                match = 0.5
            features.append(match)
        
        # Task requirements
        all_task_caps = set()
        for task in tasks:
            all_task_caps.update(getattr(task, 'required_capabilities', []))
        
        task_match = len(agent_caps & all_task_caps) / max(len(all_task_caps), 1)
        features.append(task_match)
        
        # Pad features
        while len(features) < 15:
            features.append(0.0)
        
        # Neural role scoring
        features_tensor = torch.tensor(features[:15])
        role_logits = self.role_net(features_tensor)
        role_probs = F.softmax(role_logits, dim=-1)
        
        # Convert to role scores
        scores = {}
        for i, role in enumerate(TeamRole):
            base_score = role_probs[i].item()
            
            # Adjust based on capability match
            if role in self.role_requirements:
                required = set(self.role_requirements[role])
                if required:
                    cap_match = len(agent_caps & required) / len(required)
                    scores[role] = base_score * (0.5 + 0.5 * cap_match)
                else:
                    scores[role] = base_score
            else:
                scores[role] = base_score
        
        return scores
    
    def _optimize_role_assignment(
        self,
        suitability: Dict[int, Dict[TeamRole, float]],
        team_agents: List[int],
        tasks: List[Any]
    ) -> Dict[int, TeamRole]:
        """Optimize role assignments
        
        Args:
            suitability: Role suitability scores
            team_agents: Team agents
            tasks: Tasks
            
        Returns:
            Optimal assignments
        """
        assignments = {}
        assigned_roles = set()
        
        # Priority roles that must be filled
        priority_roles = [TeamRole.LEADER]
        
        # Assign priority roles first
        for role in priority_roles:
            if len(team_agents) > len(assigned_roles):
                # Find best agent for role
                best_agent = None
                best_score = -1
                
                for agent_id in team_agents:
                    if agent_id not in assignments and agent_id in suitability:
                        score = suitability[agent_id].get(role, 0)
                        if score > best_score:
                            best_score = score
                            best_agent = agent_id
                
                if best_agent is not None:
                    assignments[best_agent] = role
                    assigned_roles.add(role)
        
        # Assign remaining agents
        remaining_agents = [a for a in team_agents if a not in assignments]
        remaining_roles = [r for r in TeamRole if r not in assigned_roles]
        
        # Greedy assignment
        while remaining_agents and remaining_roles:
            # Find best agent-role pair
            best_pair = None
            best_score = -1
            
            for agent_id in remaining_agents:
                if agent_id in suitability:
                    for role in remaining_roles:
                        score = suitability[agent_id].get(role, 0)
                        if score > best_score:
                            best_score = score
                            best_pair = (agent_id, role)
            
            if best_pair:
                agent_id, role = best_pair
                assignments[agent_id] = role
                remaining_agents.remove(agent_id)
                remaining_roles.remove(role)
            else:
                break
        
        # Assign default role to any remaining
        for agent_id in remaining_agents:
            assignments[agent_id] = TeamRole.WORKER
        
        return assignments


class CoordinationProtocol:
    """Defines coordination protocols for teams"""
    
    def __init__(self, communication_range: float):
        """Initialize coordination protocol
        
        Args:
            communication_range: Communication range
        """
        self.communication_range = communication_range
        
        # Protocol types
        self.protocols = {
            'centralized': self._centralized_protocol,
            'distributed': self._distributed_protocol,
            'hierarchical': self._hierarchical_protocol,
            'consensus': self._consensus_protocol
        }
    
    def create_plan(
        self,
        teams: Dict[str, List[int]],
        roles: Dict[str, Dict[int, TeamRole]],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Create coordination plan
        
        Args:
            teams: Team assignments
            roles: Role assignments
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Coordination plan
        """
        plan = {}
        
        for team_id, team_agents in teams.items():
            # Select protocol based on team size and task
            if len(team_agents) <= 2:
                protocol = 'distributed'
            elif len(team_agents) <= 5:
                protocol = 'hierarchical'
            else:
                protocol = 'consensus'
            
            # Create team plan
            team_roles = roles.get(team_id, {})
            team_plan = self.protocols[protocol](
                team_agents, team_roles, tasks, agent_states
            )
            
            plan[team_id] = {
                'protocol': protocol,
                'details': team_plan,
                'communication_graph': self._build_comm_graph(
                    team_agents, agent_states
                )
            }
        
        return plan
    
    def _centralized_protocol(
        self,
        agents: List[int],
        roles: Dict[int, TeamRole],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Centralized coordination protocol
        
        Args:
            agents: Team agents
            roles: Agent roles
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Protocol details
        """
        # Find leader
        leader = None
        for agent_id, role in roles.items():
            if role == TeamRole.LEADER:
                leader = agent_id
                break
        
        if leader is None and agents:
            leader = agents[0]
        
        return {
            'leader': leader,
            'followers': [a for a in agents if a != leader],
            'decision_making': 'leader_decides',
            'information_flow': 'star_topology',
            'update_protocol': {
                'followers_to_leader': {'frequency': 5.0, 'type': 'status'},
                'leader_to_followers': {'frequency': 10.0, 'type': 'commands'}
            }
        }
    
    def _distributed_protocol(
        self,
        agents: List[int],
        roles: Dict[int, TeamRole],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Distributed coordination protocol
        
        Args:
            agents: Team agents
            roles: Agent roles
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Protocol details
        """
        return {
            'topology': 'mesh',
            'decision_making': 'local_decisions',
            'information_sharing': 'broadcast',
            'consensus_required': False,
            'update_protocol': {
                'broadcast_interval': 10.0,
                'local_decision_threshold': 0.8
            }
        }
    
    def _hierarchical_protocol(
        self,
        agents: List[int],
        roles: Dict[int, TeamRole],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Hierarchical coordination protocol
        
        Args:
            agents: Team agents
            roles: Agent roles
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Protocol details
        """
        # Build hierarchy based on roles
        hierarchy = {
            'level_0': [],  # Leaders
            'level_1': [],  # Coordinators
            'level_2': []   # Workers
        }
        
        for agent_id, role in roles.items():
            if role == TeamRole.LEADER:
                hierarchy['level_0'].append(agent_id)
            elif role in [TeamRole.COORDINATOR, TeamRole.SUPPORT]:
                hierarchy['level_1'].append(agent_id)
            else:
                hierarchy['level_2'].append(agent_id)
        
        return {
            'hierarchy': hierarchy,
            'decision_flow': 'top_down',
            'information_flow': 'bottom_up',
            'delegation_enabled': True,
            'escalation_threshold': 0.3
        }
    
    def _consensus_protocol(
        self,
        agents: List[int],
        roles: Dict[int, TeamRole],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Consensus-based coordination protocol
        
        Args:
            agents: Team agents
            roles: Agent roles
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Protocol details
        """
        return {
            'consensus_algorithm': 'weighted_voting',
            'voting_weights': self._compute_voting_weights(agents, roles),
            'decision_threshold': 0.7,
            'timeout': 30.0,
            'fallback': 'majority_vote'
        }
    
    def _compute_voting_weights(
        self,
        agents: List[int],
        roles: Dict[int, TeamRole]
    ) -> Dict[int, float]:
        """Compute voting weights for consensus
        
        Args:
            agents: Agents
            roles: Roles
            
        Returns:
            Voting weights
        """
        weights = {}
        
        # Role-based weights
        role_weights = {
            TeamRole.LEADER: 2.0,
            TeamRole.COORDINATOR: 1.5,
            TeamRole.SPECIALIST: 1.5,
            TeamRole.SUPPORT: 1.0,
            TeamRole.SCOUT: 1.0,
            TeamRole.WORKER: 1.0
        }
        
        for agent_id in agents:
            role = roles.get(agent_id, TeamRole.WORKER)
            weights[agent_id] = role_weights.get(role, 1.0)
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            for agent_id in weights:
                weights[agent_id] /= total_weight
        
        return weights
    
    def _build_comm_graph(
        self,
        agents: List[int],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> nx.Graph:
        """Build communication graph
        
        Args:
            agents: Agents
            agent_states: Agent states
            
        Returns:
            Communication graph
        """
        graph = nx.Graph()
        
        # Add nodes
        for agent_id in agents:
            if agent_id in agent_states:
                pos = agent_states[agent_id].get('position', torch.zeros(3))
                graph.add_node(agent_id, position=pos)
        
        # Add edges based on communication range
        for i, agent_i in enumerate(agents):
            if agent_i not in agent_states:
                continue
            
            pos_i = agent_states[agent_i].get('position', torch.zeros(3))
            
            for agent_j in agents[i+1:]:
                if agent_j not in agent_states:
                    continue
                
                pos_j = agent_states[agent_j].get('position', torch.zeros(3))
                distance = torch.norm(pos_i - pos_j).item()
                
                if distance <= self.communication_range:
                    # Add edge with signal strength
                    signal_strength = 1.0 - (distance / self.communication_range)
                    graph.add_edge(
                        agent_i, agent_j,
                        distance=distance,
                        signal_strength=signal_strength
                    )
        
        return graph


class SynchronizationManager:
    """Manages synchronization between agents"""
    
    def __init__(self, num_agents: int):
        """Initialize synchronization manager
        
        Args:
            num_agents: Number of agents
        """
        self.num_agents = num_agents
        
        # Sync prediction network
        self.sync_net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Sync importance
        )
    
    def identify_sync_points(
        self,
        teams: Dict[str, List[int]],
        tasks: List[Any],
        coordination_plan: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Identify synchronization points
        
        Args:
            teams: Team assignments
            tasks: Tasks
            coordination_plan: Coordination plan
            
        Returns:
            Synchronization points
        """
        sync_points = {}
        
        for team_id, team_agents in teams.items():
            team_sync_points = []
            
            # Task-based sync points
            task_syncs = self._identify_task_syncs(tasks)
            team_sync_points.extend(task_syncs)
            
            # Time-based sync points
            time_syncs = self._identify_time_syncs(
                team_agents, coordination_plan.get(team_id, {})
            )
            team_sync_points.extend(time_syncs)
            
            # Critical event sync points
            event_syncs = self._identify_event_syncs(tasks)
            team_sync_points.extend(event_syncs)
            
            sync_points[team_id] = team_sync_points
        
        return sync_points
    
    def _identify_task_syncs(
        self,
        tasks: List[Any]
    ) -> List[Dict[str, Any]]:
        """Identify task-based sync points
        
        Args:
            tasks: Tasks
            
        Returns:
            Task sync points
        """
        sync_points = []
        
        # Sync at task dependencies
        for task in tasks:
            deps = getattr(task, 'dependencies', [])
            if deps:
                sync_points.append({
                    'type': 'task_dependency',
                    'trigger': f"complete_{task.subtask_id}",
                    'wait_for': deps,
                    'timeout': 300.0,
                    'priority': 'high'
                })
        
        # Sync at multi-agent tasks
        for task in tasks:
            if getattr(task, 'required_agents', 1) > 1:
                sync_points.append({
                    'type': 'multi_agent_coordination',
                    'trigger': f"start_{task.subtask_id}",
                    'participants': 'assigned_agents',
                    'sync_data': ['position', 'readiness'],
                    'priority': 'critical'
                })
        
        return sync_points
    
    def _identify_time_syncs(
        self,
        agents: List[int],
        team_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify time-based sync points
        
        Args:
            agents: Team agents
            team_plan: Team coordination plan
            
        Returns:
            Time sync points
        """
        sync_points = []
        
        # Regular sync intervals
        protocol = team_plan.get('protocol', 'distributed')
        
        if protocol == 'centralized':
            interval = 60.0  # 1 minute
        elif protocol == 'hierarchical':
            interval = 120.0  # 2 minutes
        else:
            interval = 300.0  # 5 minutes
        
        sync_points.append({
            'type': 'periodic',
            'interval': interval,
            'participants': agents,
            'sync_data': ['status', 'position', 'energy'],
            'priority': 'normal'
        })
        
        return sync_points
    
    def _identify_event_syncs(
        self,
        tasks: List[Any]
    ) -> List[Dict[str, Any]]:
        """Identify event-based sync points
        
        Args:
            tasks: Tasks
            
        Returns:
            Event sync points
        """
        sync_points = []
        
        # Critical events requiring sync
        critical_events = [
            ('target_detected', ['position', 'target_info']),
            ('emergency', ['position', 'status', 'assistance_needed']),
            ('task_completed', ['results', 'next_task']),
            ('energy_critical', ['soc', 'position', 'eta_base'])
        ]
        
        for event, sync_data in critical_events:
            sync_points.append({
                'type': 'event_triggered',
                'event': event,
                'broadcast': True,
                'sync_data': sync_data,
                'priority': 'critical' if 'emergency' in event else 'high'
            })
        
        return sync_points