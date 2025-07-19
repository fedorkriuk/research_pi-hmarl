"""Task Assignment Optimization

This module implements various algorithms for optimal task assignment
to agents based on capabilities, availability, and constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
import pulp
import logging

logger = logging.getLogger(__name__)


class AssignmentOptimizer:
    """Optimizes task assignments to agents"""
    
    def __init__(
        self,
        num_agents: int,
        optimization_method: str = "hungarian",
        consider_energy: bool = True,
        consider_capabilities: bool = True
    ):
        """Initialize assignment optimizer
        
        Args:
            num_agents: Number of agents
            optimization_method: Optimization algorithm
            consider_energy: Whether to consider energy constraints
            consider_capabilities: Whether to consider capability matching
        """
        self.num_agents = num_agents
        self.optimization_method = optimization_method
        self.consider_energy = consider_energy
        self.consider_capabilities = consider_capabilities
        
        # Components
        self.capability_matcher = CapabilityMatcher()
        self.load_balancer = LoadBalancer(num_agents)
        
        # Assignment methods
        self.assignment_methods = {
            'hungarian': HungarianAssignment(),
            'auction': AuctionBasedAssignment(num_agents),
            'greedy': self._greedy_assignment,
            'ilp': self._ilp_assignment
        }
        
        # Assignment history
        self.assignment_history = []
        
        logger.info(f"Initialized AssignmentOptimizer with {optimization_method} method")
    
    def optimize_assignment(
        self,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[int, List[str]]:
        """Optimize task assignment to agents
        
        Args:
            tasks: List of tasks/subtasks to assign
            agent_states: Current state of each agent
            constraints: Assignment constraints
            
        Returns:
            Agent ID to task ID mapping
        """
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(tasks, agent_states)
        
        # Apply capability constraints
        if self.consider_capabilities:
            capability_matrix = self.capability_matcher.match_matrix(
                tasks, agent_states
            )
            # Set infeasible assignments to high cost
            cost_matrix = torch.where(
                capability_matrix > 0,
                cost_matrix,
                torch.tensor(1e6)
            )
        
        # Apply energy constraints
        if self.consider_energy:
            energy_feasible = self._check_energy_feasibility(
                tasks, agent_states
            )
            cost_matrix = torch.where(
                energy_feasible,
                cost_matrix,
                torch.tensor(1e6)
            )
        
        # Get assignment method
        method = self.assignment_methods.get(
            self.optimization_method,
            self._greedy_assignment
        )
        
        # Optimize assignment
        if self.optimization_method in ['hungarian', 'auction']:
            assignment = method.assign(cost_matrix, tasks, agent_states)
        else:
            assignment = method(cost_matrix, tasks, agent_states)
        
        # Apply load balancing
        assignment = self.load_balancer.balance(
            assignment, tasks, agent_states
        )
        
        # Validate assignment
        assignment = self._validate_assignment(
            assignment, tasks, agent_states, constraints
        )
        
        # Record assignment
        self.assignment_history.append({
            'assignment': assignment,
            'cost': self._compute_assignment_cost(assignment, cost_matrix),
            'timestamp': torch.tensor(len(self.assignment_history))
        })
        
        return assignment
    
    def _build_cost_matrix(
        self,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Build cost matrix for assignment
        
        Args:
            tasks: Tasks to assign
            agent_states: Agent states
            
        Returns:
            Cost matrix [agents x tasks]
        """
        num_tasks = len(tasks)
        cost_matrix = torch.zeros(self.num_agents, num_tasks)
        
        for i in range(self.num_agents):
            if i not in agent_states:
                # Agent not available
                cost_matrix[i, :] = 1e6
                continue
            
            agent_state = agent_states[i]
            
            for j, task in enumerate(tasks):
                # Distance cost
                if hasattr(task, 'location') and task.location is not None:
                    agent_pos = agent_state.get('position', torch.zeros(2))
                    distance = torch.norm(task.location[:2] - agent_pos[:2])
                    distance_cost = distance / 1000.0  # Normalize
                else:
                    distance_cost = 0.0
                
                # Energy cost
                if self.consider_energy:
                    task_energy = getattr(task, 'estimated_energy', 5.0)
                    agent_energy = agent_state['soc'] * 18.5  # Wh
                    energy_cost = task_energy / (agent_energy + 1e-6)
                else:
                    energy_cost = 0.0
                
                # Capability mismatch cost
                if self.consider_capabilities:
                    capability_cost = self.capability_matcher.mismatch_cost(
                        task, agent_state
                    )
                else:
                    capability_cost = 0.0
                
                # Workload cost
                current_tasks = agent_state.get('assigned_tasks', 0)
                workload_cost = current_tasks * 0.1
                
                # Total cost
                cost_matrix[i, j] = (
                    distance_cost * 0.3 +
                    energy_cost * 0.3 +
                    capability_cost * 0.3 +
                    workload_cost * 0.1
                )
        
        return cost_matrix
    
    def _check_energy_feasibility(
        self,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Check energy feasibility for assignments
        
        Args:
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Feasibility matrix
        """
        num_tasks = len(tasks)
        feasible = torch.ones(self.num_agents, num_tasks, dtype=torch.bool)
        
        for i in range(self.num_agents):
            if i not in agent_states:
                feasible[i, :] = False
                continue
            
            agent_energy = agent_states[i]['soc'] * 18.5  # Wh
            
            for j, task in enumerate(tasks):
                # Task energy requirement
                task_energy = getattr(task, 'estimated_energy', 5.0)
                
                # Return energy
                if hasattr(task, 'location') and task.location is not None:
                    agent_pos = agent_states[i].get('position', torch.zeros(2))
                    return_distance = torch.norm(task.location[:2] - agent_pos[:2])
                    return_energy = return_distance / 1000.0 * 0.1  # Simplified
                else:
                    return_energy = 2.0
                
                # Total energy needed
                total_energy = task_energy + return_energy + 2.0  # Reserve
                
                feasible[i, j] = agent_energy > total_energy
        
        return feasible
    
    def _greedy_assignment(
        self,
        cost_matrix: torch.Tensor,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, List[str]]:
        """Greedy task assignment
        
        Args:
            cost_matrix: Cost matrix
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Assignment
        """
        assignment = {i: [] for i in range(self.num_agents)}
        assigned_tasks = set()
        
        # Flatten and sort by cost
        costs = []
        for i in range(self.num_agents):
            for j in range(len(tasks)):
                if cost_matrix[i, j] < 1e5:  # Feasible
                    costs.append((cost_matrix[i, j].item(), i, j))
        
        costs.sort(key=lambda x: x[0])
        
        # Assign greedily
        for cost, agent_id, task_idx in costs:
            if task_idx not in assigned_tasks:
                task_id = getattr(tasks[task_idx], 'subtask_id', str(task_idx))
                assignment[agent_id].append(task_id)
                assigned_tasks.add(task_idx)
        
        return assignment
    
    def _ilp_assignment(
        self,
        cost_matrix: torch.Tensor,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, List[str]]:
        """Integer Linear Programming assignment
        
        Args:
            cost_matrix: Cost matrix
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Assignment
        """
        num_tasks = len(tasks)
        
        # Create ILP problem
        prob = pulp.LpProblem("TaskAssignment", pulp.LpMinimize)
        
        # Decision variables
        x = {}
        for i in range(self.num_agents):
            for j in range(num_tasks):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
        
        # Objective: minimize total cost
        prob += pulp.lpSum([
            cost_matrix[i, j].item() * x[i, j]
            for i in range(self.num_agents)
            for j in range(num_tasks)
        ])
        
        # Constraints: each task assigned to exactly one agent
        for j in range(num_tasks):
            prob += pulp.lpSum([x[i, j] for i in range(self.num_agents)]) == 1
        
        # Constraints: agent capacity
        for i in range(self.num_agents):
            if i in agent_states:
                capacity = agent_states[i].get('capacity', 3)
            else:
                capacity = 0
            prob += pulp.lpSum([x[i, j] for j in range(num_tasks)]) <= capacity
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract assignment
        assignment = {i: [] for i in range(self.num_agents)}
        
        if prob.status == pulp.LpStatusOptimal:
            for i in range(self.num_agents):
                for j in range(num_tasks):
                    if x[i, j].varValue > 0.5:
                        task_id = getattr(tasks[j], 'subtask_id', str(j))
                        assignment[i].append(task_id)
        
        return assignment
    
    def _validate_assignment(
        self,
        assignment: Dict[int, List[str]],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[int, List[str]]:
        """Validate and fix assignment
        
        Args:
            assignment: Proposed assignment
            tasks: Tasks
            agent_states: Agent states
            constraints: Constraints
            
        Returns:
            Valid assignment
        """
        # Check all tasks are assigned
        assigned_tasks = set()
        for agent_tasks in assignment.values():
            assigned_tasks.update(agent_tasks)
        
        all_task_ids = {getattr(t, 'subtask_id', str(i)) for i, t in enumerate(tasks)}
        unassigned = all_task_ids - assigned_tasks
        
        # Assign unassigned tasks to least loaded agents
        if unassigned:
            agent_loads = {
                i: len(assignment.get(i, []))
                for i in range(self.num_agents)
                if i in agent_states
            }
            
            for task_id in unassigned:
                if agent_loads:
                    min_load_agent = min(agent_loads, key=agent_loads.get)
                    if min_load_agent not in assignment:
                        assignment[min_load_agent] = []
                    assignment[min_load_agent].append(task_id)
                    agent_loads[min_load_agent] += 1
        
        return assignment
    
    def _compute_assignment_cost(
        self,
        assignment: Dict[int, List[str]],
        cost_matrix: torch.Tensor
    ) -> float:
        """Compute total cost of assignment
        
        Args:
            assignment: Assignment
            cost_matrix: Cost matrix
            
        Returns:
            Total cost
        """
        total_cost = 0.0
        
        task_id_to_idx = {}  # Map task IDs to indices
        
        for agent_id, task_ids in assignment.items():
            for task_id in task_ids:
                if task_id in task_id_to_idx:
                    task_idx = task_id_to_idx[task_id]
                    total_cost += cost_matrix[agent_id, task_idx].item()
        
        return total_cost


class CapabilityMatcher:
    """Matches agent capabilities to task requirements"""
    
    def __init__(self):
        """Initialize capability matcher"""
        # Capability embeddings
        self.capability_embedder = nn.Embedding(20, 16)  # 20 capability types
        
        # Capability names
        self.capabilities = [
            'navigation', 'camera', 'lidar', 'thermal_camera',
            'cargo', 'communication', 'computation', 'hover',
            'precision_landing', 'obstacle_avoidance', 'tracking',
            'detection', 'planning', 'coordination', 'stabilization',
            'anomaly_detection', 'manipulation', 'sensor_fusion',
            'long_range', 'high_speed'
        ]
        
        self.capability_to_idx = {
            cap: i for i, cap in enumerate(self.capabilities)
        }
    
    def match_matrix(
        self,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Build capability match matrix
        
        Args:
            tasks: Tasks with requirements
            agent_states: Agent states with capabilities
            
        Returns:
            Match matrix (1 = match, 0 = no match)
        """
        num_agents = len(agent_states)
        num_tasks = len(tasks)
        
        match_matrix = torch.zeros(num_agents, num_tasks)
        
        for i, agent_state in agent_states.items():
            agent_caps = set(agent_state.get('capabilities', []))
            
            for j, task in enumerate(tasks):
                task_caps = set(getattr(task, 'required_capabilities', []))
                
                # Check if agent has all required capabilities
                if task_caps.issubset(agent_caps):
                    match_matrix[i, j] = 1.0
        
        return match_matrix
    
    def mismatch_cost(
        self,
        task: Any,
        agent_state: Dict[str, torch.Tensor]
    ) -> float:
        """Compute capability mismatch cost
        
        Args:
            task: Task with requirements
            agent_state: Agent state with capabilities
            
        Returns:
            Mismatch cost
        """
        task_caps = set(getattr(task, 'required_capabilities', []))
        agent_caps = set(agent_state.get('capabilities', []))
        
        # Missing capabilities
        missing = task_caps - agent_caps
        
        if not missing:
            return 0.0
        
        # Cost based on number and importance of missing capabilities
        base_cost = len(missing) * 10.0
        
        # Additional cost for critical capabilities
        critical_caps = {'navigation', 'communication', 'obstacle_avoidance'}
        critical_missing = missing & critical_caps
        
        critical_cost = len(critical_missing) * 20.0
        
        return base_cost + critical_cost
    
    def get_capability_embedding(
        self,
        capabilities: List[str]
    ) -> torch.Tensor:
        """Get embedding for capability set
        
        Args:
            capabilities: List of capabilities
            
        Returns:
            Capability embedding
        """
        if not capabilities:
            return torch.zeros(16)
        
        # Convert to indices
        indices = [
            self.capability_to_idx.get(cap, 0)
            for cap in capabilities
        ]
        
        # Get embeddings and average
        embeddings = self.capability_embedder(torch.tensor(indices))
        
        return embeddings.mean(dim=0)


class LoadBalancer:
    """Balances workload across agents"""
    
    def __init__(self, num_agents: int):
        """Initialize load balancer
        
        Args:
            num_agents: Number of agents
        """
        self.num_agents = num_agents
        
        # Load tracking
        self.agent_loads = {i: 0.0 for i in range(num_agents)}
        self.load_history = []
    
    def balance(
        self,
        assignment: Dict[int, List[str]],
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, List[str]]:
        """Balance task assignment
        
        Args:
            assignment: Initial assignment
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Balanced assignment
        """
        # Compute current loads
        loads = self._compute_loads(assignment, tasks)
        
        # Check if balancing needed
        if not self._needs_balancing(loads):
            return assignment
        
        # Balance iteratively
        balanced = assignment.copy()
        
        for _ in range(10):  # Max iterations
            # Find most and least loaded agents
            max_agent = max(loads, key=loads.get)
            min_agent = min(loads, key=loads.get)
            
            if loads[max_agent] - loads[min_agent] < 0.2:
                break
            
            # Try to move a task
            if max_agent in balanced and balanced[max_agent]:
                # Find task to move
                task_to_move = self._select_task_to_move(
                    balanced[max_agent], tasks, agent_states[min_agent]
                )
                
                if task_to_move:
                    balanced[max_agent].remove(task_to_move)
                    if min_agent not in balanced:
                        balanced[min_agent] = []
                    balanced[min_agent].append(task_to_move)
                    
                    # Update loads
                    loads = self._compute_loads(balanced, tasks)
        
        # Update history
        self.load_history.append(loads)
        
        return balanced
    
    def _compute_loads(
        self,
        assignment: Dict[int, List[str]],
        tasks: List[Any]
    ) -> Dict[int, float]:
        """Compute agent loads
        
        Args:
            assignment: Task assignment
            tasks: Tasks
            
        Returns:
            Agent loads
        """
        loads = {i: 0.0 for i in range(self.num_agents)}
        
        # Create task lookup
        task_lookup = {
            getattr(t, 'subtask_id', str(i)): t
            for i, t in enumerate(tasks)
        }
        
        for agent_id, task_ids in assignment.items():
            for task_id in task_ids:
                if task_id in task_lookup:
                    task = task_lookup[task_id]
                    duration = getattr(task, 'estimated_duration', 300)
                    loads[agent_id] += duration / 3600.0  # Hours
        
        return loads
    
    def _needs_balancing(self, loads: Dict[int, float]) -> bool:
        """Check if load balancing is needed
        
        Args:
            loads: Current loads
            
        Returns:
            Whether balancing is needed
        """
        if not loads:
            return False
        
        load_values = list(loads.values())
        max_load = max(load_values)
        min_load = min(load_values)
        avg_load = np.mean(load_values)
        
        # Balance if difference is more than 50% of average
        return (max_load - min_load) > 0.5 * avg_load
    
    def _select_task_to_move(
        self,
        agent_tasks: List[str],
        all_tasks: List[Any],
        target_agent_state: Dict[str, torch.Tensor]
    ) -> Optional[str]:
        """Select task to move to another agent
        
        Args:
            agent_tasks: Tasks assigned to overloaded agent
            all_tasks: All tasks
            target_agent_state: Target agent state
            
        Returns:
            Task ID to move
        """
        # Create task lookup
        task_lookup = {
            getattr(t, 'subtask_id', str(i)): t
            for i, t in enumerate(all_tasks)
        }
        
        target_caps = set(target_agent_state.get('capabilities', []))
        
        # Find movable tasks
        movable = []
        
        for task_id in agent_tasks:
            if task_id in task_lookup:
                task = task_lookup[task_id]
                task_caps = set(getattr(task, 'required_capabilities', []))
                
                # Check if target agent can handle task
                if task_caps.issubset(target_caps):
                    duration = getattr(task, 'estimated_duration', 300)
                    movable.append((task_id, duration))
        
        if not movable:
            return None
        
        # Select task with median duration
        movable.sort(key=lambda x: x[1])
        return movable[len(movable) // 2][0]


class HungarianAssignment:
    """Hungarian algorithm for optimal assignment"""
    
    def __init__(self):
        """Initialize Hungarian assignment"""
        pass
    
    def assign(
        self,
        cost_matrix: torch.Tensor,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, List[str]]:
        """Assign tasks using Hungarian algorithm
        
        Args:
            cost_matrix: Cost matrix
            tasks: Tasks
            agent_states: Agent states
            
        Returns:
            Assignment
        """
        # Convert to numpy
        cost_np = cost_matrix.numpy()
        
        # Handle case where we have more agents than tasks
        num_agents, num_tasks = cost_np.shape
        
        if num_agents > num_tasks:
            # Pad with dummy tasks
            padding = num_agents - num_tasks
            dummy_costs = np.zeros((num_agents, padding))
            cost_np = np.hstack([cost_np, dummy_costs])
        
        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_np)
        
        # Build assignment
        assignment = {i: [] for i in range(num_agents)}
        
        for agent_idx, task_idx in zip(row_indices, col_indices):
            if task_idx < num_tasks:  # Not a dummy task
                task_id = getattr(tasks[task_idx], 'subtask_id', str(task_idx))
                assignment[agent_idx].append(task_id)
        
        return assignment


class AuctionBasedAssignment:
    """Market-based task assignment through auctions"""
    
    def __init__(
        self,
        num_agents: int,
        auction_rounds: int = 10
    ):
        """Initialize auction-based assignment
        
        Args:
            num_agents: Number of agents
            auction_rounds: Maximum auction rounds
        """
        self.num_agents = num_agents
        self.auction_rounds = auction_rounds
        
        # Bidding network
        self.bid_net = nn.Sequential(
            nn.Linear(10, 32),  # Features: task + agent state
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Bid value
        )
    
    def assign(
        self,
        cost_matrix: torch.Tensor,
        tasks: List[Any],
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, List[str]]:
        """Assign tasks through auction
        
        Args:
            cost_matrix: Cost matrix (used for initial valuations)
            tasks: Tasks to auction
            agent_states: Agent states
            
        Returns:
            Assignment
        """
        assignment = {i: [] for i in range(self.num_agents)}
        unassigned_tasks = list(range(len(tasks)))
        
        # Agent budgets
        budgets = {
            i: agent_states[i].get('budget', 100.0)
            for i in agent_states
        }
        
        # Run auction rounds
        for round_num in range(self.auction_rounds):
            if not unassigned_tasks:
                break
            
            # Auction each unassigned task
            tasks_to_remove = []
            
            for task_idx in unassigned_tasks:
                task = tasks[task_idx]
                
                # Collect bids
                bids = {}
                
                for agent_id in agent_states:
                    if budgets.get(agent_id, 0) > 0:
                        bid = self._compute_bid(
                            agent_id, task, agent_states[agent_id],
                            cost_matrix[agent_id, task_idx]
                        )
                        
                        # Ensure bid is within budget
                        bid = min(bid, budgets[agent_id])
                        
                        if bid > 0:
                            bids[agent_id] = bid
                
                # Award to highest bidder
                if bids:
                    winner = max(bids, key=bids.get)
                    winning_bid = bids[winner]
                    
                    # Update assignment
                    task_id = getattr(task, 'subtask_id', str(task_idx))
                    assignment[winner].append(task_id)
                    
                    # Update budget
                    budgets[winner] -= winning_bid
                    
                    # Mark task as assigned
                    tasks_to_remove.append(task_idx)
            
            # Remove assigned tasks
            for task_idx in tasks_to_remove:
                unassigned_tasks.remove(task_idx)
        
        return assignment
    
    def _compute_bid(
        self,
        agent_id: int,
        task: Any,
        agent_state: Dict[str, torch.Tensor],
        base_cost: torch.Tensor
    ) -> float:
        """Compute agent's bid for task
        
        Args:
            agent_id: Agent ID
            task: Task to bid on
            agent_state: Agent state
            base_cost: Base cost from cost matrix
            
        Returns:
            Bid amount
        """
        # Extract features
        features = []
        
        # Task features
        features.append(getattr(task, 'priority', 0.5))
        features.append(getattr(task, 'estimated_duration', 300) / 1000)
        features.append(len(getattr(task, 'required_capabilities', [])) / 5)
        
        # Agent features
        features.append(agent_state.get('soc', 0.5))
        features.append(len(agent_state.get('capabilities', [])) / 10)
        features.append(agent_state.get('workload', 0.0))
        
        # Cost feature
        features.append(1.0 / (base_cost.item() + 1.0))
        
        # Pad features
        while len(features) < 10:
            features.append(0.0)
        
        features_tensor = torch.tensor(features[:10])
        
        # Compute bid with neural network
        bid_value = torch.sigmoid(self.bid_net(features_tensor)) * 50.0
        
        # Adjust based on task priority
        priority_factor = 1.0 + getattr(task, 'priority', 0.5)
        bid_value = bid_value * priority_factor
        
        return bid_value.item()