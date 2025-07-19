"""Collaborative Energy Management for Multi-Agent Teams

This module implements collaborative energy management strategies that
optimize energy usage across the entire team.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class CollaborativeEnergyManager:
    """Manages energy resources across multi-agent team"""
    
    def __init__(
        self,
        num_agents: int,
        sharing_enabled: bool = True,
        optimization_interval: float = 10.0,  # seconds
        fairness_weight: float = 0.3
    ):
        """Initialize collaborative energy manager
        
        Args:
            num_agents: Number of agents in team
            sharing_enabled: Whether energy sharing is enabled
            optimization_interval: How often to reoptimize
            fairness_weight: Weight for fairness in allocation
        """
        self.num_agents = num_agents
        self.sharing_enabled = sharing_enabled
        self.optimization_interval = optimization_interval
        self.fairness_weight = fairness_weight
        
        # Energy sharing network
        self.sharing_network = EnergySharingNetwork(num_agents)
        
        # Team optimizer
        self.team_optimizer = TeamEnergyOptimizer(
            num_agents=num_agents,
            fairness_weight=fairness_weight
        )
        
        # Charging scheduler
        self.charging_scheduler = ChargingScheduler(num_agents)
        
        # Energy balancer
        self.energy_balancer = EnergyBalancer(num_agents)
        
        # Metrics tracking
        self.metrics = {
            'total_energy': [],
            'energy_variance': [],
            'sharing_events': 0,
            'critical_events': 0
        }
        
        logger.info(f"Initialized CollaborativeEnergyManager for {num_agents} agents")
    
    def update(
        self,
        agent_states: Dict[int, Dict[str, torch.Tensor]],
        tasks: Dict[int, Any],
        dt: float = 0.1
    ) -> Dict[int, Dict[str, Any]]:
        """Update collaborative energy management
        
        Args:
            agent_states: Energy states for each agent
            tasks: Current tasks for each agent
            dt: Time step
            
        Returns:
            Energy management decisions for each agent
        """
        # Collect team energy state
        team_state = self._collect_team_state(agent_states)
        
        # Check for critical situations
        critical_agents = self._identify_critical_agents(team_state)
        
        decisions = {}
        
        # Energy sharing decisions
        if self.sharing_enabled and critical_agents:
            sharing_plan = self.sharing_network.plan_energy_transfer(
                team_state, critical_agents
            )
            
            for agent_id in range(self.num_agents):
                decisions[agent_id] = {'sharing': sharing_plan.get(agent_id, {})}
        
        # Team-level optimization
        optimization = self.team_optimizer.optimize_allocation(
            team_state, tasks
        )
        
        for agent_id, opt in optimization.items():
            if agent_id not in decisions:
                decisions[agent_id] = {}
            decisions[agent_id]['power_limit'] = opt['power_limit']
            decisions[agent_id]['priority'] = opt['priority']
        
        # Charging scheduling
        charging_schedule = self.charging_scheduler.schedule(
            team_state, tasks
        )
        
        for agent_id, schedule in charging_schedule.items():
            decisions[agent_id]['charging'] = schedule
        
        # Energy balancing
        balancing = self.energy_balancer.balance_team_energy(
            team_state, tasks
        )
        
        for agent_id, balance in balancing.items():
            decisions[agent_id]['role_adjustment'] = balance
        
        # Update metrics
        self._update_metrics(team_state, decisions)
        
        return decisions
    
    def _collect_team_state(
        self,
        agent_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Collect team-wide energy state
        
        Args:
            agent_states: Individual agent states
            
        Returns:
            Team state
        """
        socs = []
        temperatures = []
        power_consumptions = []
        
        for agent_id in range(self.num_agents):
            if agent_id in agent_states:
                state = agent_states[agent_id]
                socs.append(state.get('soc', torch.tensor(0.0)))
                temperatures.append(state.get('temperature', torch.tensor(25.0)))
                power_consumptions.append(state.get('power', torch.tensor(0.0)))
        
        team_state = {
            'socs': torch.stack(socs),
            'temperatures': torch.stack(temperatures),
            'power_consumptions': torch.stack(power_consumptions),
            'mean_soc': torch.stack(socs).mean(),
            'min_soc': torch.stack(socs).min(),
            'max_soc': torch.stack(socs).max(),
            'soc_variance': torch.stack(socs).var()
        }
        
        return team_state
    
    def _identify_critical_agents(
        self,
        team_state: Dict[str, torch.Tensor]
    ) -> List[int]:
        """Identify agents in critical energy state
        
        Args:
            team_state: Team energy state
            
        Returns:
            List of critical agent IDs
        """
        critical_threshold = 0.2  # 20% SOC
        
        critical_agents = []
        socs = team_state['socs']
        
        for i, soc in enumerate(socs):
            if soc < critical_threshold:
                critical_agents.append(i)
        
        if critical_agents:
            self.metrics['critical_events'] += len(critical_agents)
        
        return critical_agents
    
    def _update_metrics(
        self,
        team_state: Dict[str, torch.Tensor],
        decisions: Dict[int, Dict[str, Any]]
    ):
        """Update performance metrics
        
        Args:
            team_state: Team state
            decisions: Management decisions
        """
        self.metrics['total_energy'].append(
            team_state['socs'].sum().item()
        )
        self.metrics['energy_variance'].append(
            team_state['soc_variance'].item()
        )
        
        # Count sharing events
        for agent_decisions in decisions.values():
            if 'sharing' in agent_decisions and agent_decisions['sharing']:
                self.metrics['sharing_events'] += 1


class EnergySharingNetwork(nn.Module):
    """Neural network for energy sharing decisions"""
    
    def __init__(
        self,
        num_agents: int,
        hidden_dim: int = 64,
        sharing_radius: float = 50.0  # meters
    ):
        """Initialize energy sharing network
        
        Args:
            num_agents: Number of agents
            hidden_dim: Hidden dimension
            sharing_radius: Maximum sharing distance
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.sharing_radius = sharing_radius
        
        # Graph neural network for sharing decisions
        self.gnn = EnergyShareGNN(
            node_features=5,  # soc, power, temp, x, y
            edge_features=2,  # distance, relative_soc
            hidden_dim=hidden_dim
        )
        
        # Sharing amount predictor
        self.share_amount = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Share fraction 0-1
        )
    
    def plan_energy_transfer(
        self,
        team_state: Dict[str, torch.Tensor],
        critical_agents: List[int],
        positions: Optional[torch.Tensor] = None
    ) -> Dict[int, Dict[str, float]]:
        """Plan energy transfers between agents
        
        Args:
            team_state: Team energy state
            critical_agents: Agents needing energy
            positions: Agent positions
            
        Returns:
            Energy transfer plan
        """
        if not critical_agents:
            return {}
        
        # Create graph representation
        graph_data = self._create_energy_graph(
            team_state, positions
        )
        
        # Get sharing decisions from GNN
        node_embeddings = self.gnn(graph_data)
        
        transfer_plan = {}
        
        # For each critical agent, find donors
        for receiver_id in critical_agents:
            receiver_soc = team_state['socs'][receiver_id]
            
            potential_donors = []
            
            for donor_id in range(self.num_agents):
                if donor_id == receiver_id:
                    continue
                
                donor_soc = team_state['socs'][donor_id]
                
                # Only consider donors with sufficient energy
                if donor_soc > receiver_soc + 0.2:  # At least 20% more
                    # Check distance if positions provided
                    if positions is not None:
                        dist = torch.norm(
                            positions[donor_id] - positions[receiver_id]
                        )
                        if dist > self.sharing_radius:
                            continue
                    
                    # Compute sharing amount
                    donor_embed = node_embeddings[donor_id]
                    receiver_embed = node_embeddings[receiver_id]
                    
                    pair_embed = torch.cat([donor_embed, receiver_embed])
                    share_fraction = self.share_amount(pair_embed)
                    
                    # Actual amount to share
                    max_share = (donor_soc - receiver_soc) * 0.5
                    share_amount = share_fraction * max_share
                    
                    potential_donors.append({
                        'donor_id': donor_id,
                        'amount': share_amount.item(),
                        'donor_soc': donor_soc.item()
                    })
            
            # Select best donor
            if potential_donors:
                best_donor = max(potential_donors, key=lambda x: x['amount'])
                
                if receiver_id not in transfer_plan:
                    transfer_plan[receiver_id] = {}
                
                transfer_plan[receiver_id] = {
                    'receive_from': best_donor['donor_id'],
                    'amount': best_donor['amount']
                }
                
                if best_donor['donor_id'] not in transfer_plan:
                    transfer_plan[best_donor['donor_id']] = {}
                
                transfer_plan[best_donor['donor_id']] = {
                    'send_to': receiver_id,
                    'amount': best_donor['amount']
                }
        
        return transfer_plan
    
    def _create_energy_graph(
        self,
        team_state: Dict[str, torch.Tensor],
        positions: Optional[torch.Tensor] = None
    ) -> Data:
        """Create graph representation of team energy state
        
        Args:
            team_state: Team energy state
            positions: Agent positions
            
        Returns:
            Graph data
        """
        num_agents = len(team_state['socs'])
        
        # Node features
        if positions is None:
            positions = torch.randn(num_agents, 2) * 50  # Random positions
        
        node_features = torch.stack([
            team_state['socs'],
            team_state['power_consumptions'],
            team_state['temperatures'],
            positions[:, 0],
            positions[:, 1]
        ], dim=1)
        
        # Create edges (fully connected for now)
        edge_index = []
        edge_attr = []
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    edge_index.append([i, j])
                    
                    # Edge features
                    distance = torch.norm(positions[i] - positions[j])
                    soc_diff = team_state['socs'][i] - team_state['socs'][j]
                    
                    edge_attr.append([distance, soc_diff])
        
        edge_index = torch.tensor(edge_index).T
        edge_attr = torch.tensor(edge_attr)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )


class EnergyShareGNN(MessagePassing):
    """Graph neural network for energy sharing"""
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 64
    ):
        """Initialize GNN
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension
        """
        super().__init__(aggr='mean')
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass
        
        Args:
            data: Graph data
            
        Returns:
            Node embeddings
        """
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # Message passing
        x = self.propagate(data.edge_index, x=x, edge_attr=edge_attr)
        
        return x
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Compute messages
        
        Args:
            x_i: Target node features
            x_j: Source node features
            edge_attr: Edge features
            
        Returns:
            Messages
        """
        return self.message_net(torch.cat([x_i, x_j, edge_attr], dim=-1))
    
    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Update node features
        
        Args:
            aggr_out: Aggregated messages
            x: Current node features
            
        Returns:
            Updated features
        """
        return self.update_net(torch.cat([x, aggr_out], dim=-1))


class ChargingScheduler:
    """Schedules charging for multi-agent team"""
    
    def __init__(
        self,
        num_agents: int,
        num_charging_stations: int = 2,
        charge_rate: float = 100.0  # W
    ):
        """Initialize charging scheduler
        
        Args:
            num_agents: Number of agents
            num_charging_stations: Number of charging stations
            charge_rate: Charging power in Watts
        """
        self.num_agents = num_agents
        self.num_charging_stations = num_charging_stations
        self.charge_rate = charge_rate
        
        # Scheduling network
        self.scheduler_net = nn.Sequential(
            nn.Linear(num_agents * 3 + num_charging_stations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents)  # Priority scores
        )
    
    def schedule(
        self,
        team_state: Dict[str, torch.Tensor],
        tasks: Dict[int, Any],
        station_availability: Optional[torch.Tensor] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Create charging schedule
        
        Args:
            team_state: Team energy state
            tasks: Current tasks
            station_availability: Station availability (0-1)
            
        Returns:
            Charging schedule for each agent
        """
        if station_availability is None:
            station_availability = torch.ones(self.num_charging_stations)
        
        # Compute charging priorities
        priorities = self._compute_charging_priority(
            team_state, tasks, station_availability
        )
        
        # Sort agents by priority
        sorted_agents = torch.argsort(priorities, descending=True)
        
        schedule = {}
        assigned_stations = []
        
        for agent_id in sorted_agents:
            agent_id = agent_id.item()
            
            # Check if agent needs charging
            if team_state['socs'][agent_id] < 0.8:  # Below 80%
                # Find available station
                for station_id in range(self.num_charging_stations):
                    if (station_id not in assigned_stations and
                        station_availability[station_id] > 0.5):
                        
                        # Assign to station
                        schedule[agent_id] = {
                            'charge': True,
                            'station_id': station_id,
                            'priority': priorities[agent_id].item(),
                            'estimated_time': self._estimate_charge_time(
                                team_state['socs'][agent_id]
                            )
                        }
                        
                        assigned_stations.append(station_id)
                        break
                else:
                    # No station available, queue
                    schedule[agent_id] = {
                        'charge': False,
                        'queued': True,
                        'priority': priorities[agent_id].item()
                    }
            else:
                # Doesn't need charging
                schedule[agent_id] = {
                    'charge': False,
                    'queued': False
                }
        
        return schedule
    
    def _compute_charging_priority(
        self,
        team_state: Dict[str, torch.Tensor],
        tasks: Dict[int, Any],
        station_availability: torch.Tensor
    ) -> torch.Tensor:
        """Compute charging priority for each agent
        
        Args:
            team_state: Team state
            tasks: Current tasks
            station_availability: Station availability
            
        Returns:
            Priority scores
        """
        # Simple priority based on SOC and task status
        priorities = torch.zeros(self.num_agents)
        
        for i in range(self.num_agents):
            soc = team_state['socs'][i]
            
            # Base priority inversely proportional to SOC
            priority = 1.0 - soc
            
            # Boost priority if no active task
            if i not in tasks or tasks[i] is None:
                priority *= 1.5
            
            # Critical boost
            if soc < 0.2:
                priority *= 2.0
            
            priorities[i] = priority
        
        # Use neural network for more complex scheduling
        features = torch.cat([
            team_state['socs'],
            team_state['power_consumptions'],
            team_state['temperatures'],
            station_availability
        ])
        
        nn_priorities = torch.sigmoid(self.scheduler_net(features))
        
        # Combine heuristic and learned priorities
        final_priorities = 0.7 * priorities + 0.3 * nn_priorities
        
        return final_priorities
    
    def _estimate_charge_time(self, current_soc: torch.Tensor) -> float:
        """Estimate time to full charge
        
        Args:
            current_soc: Current state of charge
            
        Returns:
            Estimated minutes to full charge
        """
        # Assume 18.5 Wh capacity
        capacity = 18.5
        energy_needed = (1.0 - current_soc) * capacity
        
        # Time = Energy / Power
        time_hours = energy_needed / self.charge_rate
        time_minutes = time_hours * 60
        
        return time_minutes.item()


class TeamEnergyOptimizer:
    """Optimizes energy allocation across team"""
    
    def __init__(
        self,
        num_agents: int,
        fairness_weight: float = 0.3
    ):
        """Initialize team optimizer
        
        Args:
            num_agents: Number of agents
            fairness_weight: Weight for fairness objective
        """
        self.num_agents = num_agents
        self.fairness_weight = fairness_weight
        
        # Allocation network
        self.allocation_net = nn.Sequential(
            nn.Linear(num_agents * 4, 128),  # SOC, power, task priority, position
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents * 2)  # Power limit and priority for each
        )
    
    def optimize_allocation(
        self,
        team_state: Dict[str, torch.Tensor],
        tasks: Dict[int, Any]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Optimize energy allocation
        
        Args:
            team_state: Team energy state
            tasks: Current tasks
            
        Returns:
            Allocation for each agent
        """
        # Compute task priorities
        task_priorities = self._compute_task_priorities(tasks)
        
        # Total available energy
        total_energy = team_state['socs'].sum() * 18.5  # Wh
        
        # Optimization objectives
        allocations = {}
        
        # Neural network allocation
        features = torch.cat([
            team_state['socs'],
            team_state['power_consumptions'],
            task_priorities,
            torch.randn(self.num_agents)  # Dummy positions
        ])
        
        nn_output = self.allocation_net(features)
        power_limits = nn_output[:self.num_agents]
        priorities = nn_output[self.num_agents:]
        
        # Apply fairness constraints
        if self.fairness_weight > 0:
            power_limits = self._apply_fairness(
                power_limits, team_state['socs']
            )
        
        # Create allocation dict
        for i in range(self.num_agents):
            allocations[i] = {
                'power_limit': F.softplus(power_limits[i]) * 100,  # 0-100W
                'priority': torch.sigmoid(priorities[i]),
                'task_priority': task_priorities[i]
            }
        
        return allocations
    
    def _compute_task_priorities(
        self,
        tasks: Dict[int, Any]
    ) -> torch.Tensor:
        """Compute task priorities
        
        Args:
            tasks: Current tasks
            
        Returns:
            Task priorities
        """
        priorities = torch.zeros(self.num_agents)
        
        for agent_id, task in tasks.items():
            if task is not None:
                # Simple priority based on task type
                if hasattr(task, 'priority'):
                    priorities[agent_id] = task.priority
                else:
                    priorities[agent_id] = 0.5
        
        return priorities
    
    def _apply_fairness(
        self,
        power_limits: torch.Tensor,
        socs: torch.Tensor
    ) -> torch.Tensor:
        """Apply fairness constraints to power allocation
        
        Args:
            power_limits: Proposed power limits
            socs: State of charge for each agent
            
        Returns:
            Fair power limits
        """
        # Agents with lower SOC get priority
        soc_weights = 1.0 - socs
        soc_weights = soc_weights / soc_weights.sum()
        
        # Weighted average
        fair_limits = power_limits * (1 - self.fairness_weight)
        fair_limits = fair_limits + soc_weights * power_limits.sum() * self.fairness_weight
        
        return fair_limits


class EnergyBalancer:
    """Balances energy consumption across team"""
    
    def __init__(self, num_agents: int):
        """Initialize energy balancer
        
        Args:
            num_agents: Number of agents
        """
        self.num_agents = num_agents
        
        # Role adjustment network
        self.role_net = nn.Sequential(
            nn.Linear(num_agents * 3 + 1, 64),  # Team state + variance
            nn.ReLU(),
            nn.Linear(64, num_agents * 3)  # Role adjustments
        )
    
    def balance_team_energy(
        self,
        team_state: Dict[str, torch.Tensor],
        tasks: Dict[int, Any]
    ) -> Dict[int, Dict[str, Any]]:
        """Balance energy across team
        
        Args:
            team_state: Team energy state
            tasks: Current tasks
            
        Returns:
            Role adjustments for each agent
        """
        # Compute energy variance
        variance = team_state['soc_variance']
        
        # High variance triggers rebalancing
        if variance > 0.1:  # Threshold
            adjustments = self._compute_role_adjustments(
                team_state, tasks, variance
            )
        else:
            # No adjustment needed
            adjustments = {
                i: {'adjust_role': False}
                for i in range(self.num_agents)
            }
        
        return adjustments
    
    def _compute_role_adjustments(
        self,
        team_state: Dict[str, torch.Tensor],
        tasks: Dict[int, Any],
        variance: torch.Tensor
    ) -> Dict[int, Dict[str, Any]]:
        """Compute role adjustments to balance energy
        
        Args:
            team_state: Team state
            tasks: Current tasks
            variance: SOC variance
            
        Returns:
            Role adjustments
        """
        # Neural network features
        features = torch.cat([
            team_state['socs'],
            team_state['power_consumptions'],
            team_state['temperatures'],
            variance.unsqueeze(0)
        ])
        
        # Get role adjustments
        nn_output = self.role_net(features).view(self.num_agents, 3)
        
        adjustments = {}
        
        for i in range(self.num_agents):
            soc = team_state['socs'][i]
            
            # Low SOC agents get lighter roles
            if soc < team_state['mean_soc'] - 0.1:
                role_change = 'reduce_workload'
                factor = torch.sigmoid(nn_output[i, 0])
            # High SOC agents can take more work
            elif soc > team_state['mean_soc'] + 0.1:
                role_change = 'increase_workload'
                factor = torch.sigmoid(nn_output[i, 1])
            else:
                role_change = 'maintain'
                factor = torch.sigmoid(nn_output[i, 2])
            
            adjustments[i] = {
                'adjust_role': True,
                'role_change': role_change,
                'adjustment_factor': factor.item(),
                'current_soc': soc.item(),
                'team_mean_soc': team_state['mean_soc'].item()
            }
        
        return adjustments