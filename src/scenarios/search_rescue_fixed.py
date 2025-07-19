"""GENIUS FIX: Enhanced Search and Rescue Scenario with Proper Success Criteria
This module fixes the success criteria and coordination mechanisms.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict
import heapq

# Import original components
from search_rescue import (
    VictimStatus, AgentRole, VictimModel, SearchPattern,
    SearchAgent, RescueCoordinator, RescueMission
)

logger = logging.getLogger(__name__)


class SearchRescueScenarioFixed:
    """Fixed Search and Rescue Scenario with proper success criteria"""
    
    def __init__(
        self,
        area_size: Tuple[float, float] = (100.0, 100.0),
        num_victims: int = 10,
        num_agents: int = 6,
        obstacle_density: float = 0.1,
        success_threshold: float = 0.85,  # 85% rescue rate for success
        min_agents_for_rescue: int = 2  # Coordination requirement
    ):
        """Initialize fixed search and rescue scenario"""
        
        # Original parameters
        self.area_size = area_size
        self.num_victims = num_victims
        self.num_agents = num_agents
        self.obstacle_density = obstacle_density
        
        # Fixed parameters
        self.success_threshold = success_threshold
        self.min_agents_for_rescue = min_agents_for_rescue
        
        # Components
        self.victims: Dict[str, VictimModel] = {}
        self.agents: List[SearchAgent] = []
        self.obstacles: List[np.ndarray] = []
        self.coordinator = RescueCoordinator()
        
        # Enhanced tracking
        self.rescue_teams: Dict[str, Set[str]] = {}  # victim_id -> agent_ids
        self.coordination_events: List[Dict[str, Any]] = []
        self.episode_stats = {
            "victims_discovered": 0,
            "victims_rescued": 0,
            "victims_lost": 0,
            "total_health_saved": 0.0,
            "coordination_successes": 0,
            "avg_rescue_time": 0.0,
            "area_coverage": 0.0
        }
        
        # State
        self.time = 0.0
        self.completed = False
        self.success = False
        
        logger.info(f"Initialized Fixed SearchRescueScenario with {num_victims} victims")
    
    def reset(self):
        """Reset scenario with proper initialization"""
        self.time = 0.0
        self.completed = False
        self.success = False
        
        # Clear tracking
        self.victims.clear()
        self.agents.clear()
        self.obstacles.clear()
        self.rescue_teams.clear()
        self.coordination_events.clear()
        
        # Reset stats
        for key in self.episode_stats:
            self.episode_stats[key] = 0
        
        # Initialize components
        self._initialize_victims()
        self._initialize_agents()
        self._initialize_obstacles(self.obstacle_density)
        
        # Assign initial search patterns
        self.assign_search_patterns()
        
        return self.get_observations()
    
    def _initialize_victims(self):
        """Initialize victims with varying severity"""
        for i in range(self.num_victims):
            position = np.array([
                np.random.uniform(10, self.area_size[0] - 10),
                np.random.uniform(10, self.area_size[1] - 10),
                0.0
            ])
            
            # Severity affects health decay and rescue priority
            severity = np.random.beta(2, 5)  # Skewed toward lower severity
            
            victim = VictimModel(
                victim_id=f"victim_{i}",
                position=position,
                severity=severity,
                required_rescuers=self.min_agents_for_rescue
            )
            
            self.victims[victim.victim_id] = victim
    
    def _initialize_agents(self):
        """Initialize agents with roles"""
        roles = [AgentRole.COORDINATOR if i == 0 else 
                 AgentRole.RESCUER if i < 3 else 
                 AgentRole.SEARCHER 
                 for i in range(self.num_agents)]
        
        for i in range(self.num_agents):
            # Spread agents across area
            x = (i % 3) * self.area_size[0] / 3 + 20
            y = (i // 3) * self.area_size[1] / 2 + 20
            position = np.array([x, y, 5.0])  # Start at altitude
            
            agent = SearchAgent(
                agent_id=f"agent_{i}",
                role=roles[i],
                position=position,
                sensor_range=15.0,
                move_speed=5.0,
                communication_range=50.0
            )
            
            self.agents.append(agent)
    
    def _initialize_obstacles(self, density: float):
        """Initialize obstacles"""
        num_obstacles = int(density * np.prod(self.area_size) / 100)
        
        for _ in range(num_obstacles):
            position = np.array([
                np.random.uniform(0, self.area_size[0]),
                np.random.uniform(0, self.area_size[1]),
                0.0
            ])
            self.obstacles.append(position)
    
    def assign_search_patterns(self):
        """Assign search patterns to agents"""
        patterns = self.coordinator.create_search_patterns(
            (0, 0, self.area_size[0], self.area_size[1]),
            len([a for a in self.agents if a.role == AgentRole.SEARCHER])
        )
        
        pattern_idx = 0
        for agent in self.agents:
            if agent.role == AgentRole.SEARCHER and pattern_idx < len(patterns):
                agent.search_pattern = patterns[pattern_idx]
                pattern_idx += 1
    
    def step(self, actions: Optional[Dict[str, Any]] = None, dt: float = 0.1):
        """Execute simulation step with enhanced coordination"""
        self.time += dt
        
        # Phase 1: Update victims
        self._update_victims(dt)
        
        # Phase 2: Agent detection and communication
        self._detection_and_communication_phase()
        
        # Phase 3: Coordination (enhanced)
        self._enhanced_coordination_phase()
        
        # Phase 4: Execute actions
        if actions:
            self._execute_actions(actions, dt)
        else:
            self._execute_autonomous_actions(dt)
        
        # Phase 5: Rescue operations
        self._rescue_phase(dt)
        
        # Phase 6: Update statistics
        self._update_statistics()
        
        # Check termination
        self._check_termination()
        
        return self.get_observations(), self.get_rewards(), self.completed, self.get_info()
    
    def _update_victims(self, dt: float):
        """Update victim states"""
        for victim in self.victims.values():
            if victim.status not in [VictimStatus.RESCUED, VictimStatus.LOST]:
                victim.update_health(dt, decay_rate=0.005 * (1 + victim.severity))
                
                if victim.health <= 0:
                    victim.status = VictimStatus.LOST
                    self.episode_stats["victims_lost"] += 1
    
    def _detection_and_communication_phase(self):
        """Agents detect victims and share information"""
        for agent in self.agents:
            # Detect victims
            for victim in self.victims.values():
                if victim.status == VictimStatus.UNDISCOVERED:
                    distance = np.linalg.norm(agent.position - victim.position)
                    
                    if distance <= agent.sensor_range:
                        victim.status = VictimStatus.DETECTED
                        victim.discovery_time = self.time
                        agent.detected_victims.add(victim.victim_id)
                        self.episode_stats["victims_discovered"] += 1
                        
                        logger.info(f"{agent.agent_id} discovered {victim.victim_id}")
                        
                        # Share with nearby agents
                        self._share_discovery(agent, victim)
    
    def _share_discovery(self, discovering_agent: SearchAgent, victim: VictimModel):
        """Share victim discovery with nearby agents"""
        for other_agent in self.agents:
            if other_agent.agent_id != discovering_agent.agent_id:
                distance = np.linalg.norm(
                    discovering_agent.position - other_agent.position
                )
                
                if distance <= discovering_agent.communication_range:
                    other_agent.detected_victims.add(victim.victim_id)
                    
                    self.coordination_events.append({
                        "time": self.time,
                        "type": "information_sharing",
                        "from": discovering_agent.agent_id,
                        "to": other_agent.agent_id,
                        "victim": victim.victim_id
                    })
    
    def _enhanced_coordination_phase(self):
        """Enhanced coordination with multi-agent rescue teams"""
        # Find victims needing rescue
        victims_needing_rescue = [
            v for v in self.victims.values()
            if v.status == VictimStatus.DETECTED
        ]
        
        # Sort by priority (health and severity)
        victims_needing_rescue.sort(
            key=lambda v: v.health * (1 - v.severity)
        )
        
        # Form rescue teams
        available_rescuers = [
            a for a in self.agents 
            if a.role in [AgentRole.RESCUER, AgentRole.SUPPORT] 
            and a.assigned_victim is None
        ]
        
        for victim in victims_needing_rescue:
            if victim.victim_id not in self.rescue_teams:
                self.rescue_teams[victim.victim_id] = set()
            
            # Find closest available rescuers
            rescuers_needed = victim.required_rescuers - len(self.rescue_teams[victim.victim_id])
            
            if rescuers_needed > 0 and available_rescuers:
                # Sort by distance
                available_rescuers.sort(
                    key=lambda a: np.linalg.norm(a.position - victim.position)
                )
                
                # Assign closest rescuers
                for i in range(min(rescuers_needed, len(available_rescuers))):
                    rescuer = available_rescuers[i]
                    self.rescue_teams[victim.victim_id].add(rescuer.agent_id)
                    rescuer.assigned_victim = victim.victim_id
                    
                    self.coordination_events.append({
                        "time": self.time,
                        "type": "team_formation",
                        "victim": victim.victim_id,
                        "agent": rescuer.agent_id,
                        "team_size": len(self.rescue_teams[victim.victim_id])
                    })
                
                # Remove assigned rescuers from available list
                available_rescuers = available_rescuers[rescuers_needed:]
                
                # Start rescue if team is complete
                if len(self.rescue_teams[victim.victim_id]) >= victim.required_rescuers:
                    victim.status = VictimStatus.BEING_RESCUED
                    self.episode_stats["coordination_successes"] += 1
    
    def _execute_autonomous_actions(self, dt: float):
        """Execute autonomous agent actions"""
        for agent in self.agents:
            # Plan action
            action = agent.plan_action(self.victims, self.obstacles, self.agents)
            
            # Execute action
            agent.execute_action(action, dt)
    
    def _rescue_phase(self, dt: float):
        """Execute coordinated rescue operations"""
        rescue_range = 5.0
        
        for victim_id, rescue_team in self.rescue_teams.items():
            victim = self.victims[victim_id]
            
            if victim.status == VictimStatus.BEING_RESCUED:
                # Check all team members are in position
                agents_in_position = []
                
                for agent_id in rescue_team:
                    agent = next(a for a in self.agents if a.agent_id == agent_id)
                    distance = np.linalg.norm(agent.position - victim.position)
                    
                    if distance <= rescue_range:
                        agents_in_position.append(agent)
                
                # Progress rescue if enough agents
                if len(agents_in_position) >= victim.required_rescuers:
                    # Collaborative rescue speed
                    rescue_speed = len(agents_in_position) * 1.0
                    victim.rescue_progress += rescue_speed * dt
                    
                    # Complete rescue
                    if victim.rescue_progress >= victim.time_to_rescue:
                        victim.status = VictimStatus.RESCUED
                        victim.rescue_time = self.time
                        self.episode_stats["victims_rescued"] += 1
                        self.episode_stats["total_health_saved"] += victim.health
                        
                        # Calculate rescue time
                        if victim.discovery_time:
                            rescue_duration = self.time - victim.discovery_time
                            self.episode_stats["avg_rescue_time"] = (
                                (self.episode_stats["avg_rescue_time"] * 
                                 (self.episode_stats["victims_rescued"] - 1) + 
                                 rescue_duration) / 
                                self.episode_stats["victims_rescued"]
                            )
                        
                        # Free up agents
                        for agent in self.agents:
                            if agent.assigned_victim == victim_id:
                                agent.assigned_victim = None
                        
                        logger.info(
                            f"Victim {victim_id} rescued by team of {len(agents_in_position)}! "
                            f"Health saved: {victim.health:.2f}"
                        )
    
    def _update_statistics(self):
        """Update episode statistics"""
        # Area coverage (simplified)
        covered_cells = set()
        cell_size = 5.0
        
        for agent in self.agents:
            grid_x = int(agent.position[0] / cell_size)
            grid_y = int(agent.position[1] / cell_size)
            
            # Mark cells within sensor range
            sensor_cells = int(agent.sensor_range / cell_size)
            for dx in range(-sensor_cells, sensor_cells + 1):
                for dy in range(-sensor_cells, sensor_cells + 1):
                    covered_cells.add((grid_x + dx, grid_y + dy))
        
        total_cells = (self.area_size[0] / cell_size) * (self.area_size[1] / cell_size)
        self.episode_stats["area_coverage"] = len(covered_cells) / total_cells
    
    def _check_termination(self):
        """Check termination conditions with proper success criteria"""
        # Count victim states
        rescued = sum(1 for v in self.victims.values() if v.status == VictimStatus.RESCUED)
        lost = sum(1 for v in self.victims.values() if v.status == VictimStatus.LOST)
        remaining = self.num_victims - rescued - lost
        
        # Check termination conditions
        all_resolved = (rescued + lost) == self.num_victims
        timeout = self.time > 300.0  # 5 minute limit
        
        if all_resolved or timeout or remaining == 0:
            self.completed = True
            
            # Calculate success based on rescue rate
            rescue_rate = rescued / self.num_victims if self.num_victims > 0 else 0.0
            self.success = rescue_rate >= self.success_threshold
            
            logger.info(
                f"Episode completed: Success={self.success}, "
                f"Rescue rate={rescue_rate:.2%} "
                f"({rescued}/{self.num_victims} rescued)"
            )
    
    def get_observations(self) -> Dict[str, Any]:
        """Get observations for all agents"""
        obs = {}
        
        for agent in self.agents:
            agent_obs = {
                "position": agent.position.tolist(),
                "energy": agent.energy,
                "role": agent.role.value,
                "detected_victims": list(agent.detected_victims),
                "assigned_victim": agent.assigned_victim,
                "nearby_agents": self._get_nearby_agents(agent),
                "victim_states": self._get_visible_victim_states(agent),
                "time": self.time,
                "victims_rescued": self.episode_stats["victims_rescued"],
                "victims_remaining": sum(
                    1 for v in self.victims.values() 
                    if v.status not in [VictimStatus.RESCUED, VictimStatus.LOST]
                )
            }
            obs[agent.agent_id] = agent_obs
        
        return obs
    
    def _get_nearby_agents(self, agent: SearchAgent) -> List[Dict[str, Any]]:
        """Get information about nearby agents"""
        nearby = []
        
        for other in self.agents:
            if other.agent_id != agent.agent_id:
                distance = np.linalg.norm(agent.position - other.position)
                if distance <= agent.communication_range:
                    nearby.append({
                        "id": other.agent_id,
                        "position": other.position.tolist(),
                        "role": other.role.value,
                        "assigned_victim": other.assigned_victim
                    })
        
        return nearby
    
    def _get_visible_victim_states(self, agent: SearchAgent) -> Dict[str, Dict[str, Any]]:
        """Get states of victims visible to agent"""
        visible = {}
        
        for victim_id in agent.detected_victims:
            if victim_id in self.victims:
                victim = self.victims[victim_id]
                visible[victim_id] = {
                    "position": victim.position.tolist(),
                    "status": victim.status.value,
                    "health": victim.health,
                    "severity": victim.severity,
                    "rescue_progress": victim.rescue_progress / victim.time_to_rescue
                }
        
        return visible
    
    def get_rewards(self) -> Dict[str, float]:
        """Calculate multi-objective rewards with coordination bonuses"""
        rewards = {}
        
        for agent in self.agents:
            reward = 0.0
            
            # Discovery reward
            discovery_bonus = len(agent.detected_victims) * 5.0
            
            # Rescue participation reward
            rescue_bonus = 0.0
            if agent.assigned_victim and agent.assigned_victim in self.victims:
                victim = self.victims[agent.assigned_victim]
                if victim.status == VictimStatus.RESCUED:
                    rescue_bonus = 50.0 * victim.health  # Proportional to health saved
                elif victim.status == VictimStatus.BEING_RESCUED:
                    rescue_bonus = 10.0 * victim.rescue_progress / victim.time_to_rescue
            
            # Coordination bonus
            coord_bonus = 0.0
            for event in self.coordination_events[-10:]:  # Recent events
                if event.get("from") == agent.agent_id or event.get("to") == agent.agent_id:
                    coord_bonus += 2.0
                if event.get("agent") == agent.agent_id and event["type"] == "team_formation":
                    coord_bonus += 5.0
            
            # Distance to assigned victim (negative if far)
            distance_penalty = 0.0
            if agent.assigned_victim and agent.assigned_victim in self.victims:
                victim = self.victims[agent.assigned_victim]
                distance = np.linalg.norm(agent.position - victim.position)
                distance_penalty = -0.1 * distance
            
            # Energy penalty
            energy_penalty = -5.0 if agent.energy < 0.2 else 0.0
            
            # Time pressure
            time_penalty = -0.01 * self.time
            
            # Total reward
            reward = (
                discovery_bonus +
                rescue_bonus +
                coord_bonus +
                distance_penalty +
                energy_penalty +
                time_penalty
            )
            
            rewards[agent.agent_id] = reward
        
        return rewards
    
    def get_info(self) -> Dict[str, Any]:
        """Get scenario information"""
        rescue_rate = self.episode_stats["victims_rescued"] / self.num_victims if self.num_victims > 0 else 0.0
        
        return {
            "time": self.time,
            "completed": self.completed,
            "success": self.success,
            "rescue_rate": rescue_rate,
            "episode_stats": self.episode_stats.copy(),
            "coordination_events": len(self.coordination_events),
            "active_rescue_teams": len([t for t in self.rescue_teams.values() if t])
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate scenario performance"""
        rescue_rate = self.episode_stats["victims_rescued"] / self.num_victims if self.num_victims > 0 else 0.0
        
        return {
            "success": rescue_rate >= self.success_threshold,
            "rescue_rate": rescue_rate,
            "victims_rescued": self.episode_stats["victims_rescued"],
            "victims_lost": self.episode_stats["victims_lost"],
            "avg_rescue_time": self.episode_stats["avg_rescue_time"],
            "area_coverage": self.episode_stats["area_coverage"],
            "coordination_successes": self.episode_stats["coordination_successes"],
            "total_health_saved": self.episode_stats["total_health_saved"]
        }


# Monkey patch to replace the original class
import sys
if 'search_rescue' in sys.modules:
    sys.modules['search_rescue'].SearchRescueScenario = SearchRescueScenarioFixed
    logger.info("GENIUS FIX APPLIED: SearchRescueScenario replaced with fixed version")