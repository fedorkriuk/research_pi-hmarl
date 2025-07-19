"""Adversarial Scenarios

This module implements competitive and adversarial multi-agent scenarios
including pursuit-evasion, territorial defense, and resource competition.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class TeamType(Enum):
    """Team types in adversarial scenarios"""
    PURSUER = "pursuer"
    EVADER = "evader"
    DEFENDER = "defender"
    ATTACKER = "attacker"
    NEUTRAL = "neutral"


class ResourceType(Enum):
    """Types of resources"""
    ENERGY = "energy"
    TERRITORY = "territory"
    TARGET = "target"
    FLAG = "flag"


@dataclass
class AdversarialAgent:
    """Agent in adversarial scenario"""
    agent_id: str
    team: TeamType
    position: np.ndarray
    velocity: np.ndarray
    max_speed: float
    sensor_range: float
    action_range: float
    health: float = 1.0
    energy: float = 1.0
    score: float = 0.0
    captured: bool = False
    
    def update_position(self, dt: float, bounds: Optional[Tuple[float, float, float, float]] = None):
        """Update agent position
        
        Args:
            dt: Time step
            bounds: Environment bounds (min_x, min_y, max_x, max_y)
        """
        self.position += self.velocity * dt
        
        # Apply bounds if provided
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            self.position[0] = np.clip(self.position[0], min_x, max_x)
            self.position[1] = np.clip(self.position[1], min_y, max_y)
            
            # Reflect velocity at boundaries
            if self.position[0] == min_x or self.position[0] == max_x:
                self.velocity[0] *= -0.5
            if self.position[1] == min_y or self.position[1] == max_y:
                self.velocity[1] *= -0.5
    
    def consume_energy(self, amount: float):
        """Consume energy
        
        Args:
            amount: Energy to consume
        """
        self.energy = max(0, self.energy - amount)
        
        # Reduce max speed if low energy
        if self.energy < 0.3:
            speed_factor = self.energy / 0.3
            current_speed = np.linalg.norm(self.velocity)
            if current_speed > self.max_speed * speed_factor:
                self.velocity = self.velocity / current_speed * self.max_speed * speed_factor


class PursuitEvasion:
    """Pursuit-evasion game scenario"""
    
    def __init__(
        self,
        num_pursuers: int = 3,
        num_evaders: int = 2,
        environment_size: Tuple[float, float] = (100.0, 100.0),
        capture_distance: float = 2.0
    ):
        """Initialize pursuit-evasion scenario
        
        Args:
            num_pursuers: Number of pursuing agents
            num_evaders: Number of evading agents
            environment_size: Size of environment
            capture_distance: Distance for capture
        """
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.environment_size = environment_size
        self.capture_distance = capture_distance
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Game state
        self.time = 0.0
        self.game_over = False
        self.winner = None
        
        # Strategy components
        self.pursuer_strategy = CoordinatedPursuit()
        self.evader_strategy = EvasionStrategy()
        
        logger.info(
            f"Initialized PursuitEvasion with {num_pursuers} pursuers "
            f"and {num_evaders} evaders"
        )
    
    def _initialize_agents(self) -> List[AdversarialAgent]:
        """Initialize agents for both teams
        
        Returns:
            List of all agents
        """
        agents = []
        
        # Initialize pursuers (start from left side)
        for i in range(self.num_pursuers):
            position = np.array([
                10.0,
                (i + 1) * self.environment_size[1] / (self.num_pursuers + 1),
                0.0
            ])
            
            agent = AdversarialAgent(
                agent_id=f"pursuer_{i}",
                team=TeamType.PURSUER,
                position=position,
                velocity=np.zeros(3),
                max_speed=4.0,
                sensor_range=20.0,
                action_range=self.capture_distance
            )
            agents.append(agent)
        
        # Initialize evaders (start from right side)
        for i in range(self.num_evaders):
            position = np.array([
                self.environment_size[0] - 10.0,
                (i + 1) * self.environment_size[1] / (self.num_evaders + 1),
                0.0
            ])
            
            agent = AdversarialAgent(
                agent_id=f"evader_{i}",
                team=TeamType.EVADER,
                position=position,
                velocity=np.zeros(3),
                max_speed=4.5,  # Slightly faster
                sensor_range=25.0,  # Better sensors
                action_range=0.0
            )
            agents.append(agent)
        
        return agents
    
    def step(self, dt: float = 0.1):
        """Run one step of pursuit-evasion
        
        Args:
            dt: Time step
        """
        if self.game_over:
            return
        
        self.time += dt
        
        # Get team agents
        pursuers = [a for a in self.agents if a.team == TeamType.PURSUER and not a.captured]
        evaders = [a for a in self.agents if a.team == TeamType.EVADER and not a.captured]
        
        # Update pursuer actions
        pursuer_actions = self.pursuer_strategy.compute_actions(pursuers, evaders)
        
        for pursuer in pursuers:
            if pursuer.agent_id in pursuer_actions:
                action = pursuer_actions[pursuer.agent_id]
                pursuer.velocity = action['velocity']
            
            # Energy consumption
            speed = np.linalg.norm(pursuer.velocity)
            pursuer.consume_energy(0.01 * speed * dt)
        
        # Update evader actions
        evader_actions = self.evader_strategy.compute_actions(evaders, pursuers)
        
        for evader in evaders:
            if evader.agent_id in evader_actions:
                action = evader_actions[evader.agent_id]
                evader.velocity = action['velocity']
            
            # Energy consumption
            speed = np.linalg.norm(evader.velocity)
            evader.consume_energy(0.01 * speed * dt)
        
        # Update positions
        bounds = (0, 0, self.environment_size[0], self.environment_size[1])
        for agent in self.agents:
            agent.update_position(dt, bounds)
        
        # Check captures
        for pursuer in pursuers:
            for evader in evaders:
                distance = np.linalg.norm(pursuer.position - evader.position)
                
                if distance <= self.capture_distance:
                    evader.captured = True
                    pursuer.score += 100
                    logger.info(f"{pursuer.agent_id} captured {evader.agent_id}!")
        
        # Check game over conditions
        active_evaders = [e for e in evaders if not e.captured]
        
        if len(active_evaders) == 0:
            self.game_over = True
            self.winner = TeamType.PURSUER
            logger.info("Pursuers win! All evaders captured.")
        
        elif self.time > 120.0:  # 2 minute time limit
            self.game_over = True
            self.winner = TeamType.EVADER
            logger.info("Evaders win! Time limit reached.")
        
        # Check if pursuers are out of energy
        elif all(p.energy == 0 for p in pursuers):
            self.game_over = True
            self.winner = TeamType.EVADER
            logger.info("Evaders win! Pursuers out of energy.")


class CoordinatedPursuit:
    """Coordinated pursuit strategy"""
    
    def __init__(self):
        """Initialize pursuit strategy"""
        self.assignment_method = "nearest"  # or "optimal"
        self.prediction_horizon = 2.0  # seconds
    
    def compute_actions(
        self,
        pursuers: List[AdversarialAgent],
        evaders: List[AdversarialAgent]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute pursuer actions
        
        Args:
            pursuers: List of pursuing agents
            evaders: List of evading agents
            
        Returns:
            Actions for each pursuer
        """
        actions = {}
        
        if not evaders:
            return actions
        
        # Assign targets
        assignments = self._assign_targets(pursuers, evaders)
        
        for pursuer in pursuers:
            if pursuer.agent_id in assignments:
                target = assignments[pursuer.agent_id]
                
                # Predict target position
                predicted_pos = self._predict_position(target, self.prediction_horizon)
                
                # Compute intercept velocity
                velocity = self._compute_intercept_velocity(
                    pursuer.position,
                    predicted_pos,
                    pursuer.max_speed
                )
                
                actions[pursuer.agent_id] = {'velocity': velocity}
            else:
                # No target, stop
                actions[pursuer.agent_id] = {'velocity': np.zeros(3)}
        
        return actions
    
    def _assign_targets(
        self,
        pursuers: List[AdversarialAgent],
        evaders: List[AdversarialAgent]
    ) -> Dict[str, AdversarialAgent]:
        """Assign targets to pursuers
        
        Args:
            pursuers: List of pursuers
            evaders: List of evaders
            
        Returns:
            Pursuer to target assignments
        """
        assignments = {}
        
        if self.assignment_method == "nearest":
            # Simple nearest target assignment
            available_evaders = evaders.copy()
            
            for pursuer in pursuers:
                if not available_evaders:
                    break
                
                # Find nearest evader
                nearest = min(
                    available_evaders,
                    key=lambda e: np.linalg.norm(pursuer.position - e.position)
                )
                
                assignments[pursuer.agent_id] = nearest
                available_evaders.remove(nearest)
        
        return assignments
    
    def _predict_position(
        self,
        agent: AdversarialAgent,
        time_horizon: float
    ) -> np.ndarray:
        """Predict agent position
        
        Args:
            agent: Agent to predict
            time_horizon: Prediction time
            
        Returns:
            Predicted position
        """
        # Simple linear prediction
        return agent.position + agent.velocity * time_horizon
    
    def _compute_intercept_velocity(
        self,
        pursuer_pos: np.ndarray,
        target_pos: np.ndarray,
        max_speed: float
    ) -> np.ndarray:
        """Compute velocity to intercept target
        
        Args:
            pursuer_pos: Pursuer position
            target_pos: Target position
            max_speed: Maximum speed
            
        Returns:
            Velocity vector
        """
        direction = target_pos - pursuer_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            return direction * max_speed
        
        return np.zeros(3)


class EvasionStrategy:
    """Evasion strategy for evaders"""
    
    def __init__(self):
        """Initialize evasion strategy"""
        self.safe_distance = 15.0
        self.corner_avoidance = 10.0
    
    def compute_actions(
        self,
        evaders: List[AdversarialAgent],
        pursuers: List[AdversarialAgent]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute evader actions
        
        Args:
            evaders: List of evading agents
            pursuers: List of pursuing agents
            
        Returns:
            Actions for each evader
        """
        actions = {}
        
        for evader in evaders:
            # Find threats
            threats = self._find_threats(evader, pursuers)
            
            if threats:
                # Compute escape velocity
                velocity = self._compute_escape_velocity(
                    evader,
                    threats,
                    evader.max_speed
                )
            else:
                # No immediate threat, maintain distance
                velocity = self._maintain_distance(evader, pursuers, evader.max_speed)
            
            actions[evader.agent_id] = {'velocity': velocity}
        
        return actions
    
    def _find_threats(
        self,
        evader: AdversarialAgent,
        pursuers: List[AdversarialAgent]
    ) -> List[AdversarialAgent]:
        """Find threatening pursuers
        
        Args:
            evader: Evading agent
            pursuers: List of pursuers
            
        Returns:
            List of threats
        """
        threats = []
        
        for pursuer in pursuers:
            distance = np.linalg.norm(pursuer.position - evader.position)
            
            if distance < self.safe_distance:
                threats.append(pursuer)
        
        return threats
    
    def _compute_escape_velocity(
        self,
        evader: AdversarialAgent,
        threats: List[AdversarialAgent],
        max_speed: float
    ) -> np.ndarray:
        """Compute escape velocity
        
        Args:
            evader: Evading agent
            threats: List of threats
            max_speed: Maximum speed
            
        Returns:
            Escape velocity
        """
        escape_direction = np.zeros(3)
        
        # Sum repulsive forces from threats
        for threat in threats:
            diff = evader.position - threat.position
            distance = np.linalg.norm(diff)
            
            if distance > 0:
                # Inverse distance weighting
                weight = 1.0 / (distance + 1.0)
                escape_direction += weight * diff / distance
        
        # Normalize and scale
        if np.linalg.norm(escape_direction) > 0:
            escape_direction = escape_direction / np.linalg.norm(escape_direction)
        
        return escape_direction * max_speed
    
    def _maintain_distance(
        self,
        evader: AdversarialAgent,
        pursuers: List[AdversarialAgent],
        max_speed: float
    ) -> np.ndarray:
        """Maintain safe distance from pursuers
        
        Args:
            evader: Evading agent
            pursuers: List of pursuers
            max_speed: Maximum speed
            
        Returns:
            Velocity to maintain distance
        """
        if not pursuers:
            return np.zeros(3)
        
        # Find center of pursuers
        pursuer_center = np.mean([p.position for p in pursuers], axis=0)
        
        # Move away from center
        away_direction = evader.position - pursuer_center
        
        if np.linalg.norm(away_direction) > 0:
            away_direction = away_direction / np.linalg.norm(away_direction)
        
        return away_direction * max_speed * 0.5  # Half speed when not threatened


class TerritorialDefense:
    """Territorial defense scenario"""
    
    def __init__(
        self,
        num_defenders: int = 4,
        num_attackers: int = 3,
        territory_radius: float = 30.0,
        environment_size: Tuple[float, float] = (100.0, 100.0)
    ):
        """Initialize territorial defense
        
        Args:
            num_defenders: Number of defending agents
            num_attackers: Number of attacking agents
            territory_radius: Radius of territory to defend
            environment_size: Size of environment
        """
        self.num_defenders = num_defenders
        self.num_attackers = num_attackers
        self.territory_radius = territory_radius
        self.environment_size = environment_size
        
        # Territory center
        self.territory_center = np.array([
            environment_size[0] / 2,
            environment_size[1] / 2,
            0.0
        ])
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Game state
        self.time = 0.0
        self.territory_health = 100.0
        self.game_over = False
        self.winner = None
        
        # Score tracking
        self.attacker_score = 0.0
        self.defender_score = 0.0
        
        logger.info(
            f"Initialized TerritorialDefense with {num_defenders} defenders "
            f"and {num_attackers} attackers"
        )
    
    def _initialize_agents(self) -> List[AdversarialAgent]:
        """Initialize agents for both teams"""
        agents = []
        
        # Initialize defenders (around territory)
        for i in range(self.num_defenders):
            angle = 2 * np.pi * i / self.num_defenders
            position = self.territory_center + (self.territory_radius + 5) * np.array([
                np.cos(angle),
                np.sin(angle),
                0.0
            ])
            
            agent = AdversarialAgent(
                agent_id=f"defender_{i}",
                team=TeamType.DEFENDER,
                position=position,
                velocity=np.zeros(3),
                max_speed=3.5,
                sensor_range=25.0,
                action_range=5.0  # Attack range
            )
            agents.append(agent)
        
        # Initialize attackers (from edges)
        for i in range(self.num_attackers):
            # Random edge position
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])
            
            if edge == 'top':
                position = np.array([np.random.uniform(0, self.environment_size[0]), 0, 0])
            elif edge == 'bottom':
                position = np.array([np.random.uniform(0, self.environment_size[0]), self.environment_size[1], 0])
            elif edge == 'left':
                position = np.array([0, np.random.uniform(0, self.environment_size[1]), 0])
            else:  # right
                position = np.array([self.environment_size[0], np.random.uniform(0, self.environment_size[1]), 0])
            
            agent = AdversarialAgent(
                agent_id=f"attacker_{i}",
                team=TeamType.ATTACKER,
                position=position,
                velocity=np.zeros(3),
                max_speed=4.0,
                sensor_range=20.0,
                action_range=3.0
            )
            agents.append(agent)
        
        return agents
    
    def step(self, dt: float = 0.1):
        """Run one step of territorial defense
        
        Args:
            dt: Time step
        """
        if self.game_over:
            return
        
        self.time += dt
        
        # Get active agents
        defenders = [a for a in self.agents if a.team == TeamType.DEFENDER and a.health > 0]
        attackers = [a for a in self.agents if a.team == TeamType.ATTACKER and a.health > 0]
        
        # Defender AI
        for defender in defenders:
            # Find nearest threat
            if attackers:
                nearest_attacker = min(
                    attackers,
                    key=lambda a: np.linalg.norm(defender.position - a.position)
                )
                
                distance = np.linalg.norm(defender.position - nearest_attacker.position)
                
                if distance < defender.action_range:
                    # Attack
                    nearest_attacker.health -= 0.1 * dt
                    defender.score += 1.0 * dt
                
                # Move to intercept
                intercept_point = self._compute_intercept_point(
                    defender.position,
                    nearest_attacker.position,
                    self.territory_center,
                    self.territory_radius
                )
                
                direction = intercept_point - defender.position
                if np.linalg.norm(direction) > 0:
                    defender.velocity = direction / np.linalg.norm(direction) * defender.max_speed
            else:
                # Patrol
                defender.velocity = self._patrol_velocity(defender, self.territory_center, self.territory_radius)
        
        # Attacker AI
        for attacker in attackers:
            # Check if in territory
            dist_to_center = np.linalg.norm(attacker.position - self.territory_center)
            
            if dist_to_center < self.territory_radius:
                # Damage territory
                self.territory_health -= 5.0 * dt
                self.attacker_score += 10.0 * dt
                attacker.score += 10.0 * dt
            
            # Find defenders in range
            threats = [
                d for d in defenders
                if np.linalg.norm(attacker.position - d.position) < d.action_range + 5
            ]
            
            if threats:
                # Evade defenders
                escape_vel = self._compute_escape_velocity(attacker, threats, attacker.max_speed)
                
                # But still try to reach territory
                to_territory = self.territory_center - attacker.position
                if np.linalg.norm(to_territory) > 0:
                    to_territory = to_territory / np.linalg.norm(to_territory) * attacker.max_speed * 0.3
                
                attacker.velocity = 0.7 * escape_vel + 0.3 * to_territory
            else:
                # Move towards territory
                direction = self.territory_center - attacker.position
                if np.linalg.norm(direction) > 0:
                    attacker.velocity = direction / np.linalg.norm(direction) * attacker.max_speed
        
        # Update positions
        bounds = (0, 0, self.environment_size[0], self.environment_size[1])
        for agent in self.agents:
            if agent.health > 0:
                agent.update_position(dt, bounds)
        
        # Check game over
        if self.territory_health <= 0:
            self.game_over = True
            self.winner = TeamType.ATTACKER
            logger.info("Attackers win! Territory destroyed.")
        elif len(attackers) == 0:
            self.game_over = True
            self.winner = TeamType.DEFENDER
            logger.info("Defenders win! All attackers eliminated.")
        elif self.time > 180.0:  # 3 minute time limit
            self.game_over = True
            if self.territory_health > 50:
                self.winner = TeamType.DEFENDER
                logger.info("Defenders win! Territory protected.")
            else:
                self.winner = TeamType.ATTACKER
                logger.info("Attackers win! Territory sufficiently damaged.")
    
    def _compute_intercept_point(
        self,
        defender_pos: np.ndarray,
        attacker_pos: np.ndarray,
        territory_center: np.ndarray,
        territory_radius: float
    ) -> np.ndarray:
        """Compute point to intercept attacker
        
        Args:
            defender_pos: Defender position
            attacker_pos: Attacker position
            territory_center: Center of territory
            territory_radius: Territory radius
            
        Returns:
            Intercept point
        """
        # Project attacker's path to territory
        to_territory = territory_center - attacker_pos
        
        # Find point on line closest to defender
        t = np.dot(defender_pos - attacker_pos, to_territory) / (np.linalg.norm(to_territory)**2 + 0.001)
        t = np.clip(t, 0, 1)
        
        intercept = attacker_pos + t * to_territory
        
        return intercept
    
    def _patrol_velocity(
        self,
        agent: AdversarialAgent,
        center: np.ndarray,
        radius: float
    ) -> np.ndarray:
        """Compute patrol velocity
        
        Args:
            agent: Patrolling agent
            center: Patrol center
            radius: Patrol radius
            
        Returns:
            Patrol velocity
        """
        # Circular patrol
        to_center = center - agent.position
        dist = np.linalg.norm(to_center[:2])
        
        if abs(dist - radius) > 2.0:
            # Move to patrol radius
            if dist > radius:
                return to_center / dist * agent.max_speed * 0.5
            else:
                return -to_center / dist * agent.max_speed * 0.5
        else:
            # Patrol along circle
            tangent = np.array([-to_center[1], to_center[0], 0])
            if np.linalg.norm(tangent) > 0:
                return tangent / np.linalg.norm(tangent) * agent.max_speed * 0.7
        
        return np.zeros(3)
    
    def _compute_escape_velocity(
        self,
        agent: AdversarialAgent,
        threats: List[AdversarialAgent],
        max_speed: float
    ) -> np.ndarray:
        """Compute escape velocity from threats"""
        escape_direction = np.zeros(3)
        
        for threat in threats:
            diff = agent.position - threat.position
            distance = np.linalg.norm(diff)
            
            if distance > 0:
                weight = 1.0 / (distance + 1.0)
                escape_direction += weight * diff / distance
        
        if np.linalg.norm(escape_direction) > 0:
            escape_direction = escape_direction / np.linalg.norm(escape_direction)
        
        return escape_direction * max_speed


class CompetitiveResourceGathering:
    """Competitive resource gathering scenario"""
    
    def __init__(
        self,
        num_teams: int = 2,
        agents_per_team: int = 3,
        num_resources: int = 20,
        environment_size: Tuple[float, float] = (100.0, 100.0)
    ):
        """Initialize resource gathering scenario
        
        Args:
            num_teams: Number of competing teams
            agents_per_team: Agents per team
            num_resources: Number of resources
            environment_size: Size of environment
        """
        self.num_teams = num_teams
        self.agents_per_team = agents_per_team
        self.num_resources = num_resources
        self.environment_size = environment_size
        
        # Initialize components
        self.agents = self._initialize_agents()
        self.resources = self._initialize_resources()
        self.bases = self._initialize_bases()
        
        # Game state
        self.time = 0.0
        self.team_scores = defaultdict(float)
        self.game_duration = 180.0  # 3 minutes
        
        logger.info(
            f"Initialized CompetitiveResourceGathering with {num_teams} teams"
        )
    
    def _initialize_agents(self) -> List[AdversarialAgent]:
        """Initialize agents for all teams"""
        agents = []
        
        team_colors = [TeamType.PURSUER, TeamType.EVADER]  # Reusing for teams
        
        for team_idx in range(self.num_teams):
            team = team_colors[team_idx % len(team_colors)]
            
            # Start positions near team base
            base_x = 10 if team_idx == 0 else self.environment_size[0] - 10
            base_y = self.environment_size[1] / 2
            
            for i in range(self.agents_per_team):
                offset_y = (i - self.agents_per_team / 2) * 5
                position = np.array([base_x, base_y + offset_y, 0])
                
                agent = AdversarialAgent(
                    agent_id=f"team{team_idx}_agent{i}",
                    team=team,
                    position=position,
                    velocity=np.zeros(3),
                    max_speed=4.0,
                    sensor_range=20.0,
                    action_range=2.0
                )
                
                # Carrying capacity
                agent.carrying = None
                agent.carry_capacity = 1
                
                agents.append(agent)
        
        return agents
    
    def _initialize_resources(self) -> List[Dict[str, Any]]:
        """Initialize resources in environment"""
        resources = []
        
        for i in range(self.num_resources):
            position = np.array([
                np.random.uniform(20, self.environment_size[0] - 20),
                np.random.uniform(20, self.environment_size[1] - 20),
                0
            ])
            
            resource = {
                'id': f"resource_{i}",
                'position': position,
                'value': np.random.uniform(5, 20),
                'collected': False,
                'carried_by': None
            }
            
            resources.append(resource)
        
        return resources
    
    def _initialize_bases(self) -> Dict[int, Dict[str, Any]]:
        """Initialize team bases"""
        bases = {}
        
        for team_idx in range(self.num_teams):
            base_x = 10 if team_idx == 0 else self.environment_size[0] - 10
            base_y = self.environment_size[1] / 2
            
            bases[team_idx] = {
                'position': np.array([base_x, base_y, 0]),
                'radius': 10.0,
                'team': team_idx
            }
        
        return bases
    
    def step(self, dt: float = 0.1):
        """Run one step of resource gathering
        
        Args:
            dt: Time step
        """
        self.time += dt
        
        # Agent AI for each team
        for team_idx in range(self.num_teams):
            team_agents = [
                a for a in self.agents
                if a.agent_id.startswith(f"team{team_idx}")
            ]
            
            for agent in team_agents:
                if agent.carrying is None:
                    # Find nearest uncollected resource
                    available_resources = [
                        r for r in self.resources
                        if not r['collected'] and r['carried_by'] is None
                    ]
                    
                    if available_resources:
                        nearest = min(
                            available_resources,
                            key=lambda r: np.linalg.norm(agent.position - r['position'])
                        )
                        
                        # Move towards resource
                        direction = nearest['position'] - agent.position
                        distance = np.linalg.norm(direction)
                        
                        if distance < agent.action_range:
                            # Collect resource
                            agent.carrying = nearest
                            nearest['collected'] = True
                            nearest['carried_by'] = agent.agent_id
                        else:
                            # Move towards resource
                            agent.velocity = direction / distance * agent.max_speed
                    else:
                        # No resources, defend or interfere
                        agent.velocity = np.zeros(3)
                else:
                    # Carrying resource, return to base
                    base = self.bases[team_idx]
                    direction = base['position'] - agent.position
                    distance = np.linalg.norm(direction)
                    
                    if distance < base['radius']:
                        # Deliver resource
                        resource = agent.carrying
                        self.team_scores[team_idx] += resource['value']
                        agent.carrying = None
                        
                        # Remove delivered resource
                        self.resources.remove(resource)
                    else:
                        # Move to base
                        agent.velocity = direction / distance * agent.max_speed * 0.8  # Slower when carrying
        
        # Update positions
        bounds = (0, 0, self.environment_size[0], self.environment_size[1])
        for agent in self.agents:
            agent.update_position(dt, bounds)
            
            # Update carried resource position
            if agent.carrying:
                agent.carrying['position'] = agent.position.copy()
    
    def get_winner(self) -> Optional[int]:
        """Get winning team
        
        Returns:
            Winning team index or None
        """
        if self.time >= self.game_duration or len(self.resources) == 0:
            # Game over
            if self.team_scores:
                return max(self.team_scores.items(), key=lambda x: x[1])[0]
        
        return None


class StrategicPlanning:
    """Strategic planning for adversarial scenarios"""
    
    def __init__(self):
        """Initialize strategic planner"""
        self.planning_horizon = 10.0  # seconds
        self.risk_tolerance = 0.5
        
    def plan_team_strategy(
        self,
        team_agents: List[AdversarialAgent],
        opponents: List[AdversarialAgent],
        objectives: List[Dict[str, Any]],
        environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan team strategy
        
        Args:
            team_agents: Agents in team
            opponents: Opponent agents
            objectives: List of objectives
            environment: Environment information
            
        Returns:
            Strategic plan
        """
        # Analyze situation
        situation = self._analyze_situation(
            team_agents,
            opponents,
            objectives,
            environment
        )
        
        # Select strategy
        if situation['team_strength'] > situation['opponent_strength']:
            strategy = 'aggressive'
        elif situation['team_strength'] < situation['opponent_strength'] * 0.7:
            strategy = 'defensive'
        else:
            strategy = 'balanced'
        
        # Create plan
        plan = {
            'strategy': strategy,
            'formation': self._select_formation(strategy, len(team_agents)),
            'roles': self._assign_roles(team_agents, strategy),
            'objectives': self._prioritize_objectives(objectives, situation),
            'contingencies': self._plan_contingencies(situation)
        }
        
        return plan
    
    def _analyze_situation(
        self,
        team_agents: List[AdversarialAgent],
        opponents: List[AdversarialAgent],
        objectives: List[Dict[str, Any]],
        environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze current situation
        
        Returns:
            Situation analysis
        """
        # Team strength
        team_strength = sum(a.health * a.energy for a in team_agents)
        
        # Opponent strength
        opponent_strength = sum(a.health * a.energy for a in opponents) if opponents else 0
        
        # Objective status
        objective_distances = []
        for obj in objectives:
            min_dist = min(
                np.linalg.norm(a.position - obj['position'])
                for a in team_agents
            )
            objective_distances.append(min_dist)
        
        # Spatial distribution
        team_center = np.mean([a.position for a in team_agents], axis=0)
        team_spread = np.mean([
            np.linalg.norm(a.position - team_center)
            for a in team_agents
        ])
        
        return {
            'team_strength': team_strength,
            'opponent_strength': opponent_strength,
            'objective_distances': objective_distances,
            'team_spread': team_spread,
            'num_healthy_agents': sum(1 for a in team_agents if a.health > 0.5)
        }
    
    def _select_formation(self, strategy: str, num_agents: int) -> str:
        """Select formation based on strategy
        
        Args:
            strategy: Strategic approach
            num_agents: Number of agents
            
        Returns:
            Formation type
        """
        if strategy == 'aggressive':
            return 'wedge' if num_agents >= 3 else 'line'
        elif strategy == 'defensive':
            return 'circle' if num_agents >= 4 else 'line'
        else:  # balanced
            return 'diamond' if num_agents == 4 else 'line'
    
    def _assign_roles(
        self,
        agents: List[AdversarialAgent],
        strategy: str
    ) -> Dict[str, str]:
        """Assign roles to agents
        
        Args:
            agents: Team agents
            strategy: Strategic approach
            
        Returns:
            Agent role assignments
        """
        roles = {}
        
        # Sort agents by capability
        sorted_agents = sorted(
            agents,
            key=lambda a: a.health * a.energy * a.max_speed,
            reverse=True
        )
        
        if strategy == 'aggressive':
            # Best agents attack
            for i, agent in enumerate(sorted_agents):
                if i < len(agents) // 2:
                    roles[agent.agent_id] = 'attacker'
                else:
                    roles[agent.agent_id] = 'support'
        
        elif strategy == 'defensive':
            # Best agents defend
            for i, agent in enumerate(sorted_agents):
                if i == 0:
                    roles[agent.agent_id] = 'anchor'
                else:
                    roles[agent.agent_id] = 'defender'
        
        else:  # balanced
            for i, agent in enumerate(sorted_agents):
                if i < len(agents) // 3:
                    roles[agent.agent_id] = 'attacker'
                elif i < 2 * len(agents) // 3:
                    roles[agent.agent_id] = 'midfielder'
                else:
                    roles[agent.agent_id] = 'defender'
        
        return roles
    
    def _prioritize_objectives(
        self,
        objectives: List[Dict[str, Any]],
        situation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prioritize objectives
        
        Args:
            objectives: List of objectives
            situation: Situation analysis
            
        Returns:
            Prioritized objectives
        """
        # Score each objective
        scored_objectives = []
        
        for i, obj in enumerate(objectives):
            # Base score on value and distance
            distance = situation['objective_distances'][i]
            value = obj.get('value', 1.0)
            
            score = value / (distance + 1.0)
            
            # Adjust for risk
            risk = obj.get('risk', 0.0)
            score *= (1 - risk * self.risk_tolerance)
            
            scored_objectives.append({
                **obj,
                'priority_score': score
            })
        
        # Sort by priority
        return sorted(scored_objectives, key=lambda x: x['priority_score'], reverse=True)
    
    def _plan_contingencies(
        self,
        situation: Dict[str, Any]
    ) -> Dict[str, str]:
        """Plan contingency actions
        
        Args:
            situation: Situation analysis
            
        Returns:
            Contingency plans
        """
        contingencies = {}
        
        # Low health contingency
        if situation['num_healthy_agents'] < situation.get('total_agents', 1) * 0.5:
            contingencies['low_health'] = 'retreat_and_regroup'
        
        # Outnumbered contingency
        if situation['opponent_strength'] > situation['team_strength'] * 1.5:
            contingencies['outnumbered'] = 'defensive_formation'
        
        # Objective at risk
        if min(situation['objective_distances']) < 10.0:
            contingencies['objective_threatened'] = 'prioritize_defense'
        
        return contingencies


class AdversarialScenario:
    """Main adversarial scenario manager"""
    
    def __init__(
        self,
        scenario_type: str = "pursuit_evasion",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize adversarial scenario
        
        Args:
            scenario_type: Type of scenario
            config: Scenario configuration
        """
        self.scenario_type = scenario_type
        self.config = config or {}
        
        # Initialize specific scenario
        if scenario_type == "pursuit_evasion":
            self.scenario = PursuitEvasion(**self.config)
        elif scenario_type == "territorial_defense":
            self.scenario = TerritorialDefense(**self.config)
        elif scenario_type == "resource_gathering":
            self.scenario = CompetitiveResourceGathering(**self.config)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        # Strategic planner
        self.strategic_planner = StrategicPlanning()
        
        logger.info(f"Initialized AdversarialScenario: {scenario_type}")
    
    def run_episode(self, max_time: float = 300.0, dt: float = 0.1):
        """Run complete episode
        
        Args:
            max_time: Maximum episode time
            dt: Time step
        """
        steps = 0
        
        while self.scenario.time < max_time:
            self.scenario.step(dt)
            steps += 1
            
            # Check termination
            if hasattr(self.scenario, 'game_over') and self.scenario.game_over:
                break
            
            # Log periodically
            if steps % 100 == 0:
                self._log_status()
        
        # Final summary
        self._log_summary()
    
    def _log_status(self):
        """Log current status"""
        if self.scenario_type == "pursuit_evasion":
            active_evaders = sum(
                1 for a in self.scenario.agents
                if a.team == TeamType.EVADER and not a.captured
            )
            logger.info(
                f"Time: {self.scenario.time:.1f}s, "
                f"Active evaders: {active_evaders}"
            )
        
        elif self.scenario_type == "territorial_defense":
            logger.info(
                f"Time: {self.scenario.time:.1f}s, "
                f"Territory health: {self.scenario.territory_health:.1f}%"
            )
        
        elif self.scenario_type == "resource_gathering":
            scores = dict(self.scenario.team_scores)
            logger.info(
                f"Time: {self.scenario.time:.1f}s, "
                f"Scores: {scores}"
            )
    
    def _log_summary(self):
        """Log episode summary"""
        logger.info(f"Episode completed at time {self.scenario.time:.1f}s")
        
        if hasattr(self.scenario, 'winner'):
            logger.info(f"Winner: {self.scenario.winner}")
        
        if hasattr(self.scenario, 'team_scores'):
            logger.info(f"Final scores: {dict(self.scenario.team_scores)}")


# Example usage
def run_adversarial_scenarios():
    """Run various adversarial scenarios"""
    
    # Pursuit-Evasion
    logger.info("\n=== Pursuit-Evasion Scenario ===")
    pe_scenario = AdversarialScenario(
        "pursuit_evasion",
        {'num_pursuers': 4, 'num_evaders': 2}
    )
    pe_scenario.run_episode(max_time=120.0)
    
    # Territorial Defense
    logger.info("\n=== Territorial Defense Scenario ===")
    td_scenario = AdversarialScenario(
        "territorial_defense",
        {'num_defenders': 5, 'num_attackers': 4}
    )
    td_scenario.run_episode(max_time=180.0)
    
    # Resource Gathering
    logger.info("\n=== Resource Gathering Scenario ===")
    rg_scenario = AdversarialScenario(
        "resource_gathering",
        {'num_teams': 3, 'agents_per_team': 3}
    )
    rg_scenario.run_episode(max_time=180.0)