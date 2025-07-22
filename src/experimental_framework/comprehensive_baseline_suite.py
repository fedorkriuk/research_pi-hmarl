"""
Comprehensive Baseline Suite for Q1 Publication Standards
All required baselines for fair comparison in top-tier venues
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import json

@dataclass
class BaselineConfig:
    """Configuration for baseline algorithms"""
    num_agents: int
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    update_interval: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaselineAlgorithm(ABC):
    """Abstract base class for all baseline algorithms"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.training_steps = 0
        
    @abstractmethod
    def select_action(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select actions for all agents"""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update the algorithm with a batch of experiences"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model checkpoint"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model checkpoint"""
        pass

class IPPOBaseline(BaselineAlgorithm):
    """
    Independent PPO - Essential Q1 baseline
    Each agent runs PPO independently without communication
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__(config)
        
        # Independent actor-critic networks for each agent
        self.actors = []
        self.critics = []
        self.optimizers = []
        
        for i in range(config.num_agents):
            actor = self._build_actor_network()
            critic = self._build_critic_network()
            
            self.actors.append(actor)
            self.critics.append(critic)
            
            optimizer = optim.Adam(
                list(actor.parameters()) + list(critic.parameters()),
                lr=config.learning_rate
            )
            self.optimizers.append(optimizer)
    
    def _build_actor_network(self) -> nn.Module:
        """Build independent actor network"""
        return nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
            nn.Tanh()
        ).to(self.device)
    
    def _build_critic_network(self) -> nn.Module:
        """Build independent critic network"""
        return nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        ).to(self.device)
    
    def select_action(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Independent action selection for each agent"""
        actions = []
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            for i, (actor, state) in enumerate(zip(self.actors, states_tensor)):
                action = actor(state)
                if not deterministic:
                    # Add exploration noise
                    action += torch.randn_like(action) * 0.1
                actions.append(action.cpu().numpy())
        
        return np.array(actions)
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """PPO update for each agent independently"""
        losses = {'actor_loss': [], 'critic_loss': []}
        
        # Update each agent independently
        for i in range(self.config.num_agents):
            # Extract agent i's data
            states = torch.FloatTensor(batch['states'][:, i]).to(self.device)
            actions = torch.FloatTensor(batch['actions'][:, i]).to(self.device)
            rewards = torch.FloatTensor(batch['rewards'][:, i]).to(self.device)
            next_states = torch.FloatTensor(batch['next_states'][:, i]).to(self.device)
            dones = torch.FloatTensor(batch['dones'][:, i]).to(self.device)
            
            # Compute advantages
            with torch.no_grad():
                values = self.critics[i](states).squeeze()
                next_values = self.critics[i](next_states).squeeze()
                advantages = rewards + self.config.gamma * next_values * (1 - dones) - values
            
            # PPO actor loss
            log_probs = -((self.actors[i](states) - actions) ** 2).sum(dim=1)
            actor_loss = -(log_probs * advantages).mean()
            
            # Critic loss
            value_pred = self.critics[i](states).squeeze()
            value_target = rewards + self.config.gamma * next_values * (1 - dones)
            critic_loss = nn.MSELoss()(value_pred, value_target.detach())
            
            # Update
            total_loss = actor_loss + 0.5 * critic_loss
            self.optimizers[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actors[i].parameters()) + list(self.critics[i].parameters()), 
                0.5
            )
            self.optimizers[i].step()
            
            losses['actor_loss'].append(actor_loss.item())
            losses['critic_loss'].append(critic_loss.item())
        
        self.training_steps += 1
        
        return {
            'actor_loss': np.mean(losses['actor_loss']),
            'critic_loss': np.mean(losses['critic_loss'])
        }
    
    def save(self, path: str):
        """Save all agent models"""
        checkpoint = {
            'config': self.config,
            'training_steps': self.training_steps,
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'optimizers': [opt.state_dict() for opt in self.optimizers]
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load all agent models"""
        checkpoint = torch.load(path, map_location=self.device)
        self.training_steps = checkpoint['training_steps']
        
        for i in range(self.config.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizers'][i])

class IQLBaseline(BaselineAlgorithm):
    """
    Independent Q-Learning - Second essential Q1 baseline
    Each agent runs Q-learning independently
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__(config)
        
        # Independent Q-networks for each agent
        self.q_networks = []
        self.target_networks = []
        self.optimizers = []
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        for i in range(config.num_agents):
            q_net = self._build_q_network()
            target_net = self._build_q_network()
            target_net.load_state_dict(q_net.state_dict())
            
            self.q_networks.append(q_net)
            self.target_networks.append(target_net)
            self.optimizers.append(optim.Adam(q_net.parameters(), lr=config.learning_rate))
    
    def _build_q_network(self) -> nn.Module:
        """Build Q-network for discrete actions"""
        return nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim)
        ).to(self.device)
    
    def select_action(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Epsilon-greedy action selection"""
        actions = []
        
        for i, state in enumerate(states):
            if not deterministic and np.random.rand() < self.epsilon:
                # Random action
                action = np.random.randint(0, self.config.action_dim)
            else:
                # Greedy action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_networks[i](state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            actions.append(action)
        
        return np.array(actions)
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Q-learning update for each agent"""
        losses = []
        
        for i in range(self.config.num_agents):
            states = torch.FloatTensor(batch['states'][:, i]).to(self.device)
            actions = torch.LongTensor(batch['actions'][:, i]).to(self.device)
            rewards = torch.FloatTensor(batch['rewards'][:, i]).to(self.device)
            next_states = torch.FloatTensor(batch['next_states'][:, i]).to(self.device)
            dones = torch.FloatTensor(batch['dones'][:, i]).to(self.device)
            
            # Current Q values
            current_q = self.q_networks[i](states).gather(1, actions.unsqueeze(1)).squeeze()
            
            # Target Q values
            with torch.no_grad():
                next_q = self.target_networks[i](next_states).max(dim=1)[0]
                target_q = rewards + self.config.gamma * next_q * (1 - dones)
            
            # Loss
            loss = nn.MSELoss()(current_q, target_q)
            
            # Update
            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), 1.0)
            self.optimizers[i].step()
            
            losses.append(loss.item())
        
        # Update target networks
        if self.training_steps % self.config.update_interval == 0:
            for i in range(self.config.num_agents):
                self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_steps += 1
        
        return {'q_loss': np.mean(losses), 'epsilon': self.epsilon}
    
    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'config': self.config,
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'q_networks': [net.state_dict() for net in self.q_networks],
            'target_networks': [net.state_dict() for net in self.target_networks],
            'optimizers': [opt.state_dict() for opt in self.optimizers]
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
        
        for i in range(self.config.num_agents):
            self.q_networks[i].load_state_dict(checkpoint['q_networks'][i])
            self.target_networks[i].load_state_dict(checkpoint['target_networks'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizers'][i])

class PhysicsPenaltyMAPPO(BaselineAlgorithm):
    """
    Physics-aware MAPPO baseline with constraint penalties
    Strong baseline that includes physics but without hierarchical structure
    """
    
    def __init__(self, config: BaselineConfig, physics_weight: float = 0.1):
        super().__init__(config)
        self.physics_weight = physics_weight
        
        # Centralized critic, decentralized actors
        self.actor = self._build_actor_network()
        self.critic = self._build_centralized_critic()
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
    
    def _build_actor_network(self) -> nn.Module:
        """Decentralized actor with physics awareness"""
        return nn.Sequential(
            nn.Linear(self.config.state_dim + 3, self.config.hidden_dim),  # +3 for physics state
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
            nn.Tanh()
        ).to(self.device)
    
    def _build_centralized_critic(self) -> nn.Module:
        """Centralized critic sees all agent states"""
        input_dim = self.config.state_dim * self.config.num_agents
        return nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        ).to(self.device)
    
    def compute_physics_penalty(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute physics constraint violations"""
        # Velocity constraints
        velocities = np.linalg.norm(states[:, :, -3:], axis=2)  # Assuming last 3 dims are velocity
        max_velocity = 10.0
        velocity_penalty = np.maximum(0, velocities - max_velocity)
        
        # Action magnitude constraints
        action_magnitude = np.linalg.norm(actions, axis=2)
        max_action = 1.0
        action_penalty = np.maximum(0, action_magnitude - max_action)
        
        # Energy constraints (simplified)
        energy_penalty = 0.01 * action_magnitude ** 2
        
        total_penalty = velocity_penalty + action_penalty + energy_penalty
        return total_penalty
    
    def select_action(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select actions with physics awareness"""
        batch_size = states.shape[0] if states.ndim > 2 else 1
        actions = []
        
        with torch.no_grad():
            for i in range(self.config.num_agents):
                # Add physics features
                physics_features = self._extract_physics_features(states[..., i, :])
                augmented_state = np.concatenate([states[..., i, :], physics_features], axis=-1)
                
                state_tensor = torch.FloatTensor(augmented_state).to(self.device)
                action = self.actor(state_tensor)
                
                if not deterministic:
                    action += torch.randn_like(action) * 0.1
                
                actions.append(action.cpu().numpy())
        
        return np.array(actions).transpose(1, 0, 2) if batch_size > 1 else np.array(actions)
    
    def _extract_physics_features(self, states: np.ndarray) -> np.ndarray:
        """Extract physics-relevant features"""
        # Velocity magnitude
        velocity_mag = np.linalg.norm(states[..., -3:], axis=-1, keepdims=True)
        
        # Acceleration estimate (would need previous state in practice)
        accel_estimate = np.zeros_like(velocity_mag)
        
        # Energy estimate
        energy_estimate = 0.5 * velocity_mag ** 2
        
        return np.concatenate([velocity_mag, accel_estimate, energy_estimate], axis=-1)
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """MAPPO update with physics penalties"""
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Compute physics penalties
        physics_penalties = self.compute_physics_penalty(
            batch['states'], 
            batch['actions']
        )
        physics_penalties = torch.FloatTensor(physics_penalties).to(self.device)
        
        # Modify rewards with physics penalties
        physics_aware_rewards = rewards - self.physics_weight * physics_penalties
        
        # Centralized critic update
        all_states_flat = states.view(states.shape[0], -1)
        all_next_states_flat = next_states.view(next_states.shape[0], -1)
        
        values = self.critic(all_states_flat).squeeze()
        next_values = self.critic(all_next_states_flat).squeeze()
        
        # Compute advantages
        advantages = physics_aware_rewards.mean(dim=1) + \
                     self.config.gamma * next_values * (1 - dones.mean(dim=1)) - values
        
        # Critic loss
        value_target = physics_aware_rewards.mean(dim=1) + \
                      self.config.gamma * next_values * (1 - dones.mean(dim=1))
        critic_loss = nn.MSELoss()(values, value_target.detach())
        
        # Actor loss (simplified PPO)
        actor_losses = []
        for i in range(self.config.num_agents):
            physics_features = self._extract_physics_features(batch['states'][:, i])
            augmented_states = np.concatenate([batch['states'][:, i], physics_features], axis=-1)
            augmented_states = torch.FloatTensor(augmented_states).to(self.device)
            
            pred_actions = self.actor(augmented_states)
            log_probs = -((pred_actions - actions[:, i]) ** 2).sum(dim=1)
            actor_loss = -(log_probs * advantages.detach()).mean()
            actor_losses.append(actor_loss)
        
        total_actor_loss = torch.stack(actor_losses).mean()
        
        # Update networks
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        self.training_steps += 1
        
        return {
            'actor_loss': total_actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'physics_penalty': physics_penalties.mean().item()
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'config': self.config,
            'physics_weight': self.physics_weight,
            'training_steps': self.training_steps,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.physics_weight = checkpoint['physics_weight']
        self.training_steps = checkpoint['training_steps']
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

class ComprehensiveBaselineSuite:
    """
    Complete baseline suite for Q1 publication standards
    """
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.baselines = {
            'IPPO': IPPOBaseline(config),
            'IQL': IQLBaseline(config),
            'Physics-MAPPO': PhysicsPenaltyMAPPO(config),
            # Additional baselines would be implemented similarly:
            # 'QMIX': QMIXBaseline(config),
            # 'MADDPG': MADDPGBaseline(config),
            # 'SOTA-Physics-RL': SOTAPhysicsRLBaseline(config),
            # 'HAD': HierarchicalActorDecomposition(config),
            # 'HC-MARL': HierarchicalCentralizedMARL(config),
            # 'Random': RandomBaseline(config),
            # 'Centralized-Optimal': CentralizedOptimalBaseline(config),
            # 'Human-Expert': HumanExpertBaseline(config)
        }
    
    def get_baseline(self, name: str) -> BaselineAlgorithm:
        """Get specific baseline algorithm"""
        if name not in self.baselines:
            raise ValueError(f"Baseline {name} not found. Available: {list(self.baselines.keys())}")
        return self.baselines[name]
    
    def list_baselines(self) -> List[str]:
        """List all available baselines"""
        return list(self.baselines.keys())
    
    def validate_q1_requirements(self) -> Dict[str, bool]:
        """Check if Q1 baseline requirements are met"""
        required_baselines = [
            'IPPO', 'IQL', 'QMIX', 'MADDPG', 'MAPPO',
            'Physics-MAPPO', 'SOTA-Physics-RL', 'HAD', 'HC-MARL',
            'Random', 'Centralized-Optimal'
        ]
        
        validation = {
            baseline: baseline in self.baselines 
            for baseline in required_baselines
        }
        
        validation['all_required_present'] = all(validation.values())
        validation['num_baselines'] = len(self.baselines)
        validation['q1_compliant'] = validation['all_required_present'] and \
                                    validation['num_baselines'] >= 8
        
        return validation