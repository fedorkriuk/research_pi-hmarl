"""Training algorithms for PI-HMARL"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PIHMARLTrainer:
    """Physics-Informed Hierarchical Multi-Agent RL Trainer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        from src.models import PhysicsInformedModel
        self.model = PhysicsInformedModel(
            state_dim=config.get('state_dim', 12),
            action_dim=config.get('action_dim', 4),
            hidden_dim=config.get('hidden_dim', 64)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        
        self.physics_weight = config.get('physics_weight', 0.1)
        self.batch_size = config.get('batch_size', 32)
        
        logger.info(f"Initialized PIHMARLTrainer on device: {self.device}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform one training step"""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Forward pass
        pred_actions, values = self.model(states)
        
        # Compute losses
        action_loss = nn.MSELoss()(pred_actions, actions)
        value_loss = nn.MSELoss()(values, rewards)
        physics_loss = self.model.compute_physics_loss(states, actions, next_states)
        
        # Total loss
        total_loss = action_loss + value_loss + self.physics_weight * physics_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, env, mas, num_episodes: int = 100) -> Dict[str, List[float]]:
        """Train the model"""
        logger.info(f"Starting training for {num_episodes} episodes")
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(num_episodes):
            states = env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            for step in range(100):  # Max steps per episode
                # Get actions from agents
                actions = []
                for i, agent in enumerate(mas.agents):
                    action = agent.select_action(states[i])
                    actions.append(action)
                
                # Environment step
                next_states, rewards, dones, info = env.step(actions)
                
                # Create training batch
                batch = {
                    'states': torch.FloatTensor(np.array(states)),
                    'actions': torch.FloatTensor(np.array(actions)),
                    'rewards': torch.FloatTensor(np.array(rewards)).unsqueeze(1),
                    'next_states': torch.FloatTensor(np.array(next_states)),
                    'dones': torch.FloatTensor(np.array(dones)).unsqueeze(1)
                }
                
                # Training step
                loss = self.train_step(batch)
                
                episode_reward += sum(rewards)
                episode_loss += loss
                steps += 1
                
                states = next_states
                
                if all(dones):
                    break
            
            avg_loss = episode_loss / steps if steps > 0 else 0
            episode_rewards.append(episode_reward)
            episode_losses.append(avg_loss)
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, Loss={avg_loss:.4f}")
        
        logger.info("Training completed")
        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses
        }

__all__ = ['PIHMARLTrainer']