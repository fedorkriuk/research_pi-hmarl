"""Configuration Manager for PI-HMARL Framework

This module provides a centralized configuration management system
using YAML files with support for hierarchical configs and runtime overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import logging
from omegaconf import OmegaConf, DictConfig


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and logging"""
    name: str = "pi_hmarl_experiment"
    project: str = "pi-hmarl"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    log_dir: str = "./experiments"
    wandb_entity: Optional[str] = None
    wandb_project: str = "pi-hmarl"
    save_frequency: int = 1000
    checkpoint_frequency: int = 5000


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    algorithm: str = "PI-HMARL"
    num_agents: int = 10
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1000000
    warmup_steps: int = 10000
    update_frequency: int = 4
    gradient_clip: float = 1.0
    num_workers: int = 4
    use_gpu: bool = True
    seed: int = 42


@dataclass
class PhysicsConfig:
    """Configuration for physics simulation parameters"""
    engine: str = "pybullet"
    timestep: float = 0.01
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    enable_collision: bool = True
    enable_energy_constraints: bool = True
    enable_dynamics_constraints: bool = True
    max_velocity: float = 20.0  # m/s
    max_acceleration: float = 10.0  # m/s^2
    min_separation_distance: float = 2.0  # meters
    energy_penalty_weight: float = 0.1
    physics_loss_weight: float = 0.3


@dataclass
class AgentConfig:
    """Configuration for agent parameters"""
    observation_dim: int = 64
    action_dim: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    attention_heads: int = 8
    attention_dim: int = 64
    use_hierarchical: bool = True
    meta_controller_update_freq: int = 10
    skill_duration: int = 5
    communication_range: float = 50.0  # meters
    sensor_range: float = 30.0  # meters
    battery_capacity: float = 5000.0  # mAh
    max_speed: float = 19.0  # m/s (DJI Mavic 3 spec)
    mass: float = 0.895  # kg (DJI Mavic 3 spec)


class ConfigManager:
    """Manages configuration loading, validation, and access for PI-HMARL"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize ConfigManager with optional config file path
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(__file__).parent.parent.parent / "configs"
        self.default_config_path = self.config_dir / "default_config.yaml"
        
        # Initialize configuration
        self.config = self._load_default_config()
        
        # Load custom config if provided
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> DictConfig:
        """Load default configuration"""
        default_config = OmegaConf.structured({
            "experiment": ExperimentConfig(),
            "training": TrainingConfig(),
            "physics": PhysicsConfig(),
            "agent": AgentConfig(),
        })
        
        # Load from default YAML if exists
        if self.default_config_path.exists():
            yaml_config = OmegaConf.load(self.default_config_path)
            default_config = OmegaConf.merge(default_config, yaml_config)
        
        return default_config
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            custom_config = OmegaConf.load(config_path)
            self.config = OmegaConf.merge(self.config, custom_config)
            self.logger.info(f"Loaded configuration from: {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_config(self, save_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file
        
        Args:
            save_path: Path where to save the configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            OmegaConf.save(self.config, f)
        
        self.logger.info(f"Configuration saved to: {save_path}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
        """
        update_config = OmegaConf.create(updates)
        self.config = OmegaConf.merge(self.config, update_config)
        self.logger.info(f"Configuration updated with: {updates}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key
        
        Args:
            key: Dot-separated key path (e.g., 'training.learning_rate')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def get_experiment_config(self) -> ExperimentConfig:
        """Get experiment configuration"""
        return OmegaConf.to_object(self.config.experiment)
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration"""
        return OmegaConf.to_object(self.config.training)
    
    def get_physics_config(self) -> PhysicsConfig:
        """Get physics configuration"""
        return OmegaConf.to_object(self.config.physics)
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration"""
        return OmegaConf.to_object(self.config.agent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return OmegaConf.to_container(self.config)
    
    def validate_config(self) -> bool:
        """Validate configuration for consistency and correctness
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check required fields
            assert self.config.training.num_agents > 0
            assert self.config.training.learning_rate > 0
            assert self.config.physics.timestep > 0
            assert self.config.agent.observation_dim > 0
            assert self.config.agent.action_dim > 0
            
            # Check physics constraints
            if self.config.physics.enable_collision:
                assert self.config.physics.min_separation_distance > 0
            
            # Check hierarchical settings
            if self.config.agent.use_hierarchical:
                assert self.config.agent.meta_controller_update_freq > 0
                assert self.config.agent.skill_duration > 0
            
            self.logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(\n{OmegaConf.to_yaml(self.config)})"


# Convenience function for quick config access
def load_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Load configuration from file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)
EOF < /dev/null