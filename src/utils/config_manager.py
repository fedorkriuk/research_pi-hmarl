"""
Configuration Management System for PI-HMARL

This module provides a centralized configuration management system that supports
YAML configuration files, environment variable overrides, and validation.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
import logging


@dataclass
class Config:
    """Base configuration class for type hints and validation."""
    
    # Environment settings
    env_name: str = "CartPole-v1"
    max_episode_steps: int = 500
    
    # Training settings
    num_episodes: int = 1000
    learning_rate: float = 3e-4
    batch_size: int = 32
    gamma: float = 0.99
    
    # Physics-informed settings
    physics_loss_weight: float = 0.1
    physics_regularization: bool = True
    
    # Multi-agent settings
    num_agents: int = 2
    agent_communication: bool = True
    hierarchical_levels: int = 2
    
    # Logging and experiment settings
    log_level: str = "INFO"
    experiment_name: str = "pi_hmarl_experiment"
    save_frequency: int = 100
    
    # Device settings
    device: str = "auto"
    use_cuda: bool = True


class ConfigManager:
    """
    Centralized configuration management system for PI-HMARL.
    
    Features:
    - YAML configuration file loading
    - Environment variable overrides
    - Configuration validation
    - Hierarchical configuration merging
    - Type conversion and validation
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self._config: Optional[DictConfig] = None
        
        # Default configuration paths
        self.default_config_paths = [
            Path(__file__).parent.parent.parent / "configs" / "default_config.yaml",
            Path.cwd() / "configs" / "default_config.yaml",
            Path.cwd() / "config.yaml"
        ]
        
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            # Load base configuration
            base_config = self._load_base_config()
            
            # Load user configuration if provided
            if self.config_path:
                user_config = self._load_yaml_config(self.config_path)
                base_config = OmegaConf.merge(base_config, user_config)
            
            # Apply environment variable overrides
            env_overrides = self._load_env_overrides()
            if env_overrides:
                base_config = OmegaConf.merge(base_config, env_overrides)
            
            self._config = base_config
            self.logger.info(f"Configuration loaded successfully from {self.config_path or 'default'}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _load_base_config(self) -> DictConfig:
        """Load base configuration from default paths."""
        for config_path in self.default_config_paths:
            if config_path.exists():
                self.logger.info(f"Loading base configuration from {config_path}")
                return self._load_yaml_config(config_path)
        
        # If no config file found, use default values
        self.logger.warning("No configuration file found, using default values")
        return OmegaConf.structured(Config())
    
    def _load_yaml_config(self, config_path: Union[str, Path]) -> DictConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Loaded configuration as DictConfig
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            return OmegaConf.create(yaml_config)
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading {config_path}: {e}")
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.
        
        Environment variables should be prefixed with 'PIHMARL_' and use
        double underscores for nested configurations.
        
        Example:
            PIHMARL_LEARNING_RATE=0.001
            PIHMARL_AGENT__NUM_AGENTS=4
        """
        overrides = {}
        prefix = "PIHMARL_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Handle nested configurations
                if "__" in config_key:
                    keys = config_key.split("__")
                    nested_dict = overrides
                    for k in keys[:-1]:
                        if k not in nested_dict:
                            nested_dict[k] = {}
                        nested_dict = nested_dict[k]
                    nested_dict[keys[-1]] = self._convert_env_value(value)
                else:
                    overrides[config_key] = self._convert_env_value(value)
        
        return overrides
    
    def _convert_env_value(self, value: str) -> Any:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value with appropriate type
        """
        # Handle boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Handle numeric values
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Handle lists (comma-separated)
        if "," in value:
            return [self._convert_env_value(v.strip()) for v in value.split(",")]
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested access)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception as e:
            self.logger.warning(f"Error accessing config key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested access)
            value: Value to set
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        
        try:
            OmegaConf.update(self._config, key, value)
        except Exception as e:
            raise ConfigurationError(f"Error setting config key '{key}': {e}")
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration file
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w') as f:
                OmegaConf.save(self._config, f)
            
            self.logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration to {path}: {e}")
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if self._config is None:
            return False
        
        try:
            # Validate required fields
            required_fields = ["env_name", "num_episodes", "learning_rate"]
            for field in required_fields:
                if self.get(field) is None:
                    self.logger.error(f"Required field '{field}' is missing")
                    return False
            
            # Validate value ranges
            if self.get("learning_rate", 0) <= 0:
                self.logger.error("Learning rate must be positive")
                return False
            
            if self.get("num_episodes", 0) <= 0:
                self.logger.error("Number of episodes must be positive")
                return False
            
            if self.get("batch_size", 0) <= 0:
                self.logger.error("Batch size must be positive")
                return False
            
            if not 0 <= self.get("gamma", 0) <= 1:
                self.logger.error("Gamma must be between 0 and 1")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        
        return OmegaConf.to_container(self._config, resolve=True)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of configuration."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Global ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Load configuration and return manager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value using global manager.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    Set configuration value using global manager.
    
    Args:
        key: Configuration key
        value: Value to set
    """
    get_config_manager().set(key, value)