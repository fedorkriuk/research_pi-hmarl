"""Configuration Management System

This module handles configuration management, validation, and
environment-specific settings for the PI-HMARL system.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from jsonschema import validate, ValidationError
import hashlib
import time

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    SIMULATION = "simulation"


@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    name: str
    version: str
    description: str
    properties: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    environment: Environment
    
    # System settings
    num_agents: int = 5
    map_size: Tuple[float, float, float] = (1000.0, 1000.0, 200.0)
    physics_enabled: bool = True
    real_time: bool = False
    
    # Hardware settings
    device: str = "cpu"
    num_workers: int = 4
    memory_limit: Optional[int] = None  # MB
    
    # Network settings
    communication_enabled: bool = True
    communication_range: float = 5000.0  # meters
    bandwidth_limit: float = 10e6  # bps
    latency_target: float = 0.1  # seconds
    
    # Safety settings
    safety_checks: bool = True
    emergency_protocols: bool = True
    geofence_enabled: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    metrics_enabled: bool = True
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        'hierarchical_control': True,
        'energy_optimization': True,
        'adaptive_formation': True,
        'predictive_maintenance': False,
        'swarm_intelligence': True
    })


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    name: str
    version: str
    environment: Environment
    
    # Service configuration
    services: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Resource allocation
    resources: Dict[str, Any] = field(default_factory=lambda: {
        'cpu_limit': '4',
        'memory_limit': '8Gi',
        'gpu_enabled': False,
        'storage': '10Gi'
    })
    
    # Scaling configuration
    scaling: Dict[str, Any] = field(default_factory=lambda: {
        'min_replicas': 1,
        'max_replicas': 10,
        'target_cpu_utilization': 70,
        'scale_up_rate': 2,
        'scale_down_rate': 1
    })
    
    # Security configuration
    security: Dict[str, Any] = field(default_factory=lambda: {
        'encryption_enabled': True,
        'tls_enabled': True,
        'auth_required': True,
        'api_rate_limit': 1000
    })
    
    # Monitoring configuration
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'metrics_port': 9090,
        'health_check_interval': 30,
        'log_retention_days': 30,
        'alerts_enabled': True
    })


class ConfigValidator:
    """Configuration validator"""
    
    def __init__(self):
        """Initialize config validator"""
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration schemas"""
        schemas = {
            'environment': {
                'type': 'object',
                'properties': {
                    'environment': {'type': 'string', 'enum': [e.value for e in Environment]},
                    'num_agents': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
                    'map_size': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'minItems': 3,
                        'maxItems': 3
                    },
                    'device': {'type': 'string', 'enum': ['cpu', 'cuda', 'mps']},
                    'communication_range': {'type': 'number', 'minimum': 0},
                    'safety_checks': {'type': 'boolean'}
                },
                'required': ['environment', 'num_agents']
            },
            'deployment': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'version': {'type': 'string', 'pattern': '^\\d+\\.\\d+\\.\\d+$'},
                    'environment': {'type': 'string'},
                    'services': {'type': 'object'},
                    'resources': {
                        'type': 'object',
                        'properties': {
                            'cpu_limit': {'type': 'string'},
                            'memory_limit': {'type': 'string'},
                            'gpu_enabled': {'type': 'boolean'}
                        }
                    }
                },
                'required': ['name', 'version', 'environment']
            }
        }
        
        return schemas
    
    def validate(self, config: Dict[str, Any], schema_name: str) -> bool:
        """Validate configuration against schema
        
        Args:
            config: Configuration to validate
            schema_name: Schema name
            
        Returns:
            Validation status
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        try:
            validate(instance=config, schema=self.schemas[schema_name])
            return True
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def validate_constraints(self, config: Dict[str, Any]) -> List[str]:
        """Validate additional constraints
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Check resource constraints
        if 'num_agents' in config and 'resources' in config:
            num_agents = config['num_agents']
            cpu_limit = float(config['resources'].get('cpu_limit', '1').rstrip('m'))
            
            if num_agents > 10 and cpu_limit < 2:
                violations.append(
                    f"Insufficient CPU for {num_agents} agents (need at least 2 cores)"
                )
        
        # Check communication constraints
        if config.get('communication_enabled') and config.get('num_agents', 0) > 50:
            bandwidth = config.get('bandwidth_limit', 10e6)
            if bandwidth < 100e6:
                violations.append(
                    "Bandwidth may be insufficient for large-scale communication"
                )
        
        # Check safety constraints
        if config.get('environment') == 'production' and not config.get('safety_checks'):
            violations.append("Safety checks must be enabled in production")
        
        return violations


class ConfigManager:
    """Main configuration manager"""
    
    def __init__(
        self,
        config_dir: Union[str, Path] = "configs",
        environment: Optional[Environment] = None
    ):
        """Initialize configuration manager
        
        Args:
            config_dir: Configuration directory
            environment: Current environment
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.environment = environment or self._detect_environment()
        self.validator = ConfigValidator()
        
        # Configuration cache
        self._config_cache = {}
        self._config_hash = {}
        
        # Load base configurations
        self.base_config = self._load_base_config()
        self.env_config = self._load_environment_config()
        
        logger.info(f"ConfigManager initialized for {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env_var = os.environ.get('PI_HMARL_ENV', 'development').lower()
        
        try:
            return Environment(env_var)
        except ValueError:
            logger.warning(f"Unknown environment: {env_var}, defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        base_path = self.config_dir / "base.yaml"
        
        if base_path.exists():
            with open(base_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default base configuration
            return {
                'system': {
                    'name': 'PI-HMARL',
                    'version': '1.0.0',
                    'description': 'Physics-Informed Hierarchical Multi-Agent RL'
                },
                'defaults': {
                    'num_agents': 5,
                    'episode_length': 1000,
                    'physics_timestep': 0.01
                }
            }
    
    def _load_environment_config(self) -> EnvironmentConfig:
        """Load environment-specific configuration"""
        env_path = self.config_dir / f"{self.environment.value}.yaml"
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                config_data = yaml.safe_load(f)
                return EnvironmentConfig(**config_data)
        else:
            # Default environment configuration
            defaults = {
                Environment.DEVELOPMENT: {
                    'num_agents': 5,
                    'real_time': False,
                    'safety_checks': True,
                    'log_level': 'DEBUG'
                },
                Environment.TESTING: {
                    'num_agents': 3,
                    'real_time': False,
                    'safety_checks': True,
                    'log_level': 'INFO'
                },
                Environment.PRODUCTION: {
                    'num_agents': 10,
                    'real_time': True,
                    'safety_checks': True,
                    'log_level': 'WARNING',
                    'emergency_protocols': True
                }
            }
            
            env_defaults = defaults.get(self.environment, {})
            return EnvironmentConfig(environment=self.environment, **env_defaults)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Args:
            key: Configuration key (dot notation)
            default: Default value
            
        Returns:
            Configuration value
        """
        # Check cache
        if key in self._config_cache:
            return self._config_cache[key]
        
        # Parse key
        parts = key.split('.')
        
        # Search in configs
        value = self._search_config(parts, self.env_config)
        if value is None:
            value = self._search_config(parts, self.base_config)
        
        # Cache result
        if value is not None:
            self._config_cache[key] = value
        
        return value if value is not None else default
    
    def _search_config(self, parts: List[str], config: Any) -> Any:
        """Search configuration recursively
        
        Args:
            parts: Key parts
            config: Configuration object
            
        Returns:
            Value or None
        """
        if not parts:
            return config
        
        if isinstance(config, dict):
            if parts[0] in config:
                return self._search_config(parts[1:], config[parts[0]])
        elif hasattr(config, parts[0]):
            return self._search_config(parts[1:], getattr(config, parts[0]))
        
        return None
    
    def set(self, key: str, value: Any):
        """Set configuration value
        
        Args:
            key: Configuration key
            value: Value to set
        """
        # Update cache
        self._config_cache[key] = value
        
        # Update environment config
        parts = key.split('.')
        self._update_config(parts, value, self.env_config)
        
        # Clear hash to trigger re-save
        self._config_hash.clear()
    
    def _update_config(self, parts: List[str], value: Any, config: Any):
        """Update configuration recursively
        
        Args:
            parts: Key parts
            value: Value to set
            config: Configuration object
        """
        if len(parts) == 1:
            if isinstance(config, dict):
                config[parts[0]] = value
            else:
                setattr(config, parts[0], value)
        else:
            if isinstance(config, dict):
                if parts[0] not in config:
                    config[parts[0]] = {}
                self._update_config(parts[1:], value, config[parts[0]])
            else:
                if not hasattr(config, parts[0]):
                    setattr(config, parts[0], {})
                self._update_config(parts[1:], value, getattr(config, parts[0]))
    
    def save(self):
        """Save current configuration"""
        # Save environment config
        env_path = self.config_dir / f"{self.environment.value}.yaml"
        
        config_dict = asdict(self.env_config)
        
        # Calculate hash
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        
        # Only save if changed
        if self._config_hash.get(self.environment.value) != config_hash:
            with open(env_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            self._config_hash[self.environment.value] = config_hash
            
            # Backup
            backup_path = self.config_dir / f"{self.environment.value}_{int(time.time())}.yaml.bak"
            with open(backup_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {env_path}")
    
    def load_deployment_config(self, name: str) -> DeploymentConfig:
        """Load deployment configuration
        
        Args:
            name: Deployment name
            
        Returns:
            Deployment configuration
        """
        deploy_path = self.config_dir / "deployments" / f"{name}.yaml"
        
        if not deploy_path.exists():
            raise FileNotFoundError(f"Deployment config not found: {name}")
        
        with open(deploy_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Validate
        if not self.validator.validate(config_data, 'deployment'):
            raise ValueError(f"Invalid deployment configuration: {name}")
        
        return DeploymentConfig(**config_data)
    
    def create_deployment_config(
        self,
        name: str,
        version: str,
        **kwargs
    ) -> DeploymentConfig:
        """Create new deployment configuration
        
        Args:
            name: Deployment name
            version: Version
            **kwargs: Additional parameters
            
        Returns:
            Deployment configuration
        """
        config = DeploymentConfig(
            name=name,
            version=version,
            environment=self.environment,
            **kwargs
        )
        
        # Validate
        config_dict = asdict(config)
        if not self.validator.validate(config_dict, 'deployment'):
            raise ValueError("Invalid deployment configuration")
        
        # Check constraints
        violations = self.validator.validate_constraints(config_dict)
        if violations:
            logger.warning(f"Configuration constraint violations: {violations}")
        
        # Save
        deploy_dir = self.config_dir / "deployments"
        deploy_dir.mkdir(exist_ok=True)
        
        deploy_path = deploy_dir / f"{name}.yaml"
        with open(deploy_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Created deployment configuration: {name}")
        
        return config
    
    def export_config(self, format: str = "yaml") -> str:
        """Export current configuration
        
        Args:
            format: Export format (yaml, json)
            
        Returns:
            Configuration string
        """
        config = {
            'environment': self.environment.value,
            'base': self.base_config,
            'env_specific': asdict(self.env_config),
            'timestamp': time.time()
        }
        
        if format == "yaml":
            return yaml.dump(config, default_flow_style=False)
        elif format == "json":
            return json.dumps(config, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def import_config(self, config_str: str, format: str = "yaml"):
        """Import configuration
        
        Args:
            config_str: Configuration string
            format: Import format
        """
        if format == "yaml":
            config = yaml.safe_load(config_str)
        elif format == "json":
            config = json.loads(config_str)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Update base config
        if 'base' in config:
            self.base_config.update(config['base'])
        
        # Update environment config
        if 'env_specific' in config:
            self.env_config = EnvironmentConfig(**config['env_specific'])
        
        # Clear cache
        self._config_cache.clear()
        
        logger.info("Configuration imported successfully")
    
    def diff_configs(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two configurations
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Differences
        """
        diffs = {}
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            if key not in config1:
                diffs[key] = {'added': config2[key]}
            elif key not in config2:
                diffs[key] = {'removed': config1[key]}
            elif config1[key] != config2[key]:
                if isinstance(config1[key], dict) and isinstance(config2[key], dict):
                    nested_diff = self.diff_configs(config1[key], config2[key])
                    if nested_diff:
                        diffs[key] = nested_diff
                else:
                    diffs[key] = {
                        'old': config1[key],
                        'new': config2[key]
                    }
        
        return diffs


# Configuration templates
class ConfigTemplates:
    """Pre-defined configuration templates"""
    
    @staticmethod
    def small_scale_testing() -> EnvironmentConfig:
        """Configuration for small-scale testing"""
        return EnvironmentConfig(
            environment=Environment.TESTING,
            num_agents=3,
            map_size=(500.0, 500.0, 100.0),
            real_time=False,
            device="cpu",
            communication_range=1000.0,
            safety_checks=True,
            log_level="DEBUG"
        )
    
    @staticmethod
    def large_scale_simulation() -> EnvironmentConfig:
        """Configuration for large-scale simulation"""
        return EnvironmentConfig(
            environment=Environment.SIMULATION,
            num_agents=100,
            map_size=(5000.0, 5000.0, 500.0),
            real_time=False,
            device="cuda",
            num_workers=8,
            communication_enabled=True,
            communication_range=2000.0,
            safety_checks=True,
            features={
                'hierarchical_control': True,
                'energy_optimization': True,
                'adaptive_formation': True,
                'swarm_intelligence': True
            }
        )
    
    @staticmethod
    def production_deployment() -> EnvironmentConfig:
        """Configuration for production deployment"""
        return EnvironmentConfig(
            environment=Environment.PRODUCTION,
            num_agents=20,
            map_size=(2000.0, 2000.0, 200.0),
            real_time=True,
            device="cuda",
            safety_checks=True,
            emergency_protocols=True,
            geofence_enabled=True,
            log_level="WARNING",
            metrics_enabled=True,
            features={
                'hierarchical_control': True,
                'energy_optimization': True,
                'adaptive_formation': True,
                'predictive_maintenance': True,
                'swarm_intelligence': True
            }
        )