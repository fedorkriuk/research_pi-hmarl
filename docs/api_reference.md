# API Reference

## Core Modules

### pi_hmarl.environment

#### MultiAgentEnvironment

Main environment class for multi-agent drone simulation.

```python
class MultiAgentEnvironment:
    """Multi-agent drone environment with physics simulation"""
    
    def __init__(
        self,
        num_agents: int,
        map_size: Tuple[float, float, float],
        physics_config: Optional[Dict[str, Any]] = None,
        safety_config: Optional[Dict[str, Any]] = None,
        communication_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize environment.
        
        Args:
            num_agents: Number of agents in environment
            map_size: Environment dimensions (x, y, z) in meters
            physics_config: Physics simulation parameters
            safety_config: Safety constraint parameters
            communication_config: Communication system parameters
        """
    
    def reset(self) -> Dict[int, np.ndarray]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observations for all agents
        """
    
    def step(
        self, 
        actions: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            actions: Actions for each agent
            
        Returns:
            observations: Next observations
            rewards: Rewards for each agent
            dones: Episode termination flags
            info: Additional information
        """
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array')
            
        Returns:
            Rendered frame (if mode='rgb_array')
        """
```

### pi_hmarl.agents

#### HierarchicalAgent

Hierarchical reinforcement learning agent.

```python
class HierarchicalAgent:
    """Hierarchical multi-level agent"""
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        levels: List[str] = ['high', 'mid', 'low'],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hierarchical agent.
        
        Args:
            agent_id: Unique agent identifier
            obs_dim: Observation dimension
            action_dim: Action dimension
            levels: Hierarchy levels
            config: Agent configuration
        """
    
    def act(
        self, 
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Select action given observation.
        
        Args:
            observation: Current observation
            deterministic: Use deterministic policy
            
        Returns:
            Selected action
        """
    
    def update(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Update agent parameters.
        
        Args:
            batch: Training batch
            
        Returns:
            Training metrics
        """
    
    def save(self, path: str):
        """Save agent parameters"""
    
    def load(self, path: str):
        """Load agent parameters"""
```

### pi_hmarl.physics

#### DronePhysics

Physics simulation for drone dynamics.

```python
class DronePhysics:
    """Realistic drone physics simulation"""
    
    def __init__(
        self,
        mass: float = 1.5,
        arm_length: float = 0.25,
        motor_thrust_max: float = 10.0,
        drag_coefficient: float = 0.5
    ):
        """
        Initialize drone physics.
        
        Args:
            mass: Drone mass in kg
            arm_length: Distance from center to motors
            motor_thrust_max: Maximum thrust per motor
            drag_coefficient: Aerodynamic drag coefficient
        """
    
    def update(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: float,
        wind: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update drone state.
        
        Args:
            state: Current state [pos, vel, orient, ang_vel]
            control: Control inputs [thrust, roll, pitch, yaw]
            dt: Time step
            wind: Wind velocity vector
            
        Returns:
            Updated state
        """
```

### pi_hmarl.tasks

#### Task

Base task class.

```python
class Task:
    """Base class for all tasks"""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        priority: str = "medium",
        deadline: Optional[float] = None
    ):
        """
        Initialize task.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task
            priority: Task priority (low/medium/high)
            deadline: Task deadline in seconds
        """
    
    def get_reward(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any]
    ) -> float:
        """Calculate task reward"""
    
    def is_complete(
        self,
        state: Dict[str, Any]
    ) -> bool:
        """Check if task is complete"""
```

#### TaskDecomposer

Decomposes complex tasks into subtasks.

```python
class TaskDecomposer:
    """Decomposes tasks into subtasks"""
    
    def decompose(
        self,
        task: Task,
        context: Dict[str, Any]
    ) -> List[Task]:
        """
        Decompose task into subtasks.
        
        Args:
            task: Task to decompose
            context: Environmental context
            
        Returns:
            List of subtasks
        """
```

### pi_hmarl.communication

#### CommunicationProtocol

Inter-agent communication protocol.

```python
class CommunicationProtocol:
    """Communication protocol for multi-agent system"""
    
    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        bandwidth: float = 10e6,
        range_limit: float = 5000.0,
        packet_loss: float = 0.01
    ):
        """
        Initialize communication protocol.
        
        Args:
            agent_id: Agent identifier
            num_agents: Total number of agents
            bandwidth: Available bandwidth in bps
            range_limit: Communication range in meters
            packet_loss: Packet loss probability
        """
    
    def send_message(
        self,
        message: Message,
        priority: MessagePriority = MessagePriority.MEDIUM
    ) -> bool:
        """Send message to other agents"""
    
    def receive_messages(self) -> List[Message]:
        """Receive pending messages"""
```

### pi_hmarl.energy

#### EnergyModel

Battery and energy consumption model.

```python
class EnergyModel:
    """Energy consumption and battery model"""
    
    def __init__(
        self,
        battery_capacity: float = 5000.0,  # mAh
        voltage: float = 11.1,  # V
        discharge_curve: Optional[Callable] = None
    ):
        """
        Initialize energy model.
        
        Args:
            battery_capacity: Battery capacity in mAh
            voltage: Battery voltage
            discharge_curve: Custom discharge curve function
        """
    
    def update(
        self,
        power_draw: float,
        dt: float,
        temperature: float = 20.0
    ) -> Dict[str, float]:
        """
        Update battery state.
        
        Args:
            power_draw: Current power draw in Watts
            dt: Time step
            temperature: Battery temperature in Celsius
            
        Returns:
            Battery state metrics
        """
```

### pi_hmarl.visualization

#### Dashboard

Real-time monitoring dashboard.

```python
class Dashboard:
    """Real-time visualization dashboard"""
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        port: int = 5000
    ):
        """
        Initialize dashboard.
        
        Args:
            config: Dashboard configuration
            port: Web server port
        """
    
    def start(self):
        """Start dashboard server"""
    
    def update_data(
        self,
        data_type: str,
        data: Dict[str, Any]
    ):
        """Update dashboard data"""
    
    def add_visualization(
        self,
        viz_type: str,
        config: Dict[str, Any]
    ):
        """Add new visualization component"""
```

### pi_hmarl.deployment

#### DeploymentManager

Manages system deployment.

```python
class DeploymentManager:
    """Deployment management system"""
    
    def __init__(
        self,
        deployment_type: DeploymentTarget = DeploymentTarget.LOCAL
    ):
        """
        Initialize deployment manager.
        
        Args:
            deployment_type: Target deployment platform
        """
    
    async def deploy(
        self,
        config: Dict[str, Any],
        target: DeploymentTarget
    ) -> DeploymentResult:
        """
        Deploy system.
        
        Args:
            config: Deployment configuration
            target: Deployment target
            
        Returns:
            Deployment result
        """
    
    async def scale(
        self,
        service: str,
        replicas: int
    ) -> bool:
        """Scale service replicas"""
```

## Utilities

### pi_hmarl.utils.replay_buffer

```python
class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int):
        """Initialize replay buffer"""
    
    def push(self, *args):
        """Add experience to buffer"""
    
    def sample(self, batch_size: int) -> List:
        """Sample batch from buffer"""
```

### pi_hmarl.utils.metrics

```python
class MetricsTracker:
    """Track and aggregate metrics"""
    
    def __init__(self):
        """Initialize metrics tracker"""
    
    def record(self, name: str, value: float):
        """Record metric value"""
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get metric statistics"""
```

## Configuration Classes

### EnvironmentConfig

```python
@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    num_agents: int
    map_size: Tuple[float, float, float]
    physics_enabled: bool = True
    real_time: bool = False
    render_mode: str = "human"
```

### AgentConfig

```python
@dataclass
class AgentConfig:
    """Agent configuration"""
    model_type: str = "hierarchical"
    learning_rate: float = 1e-4
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    use_attention: bool = True
```

### TaskConfig

```python
@dataclass
class TaskConfig:
    """Task configuration"""
    task_type: str
    priority: str = "medium"
    timeout: Optional[float] = None
    requirements: Dict[str, Any] = field(default_factory=dict)
```

## Enumerations

### MessageType

```python
class MessageType(Enum):
    """Message types for communication"""
    BROADCAST = "broadcast"
    UNICAST = "unicast"
    MULTICAST = "multicast"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"
```

### TaskStatus

```python
class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### DroneState

```python
class DroneState(Enum):
    """Drone operational state"""
    IDLE = "idle"
    TAKEOFF = "takeoff"
    FLYING = "flying"
    HOVERING = "hovering"
    LANDING = "landing"
    EMERGENCY = "emergency"
    CHARGING = "charging"
```

## Exceptions

```python
class PIHMARLException(Exception):
    """Base exception for PI-HMARL"""
    pass

class EnvironmentException(PIHMARLException):
    """Environment-related exceptions"""
    pass

class AgentException(PIHMARLException):
    """Agent-related exceptions"""
    pass

class CommunicationException(PIHMARLException):
    """Communication-related exceptions"""
    pass

class SafetyViolation(PIHMARLException):
    """Safety constraint violations"""
    pass
```

## Type Definitions

```python
from typing import TypedDict, NewType

# Agent ID type
AgentID = NewType('AgentID', int)

# Task ID type  
TaskID = NewType('TaskID', str)

# State dictionary
class StateDict(TypedDict):
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    battery_level: float

# Action dictionary
class ActionDict(TypedDict):
    thrust: float
    roll: float
    pitch: float
    yaw_rate: float
```

## Constants

```python
# Physics constants
GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.225  # kg/m^3

# Communication constants
DEFAULT_BANDWIDTH = 10e6  # 10 Mbps
DEFAULT_COMM_RANGE = 5000.0  # 5 km
DEFAULT_FREQUENCY = 2.4e9  # 2.4 GHz

# Safety constants
MIN_SEPARATION_DISTANCE = 5.0  # meters
MAX_ALTITUDE = 150.0  # meters
MIN_BATTERY_LEVEL = 0.2  # 20%

# Default configurations
DEFAULT_HIDDEN_DIMS = [256, 128, 64]
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 64
```