# PI-HMARL User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Hardware Integration](#hardware-integration)
6. [Security Configuration](#security-configuration)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

```bash
# Basic installation
pip install pi-hmarl

# With hardware support
pip install pi-hmarl[hardware]

# With all features
pip install pi-hmarl[all]
```

### Quick Example

```python
from src.core import MultiAgentSystem
from src.environments import PhysicsEnvironment
from src.scenarios import SearchRescueScenario

# Create a search and rescue scenario
scenario = SearchRescueScenario(
    area_size=(100.0, 100.0),
    num_victims=5,
    num_agents=3
)

# Run the scenario
for _ in range(1000):
    scenario.step(dt=0.1)
    state = scenario.get_state()
    print(f"Time: {state['time']:.1f}s, Victims rescued: {state['rescued_count']}")
```

## Core Concepts

### 1. Multi-Agent System

The `MultiAgentSystem` class manages a collection of agents:

```python
from src.core import MultiAgentSystem

config = {
    'num_agents': 4,
    'state_dim': 12,
    'action_dim': 4,
    'communication_range': 30.0
}

mas = MultiAgentSystem(config)

# Access individual agents
for agent in mas.agents:
    print(f"Agent {agent.agent_id} at position {agent.position}")
```

### 2. Hierarchical Control

PI-HMARL uses a two-level control hierarchy:

```python
from src.core import HierarchicalController

controller = HierarchicalController(
    state_dim=12,
    action_dim=4,
    num_agents=4
)

# High-level planning
global_state = get_global_state()
high_level_actions = controller.high_level_policy(global_state)

# Low-level control
local_states = get_local_states()
low_level_actions = controller.low_level_policy(local_states, high_level_actions)
```

### 3. Physics-Informed Learning

Integrate physics constraints into the learning process:

```python
from src.models import PhysicsInformedModel

model = PhysicsInformedModel(
    state_dim=12,
    action_dim=4,
    physics_weight=0.1
)

# Training with physics constraints
loss = model.compute_loss(states, actions, rewards, next_states)
physics_loss = model.compute_physics_loss(states, actions, next_states)
total_loss = loss + physics_loss
```

## Basic Usage

### Creating Environments

```python
from src.environments import PhysicsEnvironment

env_config = {
    'num_agents': 3,
    'world_size': (100.0, 100.0, 50.0),
    'physics_dt': 0.01,
    'render': True
}

env = PhysicsEnvironment(env_config)
states = env.reset()

# Run simulation
for _ in range(100):
    actions = [agent.select_action(state) for agent, state in zip(agents, states)]
    next_states, rewards, dones, info = env.step(actions)
```

### Training Agents

```python
from src.training import PIHMARLTrainer

trainer_config = {
    'num_agents': 4,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'physics_weight': 0.1,
    'distributed': True
}

trainer = PIHMARLTrainer(trainer_config)
trainer.train(env, num_episodes=1000)
```

### Using Pre-built Scenarios

```python
from src.scenarios import (
    SearchRescueScenario,
    SwarmExplorationScenario,
    FormationControlScenario
)

# Search and Rescue
sar_scenario = SearchRescueScenario(
    area_size=(200.0, 200.0),
    num_victims=10,
    num_agents=5
)

# Swarm Exploration
exploration = SwarmExplorationScenario(
    environment_size=(150, 150),
    num_agents=6,
    obstacle_complexity=0.3
)

# Formation Control
formation = FormationControlScenario(
    num_agents=7,
    environment_size=(300.0, 300.0)
)
```

## Advanced Features

### Custom Scenarios

Create your own scenarios by extending the base classes:

```python
from src.scenarios.base import BaseScenario

class CustomScenario(BaseScenario):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your scenario
    
    def reset(self):
        # Reset scenario state
        pass
    
    def step(self, actions):
        # Execute one step
        pass
    
    def get_observation(self, agent_id):
        # Get observation for specific agent
        pass
```

### Communication Protocols

Implement custom communication between agents:

```python
from src.core.communication import CommunicationProtocol

class CustomProtocol(CommunicationProtocol):
    def encode_message(self, content, sender_id):
        # Encode message
        return encoded_message
    
    def decode_message(self, message):
        # Decode message
        return sender_id, content
```

### Physics Constraints

Define custom physics constraints:

```python
from src.models.physics import PhysicsConstraint

class CustomConstraint(PhysicsConstraint):
    def compute_violation(self, state, action, next_state):
        # Compute constraint violation
        return violation
    
    def compute_jacobian(self, state, action):
        # Compute constraint Jacobian
        return jacobian
```

## Hardware Integration

### Drone Control

```python
from src.hardware import DroneInterface, DroneController

# Connect to PX4 drone
drone = DroneInterface.create_px4_interface(
    connection_string="udp://127.0.0.1:14550"
)

controller = DroneController(drone)

# Arm and takeoff
controller.arm()
controller.takeoff(altitude=10.0)

# Navigate waypoints
waypoints = [
    (10.0, 0.0, 10.0),
    (10.0, 10.0, 10.0),
    (0.0, 10.0, 10.0),
    (0.0, 0.0, 10.0)
]

controller.navigate_waypoints(waypoints)
```

### Robot Control

```python
from src.hardware import RobotInterface, RobotController

# Connect to ROS robot
robot = RobotInterface.create_ros_interface(
    node_name="pi_hmarl_robot",
    namespace="/robot"
)

controller = RobotController(robot)

# Move to position
controller.move_to_position(x=2.0, y=3.0, theta=0.0)

# Execute path
path = generate_path(start, goal)
controller.follow_path(path)
```

### Sensor Integration

```python
from src.hardware import SensorInterface, SensorFusion

# Create sensor interfaces
camera = SensorInterface.create_camera(device_id=0)
lidar = SensorInterface.create_lidar(port="/dev/ttyUSB0")
imu = SensorInterface.create_imu(i2c_address=0x68)

# Sensor fusion
fusion = SensorFusion()
fusion.add_sensor("camera", camera)
fusion.add_sensor("lidar", lidar)
fusion.add_sensor("imu", imu)

# Get fused data
fused_state = fusion.get_fused_state()
```

## Security Configuration

### Authentication Setup

```python
from src.security import AuthenticationManager

auth_manager = AuthenticationManager(
    secret_key="your-secret-key",
    algorithm="HS256"
)

# Create user with role
token = auth_manager.create_user(
    username="operator",
    role="operator",
    permissions=["read", "write", "control"]
)

# Verify token
user_info = auth_manager.verify_token(token)
```

### Encryption

```python
from src.security import EncryptionManager

encryption = EncryptionManager()

# Encrypt sensitive data
encrypted = encryption.encrypt_data(
    data={"position": [10.0, 20.0], "target": [30.0, 40.0]},
    recipient_public_key=recipient_key
)

# Decrypt data
decrypted = encryption.decrypt_data(encrypted)
```

### Secure Communication

```python
from src.security import SecureCommunicationProtocol

secure_comm = SecureCommunicationProtocol(
    node_id="agent_1",
    private_key=private_key
)

# Send secure message
secure_comm.send_message(
    recipient="agent_2",
    message={"type": "position_update", "data": position},
    priority="high"
)

# Receive and verify
message, verified = secure_comm.receive_message()
```

## Performance Optimization

### GPU Acceleration

```python
from src.optimization import GPUAccelerator

accelerator = GPUAccelerator()

# Check GPU availability
if accelerator.is_available():
    # Move computation to GPU
    gpu_result = accelerator.accelerate(compute_function, data)
```

### Model Optimization

```python
from src.optimization import ModelOptimizer

optimizer = ModelOptimizer()

# Optimize model for deployment
optimized_model = optimizer.optimize_for_inference(
    model,
    optimization_level="O2",  # O1, O2, or O3
    target_device="gpu"
)

# Quantize model
quantized_model = optimizer.quantize_model(
    model,
    quantization_type="int8"
)
```

### Caching

```python
from src.optimization import CacheManager

cache = CacheManager(
    cache_size=1000,
    persistent=True,
    cache_dir="./cache"
)

# Cache expensive computations
@cache.cache_computation
def expensive_function(x, y):
    # Expensive computation
    return result

# Results are automatically cached
result = expensive_function(10, 20)
```

## Troubleshooting

### Common Issues

**1. GPU Not Detected**
```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Set device explicitly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

**2. Communication Failures**
```python
# Enable debug logging
import logging
logging.getLogger("pi_hmarl.communication").setLevel(logging.DEBUG)

# Test connectivity
from src.utils import test_communication
test_communication(agent1, agent2)
```

**3. Performance Issues**
```python
# Profile code
from src.optimization import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.start()

# Your code here
run_simulation()

profiler.stop()
report = profiler.get_report()
print(report)
```

### Debug Mode

Enable debug mode for detailed logging:

```python
from src.utils import setup_logging

setup_logging(
    log_level="DEBUG",
    log_file="debug.log",
    console_output=True
)
```

### Getting Help

1. Check the [API Documentation](https://pi-hmarl.readthedocs.io)
2. Search [GitHub Issues](https://github.com/your-org/pi-hmarl/issues)
3. Join our [Discord Community](https://discord.gg/pi-hmarl)
4. Email support: support@pi-hmarl.org

## Best Practices

1. **Start Simple**: Begin with pre-built scenarios before creating custom ones
2. **Monitor Performance**: Use the profiler regularly during development
3. **Test Hardware Integration**: Always test in simulation before deploying to real hardware
4. **Secure by Default**: Enable authentication and encryption for production deployments
5. **Version Control**: Track your configurations and models in version control

## Next Steps

- Explore the [Examples Directory](https://github.com/your-org/pi-hmarl/tree/main/examples)
- Read the [Developer Guide](developer_guide.md) for extending PI-HMARL
- Check out [Research Papers](papers.md) using PI-HMARL
- Contribute to the project on [GitHub](https://github.com/your-org/pi-hmarl)