# Getting Started with PI-HMARL

This guide will help you get up and running with PI-HMARL quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM
- Ubuntu 20.04+ / macOS 11+ / Windows 10+

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-org/pi-hmarl.git
cd pi-hmarl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PI-HMARL
pip install -e .
```

### Docker Installation

```bash
# Build Docker image
docker build -t pi-hmarl:latest .

# Run container
docker run -it --gpus all -p 5000:5000 -p 8080:8080 pi-hmarl:latest
```

## Basic Concepts

### 1. Environment

The environment simulates the physical world where drones operate:

```python
from pi_hmarl.environment import MultiAgentEnvironment

# Create environment
env = MultiAgentEnvironment(
    num_agents=5,
    map_size=(1000, 1000, 200),  # 1km x 1km x 200m
    physics_config={
        'gravity': 9.81,
        'air_density': 1.225,
        'wind_enabled': True
    }
)

# Reset environment
observations = env.reset()

# Step through environment
actions = {...}  # Agent actions
next_obs, rewards, dones, info = env.step(actions)
```

### 2. Agents

Agents are the decision-making entities that control drones:

```python
from pi_hmarl.agents import HierarchicalAgent

# Create agent
agent = HierarchicalAgent(
    agent_id=0,
    obs_dim=64,
    action_dim=4,
    levels=['strategic', 'tactical', 'operational']
)

# Get action
observation = env.get_observation(agent_id=0)
action = agent.act(observation)

# Train agent
agent.train(batch_size=32, episodes=1000)
```

### 3. Tasks

Tasks represent objectives that agents need to complete:

```python
from pi_hmarl.tasks import SurveillanceTask, DeliveryTask

# Create surveillance task
surveillance = SurveillanceTask(
    task_id="surv_001",
    area=[(0, 0), (500, 0), (500, 500), (0, 500)],
    duration=600,  # 10 minutes
    priority="high"
)

# Create delivery task
delivery = DeliveryTask(
    task_id="del_001",
    pickup_location=(100, 100, 0),
    delivery_location=(800, 800, 0),
    package_weight=2.0,  # kg
    deadline=300  # 5 minutes
)
```

## Your First Simulation

Here's a complete example to get you started:

```python
import numpy as np
from pi_hmarl import MultiAgentSystem
from pi_hmarl.environment import MultiAgentEnvironment
from pi_hmarl.agents import HierarchicalAgent
from pi_hmarl.tasks import TaskGenerator
from pi_hmarl.visualization import Dashboard

# 1. Setup Environment
env = MultiAgentEnvironment(
    num_agents=3,
    map_size=(500, 500, 100),
    physics_enabled=True,
    real_time=False
)

# 2. Create Agents
agents = []
for i in range(3):
    agent = HierarchicalAgent(
        agent_id=i,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config={
            'learning_rate': 1e-4,
            'hidden_dims': [256, 128, 64],
            'use_attention': True
        }
    )
    agents.append(agent)

# 3. Generate Tasks
task_gen = TaskGenerator(seed=42)
tasks = task_gen.generate_scenario(
    scenario_type='mixed',
    num_tasks=5,
    difficulty='medium'
)

# 4. Create System
system = MultiAgentSystem(
    environment=env,
    agents=agents,
    config={
        'enable_communication': True,
        'enable_visualization': True,
        'save_metrics': True
    }
)

# 5. Add Tasks
for task in tasks:
    system.add_task(task)

# 6. Run Simulation
print("Starting simulation...")
results = system.run(
    episodes=10,
    max_steps_per_episode=1000,
    render=True
)

# 7. Analyze Results
print(f"Average reward: {results['avg_reward']:.2f}")
print(f"Task completion rate: {results['completion_rate']:.2%}")
print(f"Average episode length: {results['avg_length']:.0f} steps")

# 8. Visualize Results (Optional)
dashboard = Dashboard()
dashboard.plot_results(results)
dashboard.show()
```

## Configuration Options

### Environment Configuration

```python
env_config = {
    # Map settings
    'map_size': (1000, 1000, 200),
    'obstacles': [
        {'type': 'building', 'position': (200, 200), 'size': (50, 50, 80)},
        {'type': 'no_fly_zone', 'center': (500, 500), 'radius': 100}
    ],
    
    # Physics settings
    'physics': {
        'timestep': 0.01,  # 10ms
        'gravity': 9.81,
        'air_density': 1.225,
        'wind_model': 'turbulent'
    },
    
    # Safety settings
    'safety': {
        'min_distance': 5.0,  # meters
        'max_altitude': 150.0,  # meters
        'geofence': [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    }
}
```

### Agent Configuration

```python
agent_config = {
    # Model architecture
    'model': {
        'type': 'hierarchical',
        'levels': ['strategic', 'tactical', 'operational'],
        'hidden_dims': [512, 256, 128],
        'activation': 'relu',
        'use_lstm': True
    },
    
    # Training settings
    'training': {
        'learning_rate': 1e-4,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 100000
    },
    
    # Exploration
    'exploration': {
        'type': 'epsilon_greedy',
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995
    }
}
```

## Running Experiments

### Training Mode

```python
# Train agents on scenarios
from pi_hmarl.training import Trainer

trainer = Trainer(
    agents=agents,
    environment=env,
    config={
        'num_episodes': 1000,
        'save_interval': 100,
        'eval_interval': 50,
        'checkpoint_dir': './checkpoints'
    }
)

trainer.train()
```

### Evaluation Mode

```python
# Evaluate trained agents
from pi_hmarl.evaluation import Evaluator

evaluator = Evaluator(
    agents=agents,
    environment=env,
    scenarios=['surveillance', 'delivery', 'search_rescue']
)

metrics = evaluator.evaluate(num_episodes=100)
evaluator.generate_report(metrics, output_path='./evaluation_report.pdf')
```

### Visualization

```python
# Real-time visualization
from pi_hmarl.visualization import RealtimeVisualizer

visualizer = RealtimeVisualizer(
    environment=env,
    agents=agents,
    config={
        'fps': 30,
        'resolution': (1920, 1080),
        'show_paths': True,
        'show_communication': True,
        'show_metrics': True
    }
)

visualizer.run()
```

## Next Steps

1. **Explore Examples**: Check out the [examples](examples.md) directory for more complex scenarios
2. **Read API Documentation**: Detailed API reference is available [here](api_reference.md)
3. **Try Tutorials**: Step-by-step tutorials for specific use cases [here](tutorials.md)
4. **Join Community**: Connect with other users on our [Discord server](https://discord.gg/pi-hmarl)

## Common Issues

### Performance Issues

If simulation is running slowly:
- Enable GPU acceleration: `env.enable_gpu()`
- Reduce physics timestep: `env.set_timestep(0.02)`
- Disable visualization during training: `system.set_render(False)`

### Memory Issues

For large-scale simulations:
- Use experience replay buffer limits: `buffer_size=50000`
- Enable gradient accumulation: `accumulate_gradients=True`
- Use mixed precision training: `use_amp=True`

### Communication Issues

If agents aren't communicating:
- Check communication range: `env.set_comm_range(100.0)`
- Verify protocol settings: `protocol.debug_mode = True`
- Monitor packet loss: `env.get_comm_stats()`

## Resources

- **Video Tutorials**: [YouTube Playlist](https://youtube.com/pi-hmarl)
- **Research Papers**: [ArXiv Collection](https://arxiv.org/pi-hmarl)
- **Blog Posts**: [Medium Publication](https://medium.com/pi-hmarl)
- **Code Examples**: [GitHub Examples](https://github.com/your-org/pi-hmarl/examples)