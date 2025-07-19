# PI-HMARL Examples

This directory contains comprehensive examples demonstrating various use cases and capabilities of the PI-HMARL framework.

## Available Examples

### 1. Surveillance Mission (`surveillance_mission.py`)

Demonstrates area surveillance with multiple drones:
- **Efficient area coverage** using sector assignment
- **Real-time coverage tracking** with grid-based monitoring
- **Coordinated patrol patterns** (lawnmower pattern)
- **Battery management** with return-to-base logic
- **Inter-drone communication** for coverage sharing

**Key Features:**
- Dynamic sector assignment
- Coverage heatmap visualization
- Energy-aware operations
- Performance metrics tracking

**Usage:**
```bash
python examples/surveillance_mission.py
```

**Expected Output:**
- Coverage progression plot
- Patrol path visualization
- Energy consumption analysis
- Mission report (PNG file)

---

### 2. Delivery Coordination (`delivery_coordination.py`)

Multi-drone package delivery system:
- **Dynamic package assignment** using Hungarian algorithm
- **Energy-aware route planning** with battery constraints
- **Collision avoidance** in shared airspace
- **Priority-based delivery** (high/medium/low)
- **Real-time reallocation** for new packages

**Key Features:**
- Multiple delivery hubs
- Weight capacity constraints
- Optimized route planning
- Airspace management with altitude layers

**Usage:**
```bash
python examples/delivery_coordination.py
```

**Expected Output:**
- Delivery timeline
- Energy efficiency metrics
- Drone utilization statistics
- Delivery map visualization

---

### 3. Formation Flying (`formation_flying.py`)

Coordinated formation flying demonstration:
- **Multiple formation patterns**: V-shape, line, circle, diamond
- **Dynamic formation switching** based on schedule
- **Wind compensation** with physics-based modeling
- **Obstacle avoidance** while maintaining formation
- **Leader-follower dynamics** with position feedback

**Key Features:**
- Real-time formation error tracking
- Collision avoidance between drones
- Wind field visualization
- Smooth formation transitions

**Usage:**
```bash
python examples/formation_flying.py
```

**Expected Output:**
- Formation error over time
- Pattern demonstrations
- Performance metrics
- Mission trajectory plot

---

### 4. Search and Rescue (`search_rescue.py`)

Emergency search and rescue operations:
- **Systematic area search** with multiple patterns
- **Victim detection** using simulated sensors
- **Priority-based rescue** (critical/high/medium/low)
- **Dynamic task reallocation** when victims found
- **Health degradation modeling** for time-critical operations

**Key Features:**
- Terrain-aware search patterns
- Priority zone management
- Multi-drone rescue coordination
- Real-time victim tracking

**Usage:**
```bash
python examples/search_rescue.py
```

**Expected Output:**
- Search coverage progression
- Detection/rescue timeline
- Victim status report
- Search area heatmap

---

## Running the Examples

### Prerequisites

1. Install PI-HMARL:
```bash
pip install -e .
```

2. Install example dependencies:
```bash
pip install matplotlib seaborn
```

### Basic Execution

Run any example directly:
```bash
python examples/<example_name>.py
```

### Advanced Configuration

Each example accepts configuration parameters that can be modified:

```python
# Example: Modify surveillance_mission.py
num_drones = 6  # Instead of 4
surveillance_area = [(0, 0), (2000, 0), (2000, 2000), (0, 2000)]  # 2km x 2km
mission_duration = 1200  # 20 minutes
```

### Output Files

Each example generates:
- **Console output**: Real-time mission progress
- **Report PNG**: Visual summary saved to current directory
- **Metrics**: Performance statistics and analysis

---

## Example Components

### Common Patterns

All examples demonstrate:
1. **Environment Setup**: Creating the simulation environment
2. **Agent Initialization**: Configuring hierarchical agents
3. **Task Definition**: Setting up mission objectives
4. **Communication**: Inter-agent message passing
5. **Visualization**: Real-time monitoring
6. **Metrics Collection**: Performance tracking
7. **Report Generation**: Visual summaries

### Customization

Examples can be customized by modifying:
- Number of agents
- Environment parameters (size, obstacles, wind)
- Task parameters (priorities, deadlines)
- Agent capabilities (sensors, battery, speed)
- Communication settings (range, bandwidth)

---

## Creating Your Own Examples

To create a custom example:

1. **Import necessary modules**:
```python
from src.environment import MultiAgentEnvironment
from src.agents.hierarchical_agent import HierarchicalAgent
from src.tasks.base_task import Task
```

2. **Define your task**:
```python
class CustomTask(Task):
    def __init__(self, ...):
        super().__init__(...)
```

3. **Setup environment and agents**:
```python
env = MultiAgentEnvironment(...)
agents = [HierarchicalAgent(...) for _ in range(num_agents)]
```

4. **Run simulation loop**:
```python
for step in range(duration):
    actions = {}
    for i, agent in enumerate(agents):
        actions[i] = agent.act(observations[i])
    observations, rewards, dones, info = env.step(actions)
```

5. **Generate report**:
```python
generate_mission_report(metrics)
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PI-HMARL is installed:
   ```bash
   pip install -e .
   ```

2. **Visualization Issues**: Check matplotlib backend:
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'
   ```

3. **Performance**: Reduce number of agents or simulation frequency:
   ```python
   num_drones = 3  # Instead of 8
   update_rate = 5  # Instead of 10 Hz
   ```

### Getting Help

- Check example comments and docstrings
- Review the [API documentation](../docs/api_reference.md)
- Join our [Discord community](https://discord.gg/pi-hmarl)
- Open an issue on [GitHub](https://github.com/your-org/pi-hmarl)

---

## Contributing Examples

We welcome new examples! To contribute:

1. Fork the repository
2. Create your example following the patterns above
3. Include comprehensive comments
4. Add documentation to this README
5. Submit a pull request

Example requirements:
- Clear objective and use case
- Well-commented code
- Visualization of results
- Performance metrics
- Error handling

---

## Performance Benchmarks

Typical performance on standard hardware (8-core CPU, 16GB RAM):

| Example | Agents | Real-time Factor | Memory Usage |
|---------|--------|------------------|--------------|
| Surveillance | 4 | 1.2x | 500 MB |
| Delivery | 6 | 1.0x | 700 MB |
| Formation | 7 | 1.1x | 600 MB |
| Search & Rescue | 8 | 0.9x | 800 MB |

Real-time factor: 1.0x means simulation runs at real-world speed.

---

## Future Examples

Planned examples for future releases:
- **Swarm exploration**: Unknown environment mapping
- **Adversarial scenarios**: Defensive drone operations
- **Agricultural monitoring**: Precision farming
- **Infrastructure inspection**: Bridge/powerline inspection
- **Disaster response**: Multi-hazard scenarios
- **Racing**: Competitive drone racing

Stay tuned for updates!