# Core Concepts

## Overview

PI-HMARL introduces several key concepts that enable effective multi-agent drone coordination. This guide explains these fundamental concepts and how they work together.

## 1. Hierarchical Multi-Agent System

### Concept

The hierarchical structure enables decision-making at multiple levels of abstraction, from high-level strategic planning to low-level motor control.

```
Strategic Level (Minutes-Hours)
    ↓
Tactical Level (Seconds-Minutes)  
    ↓
Operational Level (Milliseconds-Seconds)
```

### Implementation

```python
class HierarchicalAgent:
    def __init__(self):
        self.strategic = StrategicPlanner()
        self.tactical = TacticalController()
        self.operational = OperationalController()
    
    def decide(self, observation):
        # Top-down decision flow
        mission = self.strategic.plan(observation)
        tasks = self.tactical.decompose(mission, observation)
        actions = self.operational.execute(tasks, observation)
        return actions
```

### Benefits

- **Temporal Abstraction**: Different time scales for different decisions
- **Modularity**: Each level can be developed independently
- **Scalability**: Add or remove levels as needed
- **Interpretability**: Clear decision hierarchy

## 2. Physics-Informed Learning

### Concept

Instead of learning physics from scratch, we incorporate known physical laws into the learning process.

### Key Physics Models

1. **Aerodynamics**
   ```python
   # Lift force
   F_lift = 0.5 * ρ * v² * S * C_L
   
   # Drag force  
   F_drag = 0.5 * ρ * v² * S * C_D
   
   # Where:
   # ρ = air density
   # v = velocity
   # S = reference area
   # C_L, C_D = lift and drag coefficients
   ```

2. **Propeller Dynamics**
   ```python
   # Thrust from propeller
   T = C_T * ρ * n² * D⁴
   
   # Power required
   P = C_P * ρ * n³ * D⁵
   
   # Where:
   # n = rotational speed
   # D = propeller diameter
   # C_T, C_P = thrust and power coefficients
   ```

3. **Battery Model**
   ```python
   # Discharge equation
   V(t) = V_nominal - R_internal * I(t) - K * (Q_max - Q(t))
   
   # State of charge
   SoC(t) = Q(t) / Q_max
   ```

### Integration with RL

```python
class PhysicsInformedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.physics_model = DronePhysics()
        self.neural_correction = nn.Sequential(...)
    
    def forward(self, state, action):
        # Physics-based prediction
        physics_pred = self.physics_model(state, action)
        
        # Neural network correction
        correction = self.neural_correction(state, action)
        
        # Combine predictions
        return physics_pred + correction
```

## 3. Multi-Agent Coordination

### Coordination Mechanisms

1. **Centralized Training, Decentralized Execution (CTDE)**
   - Train with global information
   - Execute with local observations

2. **Communication Protocols**
   - Message passing between agents
   - Bandwidth-aware communication
   - Byzantine fault tolerance

3. **Consensus Algorithms**
   - Distributed agreement on decisions
   - Robust to agent failures

### Example: Formation Flying

```python
class FormationController:
    def __init__(self, formation_type='v_shape'):
        self.formation = self.get_formation(formation_type)
    
    def get_desired_positions(self, leader_pos, num_agents):
        positions = []
        for i in range(num_agents):
            offset = self.formation.get_offset(i)
            desired_pos = leader_pos + offset
            positions.append(desired_pos)
        return positions
    
    def compute_control(self, current_pos, desired_pos):
        # PID control to maintain formation
        error = desired_pos - current_pos
        return self.pid_controller(error)
```

## 4. Task Decomposition

### Concept

Complex missions are broken down into manageable subtasks that can be assigned to individual agents.

### Task Hierarchy

```
Mission: Survey Industrial Complex
    ├── Task 1: Perimeter Surveillance
    │   ├── Subtask 1.1: North boundary
    │   ├── Subtask 1.2: East boundary
    │   └── ...
    ├── Task 2: Building Inspection
    │   ├── Subtask 2.1: Roof inspection
    │   ├── Subtask 2.2: Wall inspection
    │   └── ...
    └── Task 3: Anomaly Detection
        └── ...
```

### Assignment Optimization

```python
class TaskAssignment:
    def __init__(self):
        self.optimizer = HungarianAlgorithm()
    
    def assign_tasks(self, tasks, agents):
        # Build cost matrix
        cost_matrix = np.zeros((len(agents), len(tasks)))
        
        for i, agent in enumerate(agents):
            for j, task in enumerate(tasks):
                # Consider distance, capability, battery
                cost = self.compute_cost(agent, task)
                cost_matrix[i, j] = cost
        
        # Optimize assignment
        assignments = self.optimizer.solve(cost_matrix)
        return assignments
```

## 5. Energy Management

### Energy-Aware Planning

Energy is a critical constraint for drone operations. The system considers:

1. **Battery State**: Current charge level and health
2. **Power Consumption**: Based on flight dynamics
3. **Environmental Factors**: Wind, temperature effects
4. **Mission Requirements**: Energy needed for tasks

### Optimization Strategy

```python
class EnergyOptimizer:
    def optimize_path(self, start, goal, battery_state):
        # A* search with energy cost
        def energy_cost(path):
            cost = 0
            for i in range(len(path)-1):
                # Physics-based energy calculation
                segment_cost = self.compute_segment_energy(
                    path[i], path[i+1], 
                    wind_field, drone_params
                )
                cost += segment_cost
            return cost
        
        # Find energy-optimal path
        optimal_path = self.astar_search(
            start, goal, 
            cost_function=energy_cost,
            constraint=battery_state
        )
        return optimal_path
```

## 6. Communication Architecture

### Message Types

1. **State Sharing**: Position, velocity, battery status
2. **Task Coordination**: Task claims, progress updates
3. **Emergency**: Collision warnings, system failures

### Protocol Stack

```
Application Layer    - Task coordination messages
    ↓
Transport Layer     - Reliable delivery, QoS
    ↓  
Network Layer       - Multi-hop routing
    ↓
Physical Layer      - Radio communication
```

### Bandwidth Management

```python
class BandwidthManager:
    def __init__(self, total_bandwidth):
        self.total_bandwidth = total_bandwidth
        self.allocations = {}
    
    def allocate(self, agent_id, priority):
        # Dynamic bandwidth allocation
        if priority == 'emergency':
            bandwidth = self.total_bandwidth * 0.3
        elif priority == 'high':
            bandwidth = self.total_bandwidth * 0.2
        else:
            bandwidth = self.total_bandwidth * 0.1
        
        self.allocations[agent_id] = bandwidth
        return bandwidth
```

## 7. Safety Constraints

### Safety Layers

1. **Geofencing**: Spatial boundaries
2. **Collision Avoidance**: Minimum separation
3. **Emergency Protocols**: Failsafe behaviors
4. **Energy Safety**: Minimum reserve requirements

### Implementation

```python
class SafetyMonitor:
    def __init__(self):
        self.constraints = {
            'min_altitude': 10.0,
            'max_altitude': 150.0,
            'min_separation': 5.0,
            'min_battery': 0.2,
            'geofence': Polygon(...)
        }
    
    def check_action(self, state, action):
        # Predict next state
        next_state = self.dynamics_model(state, action)
        
        # Check all constraints
        violations = []
        
        if next_state.altitude < self.constraints['min_altitude']:
            violations.append('altitude_low')
        
        if next_state.battery < self.constraints['min_battery']:
            violations.append('battery_critical')
        
        # Return safe action if violations
        if violations:
            return self.get_safe_action(state, violations)
        
        return action
```

## 8. Learning Paradigms

### Multi-Agent Reinforcement Learning

1. **Independent Learning**: Each agent learns separately
2. **Joint Learning**: Agents learn together
3. **Communication Learning**: Learn what and when to communicate

### Curriculum Learning

```python
class CurriculumManager:
    def __init__(self):
        self.stages = [
            {'name': 'hover', 'difficulty': 0.1},
            {'name': 'waypoint', 'difficulty': 0.3},
            {'name': 'formation', 'difficulty': 0.5},
            {'name': 'complex_mission', 'difficulty': 1.0}
        ]
        self.current_stage = 0
    
    def get_training_scenario(self, performance):
        # Progress based on performance
        if performance > 0.8:
            self.current_stage = min(
                self.current_stage + 1, 
                len(self.stages) - 1
            )
        
        return self.stages[self.current_stage]
```

## 9. Distributed Consensus

### Byzantine Fault Tolerance

Handle up to f faulty agents in a system of n agents where n > 3f.

```python
class ByzantineConsensus:
    def __init__(self, agent_id, num_agents, fault_tolerance=0.33):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.max_faulty = int(num_agents * fault_tolerance)
    
    def reach_consensus(self, proposals):
        # Collect proposals from all agents
        votes = self.collect_votes(proposals)
        
        # Byzantine agreement protocol
        for round in range(self.max_faulty + 1):
            # Exchange votes
            all_votes = self.exchange_votes(votes)
            
            # Update based on majority
            votes = self.majority_vote(all_votes)
        
        return self.final_decision(votes)
```

## 10. Scalability Patterns

### Hierarchical Clustering

```python
class HierarchicalClustering:
    def __init__(self, num_agents):
        self.clusters = self.form_clusters(num_agents)
        self.leaders = self.elect_leaders()
    
    def form_clusters(self, num_agents):
        # Proximity-based clustering
        cluster_size = 5  # agents per cluster
        num_clusters = (num_agents + cluster_size - 1) // cluster_size
        
        clusters = []
        for i in range(num_clusters):
            cluster = AgentCluster(
                id=i,
                members=list(range(
                    i * cluster_size,
                    min((i + 1) * cluster_size, num_agents)
                ))
            )
            clusters.append(cluster)
        
        return clusters
```

### Load Balancing

```python
class LoadBalancer:
    def balance_tasks(self, tasks, clusters):
        # Compute cluster loads
        loads = {c.id: c.current_load() for c in clusters}
        
        # Assign tasks to minimize maximum load
        for task in sorted(tasks, key=lambda t: t.cost, reverse=True):
            # Find cluster with minimum load
            min_cluster = min(loads, key=loads.get)
            
            # Assign task
            clusters[min_cluster].assign(task)
            loads[min_cluster] += task.cost
        
        return clusters
```

## Summary

These core concepts work together to create a robust, scalable, and efficient multi-agent drone system:

- **Hierarchical Control** provides structured decision-making
- **Physics-Informed Learning** ensures realistic and safe behaviors
- **Multi-Agent Coordination** enables complex collaborative missions
- **Task Decomposition** breaks down complex objectives
- **Energy Management** extends operational range
- **Communication Architecture** enables information sharing
- **Safety Constraints** ensure reliable operation
- **Learning Paradigms** enable continuous improvement
- **Distributed Consensus** provides fault tolerance
- **Scalability Patterns** support large deployments

Understanding these concepts is essential for effectively using and extending PI-HMARL.