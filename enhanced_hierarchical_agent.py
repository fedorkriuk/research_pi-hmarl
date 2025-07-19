
# GENIUS INTEGRATION: Hierarchical Multi-Agent System
import torch
from fix_hierarchical_coordination import (
    HierarchicalCoordinationModule,
    CoordinationRewardShaper,
    PhysicsInformedCoordination
)

class EnhancedHierarchicalAgent:
    """Enhanced agent with proper coordination"""
    
    def __init__(self, agent_id, state_dim=64, action_dim=4):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize coordination module
        self.coordinator = HierarchicalCoordinationModule(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=6  # Default for scenarios
        )
        
        # Physics validator
        self.physics_validator = None  # Will be set by environment
        
        # Communication buffer
        self.message_buffer = []
        
    def act(self, observation, other_agents_states=None):
        """Generate action with coordination"""
        # Convert observation to tensor
        state = torch.FloatTensor(self._obs_to_state(observation))
        
        if other_agents_states is None:
            # Solo decision
            return self._solo_action(state)
        else:
            # Coordinated decision
            states = {self.agent_id: state}
            states.update(other_agents_states)
            
            # Get coordinated output
            outputs = self.coordinator(states)
            
            action = outputs["actions"][self.agent_id]
            self.last_task_assignment = outputs["task_assignments"][self.agent_id]
            self.last_message = outputs["messages"][self.agent_id]
            
            return action.detach().numpy()
    
    def _obs_to_state(self, obs):
        """Convert observation dict to state vector"""
        # Extract relevant features
        features = []
        
        # Time
        features.append(obs.get("time", 0.0) / 300.0)  # Normalized
        
        # Victim information
        discovered = len(obs.get("discovered_victims", []))
        features.append(discovered / 10.0)  # Normalized by max victims
        
        # Assignment status
        assigned = 1.0 if obs.get("assigned_victim") is not None else 0.0
        features.append(assigned)
        
        # Team size
        team_size = obs.get("rescue_team_size", 0) / 6.0  # Normalized
        features.append(team_size)
        
        # Progress metrics
        rescued = obs.get("victims_rescued", 0) / 10.0
        critical = obs.get("victims_critical", 0) / 10.0
        features.extend([rescued, critical])
        
        # Pad to state dimension
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim])
    
    def _solo_action(self, state):
        """Generate action without coordination"""
        # Simple policy for testing
        return np.random.randn(self.action_dim) * 0.1
