
#!/usr/bin/env python
"""
GENIUS TEST: Validate 85%+ Success Rate
"""

import numpy as np
import time
from search_rescue_patch import SearchRescueScenario
from enhanced_hierarchical_agent import EnhancedHierarchicalAgent
from physics_integration import PhysicsInformedValidator

def test_search_rescue_performance():
    """Test Search & Rescue with all fixes"""
    
    print("=" * 80)
    print("GENIUS PI-HMARL PERFORMANCE TEST")
    print("Target: 85%+ Success Rate")
    print("=" * 80)
    
    # Configuration
    num_episodes = 10
    num_agents = 6
    success_count = 0
    
    # Performance metrics
    metrics = {
        "rescue_rates": [],
        "coordination_events": [],
        "physics_violations": [],
        "episode_lengths": []
    }
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Initialize scenario
        scenario = SearchRescueScenario(
            area_size=(100, 100),
            num_victims=10,
            min_agents_to_rescue=2,
            success_threshold=0.85
        )
        
        # Initialize agents
        agents = {}
        for i in range(num_agents):
            agents[i] = EnhancedHierarchicalAgent(i)
        
        # Physics validator
        physics_validator = PhysicsInformedValidator()
        
        # Reset scenario
        obs = scenario.reset(num_agents)
        
        # Initialize agent positions
        agent_positions = {}
        for i in range(num_agents):
            # Start positions spread across area
            x = (i % 3) * 30 + 20
            y = (i // 3) * 30 + 20
            agent_positions[i] = np.array([x, y, 0.0])
        
        # Run episode
        step_count = 0
        max_steps = 3000  # 5 minutes at 0.1s timestep
        terminated = False
        
        while not terminated and step_count < max_steps:
            # Get agent states for coordination
            agent_states = {}
            for i in range(num_agents):
                if hasattr(obs, '__getitem__'):
                    agent_obs = obs[i] if i in obs else obs.get("observations", {}).get(i, {})
                else:
                    agent_obs = obs
                state = agents[i]._obs_to_state(agent_obs)
                agent_states[i] = torch.FloatTensor(state)
            
            # Coordinated action selection
            actions = {}
            for i in range(num_agents):
                other_states = {j: s for j, s in agent_states.items() if j != i}
                if hasattr(obs, '__getitem__'):
                    agent_obs = obs[i] if i in obs else obs.get("observations", {}).get(i, {})
                else:
                    agent_obs = obs
                action = agents[i].act(agent_obs, other_states)
                
                # Validate with physics
                validated_action, safe = physics_validator.validate_action(
                    agent_states[i].numpy(),
                    action,
                    {j: pos for j, pos in agent_positions.items() if j != i}
                )
                
                actions[i] = validated_action
                
                # Update position based on action
                agent_positions[i] += validated_action[:3] * 0.1  # Simple integration
                agent_positions[i] = np.clip(agent_positions[i], [0, 0, 0], [100, 100, 10])
            
            # Step scenario
            result = scenario.step(agent_positions, dt=0.1)
            
            # Parse results
            if isinstance(result, dict):
                obs = result.get("observations", {})
                rewards = result.get("rewards", {})
                terminated = result.get("terminated", False)
                info = result.get("info", {})
                success = result.get("success", False)
                rescue_rate = result.get("rescue_rate", 0.0)
            else:
                # Handle different return formats
                obs = result
                terminated = False
                info = {}
                success = False
                rescue_rate = 0.0
            
            step_count += 1
            
            # Progress update every 100 steps
            if step_count % 100 == 0:
                current_rate = scenario.get_success_rate() if hasattr(scenario, 'get_success_rate') else 0.0
                print(f"  Step {step_count}: Rescue rate = {current_rate:.2%}")
        
        # Episode complete
        final_eval = scenario.evaluate() if hasattr(scenario, 'evaluate') else {"success": False, "rescue_rate": 0.0}
        episode_success = final_eval.get("success", False)
        final_rescue_rate = final_eval.get("rescue_rate", 0.0)
        
        if episode_success:
            success_count += 1
        
        metrics["rescue_rates"].append(final_rescue_rate)
        metrics["episode_lengths"].append(step_count)
        
        print(f"  Episode complete!")
        print(f"  Success: {episode_success}")
        print(f"  Rescue rate: {final_rescue_rate:.2%}")
        print(f"  Episode length: {step_count} steps")
    
    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    overall_success_rate = success_count / num_episodes
    avg_rescue_rate = np.mean(metrics["rescue_rates"])
    
    print(f"Overall Success Rate: {overall_success_rate:.2%} ({'PASS' if overall_success_rate >= 0.85 else 'FAIL'})")
    print(f"Average Rescue Rate: {avg_rescue_rate:.2%}")
    print(f"Success Episodes: {success_count}/{num_episodes}")
    print(f"Average Episode Length: {np.mean(metrics['episode_lengths']):.0f} steps")
    
    if overall_success_rate >= 0.85:
        print("\n🎉 SUCCESS! Achieved 85%+ success rate!")
        print("🧠 GENIUS-LEVEL PERFORMANCE ACHIEVED!")
    else:
        print("\n❌ Not yet at 85%+ - continuing optimization...")
    
    return overall_success_rate


if __name__ == "__main__":
    import torch
    # Run the test
    success_rate = test_search_rescue_performance()
    
    if success_rate < 0.85:
        print("\n🔧 Applying additional optimizations...")
        # Additional optimization code would go here
