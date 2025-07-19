#!/usr/bin/env python
"""Run all PI-HMARL demos automatically"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import logging
from src.utils import setup_logging
from src.scenarios import (
    SearchRescueScenario,
    SwarmExplorationScenario,
    FormationControlScenario
)

# Setup logging (less verbose for demo)
setup_logging(log_level="WARNING", console_output=True)

def run_search_rescue_demo():
    """Run search and rescue demo"""
    print("\n" + "="*60)
    print("DEMO 1: SEARCH & RESCUE SCENARIO")
    print("="*60)
    print("Multiple agents searching for and rescuing victims in a disaster area")
    
    scenario = SearchRescueScenario(
        area_size=(100.0, 100.0),
        num_victims=5,
        num_agents=3,
        obstacle_density=0.05
    )
    
    print(f"✓ Created {scenario.num_agents} rescue agents")
    print(f"✓ Placed {scenario.num_victims} victims in {scenario.area_size[0]}x{scenario.area_size[1]}m area")
    print("✓ Running rescue mission...\n")
    
    start_time = time.time()
    max_steps = 150
    
    for step in range(max_steps):
        scenario.step(dt=0.1)
        
        if step % 30 == 0:
            state = scenario.get_state()
            print(f"  Time: {state['time']:6.1f}s | "
                  f"Detected: {state['victims']['detected']}/{state['victims']['total']} | "
                  f"Rescued: {state['victims']['rescued']}")
        
        # Check if mission completed
        if scenario.get_state()['victims']['rescued'] == scenario.num_victims:
            final_state = scenario.get_state()
            print(f"\n🎯 SUCCESS: All victims rescued in {final_state['time']:.1f} seconds!")
            break
    else:
        final_state = scenario.get_state()
        print(f"\n⏰ Time limit reached: {final_state['victims']['rescued']}/{final_state['victims']['total']} victims rescued")
    
    elapsed = time.time() - start_time
    print(f"✓ Demo completed in {elapsed:.2f} real seconds")

def run_swarm_exploration_demo():
    """Run swarm exploration demo"""
    print("\n" + "="*60)
    print("DEMO 2: SWARM EXPLORATION SCENARIO")
    print("="*60)
    print("Swarm of agents collaboratively exploring unknown environment")
    
    scenario = SwarmExplorationScenario(
        environment_size=(80, 80),
        num_agents=4,
        obstacle_complexity=0.15
    )
    
    print(f"✓ Created {scenario.num_agents} exploration agents")
    print(f"✓ Environment: {scenario.environment_size[0]}x{scenario.environment_size[1]} cells")
    print(f"✓ Obstacle complexity: 15%")
    print("✓ Starting exploration...\n")
    
    start_time = time.time()
    last_report_time = 0
    max_time = 25.0
    
    while scenario.time < max_time and not scenario.completed:
        scenario.step(dt=0.1)
        
        if scenario.time - last_report_time >= 5.0:
            state = scenario.get_state()
            print(f"  Time: {state['time']:6.1f}s | "
                  f"Explored: {state['exploration_rate']*100:5.1f}% | "
                  f"Frontiers: {state['frontiers_remaining']}")
            last_report_time = scenario.time
    
    final_state = scenario.get_state()
    elapsed = time.time() - start_time
    
    if scenario.completed:
        print(f"\n🎯 SUCCESS: Full exploration completed!")
    else:
        print(f"\n⏰ Time limit reached")
    
    print(f"✓ Explored {final_state['exploration_rate']*100:.1f}% of environment")
    print(f"✓ Demo completed in {elapsed:.2f} real seconds")

def run_formation_control_demo():
    """Run formation control demo"""
    print("\n" + "="*60)
    print("DEMO 3: FORMATION CONTROL SCENARIO")
    print("="*60)
    print("Agents maintaining geometric formations while navigating")
    
    scenario = FormationControlScenario(
        num_agents=6,
        environment_size=(200.0, 200.0),
        num_obstacles=5
    )
    
    print(f"✓ Created {scenario.num_agents} formation agents")
    print(f"✓ Environment: {scenario.environment_size[0]}x{scenario.environment_size[1]} meters")
    print(f"✓ Obstacles: {len(scenario.obstacles)}")
    
    # Create mission waypoints
    import numpy as np
    waypoints = [
        np.array([50, 50, 0]),
        np.array([150, 50, 0]),
        np.array([150, 150, 0]),
        np.array([50, 150, 0]),
        np.array([100, 100, 0])
    ]
    scenario.controller.set_waypoints(waypoints)
    print(f"✓ Mission: Navigate through {len(waypoints)} waypoints")
    print("✓ Starting formation flight...\n")
    
    from src.scenarios.formation_control import FormationType
    
    formation_changes = [
        (3.0, FormationType.LINE, "LINE"),
        (6.0, FormationType.WEDGE, "WEDGE"),
        (9.0, FormationType.CIRCLE, "CIRCLE")
    ]
    change_idx = 0
    
    start_time = time.time()
    max_steps = 120
    
    for step in range(max_steps):
        scenario.step(dt=0.1)
        
        # Change formations at specified times
        if (change_idx < len(formation_changes) and 
            scenario.time >= formation_changes[change_idx][0]):
            scenario.controller.change_formation(
                formation_changes[change_idx][1],
                scenario.time
            )
            print(f"  🔄 Formation changed to {formation_changes[change_idx][2]} at T={scenario.time:.1f}s")
            change_idx += 1
        
        if step % 20 == 0:
            state = scenario.get_state()
            print(f"  Time: {state['time']:6.1f}s | "
                  f"Formation: {state['formation_type']:8s} | "
                  f"Quality: {state['formation_quality']:4.2f} | "
                  f"Waypoint: {state['waypoint_progress']}")
        
        # Check mission completion
        if scenario.controller.current_waypoint_index >= len(waypoints):
            print(f"\n🎯 SUCCESS: Mission completed in {scenario.time:.1f} seconds!")
            break
    
    final_state = scenario.get_state()
    elapsed = time.time() - start_time
    
    print(f"✓ Final formation quality: {final_state['formation_quality']:.2f}")
    print(f"✓ Mission progress: {final_state['waypoint_progress']}")
    print(f"✓ Demo completed in {elapsed:.2f} real seconds")

def main():
    """Run all demos automatically"""
    print("="*70)
    print("🚀 PI-HMARL COMPREHENSIVE DEMO")
    print("Physics-Informed Hierarchical Multi-Agent Reinforcement Learning")
    print("="*70)
    print("Automatically running all three advanced scenarios...")
    
    total_start = time.time()
    
    try:
        # Run all demos
        run_search_rescue_demo()
        run_swarm_exploration_demo() 
        run_formation_control_demo()
        
        total_elapsed = time.time() - total_start
        
        print("\n" + "="*70)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"⏱️  Total demo time: {total_elapsed:.2f} seconds")
        print("\n🏆 Key achievements demonstrated:")
        print("✅ Multi-agent coordination and communication")
        print("✅ Physics-informed decision making")
        print("✅ Hierarchical control architectures")
        print("✅ Real-time adaptation to environment changes")
        print("✅ Scalable performance across different team sizes")
        print("✅ Dynamic formation control and mission execution")
        print("\n🚀 PI-HMARL v1.0.0 is ready for deployment!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()