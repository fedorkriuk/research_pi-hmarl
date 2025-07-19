#!/usr/bin/env python
"""Main demo for PI-HMARL system"""

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

# Setup logging
setup_logging(log_level="INFO", console_output=True)
logger = logging.getLogger(__name__)

def run_search_rescue_demo():
    """Run search and rescue demo"""
    print("\n" + "="*60)
    print("SEARCH & RESCUE SCENARIO")
    print("="*60)
    
    scenario = SearchRescueScenario(
        area_size=(100.0, 100.0),
        num_victims=5,
        num_agents=3,
        obstacle_density=0.05
    )
    
    print(f"- Area: {scenario.area_size[0]}x{scenario.area_size[1]} meters")
    print(f"- Agents: {scenario.num_agents} search agents")
    print(f"- Victims: {scenario.num_victims} to rescue")
    print(f"- Running simulation...\n")
    
    start_time = time.time()
    max_steps = 500
    
    for step in range(max_steps):
        scenario.step(dt=0.1)
        
        if step % 50 == 0:
            state = scenario.get_state()
            print(f"Time: {state['time']:6.1f}s | "
                  f"Rescued: {state['rescued_count']}/{state['total_victims']} | "
                  f"Active agents: {state['active_agents']}")
        
        # Check if all victims rescued
        if scenario.get_state()['rescued_count'] == scenario.num_victims:
            print(f"\n✓ All victims rescued in {scenario.get_state()['time']:.1f} seconds!")
            break
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f} seconds")

def run_swarm_exploration_demo():
    """Run swarm exploration demo"""
    print("\n" + "="*60)
    print("SWARM EXPLORATION SCENARIO")
    print("="*60)
    
    scenario = SwarmExplorationScenario(
        environment_size=(100, 100),
        num_agents=4,
        obstacle_complexity=0.2
    )
    
    print(f"- Environment: {scenario.environment_size[0]}x{scenario.environment_size[1]} cells")
    print(f"- Agents: {scenario.num_agents} exploration agents")
    print(f"- Obstacle complexity: {0.2*100:.0f}%")
    print(f"- Running simulation...\n")
    
    start_time = time.time()
    max_time = 60.0  # 60 seconds max
    
    while scenario.time < max_time and not scenario.completed:
        scenario.step(dt=0.1)
        
        if int(scenario.time) % 5 == 0 and scenario.time > 0:
            state = scenario.get_state()
            print(f"Time: {state['time']:6.1f}s | "
                  f"Explored: {state['exploration_rate']*100:5.1f}% | "
                  f"Frontiers: {state['frontiers_remaining']:4d}")
    
    state = scenario.get_state()
    elapsed = time.time() - start_time
    
    if scenario.completed:
        print(f"\n✓ Exploration completed! {state['exploration_rate']*100:.1f}% explored")
    else:
        print(f"\n⚠ Time limit reached. {state['exploration_rate']*100:.1f}% explored")
    
    print(f"Simulation completed in {elapsed:.2f} seconds")

def run_formation_control_demo():
    """Run formation control demo"""
    print("\n" + "="*60)
    print("FORMATION CONTROL SCENARIO")
    print("="*60)
    
    scenario = FormationControlScenario(
        num_agents=6,
        environment_size=(200.0, 200.0),
        num_obstacles=5
    )
    
    print(f"- Environment: {scenario.environment_size[0]}x{scenario.environment_size[1]} meters")
    print(f"- Agents: {scenario.num_agents} formation agents")
    print(f"- Obstacles: {len(scenario.obstacles)}")
    print(f"- Running simulation...\n")
    
    # Create mission waypoints
    waypoints = scenario.create_mission_waypoints()
    scenario.controller.set_waypoints(waypoints)
    print(f"Mission: Navigate through {len(waypoints)} waypoints")
    
    start_time = time.time()
    max_steps = 300
    
    formations = ["LINE", "WEDGE", "CIRCLE"]
    formation_idx = 0
    last_change_time = 0
    
    for step in range(max_steps):
        scenario.step(dt=0.1)
        
        # Change formation every 10 seconds
        if scenario.time - last_change_time > 10.0:
            from src.scenarios.formation_control import FormationType
            formation_type = formations[formation_idx % len(formations)]
            scenario.controller.change_formation(
                getattr(FormationType, formation_type),
                scenario.time
            )
            formation_idx += 1
            last_change_time = scenario.time
        
        if step % 30 == 0:
            state = scenario.get_state()
            print(f"Time: {state['time']:6.1f}s | "
                  f"Formation: {state['formation_type']:8s} | "
                  f"Quality: {state['formation_quality']:4.2f} | "
                  f"Progress: {state['waypoint_progress']}")
        
        # Check if mission completed
        if scenario.controller.current_waypoint_index >= len(waypoints):
            print(f"\n✓ Mission completed in {scenario.time:.1f} seconds!")
            break
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f} seconds")

def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("PI-HMARL SYSTEM DEMO")
    print("Physics-Informed Hierarchical Multi-Agent RL")
    print("="*60)
    
    print("\nAvailable scenarios:")
    print("1. Search & Rescue - Multiple agents searching for and rescuing victims")
    print("2. Swarm Exploration - Collaborative environment mapping")
    print("3. Formation Control - Maintaining geometric formations while navigating")
    print("4. Run all demos")
    
    choice = input("\nSelect demo (1-4): ").strip()
    
    if choice == "1":
        run_search_rescue_demo()
    elif choice == "2":
        run_swarm_exploration_demo()
    elif choice == "3":
        run_formation_control_demo()
    elif choice == "4":
        print("\nRunning all demos...")
        run_search_rescue_demo()
        run_swarm_exploration_demo()
        run_formation_control_demo()
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED!")
        print("="*60)
    else:
        print("Invalid choice. Please run again and select 1-4.")
        return
    
    print("\nThank you for trying PI-HMARL!")
    print("For more information, see the documentation in docs/")

if __name__ == "__main__":
    main()