#!/usr/bin/env python
"""Working main demo for PI-HMARL system"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from src.scenarios import (
    SearchRescueScenario,
    SwarmExplorationScenario,
    FormationControlScenario
)

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
    max_steps = 100
    
    for step in range(max_steps):
        scenario.step(dt=0.1)
        
        if step % 20 == 0:
            state = scenario.get_state()
            # Use correct state structure
            print(f"Time: {state['time']:6.1f}s | "
                  f"Detected: {state['victims']['detected']}/{state['victims']['total']} | "
                  f"Rescued: {state['victims']['rescued']}")
        
        # Check if all victims rescued
        if scenario.get_state()['victims']['rescued'] == scenario.num_victims:
            print(f"\n✓ All victims rescued in {scenario.get_state()['time']:.1f} seconds!")
            break
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f} seconds")

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
    
    start_time = time.time()
    max_steps = 100
    
    for step in range(max_steps):
        scenario.step(dt=0.1)
        
        if step % 20 == 0:
            state = scenario.get_state()
            print(f"Time: {state['time']:6.1f}s | "
                  f"Formation: {state['formation_type']:8s} | "
                  f"Quality: {state['formation_quality']:4.2f}")
    
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
    print("2. Formation Control - Maintaining geometric formations while navigating")
    print("3. Run both demos")
    
    try:
        choice = input("\nSelect demo (1-3): ").strip()
    except EOFError:
        choice = "3"  # Default to run all if no input
    
    if choice == "1":
        run_search_rescue_demo()
    elif choice == "2":
        run_formation_control_demo()
    else:  # choice == "3" or any other input
        print("\nRunning all demos...")
        run_search_rescue_demo()
        run_formation_control_demo()
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED!")
        print("="*60)
    
    print("\nThank you for trying PI-HMARL!")
    print("For more information, see the documentation in docs/")

if __name__ == "__main__":
    main()