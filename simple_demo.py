#!/usr/bin/env python
"""Simple demo of PI-HMARL system"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging
from src.scenarios.search_rescue import SearchRescueScenario

def main():
    print("="*60)
    print("PI-HMARL Simple Demo")
    print("="*60)
    
    # Setup logging
    setup_logging(log_level="INFO", console_output=True)
    
    print("\nRunning Search & Rescue Scenario...")
    
    # Create scenario
    scenario = SearchRescueScenario(
        area_size=(50.0, 50.0),
        num_victims=3,
        num_agents=2,
        obstacle_density=0.05
    )
    
    print(f"- Area: {scenario.area_size[0]}x{scenario.area_size[1]} meters")
    print(f"- Agents: {scenario.num_agents}")
    print(f"- Victims: {scenario.num_victims}")
    print(f"- Starting simulation...\n")
    
    # Run simulation
    max_steps = 200
    for step in range(max_steps):
        scenario.step(dt=0.1)
        
        if step % 20 == 0:
            state = scenario.get_state()
            print(f"Step {step:3d} | Time: {state['time']:5.1f}s | "
                  f"Rescued: {state['victims']['rescued']}/{state['victims']['total']} | "
                  f"Detected: {state['victims']['detected']}")
        
        # Check completion
        if scenario.get_state()['victims']['rescued'] == scenario.num_victims:
            final_state = scenario.get_state()
            print(f"\n✓ Mission completed in {final_state['time']:.1f} seconds!")
            print(f"  All {final_state['victims']['total']} victims rescued")
            break
    else:
        print(f"\n⚠ Time limit reached after {max_steps} steps")
    
    print(f"\nDemo completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()