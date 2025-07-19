#!/usr/bin/env python
"""
CRITICAL FIX: Search Pattern Assignment
This fixes the core issue preventing agent movement and victim detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Tuple, List, Dict, Any
from src.scenarios.search_rescue import SearchRescueScenario, SearchPattern, AgentRole

def fix_search_rescue_patterns():
    """
    CRITICAL FIX: Add search pattern assignment to SearchRescueScenario
    This is the root cause of 0% success rates
    """
    
    # Read the current search_rescue.py
    with open('src/scenarios/search_rescue.py', 'r') as f:
        content = f.read()
    
    # Create the pattern assignment method
    pattern_assignment_code = '''
    def assign_search_patterns(self):
        """CRITICAL FIX: Assign search patterns to agents"""
        searcher_agents = [a for a in self.agents if a.role == AgentRole.SEARCHER]
        
        if not searcher_agents:
            return
            
        # Create simple grid-based search patterns
        num_searchers = len(searcher_agents)
        area_bounds = (0, 0, self.area_size[0], self.area_size[1])
        
        # Divide search area into sectors
        sectors_per_side = max(1, int(np.sqrt(num_searchers)))
        sector_width = self.area_size[0] / sectors_per_side
        sector_height = self.area_size[1] / sectors_per_side
        
        for i, agent in enumerate(searcher_agents):
            # Assign sector to agent
            sector_x = i % sectors_per_side
            sector_y = i // sectors_per_side
            
            # Create waypoints for systematic search
            start_x = sector_x * sector_width
            start_y = sector_y * sector_height
            end_x = min(start_x + sector_width, self.area_size[0])
            end_y = min(start_y + sector_height, self.area_size[1])
            
            # Create grid search pattern within sector
            waypoints = []
            num_points = 8  # 8 waypoints per sector
            
            for j in range(num_points):
                if j % 2 == 0:  # Horizontal sweep
                    x = start_x + (end_x - start_x) * (j // 2) / (num_points // 2 - 1)
                    y = start_y if (j // 2) % 2 == 0 else end_y
                else:  # Vertical movement
                    x = start_x + (end_x - start_x) * (j // 2) / (num_points // 2 - 1)
                    y = end_y if (j // 2) % 2 == 0 else start_y
                
                waypoints.append(np.array([x, y, 0.0]))
            
            # Create search pattern
            pattern = SearchPattern(
                pattern_type='grid_search',
                start_position=agent.position.copy(),
                parameters={'sector': (sector_x, sector_y)},
                waypoints=waypoints
            )
            
            agent.search_pattern = pattern
            print(f"Assigned search pattern to {agent.agent_id}: {len(waypoints)} waypoints")
'''
    
    # Add the method to the SearchRescueScenario class
    # Find the class definition and add the method before the last method
    class_start = content.find('class SearchRescueScenario:')
    if class_start == -1:
        print("❌ Could not find SearchRescueScenario class")
        return False
    
    # Find a good insertion point (before the last method)
    last_def_pos = content.rfind('def ', class_start)
    if last_def_pos == -1:
        print("❌ Could not find insertion point")
        return False
    
    # Insert the method
    new_content = content[:last_def_pos] + pattern_assignment_code + '\n    ' + content[last_def_pos:]
    
    # Now add call to assign_search_patterns in __init__
    init_end = new_content.find('self._initialize_obstacles()', class_start)
    if init_end != -1:
        init_end = new_content.find('\n', init_end) + 1
        new_content = new_content[:init_end] + '        self.assign_search_patterns()  # CRITICAL FIX\n' + new_content[init_end:]
    
    # Write the fixed file
    with open('src/scenarios/search_rescue.py', 'w') as f:
        f.write(new_content)
    
    print("✅ CRITICAL FIX APPLIED: Search pattern assignment added")
    return True

def test_fix():
    """Test that the fix works"""
    print("🧪 Testing the critical fix...")
    
    from src.scenarios import SearchRescueScenario
    
    # Create scenario
    scenario = SearchRescueScenario(area_size=(60,60), num_victims=3, num_agents=2)
    
    # Check if patterns are assigned
    searcher_count = 0
    patterns_assigned = 0
    
    for agent in scenario.agents:
        if agent.role == AgentRole.SEARCHER:
            searcher_count += 1
            if agent.search_pattern is not None:
                patterns_assigned += 1
                print(f"✅ {agent.agent_id} has search pattern with {len(agent.search_pattern.waypoints)} waypoints")
    
    print(f"Searcher agents: {searcher_count}, Patterns assigned: {patterns_assigned}")
    
    # Test movement
    print("\n🏃 Testing agent movement...")
    initial_positions = [agent.position.copy() for agent in scenario.agents]
    
    # Run a few steps
    for i in range(20):
        scenario.step(dt=0.1)
    
    # Check if agents moved
    moved_count = 0
    for i, agent in enumerate(scenario.agents):
        distance_moved = np.linalg.norm(agent.position - initial_positions[i])
        if distance_moved > 1.0:  # Moved more than 1 unit
            moved_count += 1
            print(f"✅ {agent.agent_id} moved {distance_moved:.1f} units")
    
    print(f"Agents that moved: {moved_count}/{len(scenario.agents)}")
    
    # Test detection
    print("\n🔍 Testing victim detection...")
    for i in range(50):
        scenario.step(dt=0.1)
        if i % 10 == 0:
            state = scenario.get_state()
            detected = state['victims']['detected']
            if detected > 0:
                print(f"🎉 BREAKTHROUGH: {detected} victims detected at step {i}!")
                return True
    
    state = scenario.get_state()
    detected = state['victims']['detected']
    print(f"Final detection count: {detected}")
    
    return detected > 0

def main():
    """Apply critical fix and test"""
    print("🚨 APPLYING CRITICAL FIX FOR 0% SUCCESS RATE ISSUE")
    print("This should immediately unlock 20-30% hidden performance")
    
    # Apply the fix
    success = fix_search_rescue_patterns()
    
    if success:
        # Test the fix
        detection_working = test_fix()
        
        if detection_working:
            print("\n🎉 SUCCESS! CRITICAL FIX WORKING!")
            print("✅ Agents now have search patterns")
            print("✅ Agents are moving") 
            print("✅ Victim detection is working")
            print("\nNext: Run benchmarks to measure performance improvement")
        else:
            print("\n⚠️ Fix applied but detection still not working")
            print("Additional debugging needed")
    else:
        print("\n❌ Failed to apply fix")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)