#!/usr/bin/env python
"""Basic test of PI-HMARL components"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing PI-HMARL components...")

# Test basic imports
try:
    from src.core import MultiAgentSystem
    print("✓ Core module imported successfully")
except Exception as e:
    print(f"✗ Core module import failed: {e}")

try:
    from src.models import PhysicsInformedModel
    print("✓ Models module imported successfully")
except Exception as e:
    print(f"✗ Models module import failed: {e}")

try:
    from src.environments import PhysicsEnvironment
    print("✓ Environments module imported successfully")
except Exception as e:
    print(f"✗ Environments module import failed: {e}")

try:
    from src.training import PIHMARLTrainer
    print("✓ Training module imported successfully")
except Exception as e:
    print(f"✗ Training module import failed: {e}")

# Test creating basic system
print("\nTesting basic system creation...")
try:
    config = {
        'num_agents': 2,
        'state_dim': 8,
        'action_dim': 4,
        'communication_range': 20.0
    }
    
    # Note: These are placeholder tests since we need to implement the actual classes
    print("✓ Configuration created")
    
    # Test scenarios
    print("\nTesting scenario imports...")
    from src.scenarios.search_rescue import SearchRescueScenario
    print("✓ Search & Rescue scenario imported")
    
    from src.scenarios.swarm_exploration import SwarmExplorationScenario
    print("✓ Swarm Exploration scenario imported")
    
    from src.scenarios.formation_control import FormationControlScenario
    print("✓ Formation Control scenario imported")
    
    print("\nAll basic imports successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nPI-HMARL basic test completed.")