
# GENIUS PATCH: Fix success criteria and coordination
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fix_search_rescue_genius import SearchRescueScenarioFixed, Victim, VictimStatus

# Replace the existing SearchRescueScenario with fixed version
class SearchRescueScenario:
    """Patched Search & Rescue Scenario with proper success criteria"""
    
    def __init__(self, **kwargs):
        # Use the fixed implementation
        self._impl = SearchRescueScenarioFixed(kwargs)
        self.victims = self._impl.victims
        self.time = 0.0
        
    def reset(self, num_agents=6):
        """Reset with proper victim handling"""
        return self._impl.reset(num_agents)
    
    def step(self, agent_positions, dt=0.1):
        """Step with coordination and rescue mechanics"""
        obs, rewards, (terminated, info) = self._impl.step(agent_positions, dt)
        
        # Convert to expected format
        self.time = self._impl.time
        
        # Return in expected format with proper success criteria
        return {
            "observations": obs,
            "rewards": rewards,
            "terminated": terminated,
            "info": info,
            "success": info.get("success", False),
            "rescue_rate": info.get("rescue_rate", 0.0)
        }
    
    def get_success_rate(self):
        """Get current success rate"""
        return self._impl.episode_stats["victims_rescued"] / max(1, self._impl.num_victims)
    
    def evaluate(self):
        """Evaluate scenario performance"""
        rescue_rate = self.get_success_rate()
        return {
            "success": rescue_rate >= 0.85,
            "rescue_rate": rescue_rate,
            "victims_rescued": self._impl.episode_stats["victims_rescued"],
            "total_victims": self._impl.num_victims
        }
