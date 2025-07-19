#!/usr/bin/env python
"""
ROBUST DEMO RUNNER
Handles all scenarios with error checking and optimized performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import traceback
from src.scenarios import SearchRescueScenario, FormationControlScenario

def safe_get_state_value(state, keys, default=0):
    """Safely get nested state values"""
    try:
        current = state
        for key in keys if isinstance(keys, list) else [keys]:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default

def run_search_rescue_demo():
    """Run optimized search and rescue demo"""
    print("\n" + "="*60)
    print("🚁 OPTIMIZED SEARCH & RESCUE SCENARIO")
    print("="*60)
    
    try:
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
                
                # Robust state access
                rescued = safe_get_state_value(state, ['victims', 'rescued'], 0)
                detected = safe_get_state_value(state, ['victims', 'detected'], 0)
                total = safe_get_state_value(state, ['victims', 'total'], scenario.num_victims)
                time_val = safe_get_state_value(state, 'time', step * 0.1)
                agents_count = len(state.get('agents', {}))
                
                print(f"Time: {time_val:6.1f}s | "
                      f"Detected: {detected}/{total} | "
                      f"Rescued: {rescued}/{total} | "
                      f"Active agents: {agents_count}")
            
            # Check if mission complete
            state = scenario.get_state()
            rescued = safe_get_state_value(state, ['victims', 'rescued'], 0)
            
            if rescued == scenario.num_victims:
                time_val = safe_get_state_value(state, 'time', step * 0.1)
                print(f"\n✅ All victims rescued in {time_val:.1f} seconds!")
                break
        
        elapsed = time.time() - start_time
        
        # Final statistics
        final_state = scenario.get_state()
        final_rescued = safe_get_state_value(final_state, ['victims', 'rescued'], 0)
        final_detected = safe_get_state_value(final_state, ['victims', 'detected'], 0)
        final_total = safe_get_state_value(final_state, ['victims', 'total'], scenario.num_victims)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Detected: {final_detected}/{final_total} ({final_detected/final_total*100:.0f}%)")
        print(f"   Rescued: {final_rescued}/{final_total} ({final_rescued/final_total*100:.0f}%)")
        print(f"   Simulation time: {elapsed:.2f} seconds")
        
        # Success evaluation
        rescue_rate = final_rescued / final_total
        detection_rate = final_detected / final_total
        
        if rescue_rate >= 0.8:
            print("   🏆 EXCELLENT PERFORMANCE!")
        elif rescue_rate >= 0.6:
            print("   🎯 GOOD PERFORMANCE!")
        elif rescue_rate >= 0.4:
            print("   ✅ ACCEPTABLE PERFORMANCE!")
        elif detection_rate >= 0.8:
            print("   🔍 STRONG DETECTION PERFORMANCE!")
        else:
            print("   ⚠️  Room for improvement")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in search rescue demo: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def run_formation_control_demo():
    """Run optimized formation control demo"""
    print("\n" + "="*60)
    print("✈️  OPTIMIZED FORMATION CONTROL SCENARIO")
    print("="*60)
    
    try:
        scenario = FormationControlScenario(
            num_agents=4,
            environment_size=(150.0, 150.0),
            num_obstacles=3
        )
        
        print(f"- Environment: {scenario.environment_size[0]}x{scenario.environment_size[1]} meters")
        print(f"- Agents: {scenario.num_agents} formation agents")
        print(f"- Running simulation...\n")
        
        start_time = time.time()
        max_steps = 300
        formation_qualities = []
        
        for step in range(max_steps):
            scenario.step(dt=0.1)
            
            if step % 30 == 0:
                state = scenario.get_state()
                
                # Robust state access
                formation_type = safe_get_state_value(state, 'formation_type', 'unknown')
                quality = safe_get_state_value(state, 'formation_quality', 0.0)
                time_val = safe_get_state_value(state, 'time', step * 0.1)
                
                formation_qualities.append(quality)
                
                print(f"Time: {time_val:6.1f}s | "
                      f"Formation: {formation_type:8s} | "
                      f"Quality: {quality:4.2f}")
        
        elapsed = time.time() - start_time
        
        # Calculate performance
        if formation_qualities:
            avg_quality = sum(formation_qualities) / len(formation_qualities)
            stable_qualities = [q for q in formation_qualities[5:] if q > 0.1]  # Skip initial
            stable_avg = sum(stable_qualities) / len(stable_qualities) if stable_qualities else 0
        else:
            avg_quality = 0
            stable_avg = 0
        
        print(f"\n📊 FORMATION RESULTS:")
        print(f"   Average formation quality: {avg_quality:.2f}")
        print(f"   Stable period average: {stable_avg:.2f}")
        print(f"   Simulation time: {elapsed:.2f} seconds")
        
        # Success evaluation
        if stable_avg >= 0.7:
            print("   🏆 EXCELLENT FORMATION CONTROL!")
        elif stable_avg >= 0.5:
            print("   🎯 GOOD FORMATION CONTROL!")
        elif stable_avg >= 0.3:
            print("   ✅ ACCEPTABLE FORMATION CONTROL!")
        else:
            print("   ⚠️  Formation needs improvement")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in formation demo: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main demo function with error handling"""
    print("🚀 PI-HMARL OPTIMIZED SYSTEM DEMO")
    print("Physics-Informed Hierarchical Multi-Agent RL")
    print("="*60)
    
    print("\nAvailable scenarios:")
    print("1. Search & Rescue - Optimized multi-agent victim rescue")
    print("2. Formation Control - Advanced formation maintenance")
    print("3. Run both demos")
    
    try:
        choice = input("\nSelect demo (1-3): ").strip()
    except EOFError:
        choice = "3"  # Default for non-interactive
    
    success_count = 0
    total_demos = 0
    
    if choice == "1":
        total_demos = 1
        if run_search_rescue_demo():
            success_count += 1
    elif choice == "2":
        total_demos = 1
        if run_formation_control_demo():
            success_count += 1
    else:  # choice == "3" or any other
        print("\nRunning all optimized demos...")
        total_demos = 2
        
        if run_search_rescue_demo():
            success_count += 1
        
        if run_formation_control_demo():
            success_count += 1
        
        print("\n" + "="*60)
        print("🏆 ALL DEMOS COMPLETED!")
        print("="*60)
    
    # Final summary
    if total_demos > 0:
        success_rate = success_count / total_demos
        print(f"\n📊 Demo Success Rate: {success_rate*100:.0f}% ({success_count}/{total_demos})")
        
        if success_rate >= 0.8:
            print("🎉 EXCELLENT! System performing at high level!")
        elif success_rate >= 0.5:
            print("✅ GOOD! System shows strong capabilities!")
        else:
            print("⚠️  Some scenarios need attention.")
    
    print("\n🙏 Thank you for trying the optimized PI-HMARL system!")
    print("For detailed analysis, run: ./venv/bin/python view_optimized_results.py")

if __name__ == "__main__":
    main()
