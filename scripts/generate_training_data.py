#!/usr/bin/env python3
"""Generate synthetic training data for PI-HMARL

This script generates physics-accurate synthetic training data using
real-world parameters extracted from manufacturer specifications.
"""

import sys
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.real_parameter_extractor import RealParameterExtractor
from data.physics_accurate_synthetic import PhysicsAccurateSynthetic, SyntheticScenario
from utils.logger import get_logger

# Setup logging
logger = get_logger(name="data_generation")


def generate_search_rescue_scenarios(num_scenarios: int):
    """Generate search and rescue scenarios"""
    scenarios = []
    
    for i in range(num_scenarios):
        # Vary parameters
        num_agents = 3 + (i % 8)  # 3-10 agents
        
        scenario = SyntheticScenario(
            name=f"search_rescue_{i:04d}",
            description=f"Search and rescue with {num_agents} drones",
            num_agents=num_agents,
            duration=120.0,  # 2 minutes
            timestep=0.01,
            world_size=(200.0, 200.0, 50.0),
            mission_type="search_rescue",
            weather_scenario="nominal" if i % 10 < 7 else "windy",
            agent_types=["dji_mavic_3"] * num_agents
        )
        
        # Add random obstacles (buildings, trees)
        num_obstacles = 5 + (i % 10)
        for j in range(num_obstacles):
            obstacle = {
                "position": (
                    50 + j * 10,
                    50 + (j % 3) * 20,
                    0
                ),
                "size": (10, 10, 15 + j % 10),
                "static": True
            }
            scenario.obstacles.append(obstacle)
        
        # Add search targets
        num_targets = 1 + (i % 3)
        for j in range(num_targets):
            target = (
                100 + j * 30,
                100 + j * 20,
                0
            )
            scenario.targets.append(target)
        
        scenarios.append(scenario)
    
    return scenarios


def generate_industrial_scenarios(num_scenarios: int):
    """Generate industrial automation scenarios"""
    scenarios = []
    
    for i in range(num_scenarios):
        num_agents = 4 + (i % 7)  # 4-10 agents
        
        scenario = SyntheticScenario(
            name=f"industrial_{i:04d}",
            description=f"Industrial inspection with {num_agents} drones",
            num_agents=num_agents,
            duration=180.0,  # 3 minutes
            timestep=0.01,
            world_size=(150.0, 150.0, 30.0),
            mission_type="surveillance",
            weather_scenario="nominal",  # Indoor
            agent_types=["parrot_anafi"] * num_agents
        )
        
        # Add industrial structures
        # Create warehouse layout
        for row in range(3):
            for col in range(4):
                obstacle = {
                    "position": (
                        30 + col * 30,
                        30 + row * 40,
                        0
                    ),
                    "size": (20, 30, 10),
                    "static": True
                }
                scenario.obstacles.append(obstacle)
        
        scenarios.append(scenario)
    
    return scenarios


def generate_formation_scenarios(num_scenarios: int):
    """Generate formation flying scenarios"""
    scenarios = []
    
    for i in range(num_scenarios):
        num_agents = 5 + (i % 6)  # 5-10 agents
        
        scenario = SyntheticScenario(
            name=f"formation_{i:04d}",
            description=f"Formation flying with {num_agents} drones",
            num_agents=num_agents,
            duration=90.0,  # 1.5 minutes
            timestep=0.01,
            world_size=(300.0, 300.0, 100.0),
            mission_type="formation",
            weather_scenario="windy" if i % 5 == 0 else "nominal",
            agent_types=["dji_mavic_3"] * num_agents
        )
        
        # Initial positions in grid formation
        grid_size = int(num_agents ** 0.5) + 1
        positions = []
        for j in range(num_agents):
            row = j // grid_size
            col = j % grid_size
            pos = (
                100 + col * 10,
                100 + row * 10,
                20
            )
            positions.append(pos)
        scenario.initial_positions = positions
        
        scenarios.append(scenario)
    
    return scenarios


def main():
    """Main data generation function"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for PI-HMARL"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=100,
        help="Number of scenarios per type (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/synthetic"),
        help="Output directory for synthetic data"
    )
    parser.add_argument(
        "--scenario-types",
        nargs="+",
        default=["search_rescue", "industrial", "formation"],
        help="Types of scenarios to generate"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render simulation (slower)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate only 1 scenario per type for testing"
    )
    
    args = parser.parse_args()
    
    if args.test:
        args.num_scenarios = 1
        logger.info("Test mode: generating 1 scenario per type")
    
    # Initialize components
    logger.info("Initializing real parameter extractor...")
    param_extractor = RealParameterExtractor()
    
    # Print parameter summary
    logger.info(param_extractor.generate_parameter_report())
    
    # Save parameter specifications
    param_output_dir = args.output_dir / "parameters"
    param_output_dir.mkdir(parents=True, exist_ok=True)
    param_extractor.save_specs_to_file(param_output_dir)
    
    # Initialize synthetic generator
    logger.info("Initializing physics-accurate synthetic generator...")
    generator = PhysicsAccurateSynthetic(
        parameter_extractor=param_extractor,
        render=args.render
    )
    
    # Generate scenarios for each type
    all_scenarios = []
    
    if "search_rescue" in args.scenario_types:
        logger.info(f"Generating {args.num_scenarios} search & rescue scenarios...")
        scenarios = generate_search_rescue_scenarios(args.num_scenarios)
        all_scenarios.extend(scenarios)
    
    if "industrial" in args.scenario_types:
        logger.info(f"Generating {args.num_scenarios} industrial scenarios...")
        scenarios = generate_industrial_scenarios(args.num_scenarios)
        all_scenarios.extend(scenarios)
    
    if "formation" in args.scenario_types:
        logger.info(f"Generating {args.num_scenarios} formation scenarios...")
        scenarios = generate_formation_scenarios(args.num_scenarios)
        all_scenarios.extend(scenarios)
    
    # Generate data for all scenarios
    logger.info(f"Generating synthetic data for {len(all_scenarios)} scenarios...")
    
    for i, scenario in enumerate(all_scenarios):
        logger.info(f"Processing scenario {i+1}/{len(all_scenarios)}: {scenario.name}")
        
        # Generate trajectory data
        states = generator.generate_scenario_data(scenario)
        
        # Save to file
        output_path = args.output_dir / f"{scenario.mission_type}" / f"{scenario.name}.h5"
        generator.save_scenario_data(states, scenario, output_path)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(all_scenarios)} scenarios completed")
    
    # Cleanup
    generator.cleanup()
    
    logger.info(f"✓ Data generation complete! Generated {len(all_scenarios)} scenarios")
    logger.info(f"✓ Data saved to: {args.output_dir}")
    
    # Print summary statistics
    total_duration = sum(s.duration for s in all_scenarios)
    total_agents = sum(s.num_agents for s in all_scenarios)
    logger.info(f"✓ Total simulation time: {total_duration/60:.1f} minutes")
    logger.info(f"✓ Total agent-minutes: {total_agents * total_duration / 60:.0f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Data generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during data generation: {e}", exc_info=True)
        sys.exit(1)