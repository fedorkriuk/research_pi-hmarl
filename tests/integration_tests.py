"""Integration Tests for PI-HMARL System

This module implements integration tests for system-wide
functionality and component interactions.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import time
import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import MultiAgentEnvironment, EnvironmentConfig
from src.agents.hierarchical_agent import HierarchicalAgent
from src.physics_engine.physics_integration import PhysicsEngine, PhysicsConfig
from src.communication.protocol import CommunicationProtocol
from src.task_decomposition.task_analyzer import TaskAnalyzer
from src.energy.energy_optimizer import EnergyOptimizer
from src.visualization.dashboard import Dashboard, DashboardConfig

from .test_framework import TestCase, TestSuite, TestLevel, TestAssertions


class SystemIntegrationTests:
    """System-wide integration tests"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create system integration test suite"""
        suite = TestSuite("SystemIntegrationTests", "System-wide integration tests")
        
        suite.add_test(TestCase(
            name="SystemIntegrationTests.test_multi_agent_coordination",
            description="Test multi-agent coordination with physics",
            test_func=SystemIntegrationTests.test_multi_agent_coordination,
            level=TestLevel.INTEGRATION,
            tags=["integration", "coordination"],
            timeout=120.0
        ))
        
        suite.add_test(TestCase(
            name="SystemIntegrationTests.test_hierarchical_control",
            description="Test hierarchical control system",
            test_func=SystemIntegrationTests.test_hierarchical_control,
            level=TestLevel.INTEGRATION,
            tags=["integration", "hierarchy"],
            timeout=120.0
        ))
        
        suite.add_test(TestCase(
            name="SystemIntegrationTests.test_communication_network",
            description="Test communication network under load",
            test_func=SystemIntegrationTests.test_communication_network,
            level=TestLevel.INTEGRATION,
            tags=["integration", "communication"],
            timeout=90.0
        ))
        
        suite.add_test(TestCase(
            name="SystemIntegrationTests.test_energy_management",
            description="Test integrated energy management",
            test_func=SystemIntegrationTests.test_energy_management,
            level=TestLevel.INTEGRATION,
            tags=["integration", "energy"],
            timeout=90.0
        ))
        
        return suite
    
    @staticmethod
    def test_multi_agent_coordination() -> Dict[str, Any]:
        """Test multi-agent coordination with physics"""
        # Create environment
        env_config = EnvironmentConfig(
            num_agents=5,
            map_size=(1000, 1000, 200),
            physics_enabled=True,
            communication_enabled=True
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create agents
        agents = []
        for i in range(env_config.num_agents):
            agent = HierarchicalAgent(
                agent_id=i,
                levels=['strategic', 'tactical', 'operational'],
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
            agents.append(agent)
        
        # Reset environment
        observations = env.reset()
        
        # Run coordination test
        episode_rewards = []
        collisions = 0
        successful_coordinations = 0
        
        for step in range(100):
            # Get actions from agents
            actions = {}
            for i, agent in enumerate(agents):
                obs = observations[i]
                action = agent.act(obs)
                actions[i] = action
            
            # Step environment
            observations, rewards, dones, info = env.step(actions)
            
            # Track metrics
            episode_rewards.append(sum(rewards.values()))
            
            if 'collisions' in info:
                collisions += len(info['collisions'])
            
            if 'coordination_success' in info:
                successful_coordinations += info['coordination_success']
        
        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        coordination_rate = successful_coordinations / 100
        
        # Verify coordination
        assert coordination_rate > 0.7, f"Low coordination rate: {coordination_rate}"
        assert collisions < 5, f"Too many collisions: {collisions}"
        
        return {
            'avg_reward': avg_reward,
            'coordination_rate': coordination_rate,
            'collisions': collisions,
            'steps_completed': 100
        }
    
    @staticmethod
    def test_hierarchical_control() -> Dict[str, Any]:
        """Test hierarchical control system"""
        # Create hierarchical environment
        env_config = EnvironmentConfig(
            num_agents=4,
            hierarchical=True,
            levels=['strategic', 'tactical', 'operational']
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create hierarchical agents with different roles
        agents = []
        roles = ['commander', 'scout', 'scout', 'support']
        
        for i, role in enumerate(roles):
            agent = HierarchicalAgent(
                agent_id=i,
                levels=env_config.levels,
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                role=role
            )
            agents.append(agent)
        
        # Test hierarchical decision making
        observations = env.reset()
        
        # Strategic level decision
        strategic_goals = {}
        for agent in agents:
            if agent.role == 'commander':
                strategic_goals = agent.make_strategic_decision(observations)
                break
        
        assert strategic_goals, "Commander should make strategic decisions"
        
        # Tactical level planning
        tactical_plans = {}
        for i, agent in enumerate(agents):
            if agent.role != 'commander':
                plan = agent.make_tactical_plan(
                    observations[i],
                    strategic_goals.get(i, {})
                )
                tactical_plans[i] = plan
        
        assert len(tactical_plans) > 0, "Agents should create tactical plans"
        
        # Operational execution
        total_steps = 50
        goal_achievements = 0
        
        for step in range(total_steps):
            actions = {}
            
            for i, agent in enumerate(agents):
                # Get operational action based on tactical plan
                action = agent.execute_operational(
                    observations[i],
                    tactical_plans.get(i, {})
                )
                actions[i] = action
            
            observations, rewards, dones, info = env.step(actions)
            
            if 'goals_achieved' in info:
                goal_achievements += info['goals_achieved']
        
        achievement_rate = goal_achievements / (total_steps * len(agents))
        
        return {
            'strategic_goals': len(strategic_goals),
            'tactical_plans': len(tactical_plans),
            'achievement_rate': achievement_rate,
            'hierarchy_levels': len(env_config.levels)
        }
    
    @staticmethod
    def test_communication_network() -> Dict[str, Any]:
        """Test communication network under load"""
        # Create communication network
        num_agents = 10
        protocols = []
        
        for i in range(num_agents):
            protocol = CommunicationProtocol(agent_id=i)
            protocols.append(protocol)
        
        # Simulate network activity
        messages_sent = 0
        messages_received = 0
        latencies = []
        packet_loss = 0
        
        # Generate traffic
        start_time = time.time()
        
        for round in range(20):
            # Each agent sends messages
            for i, sender in enumerate(protocols):
                # Send to random agents
                for _ in range(3):
                    receiver_id = np.random.randint(0, num_agents)
                    if receiver_id != i:
                        message = sender.create_message(
                            msg_type='TASK_UPDATE',
                            receiver_id=receiver_id,
                            payload={'round': round, 'data': np.random.randn(10).tolist()},
                            requires_ack=True
                        )
                        
                        send_time = time.time()
                        success = sender.send_message(message)
                        
                        if success:
                            messages_sent += 1
                            
                            # Simulate reception
                            if np.random.random() > 0.05:  # 5% packet loss
                                protocols[receiver_id].receive_message(
                                    message.encode() if hasattr(message, 'encode') else b'test'
                                )
                                messages_received += 1
                                
                                # Calculate latency
                                latency = time.time() - send_time
                                latencies.append(latency)
                            else:
                                packet_loss += 1
        
        # Calculate metrics
        duration = time.time() - start_time
        throughput = messages_sent / duration
        avg_latency = np.mean(latencies) if latencies else 0
        delivery_rate = messages_received / messages_sent if messages_sent > 0 else 0
        
        # Verify network performance
        assert delivery_rate > 0.9, f"Low delivery rate: {delivery_rate}"
        assert avg_latency < 0.1, f"High latency: {avg_latency}"
        
        return {
            'messages_sent': messages_sent,
            'messages_received': messages_received,
            'throughput': throughput,
            'avg_latency': avg_latency * 1000,  # ms
            'delivery_rate': delivery_rate
        }
    
    @staticmethod
    def test_energy_management() -> Dict[str, Any]:
        """Test integrated energy management"""
        # Create environment with energy constraints
        env_config = EnvironmentConfig(
            num_agents=4,
            energy_enabled=True,
            charging_stations=[(500, 500, 0)]
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create energy optimizer
        optimizer = EnergyOptimizer(num_agents=env_config.num_agents)
        
        # Initialize agents with different battery levels
        battery_levels = [0.9, 0.7, 0.5, 0.3]
        
        observations = env.reset()
        
        # Run energy-aware mission
        total_steps = 100
        energy_violations = 0
        charging_events = 0
        mission_completions = 0
        
        for step in range(total_steps):
            # Get energy-optimized actions
            agent_states = {}
            for i in range(env_config.num_agents):
                agent_states[i] = {
                    'position': observations[i][:3],
                    'battery_soc': battery_levels[i],
                    'power_consumption': 20.0  # Watts
                }
            
            # Optimize actions considering energy
            optimization = optimizer.optimize_step(agent_states)
            
            actions = {}
            for i in range(env_config.num_agents):
                if optimization['actions'][i] == 'charge':
                    # Navigate to charging station
                    actions[i] = env.get_charging_action(i)
                    charging_events += 1
                else:
                    # Normal operation
                    actions[i] = optimization['actions'][i]
            
            # Step environment
            observations, rewards, dones, info = env.step(actions)
            
            # Update battery levels
            for i in range(env_config.num_agents):
                battery_levels[i] = info['battery_levels'][i]
                
                if battery_levels[i] < 0.1:
                    energy_violations += 1
            
            if 'missions_completed' in info:
                mission_completions += info['missions_completed']
        
        # Calculate metrics
        avg_battery = np.mean(battery_levels)
        min_battery = np.min(battery_levels)
        
        # Verify energy management
        assert energy_violations < 5, f"Too many energy violations: {energy_violations}"
        assert min_battery > 0.05, f"Battery too low: {min_battery}"
        assert charging_events > 0, "No charging occurred"
        
        return {
            'avg_battery': avg_battery,
            'min_battery': min_battery,
            'energy_violations': energy_violations,
            'charging_events': charging_events,
            'mission_completions': mission_completions
        }


class ScenarioTests:
    """Scenario-based integration tests"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create scenario test suite"""
        suite = TestSuite("ScenarioTests", "Scenario-based integration tests")
        
        suite.add_test(TestCase(
            name="ScenarioTests.test_search_and_rescue",
            description="Test search and rescue scenario",
            test_func=ScenarioTests.test_search_and_rescue,
            level=TestLevel.INTEGRATION,
            tags=["scenario", "search_rescue"],
            timeout=180.0
        ))
        
        suite.add_test(TestCase(
            name="ScenarioTests.test_surveillance_mission",
            description="Test surveillance mission scenario",
            test_func=ScenarioTests.test_surveillance_mission,
            level=TestLevel.INTEGRATION,
            tags=["scenario", "surveillance"],
            timeout=180.0
        ))
        
        suite.add_test(TestCase(
            name="ScenarioTests.test_delivery_coordination",
            description="Test multi-agent delivery scenario",
            test_func=ScenarioTests.test_delivery_coordination,
            level=TestLevel.INTEGRATION,
            tags=["scenario", "delivery"],
            timeout=150.0
        ))
        
        return suite
    
    @staticmethod
    def test_search_and_rescue() -> Dict[str, Any]:
        """Test search and rescue scenario"""
        # Create SAR environment
        env_config = EnvironmentConfig(
            num_agents=6,
            scenario='search_and_rescue',
            map_size=(2000, 2000, 200),
            targets=5,  # Number of targets to find
            time_limit=600  # 10 minutes
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create specialized agents
        agents = []
        roles = ['coordinator', 'searcher', 'searcher', 'searcher', 'rescuer', 'rescuer']
        
        for i, role in enumerate(roles):
            agent = HierarchicalAgent(
                agent_id=i,
                role=role,
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
            agents.append(agent)
        
        # Run search and rescue mission
        observations = env.reset()
        
        targets_found = 0
        targets_rescued = 0
        search_efficiency = []
        coordination_events = 0
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < env_config.time_limit:
            # Coordinator assigns search areas
            if step % 50 == 0:
                search_assignments = agents[0].assign_search_areas(observations)
                coordination_events += 1
            
            actions = {}
            for i, agent in enumerate(agents):
                if agent.role == 'searcher':
                    # Search assigned area
                    action = agent.search_area(
                        observations[i],
                        search_assignments.get(i, {})
                    )
                elif agent.role == 'rescuer':
                    # Respond to found targets
                    action = agent.rescue_target(observations[i])
                else:
                    # Coordinator monitors and updates
                    action = agent.coordinate(observations[i])
                
                actions[i] = action
            
            observations, rewards, dones, info = env.step(actions)
            
            # Track metrics
            if 'targets_found' in info:
                targets_found = info['targets_found']
            
            if 'targets_rescued' in info:
                targets_rescued = info['targets_rescued']
            
            if 'search_coverage' in info:
                search_efficiency.append(info['search_coverage'])
            
            step += 1
            
            if all(dones.values()):
                break
        
        # Calculate performance metrics
        mission_time = time.time() - start_time
        avg_search_efficiency = np.mean(search_efficiency) if search_efficiency else 0
        rescue_rate = targets_rescued / env_config.targets if env_config.targets > 0 else 0
        
        # Verify mission success
        assert targets_found >= env_config.targets * 0.8, "Not enough targets found"
        assert rescue_rate > 0.6, "Low rescue rate"
        
        return {
            'targets_found': targets_found,
            'targets_rescued': targets_rescued,
            'rescue_rate': rescue_rate,
            'mission_time': mission_time,
            'search_efficiency': avg_search_efficiency,
            'coordination_events': coordination_events
        }
    
    @staticmethod
    def test_surveillance_mission() -> Dict[str, Any]:
        """Test surveillance mission scenario"""
        # Create surveillance environment
        env_config = EnvironmentConfig(
            num_agents=4,
            scenario='surveillance',
            map_size=(1500, 1500, 150),
            surveillance_area=[[0, 0], [1500, 0], [1500, 1500], [0, 1500]],
            intruders=3,
            mission_duration=300  # 5 minutes
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create surveillance agents
        agents = []
        for i in range(env_config.num_agents):
            agent = HierarchicalAgent(
                agent_id=i,
                role='surveillance',
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                sensor_range=100.0  # meters
            )
            agents.append(agent)
        
        # Run surveillance mission
        observations = env.reset()
        
        area_coverage = []
        intruders_detected = 0
        false_alarms = 0
        response_times = []
        
        # Setup patrol patterns
        patrol_patterns = agents[0].generate_patrol_patterns(
            env_config.surveillance_area,
            env_config.num_agents
        )
        
        for step in range(env_config.mission_duration):
            actions = {}
            
            for i, agent in enumerate(agents):
                # Execute patrol with detection
                action, detection = agent.patrol_and_detect(
                    observations[i],
                    patrol_patterns[i]
                )
                
                if detection:
                    # Verify and respond to detection
                    if agent.verify_detection(observations[i]):
                        detection_time = step
                        
                        # Coordinate response
                        for j, other_agent in enumerate(agents):
                            if j != i:
                                other_agent.respond_to_detection(
                                    detection['position'],
                                    detection_time
                                )
                        
                        response_times.append(step)
                    else:
                        false_alarms += 1
                
                actions[i] = action
            
            observations, rewards, dones, info = env.step(actions)
            
            # Track metrics
            if 'area_coverage' in info:
                area_coverage.append(info['area_coverage'])
            
            if 'intruders_detected' in info:
                intruders_detected = info['intruders_detected']
        
        # Calculate performance
        avg_coverage = np.mean(area_coverage) if area_coverage else 0
        detection_rate = intruders_detected / env_config.intruders
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Verify surveillance effectiveness
        assert avg_coverage > 0.7, "Insufficient area coverage"
        assert detection_rate > 0.8, "Low detection rate"
        assert false_alarms < 5, "Too many false alarms"
        
        return {
            'area_coverage': avg_coverage,
            'intruders_detected': intruders_detected,
            'detection_rate': detection_rate,
            'false_alarms': false_alarms,
            'avg_response_time': avg_response_time
        }
    
    @staticmethod
    def test_delivery_coordination() -> Dict[str, Any]:
        """Test multi-agent delivery scenario"""
        # Create delivery environment
        env_config = EnvironmentConfig(
            num_agents=5,
            scenario='delivery',
            map_size=(1000, 1000, 100),
            delivery_points=10,
            packages=15,
            time_window=400  # seconds
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create delivery agents with different capacities
        agents = []
        capacities = [3, 3, 2, 2, 1]  # Package capacity
        
        for i, capacity in enumerate(capacities):
            agent = HierarchicalAgent(
                agent_id=i,
                role='delivery',
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                capacity=capacity
            )
            agents.append(agent)
        
        # Task decomposition and assignment
        task_analyzer = TaskAnalyzer()
        
        # Get delivery tasks
        observations = env.reset()
        delivery_tasks = env.get_delivery_tasks()
        
        # Decompose and assign tasks
        task_assignments = task_analyzer.decompose_delivery_tasks(
            delivery_tasks,
            [a.capacity for a in agents]
        )
        
        # Run delivery mission
        packages_delivered = 0
        total_distance = 0
        delivery_times = []
        coordination_conflicts = 0
        
        for step in range(env_config.time_window):
            actions = {}
            
            for i, agent in enumerate(agents):
                # Get assigned tasks
                assigned_tasks = task_assignments.get(i, [])
                
                if assigned_tasks:
                    # Plan route
                    route = agent.plan_delivery_route(
                        observations[i],
                        assigned_tasks
                    )
                    
                    # Execute delivery
                    action = agent.execute_delivery(
                        observations[i],
                        route
                    )
                else:
                    # Return to base
                    action = agent.return_to_base(observations[i])
                
                actions[i] = action
            
            observations, rewards, dones, info = env.step(actions)
            
            # Track metrics
            if 'packages_delivered' in info:
                new_deliveries = info['packages_delivered'] - packages_delivered
                packages_delivered = info['packages_delivered']
                
                if new_deliveries > 0:
                    delivery_times.extend([step] * new_deliveries)
            
            if 'total_distance' in info:
                total_distance = info['total_distance']
            
            if 'coordination_conflicts' in info:
                coordination_conflicts += info['coordination_conflicts']
        
        # Calculate performance
        delivery_rate = packages_delivered / env_config.packages
        avg_delivery_time = np.mean(delivery_times) if delivery_times else env_config.time_window
        efficiency = packages_delivered / (total_distance / 1000) if total_distance > 0 else 0
        
        # Verify delivery performance
        assert delivery_rate > 0.8, "Low delivery rate"
        assert coordination_conflicts < 5, "Too many coordination conflicts"
        
        return {
            'packages_delivered': packages_delivered,
            'delivery_rate': delivery_rate,
            'avg_delivery_time': avg_delivery_time,
            'total_distance': total_distance,
            'efficiency': efficiency,
            'coordination_conflicts': coordination_conflicts
        }


class PerformanceTests:
    """Performance and scalability tests"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create performance test suite"""
        suite = TestSuite("PerformanceTests", "Performance and scalability tests")
        
        suite.add_test(TestCase(
            name="PerformanceTests.test_scalability",
            description="Test system scalability with agent count",
            test_func=PerformanceTests.test_scalability,
            level=TestLevel.PERFORMANCE,
            tags=["performance", "scalability"],
            timeout=300.0
        ))
        
        suite.add_test(TestCase(
            name="PerformanceTests.test_real_time_performance",
            description="Test real-time performance requirements",
            test_func=PerformanceTests.test_real_time_performance,
            level=TestLevel.PERFORMANCE,
            tags=["performance", "realtime"],
            timeout=120.0
        ))
        
        suite.add_test(TestCase(
            name="PerformanceTests.test_resource_usage",
            description="Test resource usage and efficiency",
            test_func=PerformanceTests.test_resource_usage,
            level=TestLevel.PERFORMANCE,
            tags=["performance", "resources"],
            timeout=180.0
        ))
        
        return suite
    
    @staticmethod
    def test_scalability() -> Dict[str, Any]:
        """Test system scalability with agent count"""
        agent_counts = [5, 10, 20, 50]
        performance_metrics = {}
        
        for num_agents in agent_counts:
            # Create environment
            env_config = EnvironmentConfig(
                num_agents=num_agents,
                map_size=(2000, 2000, 200)
            )
            env = MultiAgentEnvironment(env_config)
            
            # Create agents
            agents = []
            for i in range(num_agents):
                agent = HierarchicalAgent(
                    agent_id=i,
                    obs_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0]
                )
                agents.append(agent)
            
            # Measure performance
            observations = env.reset()
            
            step_times = []
            communication_times = []
            coordination_times = []
            
            for _ in range(50):
                # Measure step time
                step_start = time.time()
                
                # Communication phase
                comm_start = time.time()
                for agent in agents:
                    agent.communicate(observations)
                comm_time = time.time() - comm_start
                communication_times.append(comm_time)
                
                # Coordination phase
                coord_start = time.time()
                actions = {}
                for i, agent in enumerate(agents):
                    actions[i] = agent.act(observations[i])
                coord_time = time.time() - coord_start
                coordination_times.append(coord_time)
                
                # Environment step
                observations, _, _, _ = env.step(actions)
                
                step_time = time.time() - step_start
                step_times.append(step_time)
            
            # Store metrics
            performance_metrics[num_agents] = {
                'avg_step_time': np.mean(step_times),
                'avg_comm_time': np.mean(communication_times),
                'avg_coord_time': np.mean(coordination_times),
                'max_step_time': np.max(step_times)
            }
        
        # Analyze scalability
        step_times_list = [m['avg_step_time'] for m in performance_metrics.values()]
        
        # Check sub-linear scaling
        scaling_factor = step_times_list[-1] / step_times_list[0]
        agent_factor = agent_counts[-1] / agent_counts[0]
        
        assert scaling_factor < agent_factor, "Poor scalability detected"
        
        return {
            'performance_metrics': performance_metrics,
            'scaling_factor': scaling_factor,
            'max_agents_tested': agent_counts[-1]
        }
    
    @staticmethod
    def test_real_time_performance() -> Dict[str, Any]:
        """Test real-time performance requirements"""
        # Create real-time environment
        env_config = EnvironmentConfig(
            num_agents=10,
            real_time=True,
            target_fps=30  # 30 Hz update rate
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create agents
        agents = []
        for i in range(env_config.num_agents):
            agent = HierarchicalAgent(
                agent_id=i,
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
            agents.append(agent)
        
        # Run real-time test
        observations = env.reset()
        
        frame_times = []
        missed_deadlines = 0
        target_frame_time = 1.0 / env_config.target_fps
        
        test_duration = 10.0  # seconds
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            frame_start = time.time()
            
            # Get actions
            actions = {}
            for i, agent in enumerate(agents):
                actions[i] = agent.act(observations[i])
            
            # Step environment
            observations, _, _, _ = env.step(actions)
            
            # Measure frame time
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            if frame_time > target_frame_time:
                missed_deadlines += 1
            
            # Sleep to maintain frame rate
            sleep_time = target_frame_time - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Calculate metrics
        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)
        fps_achieved = 1.0 / avg_frame_time
        deadline_miss_rate = missed_deadlines / len(frame_times)
        
        # Verify real-time performance
        assert fps_achieved > env_config.target_fps * 0.9, "Cannot maintain target FPS"
        assert deadline_miss_rate < 0.05, "Too many missed deadlines"
        
        return {
            'avg_frame_time': avg_frame_time * 1000,  # ms
            'max_frame_time': max_frame_time * 1000,  # ms
            'fps_achieved': fps_achieved,
            'deadline_miss_rate': deadline_miss_rate,
            'total_frames': len(frame_times)
        }
    
    @staticmethod
    def test_resource_usage() -> Dict[str, Any]:
        """Test resource usage and efficiency"""
        import psutil
        import os
        
        # Get process
        process = psutil.Process(os.getpid())
        
        # Initial measurements
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent(interval=1)
        
        # Create large-scale environment
        env_config = EnvironmentConfig(
            num_agents=20,
            map_size=(2000, 2000, 200),
            enable_visualization=False  # Disable for performance test
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create agents
        agents = []
        for i in range(env_config.num_agents):
            agent = HierarchicalAgent(
                agent_id=i,
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
            agents.append(agent)
        
        # Run workload
        observations = env.reset()
        
        cpu_samples = []
        memory_samples = []
        
        for step in range(100):
            # Get actions
            actions = {}
            for i, agent in enumerate(agents):
                actions[i] = agent.act(observations[i])
            
            # Step environment
            observations, _, _, _ = env.step(actions)
            
            # Sample resources
            if step % 10 == 0:
                cpu_samples.append(process.cpu_percent(interval=0.1))
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
        
        # Calculate resource usage
        avg_cpu = np.mean(cpu_samples)
        max_cpu = np.max(cpu_samples)
        avg_memory = np.mean(memory_samples)
        max_memory = np.max(memory_samples)
        memory_growth = max_memory - initial_memory
        
        # Verify resource efficiency
        assert avg_cpu < 80, "CPU usage too high"
        assert memory_growth < 500, "Excessive memory growth"
        
        return {
            'avg_cpu_usage': avg_cpu,
            'max_cpu_usage': max_cpu,
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'memory_growth_mb': memory_growth
        }


class StressTests:
    """Stress tests for edge cases and failure scenarios"""
    
    @staticmethod
    def create_test_suite() -> TestSuite:
        """Create stress test suite"""
        suite = TestSuite("StressTests", "Stress tests for edge cases")
        
        suite.add_test(TestCase(
            name="StressTests.test_communication_failure",
            description="Test system behavior under communication failures",
            test_func=StressTests.test_communication_failure,
            level=TestLevel.STRESS,
            tags=["stress", "failure"],
            timeout=120.0
        ))
        
        suite.add_test(TestCase(
            name="StressTests.test_agent_failures",
            description="Test system resilience to agent failures",
            test_func=StressTests.test_agent_failures,
            level=TestLevel.STRESS,
            tags=["stress", "resilience"],
            timeout=120.0
        ))
        
        suite.add_test(TestCase(
            name="StressTests.test_extreme_conditions",
            description="Test under extreme environmental conditions",
            test_func=StressTests.test_extreme_conditions,
            level=TestLevel.STRESS,
            tags=["stress", "environment"],
            timeout=150.0
        ))
        
        return suite
    
    @staticmethod
    def test_communication_failure() -> Dict[str, Any]:
        """Test system behavior under communication failures"""
        # Create environment with unreliable communication
        env_config = EnvironmentConfig(
            num_agents=8,
            communication_range=500.0,
            packet_loss_rate=0.3,  # 30% packet loss
            communication_delay=0.5  # 500ms delay
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create resilient agents
        agents = []
        for i in range(env_config.num_agents):
            agent = HierarchicalAgent(
                agent_id=i,
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                enable_fallback=True
            )
            agents.append(agent)
        
        # Run mission with communication failures
        observations = env.reset()
        
        communication_attempts = 0
        communication_failures = 0
        fallback_actions = 0
        mission_success = False
        
        for step in range(200):
            actions = {}
            
            for i, agent in enumerate(agents):
                # Attempt communication
                comm_success = agent.attempt_communication(observations[i])
                communication_attempts += 1
                
                if not comm_success:
                    communication_failures += 1
                    # Use fallback strategy
                    action = agent.fallback_action(observations[i])
                    fallback_actions += 1
                else:
                    # Normal coordinated action
                    action = agent.coordinated_action(observations[i])
                
                actions[i] = action
            
            observations, rewards, dones, info = env.step(actions)
            
            if 'mission_complete' in info and info['mission_complete']:
                mission_success = True
                break
        
        # Calculate metrics
        failure_rate = communication_failures / communication_attempts
        resilience_score = 1.0 if mission_success else 0.5
        
        # Verify system resilience
        assert failure_rate < 0.4, "Communication failure rate too high"
        assert fallback_actions > 0, "No fallback actions taken"
        assert mission_success or step > 150, "Mission failed too quickly"
        
        return {
            'communication_attempts': communication_attempts,
            'communication_failures': communication_failures,
            'failure_rate': failure_rate,
            'fallback_actions': fallback_actions,
            'mission_success': mission_success,
            'resilience_score': resilience_score
        }
    
    @staticmethod
    def test_agent_failures() -> Dict[str, Any]:
        """Test system resilience to agent failures"""
        # Create environment
        env_config = EnvironmentConfig(
            num_agents=10,
            enable_failures=True
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create agents
        agents = []
        for i in range(env_config.num_agents):
            agent = HierarchicalAgent(
                agent_id=i,
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
            agents.append(agent)
        
        # Track agent status
        active_agents = set(range(env_config.num_agents))
        failed_agents = set()
        
        observations = env.reset()
        
        # Simulate agent failures
        failure_times = [50, 75, 100, 125]  # Steps when agents fail
        agents_to_fail = [2, 5, 7, 9]
        
        task_completion_before = 0
        task_completion_after = 0
        redistribution_success = 0
        
        for step in range(150):
            # Simulate failures
            if step in failure_times:
                idx = failure_times.index(step)
                failed_agent = agents_to_fail[idx]
                active_agents.remove(failed_agent)
                failed_agents.add(failed_agent)
                
                # Redistribute tasks
                redistribution_success += env.redistribute_tasks(
                    failed_agent,
                    list(active_agents)
                )
            
            actions = {}
            for i in active_agents:
                actions[i] = agents[i].act(observations[i])
            
            observations, rewards, dones, info = env.step(actions)
            
            # Track task completion
            if 'tasks_completed' in info:
                if step < failure_times[0]:
                    task_completion_before = info['tasks_completed']
                else:
                    task_completion_after = info['tasks_completed']
        
        # Calculate resilience metrics
        agents_lost_ratio = len(failed_agents) / env_config.num_agents
        functionality_retained = len(active_agents) / env_config.num_agents
        task_completion_ratio = (
            task_completion_after / task_completion_before 
            if task_completion_before > 0 else 0
        )
        
        # Verify resilience
        assert functionality_retained >= 0.6, "Too many agents lost"
        assert redistribution_success >= len(failed_agents) * 0.75, "Poor task redistribution"
        assert task_completion_ratio > 0.5, "Severe performance degradation"
        
        return {
            'agents_failed': len(failed_agents),
            'agents_active': len(active_agents),
            'functionality_retained': functionality_retained,
            'task_completion_ratio': task_completion_ratio,
            'redistribution_success': redistribution_success
        }
    
    @staticmethod
    def test_extreme_conditions() -> Dict[str, Any]:
        """Test under extreme environmental conditions"""
        # Create extreme environment
        env_config = EnvironmentConfig(
            num_agents=6,
            wind_speed=20.0,  # m/s (strong wind)
            turbulence=0.5,   # High turbulence
            visibility=50.0,  # meters (low visibility)
            temperature=-10.0,  # Celsius (cold)
            obstacles=50      # Many obstacles
        )
        env = MultiAgentEnvironment(env_config)
        
        # Create robust agents
        agents = []
        for i in range(env_config.num_agents):
            agent = HierarchicalAgent(
                agent_id=i,
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                robust_control=True
            )
            agents.append(agent)
        
        # Run in extreme conditions
        observations = env.reset()
        
        position_errors = []
        control_failures = 0
        collisions = 0
        emergency_landings = 0
        
        for step in range(100):
            actions = {}
            
            for i, agent in enumerate(agents):
                # Get robust control action
                action, control_status = agent.robust_control_action(
                    observations[i],
                    env_config
                )
                
                if control_status == 'failure':
                    control_failures += 1
                
                actions[i] = action
            
            observations, rewards, dones, info = env.step(actions)
            
            # Track performance
            if 'position_errors' in info:
                position_errors.extend(info['position_errors'])
            
            if 'collisions' in info:
                collisions += len(info['collisions'])
            
            if 'emergency_landings' in info:
                emergency_landings += info['emergency_landings']
        
        # Calculate robustness metrics
        avg_position_error = np.mean(position_errors) if position_errors else 0
        control_failure_rate = control_failures / (100 * env_config.num_agents)
        survival_rate = (env_config.num_agents - emergency_landings) / env_config.num_agents
        
        # Verify robustness
        assert avg_position_error < 5.0, "Poor position control in extreme conditions"
        assert collisions < 5, "Too many collisions"
        assert survival_rate > 0.7, "Poor survival rate"
        
        return {
            'avg_position_error': avg_position_error,
            'control_failures': control_failures,
            'control_failure_rate': control_failure_rate,
            'collisions': collisions,
            'emergency_landings': emergency_landings,
            'survival_rate': survival_rate
        }