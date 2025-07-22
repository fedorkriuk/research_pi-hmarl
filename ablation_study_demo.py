import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import numpy as np
import pandas as pd

@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    name: str
    physics_weight: float = 1.0
    use_hierarchy: bool = True
    use_attention: bool = True
    description: str = ""
    
class AblationStudyFramework:
    """
    Minimal-effort ablation study framework that wraps existing training code.
    """
    
    def __init__(self, base_agent_class):
        """
        Args:
            base_agent_class: Your existing PIHMARLAgent class
        """
        self.agent_class = base_agent_class
        self.results = {}
        
    def get_ablation_configurations(self) -> List[AblationConfig]:
        """Define all ablation configurations to test."""
        return [
            AblationConfig(
                name="full_model",
                physics_weight=1.0,
                use_hierarchy=True,
                use_attention=True,
                description="Full PI-HMARL model with all components"
            ),
            AblationConfig(
                name="no_physics",
                physics_weight=0.0,
                use_hierarchy=True,
                use_attention=True,
                description="Ablation: Physics constraints disabled"
            ),
            AblationConfig(
                name="no_hierarchy",
                physics_weight=1.0,
                use_hierarchy=False,
                use_attention=True,
                description="Ablation: Flat architecture (no hierarchy)"
            ),
            AblationConfig(
                name="no_attention",
                physics_weight=1.0,
                use_hierarchy=True,
                use_attention=False,
                description="Ablation: Simple averaging instead of attention"
            ),
            AblationConfig(
                name="baseline",
                physics_weight=0.0,
                use_hierarchy=False,
                use_attention=False,
                description="Baseline: All components disabled"
            )
        ]
    
    def run_ablation_study(self, 
                          train_episodes: int = 1000,
                          eval_episodes: int = 100,
                          num_seeds: int = 5,
                          save_results: bool = True):
        """
        Run complete ablation study with multiple random seeds.
        """
        configs = self.get_ablation_configurations()
        
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Running ablation: {config.name}")
            print(f"Description: {config.description}")
            print(f"Config: physics={config.physics_weight}, "
                  f"hierarchy={config.use_hierarchy}, "
                  f"attention={config.use_attention}")
            print(f"{'='*60}")
            
            config_results = {
                'config': asdict(config),
                'runs': []
            }
            
            for seed in range(num_seeds):
                print(f"\nRun {seed + 1}/{num_seeds} (seed={seed})")
                
                # Set random seed for reproducibility
                np.random.seed(seed)
                
                # Create agent with ablation config
                agent = self.agent_class(
                    physics_weight=config.physics_weight,
                    use_hierarchy=config.use_hierarchy,
                    use_attention=config.use_attention
                )
                
                # Time the training
                start_time = time.time()
                train_results = agent.train(episodes=train_episodes)
                train_time = time.time() - start_time
                
                # Evaluate
                eval_results = agent.evaluate(test_episodes=eval_episodes)
                
                # Store results
                run_results = {
                    'seed': seed,
                    'train_time': train_time,
                    'train_results': train_results,
                    'eval_results': eval_results
                }
                config_results['runs'].append(run_results)
                
                # Print progress
                if 'success_rate' in eval_results:
                    print(f"  Success rate: {eval_results['success_rate']:.3f}")
            
            self.results[config.name] = config_results
            
            if save_results:
                self.save_results(f"ablation_results_{config.name}.json")
        
        # Generate comparison report
        self.generate_ablation_report()
        
    def save_results(self, filename: str):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filename: str):
        """Load results from JSON file."""
        with open(filename, 'r') as f:
            self.results = json.load(f)
    
    def generate_ablation_report(self):
        """Generate comprehensive ablation study report."""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS SUMMARY")
        print("="*80)
        
        # Collect metrics for each configuration
        summary_data = []
        
        for config_name, config_results in self.results.items():
            # Aggregate metrics across seeds
            success_rates = []
            physics_violations = []
            energy_efficiency = []
            decision_times = []
            
            for run in config_results['runs']:
                eval = run['eval_results']
                success_rates.append(eval.get('success_rate', 0))
                physics_violations.append(eval.get('physics_violations', 0))
                energy_efficiency.append(eval.get('energy_efficiency', 0))
                decision_times.append(eval.get('avg_decision_time', 0))
            
            # Get baseline performance for relative calculation
            baseline_success = np.mean([run['eval_results']['success_rate'] 
                                      for run in self.results.get('baseline', {'runs': [{'eval_results': {'success_rate': 0.65}}]})['runs']])
            
            summary = {
                'Configuration': config_name,
                'Success Rate': f"{np.mean(success_rates):.3f} ± {np.std(success_rates):.3f}",
                'Physics Violations': f"{np.mean(physics_violations):.1f} ± {np.std(physics_violations):.1f}",
                'Energy Efficiency': f"{np.mean(energy_efficiency):.3f} ± {np.std(energy_efficiency):.3f}",
                'Decision Time (ms)': f"{np.mean(decision_times)*1000:.1f} ± {np.std(decision_times)*1000:.1f}",
                'Relative Performance': f"{(np.mean(success_rates) / baseline_success - 1) * 100:+.1f}%"
            }
            summary_data.append(summary)
        
        # Create DataFrame for nice printing
        df = pd.DataFrame(summary_data)
        print("\nTable 2: Ablation Study Results")
        print(df.to_string(index=False))
        
        # Component contribution analysis
        print("\n\nCOMPONENT CONTRIBUTION ANALYSIS:")
        print("-" * 40)
        
        full_perf = np.mean([run['eval_results']['success_rate'] 
                            for run in self.results['full_model']['runs']])
        no_physics_perf = np.mean([run['eval_results']['success_rate'] 
                                   for run in self.results['no_physics']['runs']])
        no_hierarchy_perf = np.mean([run['eval_results']['success_rate'] 
                                    for run in self.results['no_hierarchy']['runs']])
        no_attention_perf = np.mean([run['eval_results']['success_rate'] 
                                    for run in self.results['no_attention']['runs']])
        
        physics_contribution = ((full_perf - no_physics_perf) / full_perf) * 100
        hierarchy_contribution = ((full_perf - no_hierarchy_perf) / full_perf) * 100
        attention_contribution = ((full_perf - no_attention_perf) / full_perf) * 100
        
        print(f"Physics constraints contribution: {physics_contribution:.1f}%")
        print(f"Hierarchical architecture contribution: {hierarchy_contribution:.1f}%")
        print(f"Attention mechanism contribution: {attention_contribution:.1f}%")

# Modified wrapper for your existing agent
class AblationPIHMARLAgent:
    """Wrapper that adds ablation capabilities to existing agent."""
    
    def __init__(self, physics_weight=1.0, use_hierarchy=True, use_attention=True):
        # Import your existing agent
        # from your_module import PIHMARLAgent
        # self.base_agent = PIHMARLAgent(...)
        
        self.physics_weight = physics_weight
        self.use_hierarchy = use_hierarchy
        self.use_attention = use_attention
        
    def train(self, episodes=1000):
        """Modified training with ablation settings."""
        # Simulate training with modifications
        # In reality, modify your existing training loop
        
        # Example of how to modify existing code:
        results = []
        for ep in range(episodes):
            # Your existing training code, but with modifications:
            
            # 1. Disable physics by setting weight to 0
            physics_loss = self._compute_physics_loss() * self.physics_weight
            
            # 2. Disable hierarchy by using only one level
            if self.use_hierarchy:
                action = self._hierarchical_decision()
            else:
                action = self._flat_decision()
            
            # 3. Disable attention by using simple averaging
            if self.use_attention:
                coordinated_action = self._attention_coordination(action)
            else:
                coordinated_action = self._simple_average_coordination(action)
            
            # Simulate some results
            success = np.random.random() > (0.3 + 0.2 * self.physics_weight + 
                                           0.1 * self.use_hierarchy + 
                                           0.1 * self.use_attention)
            results.append(success)
        
        return {'success_rates': results}
    
    def evaluate(self, test_episodes=100):
        """Evaluation with metrics tracking."""
        metrics = {
            'success_rate': 0.0,
            'physics_violations': 0,
            'energy_efficiency': 0.0,
            'avg_decision_time': 0.0
        }
        
        # Simulate evaluation based on configuration
        base_success = 0.65
        if self.physics_weight > 0:
            base_success += 0.15
            metrics['physics_violations'] = np.random.randint(0, 3)
        else:
            metrics['physics_violations'] = np.random.randint(10, 20)
            
        if self.use_hierarchy:
            base_success += 0.08
            metrics['avg_decision_time'] = 0.05
        else:
            metrics['avg_decision_time'] = 0.03
            
        if self.use_attention:
            base_success += 0.05
            
        metrics['success_rate'] = min(base_success + np.random.normal(0, 0.02), 1.0)
        metrics['energy_efficiency'] = 0.7 + 0.1 * self.physics_weight
        
        return metrics
    
    def _compute_physics_loss(self):
        return np.random.random() * 0.1
    
    def _hierarchical_decision(self):
        return np.random.randn(10)
    
    def _flat_decision(self):
        return np.random.randn(10)
    
    def _attention_coordination(self, action):
        return action * 1.1
    
    def _simple_average_coordination(self, action):
        return action * 0.9

# Example usage
def run_ablation_example():
    """Demonstrate ablation study framework."""
    
    # Create ablation framework with your agent
    ablation = AblationStudyFramework(AblationPIHMARLAgent)
    
    # Run complete ablation study
    ablation.run_ablation_study(
        train_episodes=100,  # Reduced for demo
        eval_episodes=50,
        num_seeds=3
    )

if __name__ == "__main__":
    print("=" * 80)
    print("2. ABLATION STUDY DEMONSTRATION")
    print("=" * 80)
    run_ablation_example()