#!/usr/bin/env python
"""
Run Q1 Publication-Grade Experiments for PI-HMARL
This script orchestrates the complete experimental evaluation meeting top-tier journal standards
"""

import os
import sys
import argparse
import yaml
import warnings
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experimental_framework.q1_experiment_orchestrator import Q1ExperimentOrchestrator, ExperimentConfig
from experimental_framework.q1_statistical_analyzer import Q1StatisticalAnalyzer

def print_banner():
    """Print Q1 experiment banner"""
    print("\n" + "="*80)
    print("🎓 PI-HMARL Q1 PUBLICATION-GRADE EXPERIMENTAL EVALUATION")
    print("="*80)
    print("Meeting standards for: JMLR, Nature Machine Intelligence, IEEE TPAMI")
    print("="*80 + "\n")

def validate_environment():
    """Validate that environment meets Q1 requirements"""
    print("🔍 Validating Experimental Environment...")
    
    issues = []
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required for reproducibility")
    
    # Check required packages
    required_packages = {
        'numpy': '1.19.0',
        'torch': '1.7.0',
        'scipy': '1.5.0',
        'pandas': '1.1.0',
        'matplotlib': '3.3.0',
        'seaborn': '0.11.0'
    }
    
    for package, min_version in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', '0.0.0')
            print(f"  ✓ {package} {version}")
        except ImportError:
            issues.append(f"{package} >= {min_version} required")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️ No GPU available - experiments will be slower")
    except:
        pass
    
    # Check disk space
    import shutil
    free_gb = shutil.disk_usage(".").free / (1024**3)
    if free_gb < 50:
        issues.append(f"Low disk space: {free_gb:.1f}GB (recommend 50GB+ for full experiments)")
    else:
        print(f"  ✓ Disk space: {free_gb:.1f}GB available")
    
    if issues:
        print("\n❌ Environment Issues:")
        for issue in issues:
            print(f"  - {issue}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print("\n✅ Environment validated successfully!\n")

def load_config(config_path: str) -> dict:
    """Load experimental configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_experiment_config(yaml_config: dict, args) -> ExperimentConfig:
    """Create experiment configuration from YAML and command line args"""
    exp_config = yaml_config['experiment']
    
    # Override with command line arguments
    if args.seeds:
        exp_config['num_seeds'] = args.seeds
    if args.episodes:
        exp_config['num_episodes'] = args.episodes
    if args.quick:
        exp_config['num_seeds'] = 3
        exp_config['num_episodes'] = 10
        exp_config['num_steps_per_episode'] = 100
    
    # Select domains
    enabled_domains = [
        domain for domain, cfg in yaml_config['domains'].items() 
        if cfg['enabled']
    ]
    
    # Select baselines
    enabled_baselines = [
        baseline for baseline, cfg in yaml_config['baselines'].items()
        if cfg['enabled']
    ]
    
    config = ExperimentConfig(
        name=exp_config['name'],
        num_seeds=exp_config['num_seeds'],
        num_episodes=exp_config['num_episodes'],
        num_steps=exp_config.get('num_steps_per_episode', 1000),
        domains=enabled_domains,
        baselines=enabled_baselines,
        pi_hmarl_config=yaml_config.get('pi_hmarl', {}),
        save_checkpoints=exp_config.get('save_checkpoints', True),
        checkpoint_interval=exp_config.get('checkpoint_interval', 100),
        parallel_seeds=exp_config.get('parallel_seeds', 5),
        compute_theory=exp_config.get('compute_theory', True),
        profile_computation=exp_config.get('profile_computation', True),
        statistical_tests=exp_config.get('statistical_tests', []),
        output_dir=args.output_dir or exp_config.get('output_dir', 'q1_experiments/')
    )
    
    return config

def run_demo_analysis():
    """Run a quick demonstration of Q1 statistical analysis"""
    print("\n📊 DEMONSTRATION: Q1 Statistical Analysis")
    print("-" * 60)
    
    # Create sample data with realistic effect sizes
    import numpy as np
    
    # Simulate realistic experimental results
    np.random.seed(42)
    
    baseline_success = 0.65 + np.random.normal(0, 0.05, 30)  # 30 seeds
    pi_hmarl_success = 0.78 + np.random.normal(0, 0.04, 30)  # ~20% improvement (realistic)
    
    analyzer = Q1StatisticalAnalyzer(num_seeds=30)
    
    # Run analysis
    print("\n1. Bootstrap Confidence Intervals (50,000 iterations):")
    mean, ci_low, ci_high = analyzer.bootstrap_confidence_interval(pi_hmarl_success)
    print(f"   PI-HMARL: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    
    print("\n2. Statistical Comparison:")
    comparison = analyzer.analyze_comparison(
        baseline_success, 
        pi_hmarl_success,
        "Baseline",
        "PI-HMARL"
    )
    
    print(f"   Test used: {comparison['test_type']}")
    print(f"   p-value: {comparison['p_value']:.4f}")
    print(f"   Significant: {comparison['significant']}")
    
    print("\n3. Effect Size Analysis:")
    effect_size = analyzer.calculate_realistic_effect_size(baseline_success, pi_hmarl_success)
    print(f"   Cohen's d: {effect_size.cohens_d:.3f}")
    print(f"   Interpretation: {effect_size.interpretation}")
    print(f"   Q1 Realistic: {effect_size.is_realistic}")
    
    if not effect_size.is_realistic:
        print(f"\n   ⚠️ WARNING: {effect_size.warning}")
    
    print("\n4. Power Analysis:")
    power = analyzer.power_analysis(effect_size=effect_size.cohens_d)
    print(f"   Required sample size: {power['required_sample_size']}")
    print(f"   Achieved power: {power['achieved_power']:.3f}")
    print(f"   Minimum detectable effect: {power['minimum_detectable_effect']:.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="Run Q1 publication-grade experiments for PI-HMARL"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='src/configs/q1_experimental_config.yaml',
        help='Path to experimental configuration file'
    )
    
    parser.add_argument(
        '--seeds',
        type=int,
        help='Override number of random seeds (min 30 for Q1)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        help='Override number of episodes per run'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with reduced parameters'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run statistical analysis demonstration only'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment without running experiments'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Validate environment
    validate_environment()
    
    if args.validate_only:
        print("✅ Validation complete. Exiting.")
        return
    
    # Run demo if requested
    if args.demo:
        run_demo_analysis()
        return
    
    # Load configuration
    print(f"📄 Loading configuration from: {args.config}")
    yaml_config = load_config(args.config)
    
    # Create experiment configuration
    config = create_experiment_config(yaml_config, args)
    
    # Display configuration
    print("\n📋 Experiment Configuration:")
    print(f"  - Name: {config.name}")
    print(f"  - Seeds: {config.num_seeds}")
    print(f"  - Episodes: {config.num_episodes}")
    print(f"  - Domains: {', '.join(config.domains)}")
    print(f"  - Baselines: {len(config.baselines)} algorithms")
    print(f"  - Output: {config.output_dir}")
    
    if config.num_seeds < 30:
        warnings.warn(
            f"Running with {config.num_seeds} seeds. "
            "Q1 venues require 30+ seeds for publication."
        )
    
    # Confirm before starting
    if not args.quick:
        total_experiments = (
            config.num_seeds * 
            len(config.domains) * 
            (len(config.baselines) + 1) *  # +1 for PI-HMARL
            3  # Approximate scenarios per domain
        )
        
        print(f"\n⚠️ This will run approximately {total_experiments} experiments.")
        print(f"Estimated time: {total_experiments * 2 / 60:.1f} hours")
        
        response = input("\nProceed with experiments? (y/n): ")
        if response.lower() != 'y':
            print("Experiments cancelled.")
            return
    
    # Create experiment orchestrator
    print("\n🚀 Starting Q1 Experimental Evaluation...")
    orchestrator = Q1ExperimentOrchestrator(config)
    
    try:
        # Run complete evaluation
        report = orchestrator.run_complete_evaluation()
        
        # Display summary
        print("\n" + "="*80)
        print("📊 EXPERIMENT SUMMARY")
        print("="*80)
        
        if 'results_summary' in report:
            summary = report['results_summary']['overall_performance']
            print(f"\nPI-HMARL Performance:")
            print(f"  Success Rate: {summary.get('pi_hmarl_mean_success', 0):.1%} ± "
                  f"{summary.get('pi_hmarl_std_success', 0):.1%}")
            
            print(f"\nBest Baseline ({summary.get('best_baseline', 'Unknown')}):")
            print(f"  Success Rate: {summary.get('best_baseline_success', 0):.1%}")
            
            improvement = (summary.get('pi_hmarl_mean_success', 0) - 
                          summary.get('best_baseline_success', 0))
            print(f"\nImprovement: {improvement:.1%}")
            
            if improvement > 0.3:  # 30% improvement
                print("\n⚠️ WARNING: Large improvement may indicate experimental issues.")
                print("Consider reviewing baseline implementations and experimental setup.")
        
        print(f"\n📁 Full results saved to: {config.output_dir}")
        print("\n✅ Q1 Experimental Evaluation Complete!")
        
    except KeyboardInterrupt:
        print("\n\n❌ Experiments interrupted by user.")
        print("Partial results may be available in the output directory.")
    except Exception as e:
        print(f"\n\n❌ Error during experiments: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()