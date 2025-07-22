import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class StatisticalAnalyzer:
    """
    Statistical analysis tools for PI-HMARL experimental results.
    Provides significance testing, effect sizes, and publication-quality plots.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def analyze_comparison(self, 
                          method1_results: List[float], 
                          method2_results: List[float],
                          method1_name: str = "Method 1",
                          method2_name: str = "Method 2") -> Dict:
        """
        Comprehensive statistical comparison between two methods.
        
        Returns:
            Dictionary containing all statistical metrics for paper reporting
        """
        # Convert to numpy arrays
        data1 = np.array(method1_results)
        data2 = np.array(method2_results)
        
        # Basic statistics
        stats_dict = {
            'method1_name': method1_name,
            'method2_name': method2_name,
            'method1_mean': np.mean(data1),
            'method1_std': np.std(data1, ddof=1),
            'method2_mean': np.mean(data2),
            'method2_std': np.std(data2, ddof=1),
            'method1_n': len(data1),
            'method2_n': len(data2)
        }
        
        # Confidence intervals
        stats_dict['method1_ci'] = self._confidence_interval(data1)
        stats_dict['method2_ci'] = self._confidence_interval(data2)
        
        # Normality test (important for choosing appropriate test)
        _, p_norm1 = stats.shapiro(data1)
        _, p_norm2 = stats.shapiro(data2)
        stats_dict['normally_distributed'] = p_norm1 > 0.05 and p_norm2 > 0.05
        
        # Statistical tests
        if stats_dict['normally_distributed'] and len(data1) >= 30:
            # Use t-test for normal distributions
            t_stat, p_value = stats.ttest_ind(data1, data2)
            stats_dict['test_type'] = 't-test'
            stats_dict['test_statistic'] = t_stat
        else:
            # Use Mann-Whitney U test for non-normal or small samples
            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            stats_dict['test_type'] = 'Mann-Whitney U'
            stats_dict['test_statistic'] = u_stat
            
        stats_dict['p_value'] = p_value
        stats_dict['significant'] = p_value < self.alpha
        
        # Effect size (Cohen's d)
        stats_dict['cohens_d'] = self._cohens_d(data1, data2)
        stats_dict['effect_size_interpretation'] = self._interpret_cohens_d(stats_dict['cohens_d'])
        
        # Relative improvement
        stats_dict['relative_improvement'] = ((stats_dict['method2_mean'] - stats_dict['method1_mean']) 
                                            / stats_dict['method1_mean'] * 100)
        
        return stats_dict
    
    def _confidence_interval(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval using t-distribution."""
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        margin = sem * stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        return (mean - margin, mean + margin)
    
    def _cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(data1), len(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        return (np.mean(data2) - np.mean(data1)) / np.sqrt(pooled_var)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    def create_comparison_table(self, comparisons: List[Dict]) -> pd.DataFrame:
        """
        Create publication-ready comparison table.
        
        Args:
            comparisons: List of results from analyze_comparison
        """
        table_data = []
        for comp in comparisons:
            row = {
                'Method': comp['method2_name'],
                'Mean ± Std': f"{comp['method2_mean']:.3f} ± {comp['method2_std']:.3f}",
                'CI (95%)': f"[{comp['method2_ci'][0]:.3f}, {comp['method2_ci'][1]:.3f}]",
                'p-value': f"{comp['p_value']:.4f}" + ("*" if comp['significant'] else ""),
                'Cohen\'s d': f"{comp['cohens_d']:.3f}",
                'Effect Size': comp['effect_size_interpretation'],
                'Improvement': f"{comp['relative_improvement']:+.1f}%"
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def plot_comparison_with_significance(self, 
                                        results_dict: Dict[str, List[float]],
                                        metric_name: str = "Performance",
                                        save_path: Optional[str] = None):
        """
        Create publication-quality plot with error bars and significance indicators.
        """
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        methods = list(results_dict.keys())
        means = [np.mean(results_dict[m]) for m in methods]
        stds = [np.std(results_dict[m], ddof=1) for m in methods]
        cis = [self._confidence_interval(np.array(results_dict[m])) for m in methods]
        
        # Create bar plot with error bars
        x = np.arange(len(methods))
        bars = plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        
        # Add confidence interval lines
        for i, (ci_low, ci_high) in enumerate(cis):
            plt.plot([i-0.2, i+0.2], [ci_low, ci_low], 'k-', linewidth=2)
            plt.plot([i-0.2, i+0.2], [ci_high, ci_high], 'k-', linewidth=2)
            plt.plot([i, i], [ci_low, ci_high], 'k-', linewidth=2)
        
        # Statistical significance indicators
        baseline_name = methods[0]
        baseline_data = results_dict[baseline_name]
        y_max = max(means) + max(stds) * 1.5
        
        for i, method in enumerate(methods[1:], 1):
            comp = self.analyze_comparison(baseline_data, results_dict[method], 
                                         baseline_name, method)
            if comp['significant']:
                # Add significance star
                plt.text(i, y_max * 1.05, '*', ha='center', va='bottom', fontsize=16)
                # Add p-value
                plt.text(i, y_max * 1.08, f"p={comp['p_value']:.3f}", 
                        ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.xlabel('Method', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} Comparison with Statistical Significance', fontsize=14)
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage:
def demonstrate_statistical_analysis():
    """Show how to use the statistical analyzer with your existing results."""
    
    # Your existing results
    results = {
        'baseline': [0.65, 0.68, 0.63, 0.66, 0.64, 0.67, 0.65, 0.66, 0.64, 0.65],
        'pi_hmarl': [0.87, 0.91, 0.88, 0.89, 0.86, 0.90, 0.88, 0.87, 0.89, 0.88],
        'pi_hmarl_no_physics': [0.78, 0.81, 0.79, 0.80, 0.77, 0.79, 0.78, 0.80, 0.79, 0.78],
        'pi_hmarl_no_hierarchy': [0.75, 0.77, 0.76, 0.74, 0.76, 0.75, 0.77, 0.75, 0.76, 0.75]
    }
    
    analyzer = StatisticalAnalyzer()
    
    # Compare all methods against baseline
    comparisons = []
    for method in ['pi_hmarl', 'pi_hmarl_no_physics', 'pi_hmarl_no_hierarchy']:
        comp = analyzer.analyze_comparison(
            results['baseline'], 
            results[method],
            'Baseline',
            method.replace('_', ' ').title()
        )
        comparisons.append(comp)
    
    # Create table for paper
    table = analyzer.create_comparison_table(comparisons)
    print("\nTable 1: Statistical Comparison of Methods")
    print(table.to_string(index=False))
    
    # Create figure for paper
    analyzer.plot_comparison_with_significance(results, "Success Rate", "statistical_comparison_plot.png")
    
    # Print detailed analysis for one comparison
    print("\n\nDetailed Statistical Analysis: Baseline vs PI-HMARL")
    comp = comparisons[0]
    print(f"Baseline: {comp['method1_mean']:.3f} ± {comp['method1_std']:.3f}")
    print(f"PI-HMARL: {comp['method2_mean']:.3f} ± {comp['method2_std']:.3f}")
    print(f"Statistical Test: {comp['test_type']}")
    print(f"p-value: {comp['p_value']:.4f} {'(significant)' if comp['significant'] else '(not significant)'}")
    print(f"Cohen's d: {comp['cohens_d']:.3f} ({comp['effect_size_interpretation']} effect)")
    print(f"Relative Improvement: {comp['relative_improvement']:+.1f}%")

if __name__ == "__main__":
    print("=" * 80)
    print("1. STATISTICAL ANALYSIS DEMONSTRATION")
    print("=" * 80)
    demonstrate_statistical_analysis()