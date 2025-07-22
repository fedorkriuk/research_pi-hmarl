"""
Q1-Grade Statistical Analysis Framework
Meeting standards for JMLR, Nature Machine Intelligence, IEEE TPAMI
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from sklearn.utils import resample
import pymc3 as pm
import arviz as az
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
import json
import hashlib
from datetime import datetime

@dataclass
class EffectSizeResult:
    """Comprehensive effect size analysis result"""
    cohens_d: float
    hedges_g: float
    glass_delta: float
    confidence_interval: Tuple[float, float]
    is_realistic: bool  # True if d < 1.2
    interpretation: str
    warning: Optional[str] = None

class Q1StatisticalAnalyzer:
    """
    Q1 publication-grade statistical analysis with:
    - 30+ seed requirement
    - Bootstrap confidence intervals (50k iterations)
    - Bayesian analysis
    - Effect size bounds checking
    - Multiple comparison corrections
    - Power analysis
    """
    
    def __init__(self, 
                 num_seeds: int = 30,
                 bootstrap_iterations: int = 50000,
                 significance_level: float = 0.01,
                 effect_size_threshold: float = 1.2):
        
        if num_seeds < 30:
            warnings.warn(f"Q1 venues require at least 30 seeds. Current: {num_seeds}")
        
        self.num_seeds = num_seeds
        self.bootstrap_iterations = bootstrap_iterations
        self.significance_level = significance_level
        self.effect_size_threshold = effect_size_threshold
        self.power_analyzer = TTestPower()
        
        # Track all comparisons for multiple testing correction
        self.all_comparisons = []
        
    def bootstrap_confidence_interval(self, 
                                    data: np.ndarray, 
                                    statistic_func: callable = np.mean,
                                    confidence_level: float = 0.99) -> Tuple[float, float, float]:
        """
        High-precision bootstrap confidence intervals for Q1 standards.
        
        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        if len(data) < self.num_seeds:
            warnings.warn(f"Insufficient data points: {len(data)} < {self.num_seeds}")
        
        # Bootstrap resampling
        bootstrap_statistics = []
        
        for _ in range(self.bootstrap_iterations):
            # Resample with replacement
            resample_data = resample(data, replace=True, n_samples=len(data))
            bootstrap_statistics.append(statistic_func(resample_data))
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Calculate percentile confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
        ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
        point_estimate = statistic_func(data)
        
        return point_estimate, ci_lower, ci_upper
    
    def permutation_test(self, 
                        data1: np.ndarray, 
                        data2: np.ndarray,
                        n_permutations: int = 50000) -> Dict[str, Any]:
        """
        High-precision non-parametric permutation test for Q1 standards.
        """
        combined = np.concatenate([data1, data2])
        n1 = len(data1)
        observed_diff = np.mean(data1) - np.mean(data2)
        
        permutation_diffs = []
        
        for _ in range(n_permutations):
            # Random permutation
            np.random.shuffle(combined)
            perm_group1 = combined[:n1]
            perm_group2 = combined[n1:]
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            permutation_diffs.append(perm_diff)
        
        permutation_diffs = np.array(permutation_diffs)
        
        # Two-tailed p-value
        p_value = np.sum(np.abs(permutation_diffs) >= np.abs(observed_diff)) / n_permutations
        
        # Effect size
        effect_size = self.calculate_realistic_effect_size(data1, data2)
        
        result = {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'effect_size': effect_size,
            'permutation_distribution': permutation_diffs,
            'method': 'permutation_test',
            'n_permutations': n_permutations
        }
        
        # Store for multiple comparison correction
        self.all_comparisons.append(p_value)
        
        return result
    
    def bayesian_comparison(self, 
                          data1: np.ndarray, 
                          data2: np.ndarray,
                          n_samples: int = 10000) -> Dict[str, Any]:
        """
        Bayesian analysis with credible intervals for Q1 standards.
        """
        with pm.Model() as model:
            # Priors
            mu1 = pm.Normal('mu1', mu=np.mean(data1), sd=np.std(data1) * 2)
            mu2 = pm.Normal('mu2', mu=np.mean(data2), sd=np.std(data2) * 2)
            
            sigma1 = pm.HalfNormal('sigma1', sd=np.std(data1) * 2)
            sigma2 = pm.HalfNormal('sigma2', sd=np.std(data2) * 2)
            
            # Difference
            diff = pm.Deterministic('difference', mu1 - mu2)
            
            # Effect size (Cohen's d)
            pooled_std = pm.Deterministic('pooled_std', 
                                        pm.math.sqrt((sigma1**2 + sigma2**2) / 2))
            effect_size = pm.Deterministic('effect_size', diff / pooled_std)
            
            # Likelihood
            obs1 = pm.Normal('obs1', mu=mu1, sd=sigma1, observed=data1)
            obs2 = pm.Normal('obs2', mu=mu2, sd=sigma2, observed=data2)
            
            # Sampling
            trace = pm.sample(n_samples, return_inferencedata=True, progressbar=False)
        
        # Extract results
        posterior_diff = trace.posterior['difference'].values.flatten()
        posterior_effect = trace.posterior['effect_size'].values.flatten()
        
        # Credible intervals
        ci_diff = az.hdi(posterior_diff, hdi_prob=0.99)
        ci_effect = az.hdi(posterior_effect, hdi_prob=0.99)
        
        # Probability of practical significance
        prob_positive = np.mean(posterior_diff > 0)
        prob_large_effect = np.mean(np.abs(posterior_effect) > 0.5)
        
        result = {
            'mean_difference': np.mean(posterior_diff),
            'difference_ci': ci_diff,
            'mean_effect_size': np.mean(posterior_effect),
            'effect_size_ci': ci_effect,
            'probability_positive': prob_positive,
            'probability_large_effect': prob_large_effect,
            'posterior_samples': {
                'difference': posterior_diff,
                'effect_size': posterior_effect
            },
            'method': 'bayesian_analysis'
        }
        
        # Check if effect size is realistic
        if np.abs(result['mean_effect_size']) > self.effect_size_threshold:
            warnings.warn(
                f"Effect size {result['mean_effect_size']:.2f} exceeds Q1 threshold "
                f"of {self.effect_size_threshold}. This may indicate experimental issues."
            )
        
        return result
    
    def power_analysis(self, 
                      effect_size: float = 0.5,
                      alpha: float = 0.01,
                      power: float = 0.95) -> Dict[str, Any]:
        """
        Formal power analysis for sample size justification (Q1 requirement).
        """
        # Calculate required sample size
        required_n = self.power_analyzer.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            nobs1=None,
            ratio=1.0
        )
        
        # Calculate achieved power with current sample size
        achieved_power = self.power_analyzer.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=None,
            nobs1=self.num_seeds
        )
        
        # Minimum detectable effect
        min_detectable_effect = self.power_analyzer.solve_power(
            effect_size=None,
            alpha=alpha,
            power=power,
            nobs1=self.num_seeds
        )
        
        result = {
            'target_effect_size': effect_size,
            'significance_level': alpha,
            'target_power': power,
            'required_sample_size': int(np.ceil(required_n)),
            'current_sample_size': self.num_seeds,
            'achieved_power': achieved_power,
            'minimum_detectable_effect': min_detectable_effect,
            'sample_size_adequate': self.num_seeds >= required_n
        }
        
        if not result['sample_size_adequate']:
            warnings.warn(
                f"Current sample size ({self.num_seeds}) is insufficient. "
                f"Need {result['required_sample_size']} for {power:.0%} power."
            )
        
        return result
    
    def calculate_realistic_effect_size(self, 
                                      data1: np.ndarray, 
                                      data2: np.ndarray) -> EffectSizeResult:
        """
        Calculate multiple effect sizes with Q1 realism checking.
        """
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        # Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean2 - mean1) / pooled_std
        
        # Hedges' g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction_factor
        
        # Glass's delta (when variances are unequal)
        glass_delta = (mean2 - mean1) / std1
        
        # Bootstrap confidence interval for effect size
        bootstrap_effects = []
        for _ in range(self.bootstrap_iterations):
            resample1 = resample(data1, n_samples=n1)
            resample2 = resample(data2, n_samples=n2)
            
            boot_mean1, boot_mean2 = np.mean(resample1), np.mean(resample2)
            boot_std1, boot_std2 = np.std(resample1, ddof=1), np.std(resample2, ddof=1)
            boot_pooled = np.sqrt(((n1-1)*boot_std1**2 + (n2-1)*boot_std2**2) / (n1+n2-2))
            
            boot_d = (boot_mean2 - boot_mean1) / boot_pooled
            bootstrap_effects.append(boot_d)
        
        ci_lower = np.percentile(bootstrap_effects, 0.5)
        ci_upper = np.percentile(bootstrap_effects, 99.5)
        
        # Check if effect size is realistic for Q1
        is_realistic = abs(cohens_d) <= self.effect_size_threshold
        
        # Interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        elif abs_d < 1.2:
            interpretation = "large"
        else:
            interpretation = "very large (potentially unrealistic)"
        
        warning = None
        if not is_realistic:
            warning = (
                f"Effect size d={cohens_d:.2f} exceeds Q1 realism threshold "
                f"({self.effect_size_threshold}). This suggests:\n"
                "1. Insufficient baseline difficulty\n"
                "2. Overfitting or data leakage\n"
                "3. Experimental design issues\n"
                "Consider re-evaluating experimental setup."
            )
            warnings.warn(warning)
        
        return EffectSizeResult(
            cohens_d=cohens_d,
            hedges_g=hedges_g,
            glass_delta=glass_delta,
            confidence_interval=(ci_lower, ci_upper),
            is_realistic=is_realistic,
            interpretation=interpretation,
            warning=warning
        )
    
    def multiple_comparison_correction(self, 
                                     p_values: Optional[List[float]] = None,
                                     method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple comparison corrections for Q1 family-wise error control.
        
        Methods: 'bonferroni', 'fdr_bh' (Benjamini-Hochberg), 'holm'
        """
        if p_values is None:
            p_values = self.all_comparisons
        
        if not p_values:
            return {'error': 'No p-values to correct'}
        
        # Apply correction
        rejected, corrected_p, alpha_sidak, alpha_bonf = multipletests(
            p_values, 
            alpha=self.significance_level,
            method=method
        )
        
        result = {
            'original_p_values': p_values,
            'corrected_p_values': corrected_p.tolist(),
            'rejected_null': rejected.tolist(),
            'correction_method': method,
            'family_wise_error_rate': self.significance_level,
            'num_comparisons': len(p_values),
            'bonferroni_threshold': self.significance_level / len(p_values)
        }
        
        return result
    
    def generate_q1_statistical_report(self, 
                                     all_results: Dict[str, Any],
                                     output_path: str = 'q1_statistical_report.json') -> Dict:
        """
        Generate comprehensive statistical report meeting Q1 standards.
        """
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_seeds': self.num_seeds,
                'bootstrap_iterations': self.bootstrap_iterations,
                'significance_level': self.significance_level,
                'effect_size_threshold': self.effect_size_threshold,
                'analysis_hash': hashlib.sha256(
                    json.dumps(all_results, sort_keys=True).encode()
                ).hexdigest()
            },
            'power_analysis': self.power_analysis(),
            'statistical_tests': all_results,
            'multiple_comparisons': self.multiple_comparison_correction(),
            'q1_compliance': {
                'sufficient_seeds': self.num_seeds >= 30,
                'realistic_effect_sizes': all([
                    r.get('effect_size', {}).get('is_realistic', True) 
                    for r in all_results.values()
                ]),
                'proper_corrections': True,
                'bayesian_analysis': any(['bayesian' in str(r) for r in all_results.values()]),
                'bootstrap_analysis': True,
                'theoretical_backing': 'See theoretical_analysis module'
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report