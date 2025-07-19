"""Core Testing Framework

This module implements the fundamental testing framework for
comprehensive validation of the PI-HMARL system.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import logging
import traceback
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestLevel(Enum):
    """Test levels"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    STRESS = "stress"


@dataclass
class TestConfig:
    """Test configuration"""
    # Test parameters
    test_level: TestLevel = TestLevel.UNIT
    timeout: float = 300.0  # seconds
    max_retries: int = 3
    parallel_execution: bool = True
    num_workers: int = 4
    
    # Environment parameters
    num_agents: int = 5
    episode_length: int = 1000
    random_seed: int = 42
    device: str = "cpu"
    
    # Validation parameters
    tolerance: float = 1e-6
    physics_accuracy: float = 0.01  # 1% error tolerance
    timing_accuracy: float = 0.1    # 100ms timing tolerance
    
    # Output parameters
    output_dir: Path = Path("test_results")
    save_logs: bool = True
    save_plots: bool = True
    verbose: bool = True


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    test_level: TestLevel
    status: TestStatus
    start_time: float
    end_time: float
    duration: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class TestCase:
    """Individual test case"""
    name: str
    description: str
    test_func: Callable
    level: TestLevel
    requirements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    skip_condition: Optional[Callable] = None


class TestSuite:
    """Collection of related test cases"""
    
    def __init__(self, name: str, description: str):
        """Initialize test suite
        
        Args:
            name: Suite name
            description: Suite description
        """
        self.name = name
        self.description = description
        self.test_cases: List[TestCase] = []
        self.setup_func: Optional[Callable] = None
        self.teardown_func: Optional[Callable] = None
    
    def add_test(self, test_case: TestCase):
        """Add test case to suite
        
        Args:
            test_case: Test case to add
        """
        self.test_cases.append(test_case)
    
    def set_setup(self, setup_func: Callable):
        """Set suite setup function
        
        Args:
            setup_func: Setup function
        """
        self.setup_func = setup_func
    
    def set_teardown(self, teardown_func: Callable):
        """Set suite teardown function
        
        Args:
            teardown_func: Teardown function
        """
        self.teardown_func = teardown_func


class TestFramework:
    """Main testing framework"""
    
    def __init__(self, config: TestConfig):
        """Initialize test framework
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: List[TestResult] = []
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Execution management
        self.executor = None
        if config.parallel_execution:
            if config.test_level in [TestLevel.UNIT, TestLevel.INTEGRATION]:
                self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
            else:
                self.executor = ProcessPoolExecutor(max_workers=config.num_workers)
        
        logger.info(f"Initialized test framework with config: {config}")
    
    def _setup_logging(self):
        """Setup test logging"""
        log_file = self.config.output_dir / f"test_run_{int(time.time())}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def register_suite(self, suite: TestSuite):
        """Register test suite
        
        Args:
            suite: Test suite to register
        """
        self.test_suites[suite.name] = suite
        logger.info(f"Registered test suite: {suite.name}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all registered tests
        
        Returns:
            Test execution summary
        """
        logger.info("Starting full test run")
        start_time = time.time()
        
        # Run each suite
        for suite_name, suite in self.test_suites.items():
            self._run_suite(suite)
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save results
        if self.config.save_logs:
            self._save_results(summary)
        
        # Generate report
        report = self._generate_report(summary)
        
        total_time = time.time() - start_time
        logger.info(f"Test run completed in {total_time:.2f} seconds")
        
        return report
    
    def run_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run specific test suite
        
        Args:
            suite_name: Name of suite to run
            
        Returns:
            Suite execution summary
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        suite = self.test_suites[suite_name]
        return self._run_suite(suite)
    
    def _run_suite(self, suite: TestSuite) -> Dict[str, Any]:
        """Execute test suite
        
        Args:
            suite: Test suite to run
            
        Returns:
            Suite results
        """
        logger.info(f"Running test suite: {suite.name}")
        suite_start = time.time()
        
        # Run setup
        if suite.setup_func:
            try:
                suite.setup_func()
            except Exception as e:
                logger.error(f"Suite setup failed: {e}")
                return {'status': 'setup_failed', 'error': str(e)}
        
        # Run test cases
        if self.config.parallel_execution and self.executor:
            futures = []
            for test_case in suite.test_cases:
                future = self.executor.submit(self._run_test_case, test_case)
                futures.append((test_case, future))
            
            # Collect results
            for test_case, future in futures:
                try:
                    result = future.result(timeout=test_case.timeout or self.config.timeout)
                    self.test_results.append(result)
                except Exception as e:
                    logger.error(f"Test execution error: {e}")
                    self.test_results.append(self._create_error_result(test_case, e))
        else:
            # Sequential execution
            for test_case in suite.test_cases:
                result = self._run_test_case(test_case)
                self.test_results.append(result)
        
        # Run teardown
        if suite.teardown_func:
            try:
                suite.teardown_func()
            except Exception as e:
                logger.error(f"Suite teardown failed: {e}")
        
        suite_duration = time.time() - suite_start
        
        # Generate suite summary
        suite_results = [r for r in self.test_results if r.test_name.startswith(suite.name)]
        
        summary = {
            'suite_name': suite.name,
            'total_tests': len(suite_results),
            'passed': sum(1 for r in suite_results if r.status == TestStatus.PASSED),
            'failed': sum(1 for r in suite_results if r.status == TestStatus.FAILED),
            'errors': sum(1 for r in suite_results if r.status == TestStatus.ERROR),
            'skipped': sum(1 for r in suite_results if r.status == TestStatus.SKIPPED),
            'duration': suite_duration,
            'results': suite_results
        }
        
        return summary
    
    def _run_test_case(self, test_case: TestCase) -> TestResult:
        """Execute individual test case
        
        Args:
            test_case: Test case to run
            
        Returns:
            Test result
        """
        logger.debug(f"Running test: {test_case.name}")
        
        # Check skip condition
        if test_case.skip_condition and test_case.skip_condition():
            return TestResult(
                test_name=test_case.name,
                test_level=test_case.level,
                status=TestStatus.SKIPPED,
                start_time=time.time(),
                end_time=time.time(),
                duration=0.0
            )
        
        # Initialize result
        result = TestResult(
            test_name=test_case.name,
            test_level=test_case.level,
            status=TestStatus.RUNNING,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0
        )
        
        # Run test with retries
        for attempt in range(self.config.max_retries):
            try:
                # Execute test function
                test_metrics = test_case.test_func()
                
                # Test passed
                result.status = TestStatus.PASSED
                result.end_time = time.time()
                result.duration = result.end_time - result.start_time
                
                if isinstance(test_metrics, dict):
                    result.metrics = test_metrics
                
                logger.info(f"Test {test_case.name} PASSED")
                break
                
            except AssertionError as e:
                # Test failed
                result.status = TestStatus.FAILED
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
                
                logger.warning(f"Test {test_case.name} FAILED: {e}")
                
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying test {test_case.name} (attempt {attempt + 2})")
                    time.sleep(1)  # Brief delay before retry
                else:
                    break
                    
            except Exception as e:
                # Test error
                result.status = TestStatus.ERROR
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
                
                logger.error(f"Test {test_case.name} ERROR: {e}")
                break
        
        result.end_time = time.time()
        result.duration = result.end_time - result.start_time
        
        return result
    
    def _create_error_result(self, test_case: TestCase, error: Exception) -> TestResult:
        """Create error result for test case
        
        Args:
            test_case: Test case
            error: Error that occurred
            
        Returns:
            Error test result
        """
        return TestResult(
            test_name=test_case.name,
            test_level=test_case.level,
            status=TestStatus.ERROR,
            start_time=time.time(),
            end_time=time.time(),
            duration=0.0,
            error_message=str(error),
            stack_trace=traceback.format_exc()
        )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test execution summary
        
        Returns:
            Execution summary
        """
        total_tests = len(self.test_results)
        
        status_counts = defaultdict(int)
        for result in self.test_results:
            status_counts[result.status.value] += 1
        
        level_counts = defaultdict(int)
        for result in self.test_results:
            level_counts[result.test_level.value] += 1
        
        # Calculate metrics
        total_duration = sum(r.duration for r in self.test_results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        summary = {
            'total_tests': total_tests,
            'status_summary': dict(status_counts),
            'level_summary': dict(level_counts),
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'success_rate': status_counts[TestStatus.PASSED.value] / total_tests if total_tests > 0 else 0,
            'timestamp': time.time()
        }
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to file
        
        Args:
            summary: Test summary
        """
        # Save detailed results
        results_file = self.config.output_dir / f"test_results_{int(time.time())}.json"
        
        results_data = {
            'config': {
                'test_level': self.config.test_level.value,
                'num_agents': self.config.num_agents,
                'timeout': self.config.timeout
            },
            'summary': summary,
            'results': [
                {
                    'test_name': r.test_name,
                    'status': r.status.value,
                    'duration': r.duration,
                    'error_message': r.error_message,
                    'metrics': r.metrics
                }
                for r in self.test_results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved test results to {results_file}")
    
    def _generate_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test report with visualizations
        
        Args:
            summary: Test summary
            
        Returns:
            Test report
        """
        report = {
            'summary': summary,
            'failed_tests': [],
            'error_tests': [],
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Collect failed and error tests
        for result in self.test_results:
            if result.status == TestStatus.FAILED:
                report['failed_tests'].append({
                    'name': result.test_name,
                    'error': result.error_message,
                    'metrics': result.metrics
                })
            elif result.status == TestStatus.ERROR:
                report['error_tests'].append({
                    'name': result.test_name,
                    'error': result.error_message,
                    'stack_trace': result.stack_trace
                })
        
        # Generate visualizations if enabled
        if self.config.save_plots:
            self._generate_visualizations(summary)
        
        # Add recommendations
        if summary['success_rate'] < 0.8:
            report['recommendations'].append(
                "Low success rate detected. Review failed tests and consider increasing timeouts."
            )
        
        if report['error_tests']:
            report['recommendations'].append(
                "Test errors detected. Check test environment and dependencies."
            )
        
        return report
    
    def _generate_visualizations(self, summary: Dict[str, Any]):
        """Generate test result visualizations
        
        Args:
            summary: Test summary
        """
        # Status distribution pie chart
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        status_counts = summary['status_summary']
        if status_counts:
            plt.pie(
                status_counts.values(),
                labels=status_counts.keys(),
                autopct='%1.1f%%',
                colors=['green', 'red', 'orange', 'gray', 'yellow']
            )
            plt.title('Test Status Distribution')
        
        # Test duration histogram
        plt.subplot(1, 2, 2)
        durations = [r.duration for r in self.test_results]
        if durations:
            plt.hist(durations, bins=20, edgecolor='black')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Number of Tests')
            plt.title('Test Duration Distribution')
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'test_summary.png')
        plt.close()
        
        # Level-wise results
        fig, ax = plt.subplots(figsize=(8, 6))
        
        level_data = defaultdict(lambda: defaultdict(int))
        for result in self.test_results:
            level_data[result.test_level.value][result.status.value] += 1
        
        if level_data:
            levels = list(level_data.keys())
            statuses = ['passed', 'failed', 'error', 'skipped']
            
            x = np.arange(len(levels))
            width = 0.2
            
            for i, status in enumerate(statuses):
                counts = [level_data[level].get(status, 0) for level in levels]
                ax.bar(x + i * width, counts, width, label=status)
            
            ax.set_xlabel('Test Level')
            ax.set_ylabel('Number of Tests')
            ax.set_title('Test Results by Level')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(levels)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.config.output_dir / 'test_levels.png')
            plt.close()
    
    def cleanup(self):
        """Cleanup test framework resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Test framework cleanup completed")


# Assertion helpers
class TestAssertions:
    """Common test assertions"""
    
    @staticmethod
    def assert_near(actual: float, expected: float, tolerance: float, message: str = ""):
        """Assert values are near each other
        
        Args:
            actual: Actual value
            expected: Expected value
            tolerance: Tolerance for comparison
            message: Optional error message
        """
        diff = abs(actual - expected)
        assert diff <= tolerance, f"{message} Expected {expected}, got {actual} (diff: {diff})"
    
    @staticmethod
    def assert_tensor_near(actual: torch.Tensor, expected: torch.Tensor, tolerance: float = 1e-6):
        """Assert tensors are near each other
        
        Args:
            actual: Actual tensor
            expected: Expected tensor
            tolerance: Tolerance for comparison
        """
        assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
        max_diff = torch.max(torch.abs(actual - expected)).item()
        assert max_diff <= tolerance, f"Tensor difference {max_diff} exceeds tolerance {tolerance}"
    
    @staticmethod
    def assert_in_range(value: float, min_val: float, max_val: float, message: str = ""):
        """Assert value is in range
        
        Args:
            value: Value to check
            min_val: Minimum value
            max_val: Maximum value
            message: Optional error message
        """
        assert min_val <= value <= max_val, f"{message} Value {value} not in range [{min_val}, {max_val}]"
    
    @staticmethod
    def assert_monotonic_increasing(values: List[float], strict: bool = False):
        """Assert values are monotonically increasing
        
        Args:
            values: List of values
            strict: Whether to require strict increase
        """
        for i in range(1, len(values)):
            if strict:
                assert values[i] > values[i-1], f"Not strictly increasing at index {i}"
            else:
                assert values[i] >= values[i-1], f"Not monotonically increasing at index {i}"
    
    @staticmethod
    def assert_convergence(values: List[float], threshold: float = 0.01, window: int = 10):
        """Assert values converge
        
        Args:
            values: List of values
            threshold: Convergence threshold
            window: Window size for checking convergence
        """
        if len(values) < window:
            return
        
        recent_values = values[-window:]
        variance = np.var(recent_values)
        assert variance < threshold, f"Values not converged: variance {variance} > {threshold}"