"""
Advanced Logging System for PI-HMARL

This module provides a comprehensive logging system with file and console output,
structured logging, experiment tracking integration, and performance monitoring.
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
import colorlog
from contextlib import contextmanager
import time
import threading
from functools import wraps


@dataclass
class LogEntry:
    """Structured log entry for experiment tracking."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line: int
    experiment_id: Optional[str] = None
    episode: Optional[int] = None
    step: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class ExperimentTracker:
    """Integration with experiment tracking systems like WandB."""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.wandb_available = False
        self.tensorboard_available = False
        
        # Try to initialize WandB
        try:
            import wandb
            self.wandb = wandb
            self.wandb_available = True
        except ImportError:
            self.wandb = None
        
        # Try to initialize TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(f"logs/tensorboard/{experiment_id}")
            self.tensorboard_available = True
        except ImportError:
            self.tensorboard_writer = None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to available tracking systems."""
        if self.wandb_available and self.wandb.run is not None:
            self.wandb.log(metrics, step=step)
        
        if self.tensorboard_available and self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, step)
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to available tracking systems."""
        if self.wandb_available and self.wandb.run is not None:
            self.wandb.config.update(params)
        
        if self.tensorboard_available and self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(params, {})
    
    def finish(self) -> None:
        """Finish experiment tracking."""
        if self.wandb_available and self.wandb.run is not None:
            self.wandb.finish()
        
        if self.tensorboard_available and self.tensorboard_writer:
            self.tensorboard_writer.close()


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        with self.lock:
            self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        with self.lock:
            if name in self.timers:
                elapsed = time.time() - self.timers[name]
                del self.timers[name]
                return elapsed
            return 0.0
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            return {
                "active_timers": list(self.timers.keys()),
                "counters": self.counters.copy()
            }
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.logger.info(f"Timer '{name}': {elapsed:.4f}s")


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_metrics: bool = True):
        super().__init__()
        self.include_metrics = include_metrics
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line=record.lineno,
            experiment_id=getattr(record, 'experiment_id', None),
            episode=getattr(record, 'episode', None),
            step=getattr(record, 'step', None),
            metrics=getattr(record, 'metrics', None) if self.include_metrics else None,
            tags=getattr(record, 'tags', None)
        )
        
        return json.dumps(asdict(log_entry), default=str)


class PIHMARLLogger:
    """
    Advanced logging system for PI-HMARL experiments.
    
    Features:
    - Structured logging with JSON output
    - Colored console output
    - File rotation
    - Experiment tracking integration
    - Performance monitoring
    - Context-aware logging
    """
    
    def __init__(
        self,
        name: str = "pi_hmarl",
        log_level: str = "INFO",
        log_dir: Union[str, Path] = "logs",
        experiment_id: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True,
        structured_output: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the PI-HMARL logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_dir: Directory for log files
            experiment_id: Unique experiment identifier
            console_output: Enable console output
            file_output: Enable file output
            structured_output: Enable structured JSON logging
            max_bytes: Maximum bytes per log file
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.console_output = console_output
        self.file_output = file_output
        self.structured_output = structured_output
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Initialize components
        self.experiment_tracker = ExperimentTracker(self.experiment_id)
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handlers(max_bytes, backup_count)
        
        # Log initialization
        self.logger.info(f"PI-HMARL Logger initialized - Experiment ID: {self.experiment_id}")
        
        # Context variables
        self._context: Dict[str, Any] = {}
        self._context_lock = threading.Lock()
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _setup_console_handler(self) -> None:
        """Setup colored console handler."""
        if not self.console_output:
            return
        
        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Colored formatter
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        console_handler.setFormatter(color_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, max_bytes: int, backup_count: int) -> None:
        """Setup file handlers for different log levels."""
        if not self.file_output:
            return
        
        # General log file
        general_file = self.log_dir / f"{self.experiment_id}.log"
        general_handler = logging.handlers.RotatingFileHandler(
            general_file, maxBytes=max_bytes, backupCount=backup_count
        )
        general_handler.setLevel(self.log_level)
        general_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        general_handler.setFormatter(general_formatter)
        self.logger.addHandler(general_handler)
        
        # Structured log file (JSON)
        if self.structured_output:
            structured_file = self.log_dir / f"{self.experiment_id}_structured.log"
            structured_handler = logging.handlers.RotatingFileHandler(
                structured_file, maxBytes=max_bytes, backupCount=backup_count
            )
            structured_handler.setLevel(self.log_level)
            structured_formatter = StructuredFormatter()
            structured_handler.setFormatter(structured_formatter)
            self.logger.addHandler(structured_handler)
        
        # Error log file
        error_file = self.log_dir / f"{self.experiment_id}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=max_bytes, backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
    
    def set_context(self, **kwargs) -> None:
        """Set logging context variables."""
        with self._context_lock:
            self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context variables."""
        with self._context_lock:
            self._context.clear()
    
    def _add_context_to_record(self, record: logging.LogRecord) -> None:
        """Add context variables to log record."""
        with self._context_lock:
            for key, value in self._context.items():
                setattr(record, key, value)
    
    def log_with_context(self, level: str, message: str, **kwargs) -> None:
        """Log message with additional context."""
        record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level.upper()),
            __file__, 0, message, (), None
        )
        
        # Add context
        self._add_context_to_record(record)
        
        # Add additional kwargs
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        self.logger.handle(record)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, **kwargs) -> None:
        """Log metrics with experiment tracking."""
        # Log to experiment tracker
        self.experiment_tracker.log_metrics(metrics, step)
        
        # Log to file
        self.log_with_context("INFO", f"Metrics: {metrics}", metrics=metrics, step=step, **kwargs)
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.experiment_tracker.log_hyperparameters(params)
        self.log_with_context("INFO", f"Hyperparameters: {params}", hyperparameters=params)
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log episode information."""
        self.set_context(episode=episode)
        self.log_metrics(metrics, step=episode)
        self.info(f"Episode {episode} completed - {metrics}")
    
    def log_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log step information."""
        self.set_context(step=step)
        self.log_metrics(metrics, step=step)
    
    def log_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log exception with context."""
        if context:
            self.set_context(**context)
        
        self.logger.exception(f"Exception occurred: {exception}")
        
        if context:
            self.clear_context()
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log_with_context("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log_with_context("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log_with_context("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log_with_context("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log_with_context("CRITICAL", message, **kwargs)
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        with self.performance_monitor.timer(name):
            yield
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.performance_monitor.start_timer(name)
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        elapsed = self.performance_monitor.stop_timer(name)
        self.info(f"Timer '{name}': {elapsed:.4f}s")
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        self.performance_monitor.increment_counter(name, value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_monitor.get_metrics()
    
    def finish(self) -> None:
        """Finish logging and cleanup."""
        self.info("Finishing experiment logging")
        self.experiment_tracker.finish()
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Global logger instance
_logger: Optional[PIHMARLLogger] = None


def get_logger(
    name: str = "pi_hmarl",
    log_level: str = "INFO",
    experiment_id: Optional[str] = None,
    **kwargs
) -> PIHMARLLogger:
    """
    Get or create the global logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level
        experiment_id: Experiment ID
        **kwargs: Additional arguments for logger initialization
    
    Returns:
        PIHMARLLogger instance
    """
    global _logger
    if _logger is None:
        _logger = PIHMARLLogger(
            name=name,
            log_level=log_level,
            experiment_id=experiment_id,
            **kwargs
        )
    return _logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: Union[str, Path] = "logs",
    experiment_id: Optional[str] = None,
    **kwargs
) -> PIHMARLLogger:
    """
    Setup logging for PI-HMARL experiments.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        experiment_id: Experiment ID
        **kwargs: Additional arguments for logger initialization
    
    Returns:
        PIHMARLLogger instance
    """
    global _logger
    _logger = PIHMARLLogger(
        log_level=log_level,
        log_dir=log_dir,
        experiment_id=experiment_id,
        **kwargs
    )
    return _logger


def timed(logger: Optional[PIHMARLLogger] = None):
    """Decorator for timing function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger()
            func_name = f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log.info(f"Function '{func_name}' completed in {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                log.error(f"Function '{func_name}' failed after {elapsed:.4f}s: {e}")
                raise
        return wrapper
    return decorator