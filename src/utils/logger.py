"""Logging utilities for PI-HMARL Framework

This module provides a comprehensive logging system with support for
file and console output, colored formatting, and integration with
experiment tracking systems.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, Any
import json
from colorlog import ColoredFormatter
import wandb


class PIHMARLLogger:
    """Custom logger for PI-HMARL with file and console output"""
    
    def __init__(
        self,
        name: str = "pi-hmarl",
        log_dir: Optional[Union[str, Path]] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        log_format: Optional[str] = None,
        colored: bool = True
    ):
        """Initialize PI-HMARL Logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files. If None, uses ./logs
            console_level: Logging level for console output
            file_level: Logging level for file output
            use_wandb: Whether to log to Weights & Biases
            wandb_project: W&B project name
            log_format: Custom log format string
            colored: Whether to use colored console output
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Setup log directory
        if log_dir is None:
            log_dir = Path("./logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup formatters
        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Console handler with color
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        
        if colored:
            color_format = '%(log_color)s' + log_format
            console_formatter = ColoredFormatter(
                color_format,
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_formatter)
        else:
            console_formatter = logging.Formatter(log_format)
            console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON file handler for structured logs
        json_log_file = self.log_dir / f"{name}_{timestamp}.json"
        self.json_file_handler = logging.FileHandler(json_log_file)
        self.json_file_handler.setLevel(logging.DEBUG)
        self.json_file_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(self.json_file_handler)
        
        # W&B integration
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        if self.use_wandb and wandb.run is None:
            self._init_wandb()
        
        self.logger.info(f"Logger initialized. Logs saved to: {log_file}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases if not already initialized"""
        try:
            if self.wandb_project:
                wandb.init(project=self.wandb_project)
            else:
                wandb.init(project="pi-hmarl")
        except Exception as e:
            self.logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
        self._log_to_wandb("debug", message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
        self._log_to_wandb("info", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
        self._log_to_wandb("warning", message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
        self._log_to_wandb("error", message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
        self._log_to_wandb("critical", message, kwargs)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to file and optionally to W&B
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/iteration number
        """
        # Log to file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "metrics",
            "step": step,
            "metrics": metrics
        }
        self.logger.info(f"Metrics: {metrics}", extra=log_entry)
        
        # Log to W&B
        if self.use_wandb and wandb.run is not None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters
        
        Args:
            params: Dictionary of hyperparameter names and values
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "hyperparameters",
            "params": params
        }
        self.logger.info(f"Hyperparameters: {params}", extra=log_entry)
        
        if self.use_wandb and wandb.run is not None:
            wandb.config.update(params)
    
    def log_model_summary(self, model_info: Dict[str, Any]):
        """Log model architecture summary
        
        Args:
            model_info: Dictionary containing model information
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "model_summary",
            "model_info": model_info
        }
        self.logger.info(f"Model Summary: {model_info}", extra=log_entry)
    
    def _log_to_wandb(self, level: str, message: str, kwargs: Dict[str, Any]):
        """Internal method to log messages to W&B"""
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                f"log/{level}": message,
                "log/timestamp": datetime.now().isoformat(),
                **{f"log/{k}": v for k, v in kwargs.items()}
            })
    
    def set_level(self, level: str, handler_type: str = "all"):
        """Change logging level
        
        Args:
            level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            handler_type: Which handler to update ('console', 'file', 'all')
        """
        level_value = getattr(logging, level.upper())
        
        if handler_type in ["console", "all"]:
            self.logger.handlers[0].setLevel(level_value)
        
        if handler_type in ["file", "all"]:
            for handler in self.logger.handlers[1:]:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(level_value)
    
    def close(self):
        """Close all handlers and finish W&B run if active"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        
        if self.use_wandb and wandb.run is not None:
            wandb.finish()


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add any extra fields
        if hasattr(record, 'extra'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'created', 'filename',
                              'funcName', 'levelname', 'levelno', 'lineno',
                              'module', 'msecs', 'message', 'pathname', 'process',
                              'processName', 'relativeCreated', 'thread',
                              'threadName', 'exc_info', 'exc_text', 'stack_info']:
                    log_data[key] = value
        
        return json.dumps(log_data)


# Global logger instance
_logger_instance = None


def get_logger(
    name: Optional[str] = None,
    log_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PIHMARLLogger:
    """Get or create a logger instance
    
    Args:
        name: Logger name. If None, returns global logger
        log_dir: Log directory
        **kwargs: Additional arguments for PIHMARLLogger
    
    Returns:
        PIHMARLLogger instance
    """
    global _logger_instance
    
    if name is None:
        if _logger_instance is None:
            _logger_instance = PIHMARLLogger(log_dir=log_dir, **kwargs)
        return _logger_instance
    else:
        return PIHMARLLogger(name=name, log_dir=log_dir, **kwargs)


# Convenience logging functions
def debug(message: str, **kwargs):
    """Log debug message using global logger"""
    logger = get_logger()
    logger.debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message using global logger"""
    logger = get_logger()
    logger.info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message using global logger"""
    logger = get_logger()
    logger.warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message using global logger"""
    logger = get_logger()
    logger.error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message using global logger"""
    logger = get_logger()
    logger.critical(message, **kwargs)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics using global logger"""
    logger = get_logger()
    logger.log_metrics(metrics, step)
EOF < /dev/null