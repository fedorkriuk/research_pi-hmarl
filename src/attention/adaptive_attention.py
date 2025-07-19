"""Adaptive Attention Selection

This module implements dynamic attention head selection based on scenario
complexity and computational constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging

from .base_attention import BaseMultiHeadAttention
from .hierarchical_attention import HierarchicalAttention
from .physics_aware_attention import PhysicsAwareAttention
from .scalable_attention import ScalableAttention

logger = logging.getLogger(__name__)


class AdaptiveAttentionSelector(nn.Module):
    """Dynamically selects attention mechanism based on context"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads_options: List[int] = [4, 8, 16],
        num_agents: int = 20,
        scenario_types: List[str] = ["simple", "moderate", "complex"],
        dropout: float = 0.1
    ):
        """Initialize adaptive attention selector
        
        Args:
            embed_dim: Embedding dimension
            num_heads_options: Options for number of attention heads
            num_agents: Maximum number of agents
            scenario_types: Types of scenarios to handle
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads_options = num_heads_options
        self.num_agents = num_agents
        self.scenario_types = scenario_types
        
        # Scenario classifier
        self.scenario_classifier = ScenarioClassifier(
            embed_dim=embed_dim,
            num_scenarios=len(scenario_types)
        )
        
        # Complexity analyzer
        self.complexity_analyzer = ComplexityAnalyzer(
            embed_dim=embed_dim,
            num_agents=num_agents
        )
        
        # Create attention modules for different scenarios
        self.attention_modules = nn.ModuleDict()
        
        # Simple scenarios - basic attention with few heads
        self.attention_modules["simple"] = BaseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads_options[0],
            dropout=dropout
        )
        
        # Moderate scenarios - hierarchical attention
        self.attention_modules["moderate"] = HierarchicalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads_options[1],
            num_agents=num_agents,
            dropout=dropout
        )
        
        # Complex scenarios - physics-aware attention with many heads
        self.attention_modules["complex"] = PhysicsAwareAttention(
            embed_dim=embed_dim,
            num_heads=num_heads_options[2],
            num_agents=num_agents,
            dropout=dropout
        )
        
        # Scalable attention for very large groups
        self.attention_modules["scalable"] = ScalableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads_options[1],
            max_agents=num_agents * 2,
            attention_type="linear",
            dropout=dropout
        )
        
        # Head selection network
        self.head_selector = HeadSelector(
            embed_dim=embed_dim,
            num_heads_options=num_heads_options
        )
        
        # Attention mixer for smooth transitions
        self.attention_mixer = AttentionMixer(embed_dim=embed_dim)
        
        logger.info(f"Initialized AdaptiveAttentionSelector with {len(self.attention_modules)} modules")
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        force_scenario: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with adaptive attention selection
        
        Args:
            x: Agent features [batch_size, num_agents, embed_dim]
            positions: Agent positions
            velocities: Agent velocities
            context: Additional context information
            force_scenario: Force specific scenario type
            
        Returns:
            Output features and attention info
        """
        batch_size, num_agents, _ = x.size()
        
        # Analyze complexity
        complexity_score = self.complexity_analyzer(x, positions, velocities)
        
        # Classify scenario if not forced
        if force_scenario is None:
            scenario_probs = self.scenario_classifier(x, complexity_score)
            scenario_idx = torch.argmax(scenario_probs, dim=-1)
            scenario_type = self.scenario_types[scenario_idx.item()]
        else:
            scenario_type = force_scenario
            scenario_probs = None
        
        # Select number of heads based on complexity
        num_heads = self.head_selector(x, complexity_score)
        
        # Special case: use scalable attention for very large groups
        if num_agents > self.num_agents:
            scenario_type = "scalable"
        
        # Apply selected attention mechanism
        if scenario_type == "simple":
            output, attention_info = self.attention_modules["simple"](x, x, x)
        
        elif scenario_type == "moderate":
            output, attention_info = self.attention_modules["moderate"](
                x, positions=positions
            )
        
        elif scenario_type == "complex":
            output, attention_info = self.attention_modules["complex"](
                x, positions, velocities
            )
        
        elif scenario_type == "scalable":
            output, attention_info = self.attention_modules["scalable"](
                x, positions=positions
            )
        
        else:
            # Default to moderate
            output, attention_info = self.attention_modules["moderate"](
                x, positions=positions
            )
        
        # Mix outputs if transitioning between scenarios
        if hasattr(self, "_prev_scenario") and self._prev_scenario != scenario_type:
            # Smooth transition between attention mechanisms
            if hasattr(self, "_prev_output"):
                output = self.attention_mixer(self._prev_output, output, alpha=0.7)
        
        self._prev_scenario = scenario_type
        self._prev_output = output.detach()
        
        # Compile info
        info = {
            "scenario_type": scenario_type,
            "scenario_probs": scenario_probs,
            "complexity_score": complexity_score,
            "num_heads": num_heads,
            "num_agents": num_agents,
            **attention_info
        }
        
        return output, info
    
    def get_computational_stats(self) -> Dict[str, Any]:
        """Get computational statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "active_scenario": getattr(self, "_prev_scenario", "unknown"),
            "modules": list(self.attention_modules.keys()),
            "num_heads_options": self.num_heads_options
        }
        
        # Add module-specific stats
        for name, module in self.attention_modules.items():
            if hasattr(module, "get_attention_stats"):
                stats[f"{name}_stats"] = module.get_attention_stats()
        
        return stats


class ScenarioClassifier(nn.Module):
    """Classifies scenario complexity"""
    
    def __init__(self, embed_dim: int, num_scenarios: int = 3):
        """Initialize scenario classifier
        
        Args:
            embed_dim: Embedding dimension
            num_scenarios: Number of scenario types
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim // 2),  # +1 for complexity score
            nn.ReLU(),
            nn.LayerNorm(embed_dim // 2),
            nn.Linear(embed_dim // 2, num_scenarios)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        complexity_score: torch.Tensor
    ) -> torch.Tensor:
        """Classify scenario
        
        Args:
            x: Agent features
            complexity_score: Complexity metric
            
        Returns:
            Scenario probabilities
        """
        # Global pooling of agent features
        global_features = x.mean(dim=1)  # [batch_size, embed_dim]
        
        # Concatenate with complexity
        features = torch.cat([global_features, complexity_score.unsqueeze(-1)], dim=-1)
        
        # Classify
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        
        return probs


class ComplexityAnalyzer(nn.Module):
    """Analyzes scenario complexity"""
    
    def __init__(self, embed_dim: int, num_agents: int):
        """Initialize complexity analyzer
        
        Args:
            embed_dim: Embedding dimension
            num_agents: Maximum number of agents
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_agents = num_agents
        
        # Feature extractors
        self.spatial_analyzer = nn.Sequential(
            nn.Linear(3, 32),  # Position features
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.dynamic_analyzer = nn.Sequential(
            nn.Linear(3, 32),  # Velocity features
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.interaction_analyzer = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute complexity score
        
        Args:
            x: Agent features
            positions: Agent positions
            velocities: Agent velocities
            
        Returns:
            Complexity score (0-1)
        """
        batch_size, num_agents, _ = x.size()
        complexity_components = []
        
        # Agent density complexity
        density_score = num_agents / self.num_agents
        complexity_components.append(density_score)
        
        # Spatial complexity
        if positions is not None:
            # Compute spatial spread
            pos_std = positions.std(dim=1)
            spatial_score = self.spatial_analyzer(pos_std).squeeze(-1)
            complexity_components.append(torch.sigmoid(spatial_score))
        
        # Dynamic complexity
        if velocities is not None:
            # Compute velocity variance
            vel_std = velocities.std(dim=1)
            dynamic_score = self.dynamic_analyzer(vel_std).squeeze(-1)
            complexity_components.append(torch.sigmoid(dynamic_score))
        
        # Interaction complexity
        interaction_features = x.var(dim=1)  # Variance across agents
        interaction_score = self.interaction_analyzer(interaction_features).squeeze(-1)
        complexity_components.append(torch.sigmoid(interaction_score))
        
        # Combine scores
        if len(complexity_components) > 1:
            complexity = torch.stack(complexity_components, dim=-1).mean(dim=-1)
        else:
            complexity = torch.tensor([density_score], device=x.device)
        
        return complexity


class HeadSelector(nn.Module):
    """Selects optimal number of attention heads"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads_options: List[int] = [4, 8, 16]
    ):
        """Initialize head selector
        
        Args:
            embed_dim: Embedding dimension
            num_heads_options: Available head counts
        """
        super().__init__()
        
        self.num_heads_options = num_heads_options
        
        self.selector = nn.Sequential(
            nn.Linear(embed_dim + 1, 64),  # +1 for complexity
            nn.ReLU(),
            nn.Linear(64, len(num_heads_options))
        )
    
    def forward(
        self,
        x: torch.Tensor,
        complexity_score: torch.Tensor
    ) -> int:
        """Select number of heads
        
        Args:
            x: Agent features
            complexity_score: Complexity metric
            
        Returns:
            Selected number of heads
        """
        # Global features
        global_features = x.mean(dim=1)
        
        # Concatenate with complexity
        features = torch.cat([global_features, complexity_score.unsqueeze(-1)], dim=-1)
        
        # Select heads
        logits = self.selector(features)
        head_idx = torch.argmax(logits, dim=-1)
        
        return self.num_heads_options[head_idx.item()]


class AttentionMixer(nn.Module):
    """Mixes outputs from different attention mechanisms"""
    
    def __init__(self, embed_dim: int):
        """Initialize attention mixer
        
        Args:
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.mixer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """Mix two attention outputs
        
        Args:
            output1: First attention output
            output2: Second attention output
            alpha: Mixing weight (0-1)
            
        Returns:
            Mixed output
        """
        # Linear interpolation
        mixed = alpha * output1 + (1 - alpha) * output2
        
        # Non-linear mixing
        combined = torch.cat([output1, output2], dim=-1)
        refined = self.mixer(combined)
        
        # Combine linear and non-linear
        output = 0.7 * mixed + 0.3 * refined
        
        return output


class ScenarioAdaptiveAttention(nn.Module):
    """Complete scenario-adaptive attention system"""
    
    def __init__(
        self,
        embed_dim: int,
        num_agents: int = 20,
        min_heads: int = 4,
        max_heads: int = 16,
        adaptation_rate: float = 0.1,
        dropout: float = 0.1
    ):
        """Initialize scenario-adaptive attention
        
        Args:
            embed_dim: Embedding dimension
            num_agents: Expected number of agents
            min_heads: Minimum attention heads
            max_heads: Maximum attention heads
            adaptation_rate: Rate of adaptation
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_agents = num_agents
        self.min_heads = min_heads
        self.max_heads = max_heads
        self.adaptation_rate = adaptation_rate
        
        # Adaptive selector
        num_heads_options = [min_heads, (min_heads + max_heads) // 2, max_heads]
        self.selector = AdaptiveAttentionSelector(
            embed_dim=embed_dim,
            num_heads_options=num_heads_options,
            num_agents=num_agents,
            dropout=dropout
        )
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Adaptation controller
        self.adaptation_controller = AdaptationController(
            adaptation_rate=adaptation_rate
        )
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        target_latency: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with performance adaptation
        
        Args:
            x: Agent features
            positions: Agent positions
            velocities: Agent velocities
            target_latency: Target inference latency
            
        Returns:
            Output and info
        """
        # Monitor performance
        start_time = torch.cuda.Event(enable_timing=True) if x.is_cuda else None
        end_time = torch.cuda.Event(enable_timing=True) if x.is_cuda else None
        
        if start_time:
            start_time.record()
        
        # Apply adaptive attention
        output, info = self.selector(x, positions, velocities)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            latency = start_time.elapsed_time(end_time)
            info["latency_ms"] = latency
        
        # Update performance statistics
        self.performance_monitor.update(info)
        
        # Adapt based on performance
        if target_latency is not None:
            adaptation = self.adaptation_controller.compute_adaptation(
                current_latency=info.get("latency_ms", 0),
                target_latency=target_latency,
                current_scenario=info["scenario_type"]
            )
            
            if adaptation:
                info["adaptation"] = adaptation
        
        return output, info


class PerformanceMonitor:
    """Monitors attention performance metrics"""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance monitor
        
        Args:
            window_size: Size of monitoring window
        """
        self.window_size = window_size
        self.metrics = {
            "latency": [],
            "memory": [],
            "scenario_distribution": {},
            "head_distribution": {}
        }
    
    def update(self, info: Dict[str, Any]):
        """Update performance metrics
        
        Args:
            info: Information from attention forward pass
        """
        # Update latency
        if "latency_ms" in info:
            self.metrics["latency"].append(info["latency_ms"])
            if len(self.metrics["latency"]) > self.window_size:
                self.metrics["latency"].pop(0)
        
        # Update scenario distribution
        scenario = info.get("scenario_type", "unknown")
        self.metrics["scenario_distribution"][scenario] = \
            self.metrics["scenario_distribution"].get(scenario, 0) + 1
        
        # Update head distribution
        num_heads = info.get("num_heads", 0)
        self.metrics["head_distribution"][num_heads] = \
            self.metrics["head_distribution"].get(num_heads, 0) + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        if self.metrics["latency"]:
            stats["avg_latency_ms"] = np.mean(self.metrics["latency"])
            stats["max_latency_ms"] = np.max(self.metrics["latency"])
            stats["min_latency_ms"] = np.min(self.metrics["latency"])
        
        stats["scenario_distribution"] = self.metrics["scenario_distribution"]
        stats["head_distribution"] = self.metrics["head_distribution"]
        
        return stats


class AdaptationController:
    """Controls adaptation based on performance"""
    
    def __init__(self, adaptation_rate: float = 0.1):
        """Initialize adaptation controller
        
        Args:
            adaptation_rate: Rate of adaptation
        """
        self.adaptation_rate = adaptation_rate
        self.adaptation_history = []
    
    def compute_adaptation(
        self,
        current_latency: float,
        target_latency: float,
        current_scenario: str
    ) -> Optional[Dict[str, Any]]:
        """Compute adaptation decision
        
        Args:
            current_latency: Current inference latency
            target_latency: Target latency
            current_scenario: Current scenario type
            
        Returns:
            Adaptation decision or None
        """
        if current_latency <= 0:
            return None
        
        # Compute latency ratio
        latency_ratio = current_latency / target_latency
        
        adaptation = None
        
        if latency_ratio > 1.2:  # Too slow
            # Suggest simpler scenario or fewer heads
            if current_scenario == "complex":
                adaptation = {
                    "action": "downgrade_scenario",
                    "target": "moderate",
                    "reason": "latency_exceeded"
                }
            elif current_scenario == "moderate":
                adaptation = {
                    "action": "downgrade_scenario",
                    "target": "simple",
                    "reason": "latency_exceeded"
                }
        
        elif latency_ratio < 0.5:  # Too fast, can do more
            # Suggest more complex scenario
            if current_scenario == "simple":
                adaptation = {
                    "action": "upgrade_scenario",
                    "target": "moderate",
                    "reason": "latency_headroom"
                }
            elif current_scenario == "moderate":
                adaptation = {
                    "action": "upgrade_scenario",
                    "target": "complex",
                    "reason": "latency_headroom"
                }
        
        if adaptation:
            self.adaptation_history.append({
                "timestamp": torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
                "adaptation": adaptation,
                "latency_ratio": latency_ratio
            })
        
        return adaptation