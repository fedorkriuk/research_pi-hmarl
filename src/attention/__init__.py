"""Multi-Head Attention Mechanism for Hierarchical Multi-Agent Coordination

This module implements scalable attention mechanisms with physics-aware
features and real communication constraints.
"""

from .base_attention import BaseMultiHeadAttention
from .hierarchical_attention import HierarchicalAttention
from .physics_aware_attention import PhysicsAwareAttention
from .scalable_attention import ScalableAttention, LinearAttention
from .adaptive_attention import AdaptiveAttentionSelector
from .attention_visualizer import AttentionVisualizer
from .attention_utils import (
    compute_attention_masks,
    create_distance_matrix,
    get_cluster_assignments
)

__all__ = [
    "BaseMultiHeadAttention",
    "HierarchicalAttention",
    "PhysicsAwareAttention",
    "ScalableAttention",
    "LinearAttention",
    "AdaptiveAttentionSelector",
    "AttentionVisualizer",
    "compute_attention_masks",
    "create_distance_matrix",
    "get_cluster_assignments"
]