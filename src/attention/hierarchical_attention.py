"""Hierarchical Attention for Multi-Agent Coordination

This module implements hierarchical attention mechanisms with intra-cluster
and inter-cluster coordination based on real communication constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging

from .base_attention import BaseMultiHeadAttention
from .attention_utils import compute_attention_masks, get_cluster_assignments

logger = logging.getLogger(__name__)


class HierarchicalAttention(nn.Module):
    """Hierarchical attention with cluster-based organization"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_agents: int,
        cluster_size: int = 5,
        dropout: float = 0.1,
        communication_range: float = 100.0,  # meters
        use_cross_level: bool = True
    ):
        """Initialize hierarchical attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_agents: Maximum number of agents
            cluster_size: Size of each cluster
            dropout: Dropout rate
            communication_range: Maximum communication range
            use_cross_level: Whether to use cross-level attention
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_agents = num_agents
        self.cluster_size = cluster_size
        self.num_clusters = (num_agents + cluster_size - 1) // cluster_size
        self.communication_range = communication_range
        self.use_cross_level = use_cross_level
        
        # Intra-cluster attention (local coordination)
        self.intra_cluster_attention = IntraClusterAttention(
            embed_dim=embed_dim,
            num_heads=num_heads // 2,  # Half heads for local
            cluster_size=cluster_size,
            dropout=dropout
        )
        
        # Inter-cluster attention (global coordination)
        self.inter_cluster_attention = InterClusterAttention(
            embed_dim=embed_dim,
            num_heads=num_heads // 2,  # Half heads for global
            num_clusters=self.num_clusters,
            dropout=dropout
        )
        
        # Optional cross-level attention
        if use_cross_level:
            self.cross_level_attention = CrossLevelAttention(
                embed_dim=embed_dim,
                num_heads=num_heads // 4,
                dropout=dropout
            )
        
        # Cluster aggregation
        self.cluster_aggregator = ClusterAggregator(
            embed_dim=embed_dim,
            aggregation_type="weighted_mean"
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        logger.info(f"Initialized HierarchicalAttention for {num_agents} agents")
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cluster_assignments: Optional[torch.Tensor] = None,
        hierarchy_level: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass
        
        Args:
            x: Agent features [batch_size, num_agents, embed_dim]
            positions: Agent positions [batch_size, num_agents, 3]
            cluster_assignments: Pre-computed cluster assignments
            hierarchy_level: Hierarchy level for each agent
            
        Returns:
            Output features and attention weights dictionary
        """
        batch_size, num_agents, _ = x.size()
        
        # Get cluster assignments if not provided
        if cluster_assignments is None:
            if positions is not None:
                cluster_assignments = get_cluster_assignments(
                    positions, self.cluster_size, self.communication_range
                )
            else:
                # Default clustering
                cluster_assignments = torch.arange(num_agents) // self.cluster_size
                cluster_assignments = cluster_assignments.unsqueeze(0).expand(batch_size, -1)
        
        # Intra-cluster attention
        intra_output, intra_weights = self.intra_cluster_attention(
            x, cluster_assignments
        )
        
        # Aggregate clusters for inter-cluster attention
        cluster_features = self.cluster_aggregator(
            intra_output, cluster_assignments, self.num_clusters
        )
        
        # Inter-cluster attention
        inter_output, inter_weights = self.inter_cluster_attention(
            cluster_features
        )
        
        # Broadcast inter-cluster features back to agents
        expanded_inter = self._expand_cluster_features(
            inter_output, cluster_assignments, num_agents
        )
        
        # Optional cross-level attention
        if self.use_cross_level and hierarchy_level is not None:
            cross_output, cross_weights = self.cross_level_attention(
                intra_output, expanded_inter, hierarchy_level
            )
            combined = torch.cat([intra_output, cross_output], dim=-1)
        else:
            combined = torch.cat([intra_output, expanded_inter], dim=-1)
        
        # Final projection
        output = self.output_projection(combined)
        
        # Collect attention weights
        attention_weights = {
            "intra_cluster": intra_weights,
            "inter_cluster": inter_weights
        }
        if self.use_cross_level and hierarchy_level is not None:
            attention_weights["cross_level"] = cross_weights
        
        return output, attention_weights
    
    def _expand_cluster_features(
        self,
        cluster_features: torch.Tensor,
        cluster_assignments: torch.Tensor,
        num_agents: int
    ) -> torch.Tensor:
        """Expand cluster features back to agent dimension
        
        Args:
            cluster_features: Features per cluster
            cluster_assignments: Agent to cluster mapping
            num_agents: Number of agents
            
        Returns:
            Expanded features
        """
        batch_size = cluster_features.size(0)
        expanded = torch.zeros(
            batch_size, num_agents, self.embed_dim,
            device=cluster_features.device
        )
        
        for b in range(batch_size):
            for agent_idx in range(num_agents):
                cluster_idx = cluster_assignments[b, agent_idx]
                expanded[b, agent_idx] = cluster_features[b, cluster_idx]
        
        return expanded


class IntraClusterAttention(nn.Module):
    """Attention within clusters for local coordination"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cluster_size: int,
        dropout: float = 0.1
    ):
        """Initialize intra-cluster attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            cluster_size: Maximum agents per cluster
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cluster_size = cluster_size
        
        self.attention = BaseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Position encoding for relative positions
        self.position_encoding = nn.Linear(3, embed_dim // 4)
    
    def forward(
        self,
        x: torch.Tensor,
        cluster_assignments: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Agent features
            cluster_assignments: Cluster assignment for each agent
            positions: Optional position information
            
        Returns:
            Output features and attention weights
        """
        batch_size, num_agents, _ = x.size()
        device = x.device
        
        # Initialize output
        output = torch.zeros_like(x)
        all_weights = []
        
        # Process each cluster
        unique_clusters = torch.unique(cluster_assignments)
        
        for cluster_id in unique_clusters:
            # Get agents in this cluster
            cluster_mask = cluster_assignments == cluster_id
            cluster_agents = []
            
            for b in range(batch_size):
                agent_indices = torch.where(cluster_mask[b])[0]
                if len(agent_indices) > 0:
                    cluster_features = x[b, agent_indices]
                    
                    # Add position encoding if available
                    if positions is not None:
                        cluster_positions = positions[b, agent_indices]
                        # Compute relative positions
                        rel_pos = cluster_positions.unsqueeze(0) - cluster_positions.unsqueeze(1)
                        pos_encoding = self.position_encoding(rel_pos.view(-1, 3))
                        pos_encoding = pos_encoding.view(len(agent_indices), len(agent_indices), -1)
                        # Add to features (simplified - in practice would modify attention)
                    
                    # Apply attention within cluster
                    cluster_output, weights = self.attention(
                        cluster_features.unsqueeze(0),
                        cluster_features.unsqueeze(0),
                        cluster_features.unsqueeze(0)
                    )
                    
                    # Store results
                    output[b, agent_indices] = cluster_output.squeeze(0)
                    all_weights.append(weights)
        
        # Aggregate weights
        if all_weights:
            attention_weights = torch.cat(all_weights, dim=0).mean(dim=0)
        else:
            attention_weights = torch.zeros(num_agents, num_agents, device=device)
        
        return output, attention_weights


class InterClusterAttention(nn.Module):
    """Attention between clusters for global coordination"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_clusters: int,
        dropout: float = 0.1
    ):
        """Initialize inter-cluster attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_clusters: Maximum number of clusters
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_clusters = num_clusters
        
        self.attention = BaseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cluster-level position encoding
        self.cluster_embedding = nn.Embedding(num_clusters, embed_dim // 4)
    
    def forward(
        self,
        cluster_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            cluster_features: Aggregated features per cluster
            
        Returns:
            Output features and attention weights
        """
        batch_size, num_clusters, _ = cluster_features.size()
        
        # Add cluster embeddings
        cluster_ids = torch.arange(num_clusters, device=cluster_features.device)
        cluster_emb = self.cluster_embedding(cluster_ids)
        cluster_emb = cluster_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate embeddings (simplified - in practice would modify attention)
        enhanced_features = cluster_features + cluster_emb.repeat(1, 1, 4)
        
        # Apply attention between clusters
        output, weights = self.attention(
            enhanced_features,
            enhanced_features,
            enhanced_features
        )
        
        return output, weights


class CrossLevelAttention(nn.Module):
    """Attention across hierarchy levels"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """Initialize cross-level attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.cross_attention = BaseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Level encoding
        self.level_embedding = nn.Embedding(3, embed_dim // 4)  # 3 levels
    
    def forward(
        self,
        low_level: torch.Tensor,
        high_level: torch.Tensor,
        hierarchy_level: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            low_level: Lower level features
            high_level: Higher level features
            hierarchy_level: Level indicator for each agent
            
        Returns:
            Output features and attention weights
        """
        # Add level embeddings
        level_emb = self.level_embedding(hierarchy_level)
        level_emb = level_emb.unsqueeze(1).expand(-1, low_level.size(1), -1)
        
        # Enhance features with level information
        low_enhanced = low_level + level_emb.repeat(1, 1, 4)
        high_enhanced = high_level + level_emb.repeat(1, 1, 4)
        
        # Cross attention from low to high level
        output, weights = self.cross_attention(
            low_enhanced,  # Query from low level
            high_enhanced, # Key/Value from high level
            high_enhanced
        )
        
        return output, weights


class ClusterAggregator(nn.Module):
    """Aggregates agent features into cluster representations"""
    
    def __init__(
        self,
        embed_dim: int,
        aggregation_type: str = "weighted_mean"
    ):
        """Initialize cluster aggregator
        
        Args:
            embed_dim: Embedding dimension
            aggregation_type: Type of aggregation
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.aggregation_type = aggregation_type
        
        if aggregation_type == "weighted_mean":
            self.importance_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        features: torch.Tensor,
        cluster_assignments: torch.Tensor,
        num_clusters: int
    ) -> torch.Tensor:
        """Aggregate features by cluster
        
        Args:
            features: Agent features
            cluster_assignments: Cluster assignment per agent
            num_clusters: Total number of clusters
            
        Returns:
            Cluster-level features
        """
        batch_size, num_agents, embed_dim = features.size()
        device = features.device
        
        # Initialize cluster features
        cluster_features = torch.zeros(
            batch_size, num_clusters, embed_dim,
            device=device
        )
        
        # Aggregate based on type
        if self.aggregation_type == "mean":
            for b in range(batch_size):
                for c in range(num_clusters):
                    mask = cluster_assignments[b] == c
                    if mask.any():
                        cluster_features[b, c] = features[b, mask].mean(dim=0)
        
        elif self.aggregation_type == "weighted_mean":
            # Compute importance weights
            importance = self.importance_net(features)
            
            for b in range(batch_size):
                for c in range(num_clusters):
                    mask = cluster_assignments[b] == c
                    if mask.any():
                        cluster_agents = features[b, mask]
                        cluster_importance = importance[b, mask]
                        
                        # Normalize importance
                        cluster_importance = F.softmax(cluster_importance, dim=0)
                        
                        # Weighted aggregation
                        cluster_features[b, c] = (cluster_agents * cluster_importance).sum(dim=0)
        
        elif self.aggregation_type == "max":
            for b in range(batch_size):
                for c in range(num_clusters):
                    mask = cluster_assignments[b] == c
                    if mask.any():
                        cluster_features[b, c] = features[b, mask].max(dim=0)[0]
        
        return cluster_features


class TemporalHierarchicalAttention(nn.Module):
    """Hierarchical attention with temporal modeling"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_agents: int,
        cluster_size: int = 5,
        history_len: int = 10,
        dropout: float = 0.1
    ):
        """Initialize temporal hierarchical attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_agents: Maximum number of agents
            cluster_size: Size of each cluster
            history_len: Length of temporal history
            dropout: Dropout rate
        """
        super().__init__()
        
        self.history_len = history_len
        
        # Spatial hierarchical attention
        self.spatial_attention = HierarchicalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_agents=num_agents,
            cluster_size=cluster_size,
            dropout=dropout
        )
        
        # Temporal attention
        self.temporal_attention = BaseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads // 2,
            dropout=dropout
        )
        
        # Temporal position encoding
        self.temporal_encoding = nn.Parameter(
            torch.randn(1, history_len, embed_dim)
        )
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        history: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        cluster_assignments: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with temporal modeling
        
        Args:
            x: Current features
            history: Historical features [batch, history_len, num_agents, embed_dim]
            positions: Agent positions
            cluster_assignments: Cluster assignments
            
        Returns:
            Output features and attention weights
        """
        batch_size, num_agents, embed_dim = x.size()
        
        # Spatial hierarchical attention
        spatial_output, spatial_weights = self.spatial_attention(
            x, positions, cluster_assignments
        )
        
        # Temporal attention if history is provided
        if history is not None:
            # Reshape for temporal attention
            history_flat = history.view(batch_size * num_agents, self.history_len, embed_dim)
            current_flat = x.view(batch_size * num_agents, 1, embed_dim)
            
            # Add temporal encoding
            history_encoded = history_flat + self.temporal_encoding[:, :history.size(1), :]
            
            # Temporal attention
            temporal_output, temporal_weights = self.temporal_attention(
                current_flat,
                history_encoded,
                history_encoded
            )
            
            # Reshape back
            temporal_output = temporal_output.view(batch_size, num_agents, embed_dim)
            
            # Combine spatial and temporal
            combined = torch.cat([spatial_output, temporal_output], dim=-1)
            output = self.output_projection(combined)
            
            weights = {
                **spatial_weights,
                "temporal": temporal_weights
            }
        else:
            output = spatial_output
            weights = spatial_weights
        
        return output, weights