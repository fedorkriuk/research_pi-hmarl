"""Utility Functions for Attention Mechanisms

This module provides utility functions for attention computation, masking,
and clustering based on real-world constraints.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def compute_attention_masks(
    positions: torch.Tensor,
    communication_range: float = 100.0,
    min_safety_distance: float = 5.0,
    velocities: Optional[torch.Tensor] = None,
    energy_levels: Optional[torch.Tensor] = None,
    energy_threshold: float = 0.2
) -> Dict[str, torch.Tensor]:
    """Compute various attention masks based on physical constraints
    
    Args:
        positions: Agent positions [batch_size, num_agents, 3]
        communication_range: Maximum communication range in meters
        min_safety_distance: Minimum safety distance in meters
        velocities: Agent velocities [batch_size, num_agents, 3]
        energy_levels: Energy levels [batch_size, num_agents]
        energy_threshold: Minimum energy for communication
        
    Returns:
        Dictionary of attention masks
    """
    batch_size, num_agents, _ = positions.size()
    device = positions.device
    
    # Distance matrix
    distance_matrix = create_distance_matrix(positions)
    
    masks = {}
    
    # Communication range mask
    comm_mask = (distance_matrix <= communication_range).float()
    masks["communication"] = comm_mask
    
    # Safety distance mask (inverse - mask out too close agents)
    safety_mask = (distance_matrix >= min_safety_distance).float()
    # Allow self-attention
    eye = torch.eye(num_agents, device=device).unsqueeze(0)
    safety_mask = safety_mask + eye
    safety_mask = torch.clamp(safety_mask, 0, 1)
    masks["safety"] = safety_mask
    
    # Collision risk mask based on velocities
    if velocities is not None:
        collision_mask = compute_collision_mask(
            positions, velocities, min_safety_distance
        )
        masks["collision"] = collision_mask
    
    # Energy-based mask
    if energy_levels is not None:
        energy_mask = compute_energy_mask(
            energy_levels, distance_matrix, energy_threshold
        )
        masks["energy"] = energy_mask
    
    # Combined mask
    combined_mask = comm_mask * safety_mask
    if "collision" in masks:
        combined_mask = combined_mask * masks["collision"]
    if "energy" in masks:
        combined_mask = combined_mask * masks["energy"]
    
    masks["combined"] = combined_mask
    
    # Convert to attention format (0 = attend, -inf = mask)
    for key in masks:
        masks[key] = convert_to_attention_mask(masks[key])
    
    return masks


def create_distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Create pairwise distance matrix
    
    Args:
        positions: Agent positions [batch_size, num_agents, dim]
        
    Returns:
        Distance matrix [batch_size, num_agents, num_agents]
    """
    # Expand for broadcasting
    pos1 = positions.unsqueeze(2)  # [batch, agents, 1, dim]
    pos2 = positions.unsqueeze(1)  # [batch, 1, agents, dim]
    
    # Compute Euclidean distance
    distances = torch.norm(pos1 - pos2, dim=-1)
    
    return distances


def compute_collision_mask(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    min_safety_distance: float,
    time_horizon: float = 5.0
) -> torch.Tensor:
    """Compute collision avoidance mask
    
    Args:
        positions: Agent positions
        velocities: Agent velocities
        min_safety_distance: Minimum safety distance
        time_horizon: Time horizon for collision prediction
        
    Returns:
        Collision mask (1 = safe, 0 = potential collision)
    """
    batch_size, num_agents, _ = positions.size()
    device = positions.device
    
    # Current distances
    current_distances = create_distance_matrix(positions)
    
    # Predict future positions
    future_positions = positions + velocities * time_horizon
    future_distances = create_distance_matrix(future_positions)
    
    # Check if agents are getting closer
    approaching = future_distances < current_distances
    
    # Dynamic safety margin based on approach speed
    vel_magnitude = torch.norm(velocities, dim=-1, keepdim=True)
    vel_mag_matrix = vel_magnitude.unsqueeze(1) + vel_magnitude.unsqueeze(2)
    
    dynamic_safety = min_safety_distance * (1 + 0.1 * vel_mag_matrix)
    
    # Safe if current distance is large enough OR not approaching
    safe_mask = (current_distances > dynamic_safety) | (~approaching)
    
    # Ensure self-attention
    eye = torch.eye(num_agents, device=device).unsqueeze(0)
    safe_mask = safe_mask.float() + eye
    safe_mask = torch.clamp(safe_mask, 0, 1)
    
    return safe_mask


def compute_energy_mask(
    energy_levels: torch.Tensor,
    distance_matrix: torch.Tensor,
    energy_threshold: float = 0.2,
    comm_cost_per_meter: float = 0.001
) -> torch.Tensor:
    """Compute energy-aware communication mask
    
    Args:
        energy_levels: Energy levels [batch_size, num_agents]
        distance_matrix: Distance matrix
        energy_threshold: Minimum energy threshold
        comm_cost_per_meter: Communication cost per meter
        
    Returns:
        Energy mask
    """
    batch_size, num_agents = energy_levels.size()
    device = energy_levels.device
    
    # Check if agents have enough energy
    has_energy = energy_levels > energy_threshold
    
    # Compute communication cost
    comm_cost = distance_matrix * comm_cost_per_meter
    
    # Check if both agents have enough energy for communication
    energy_mask = has_energy.unsqueeze(1) & has_energy.unsqueeze(2)
    
    # Prioritize closer agents when energy is low
    low_energy = energy_levels < (energy_threshold * 2)
    for b in range(batch_size):
        if low_energy[b].any():
            # For low energy agents, only communicate with nearest neighbors
            k = min(3, num_agents)
            _, nearest = distance_matrix[b].topk(k, dim=-1, largest=False)
            
            for i in range(num_agents):
                if low_energy[b, i]:
                    restricted_mask = torch.zeros(num_agents, device=device)
                    restricted_mask[nearest[i]] = 1
                    energy_mask[b, i] = energy_mask[b, i] & restricted_mask.bool()
    
    return energy_mask.float()


def get_cluster_assignments(
    positions: torch.Tensor,
    cluster_size: int = 5,
    max_distance: float = 50.0,
    method: str = "kmeans"
) -> torch.Tensor:
    """Assign agents to clusters based on positions
    
    Args:
        positions: Agent positions [batch_size, num_agents, dim]
        cluster_size: Target size of each cluster
        max_distance: Maximum distance for DBSCAN clustering
        method: Clustering method ("kmeans", "dbscan", "grid")
        
    Returns:
        Cluster assignments [batch_size, num_agents]
    """
    batch_size, num_agents, dim = positions.size()
    device = positions.device
    
    cluster_assignments = torch.zeros(batch_size, num_agents, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        pos_np = positions[b].detach().cpu().numpy()
        
        if method == "kmeans":
            # K-means clustering
            n_clusters = max(1, num_agents // cluster_size)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(pos_np)
            
        elif method == "dbscan":
            # DBSCAN for density-based clustering
            dbscan = DBSCAN(eps=max_distance, min_samples=2)
            labels = dbscan.fit_predict(pos_np)
            # Handle noise points
            labels[labels == -1] = labels.max() + 1
            
        elif method == "grid":
            # Grid-based clustering
            grid_size = np.sqrt(max_distance)
            grid_coords = (pos_np[:, :2] / grid_size).astype(int)
            # Convert grid coordinates to cluster labels
            unique_grids, labels = np.unique(
                grid_coords, axis=0, return_inverse=True
            )
        
        else:
            # Default: sequential clustering
            labels = np.arange(num_agents) // cluster_size
        
        cluster_assignments[b] = torch.tensor(labels, device=device)
    
    return cluster_assignments


def convert_to_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert binary mask to attention mask format
    
    Args:
        mask: Binary mask (1 = attend, 0 = ignore)
        
    Returns:
        Attention mask (0 = attend, -inf = ignore)
    """
    attention_mask = torch.where(
        mask > 0,
        torch.zeros_like(mask),
        torch.full_like(mask, float('-inf'))
    )
    return attention_mask


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask
    
    Args:
        seq_len: Sequence length
        device: Device for tensor
        
    Returns:
        Causal mask
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_block_diagonal_mask(
    block_sizes: List[int],
    device: torch.device
) -> torch.Tensor:
    """Create block diagonal attention mask
    
    Args:
        block_sizes: Size of each block
        device: Device for tensor
        
    Returns:
        Block diagonal mask
    """
    total_size = sum(block_sizes)
    mask = torch.full((total_size, total_size), float('-inf'), device=device)
    
    start = 0
    for block_size in block_sizes:
        end = start + block_size
        mask[start:end, start:end] = 0
        start = end
    
    return mask


def compute_attention_entropy(
    attention_weights: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """Compute entropy of attention distribution
    
    Args:
        attention_weights: Attention weights
        epsilon: Small value for numerical stability
        
    Returns:
        Entropy values
    """
    # Add epsilon for numerical stability
    weights = attention_weights + epsilon
    
    # Normalize if not already
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Compute entropy
    entropy = -(weights * torch.log(weights)).sum(dim=-1)
    
    return entropy


def compute_attention_sparsity(
    attention_weights: torch.Tensor,
    threshold: float = 0.01
) -> Dict[str, float]:
    """Compute sparsity metrics for attention
    
    Args:
        attention_weights: Attention weights
        threshold: Threshold for considering weight as zero
        
    Returns:
        Dictionary of sparsity metrics
    """
    # Count near-zero weights
    sparse_count = (attention_weights < threshold).sum()
    total_count = attention_weights.numel()
    
    # Compute metrics
    metrics = {
        "sparsity_ratio": (sparse_count / total_count).item(),
        "density_ratio": 1 - (sparse_count / total_count).item(),
        "avg_nonzero_weight": attention_weights[attention_weights >= threshold].mean().item()
        if (attention_weights >= threshold).any() else 0.0,
        "max_weight": attention_weights.max().item(),
        "effective_connections": (attention_weights >= threshold).sum().item()
    }
    
    return metrics


def apply_dropout_mask(
    attention_weights: torch.Tensor,
    dropout_rate: float = 0.1,
    training: bool = True
) -> torch.Tensor:
    """Apply structured dropout to attention weights
    
    Args:
        attention_weights: Attention weights
        dropout_rate: Dropout probability
        training: Whether in training mode
        
    Returns:
        Attention weights with dropout
    """
    if not training or dropout_rate == 0:
        return attention_weights
    
    # Structured dropout - drop entire connections
    keep_prob = 1 - dropout_rate
    
    # Create dropout mask
    mask_shape = attention_weights.shape[:-1]  # Drop last dimension
    dropout_mask = torch.bernoulli(
        torch.full(mask_shape, keep_prob, device=attention_weights.device)
    ).unsqueeze(-1)
    
    # Apply mask and rescale
    attention_weights = attention_weights * dropout_mask / keep_prob
    
    # Renormalize
    attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
    
    return attention_weights


def compute_relative_position_bias(
    positions: torch.Tensor,
    max_distance: float = 100.0,
    num_buckets: int = 32
) -> torch.Tensor:
    """Compute relative position bias for attention
    
    Args:
        positions: Agent positions
        max_distance: Maximum distance to consider
        num_buckets: Number of distance buckets
        
    Returns:
        Position bias matrix
    """
    distances = create_distance_matrix(positions)
    
    # Normalize distances
    normalized_dist = distances / max_distance
    normalized_dist = torch.clamp(normalized_dist, 0, 1)
    
    # Convert to buckets
    bucket_indices = (normalized_dist * (num_buckets - 1)).long()
    
    # Create learnable bias (in practice, this would be a parameter)
    # For now, use a simple distance-based bias
    bias = -normalized_dist * 2  # Closer agents get higher attention
    
    return bias


def merge_attention_heads(
    multi_head_attention: torch.Tensor,
    merge_type: str = "mean"
) -> torch.Tensor:
    """Merge attention from multiple heads
    
    Args:
        multi_head_attention: Attention weights [batch, heads, seq, seq]
        merge_type: How to merge ("mean", "max", "min", "sum")
        
    Returns:
        Merged attention [batch, seq, seq]
    """
    if merge_type == "mean":
        return multi_head_attention.mean(dim=1)
    elif merge_type == "max":
        return multi_head_attention.max(dim=1)[0]
    elif merge_type == "min":
        return multi_head_attention.min(dim=1)[0]
    elif merge_type == "sum":
        return multi_head_attention.sum(dim=1)
    else:
        raise ValueError(f"Unknown merge type: {merge_type}")


def compute_attention_consistency(
    attention_weights: torch.Tensor,
    prev_attention_weights: torch.Tensor
) -> float:
    """Compute consistency between consecutive attention patterns
    
    Args:
        attention_weights: Current attention weights
        prev_attention_weights: Previous attention weights
        
    Returns:
        Consistency score (0-1)
    """
    # Flatten and normalize
    curr_flat = attention_weights.flatten()
    prev_flat = prev_attention_weights.flatten()
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(curr_flat.unsqueeze(0), prev_flat.unsqueeze(0))
    
    # Convert to 0-1 range
    consistency = (cos_sim + 1) / 2
    
    return consistency.item()


def create_attention_prior(
    num_agents: int,
    prior_type: str = "uniform",
    device: torch.device = torch.device("cpu"),
    **kwargs
) -> torch.Tensor:
    """Create attention prior distribution
    
    Args:
        num_agents: Number of agents
        prior_type: Type of prior ("uniform", "local", "exponential")
        device: Device for tensor
        **kwargs: Additional arguments for specific priors
        
    Returns:
        Attention prior matrix
    """
    if prior_type == "uniform":
        # Uniform attention to all agents
        prior = torch.ones(num_agents, num_agents, device=device)
        prior = prior / num_agents
        
    elif prior_type == "local":
        # Attend mostly to nearby agents
        window_size = kwargs.get("window_size", 3)
        prior = torch.zeros(num_agents, num_agents, device=device)
        
        for i in range(num_agents):
            for j in range(max(0, i - window_size), min(num_agents, i + window_size + 1)):
                prior[i, j] = 1.0
        
        # Normalize
        prior = prior / prior.sum(dim=-1, keepdim=True)
        
    elif prior_type == "exponential":
        # Exponential decay with distance
        decay_rate = kwargs.get("decay_rate", 0.1)
        indices = torch.arange(num_agents, device=device)
        
        # Create distance matrix based on indices
        dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        
        # Apply exponential decay
        prior = torch.exp(-decay_rate * dist.float())
        
        # Normalize
        prior = prior / prior.sum(dim=-1, keepdim=True)
    
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")
    
    return prior