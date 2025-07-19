"""Scalable Attention Mechanisms for Large Agent Groups

This module implements efficient attention mechanisms with linear complexity
for handling 20+ agents in real-time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging
import math

from .base_attention import BaseMultiHeadAttention

logger = logging.getLogger(__name__)


class ScalableAttention(nn.Module):
    """Scalable attention mechanism for large agent groups"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_agents: int = 50,
        attention_type: str = "linear",
        use_sparse: bool = True,
        sparsity_ratio: float = 0.1,
        chunk_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        """Initialize scalable attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            max_agents: Maximum number of agents
            attention_type: Type of attention ("linear", "sparse", "local")
            use_sparse: Whether to use sparse attention patterns
            sparsity_ratio: Ratio of connections to maintain
            chunk_size: Size of chunks for local attention
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_agents = max_agents
        self.attention_type = attention_type
        self.use_sparse = use_sparse
        self.sparsity_ratio = sparsity_ratio
        self.chunk_size = chunk_size or int(np.sqrt(max_agents))
        
        # Select attention implementation
        if attention_type == "linear":
            self.attention = LinearAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif attention_type == "sparse":
            self.attention = SparseAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sparsity_ratio=sparsity_ratio,
                dropout=dropout
            )
        elif attention_type == "local":
            self.attention = LocalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=chunk_size,
                dropout=dropout
            )
        else:
            # Default to efficient multi-head attention
            self.attention = EfficientMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Memory optimization
        self.use_gradient_checkpointing = max_agents > 20
        
        logger.info(f"Initialized ScalableAttention with {attention_type} for {max_agents} agents")
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass
        
        Args:
            x: Agent features [batch_size, num_agents, embed_dim]
            positions: Agent positions for sparse patterns
            mask: Attention mask
            return_stats: Whether to return computational statistics
            
        Returns:
            Output features and info dictionary
        """
        batch_size, num_agents, _ = x.size()
        
        # Generate sparse mask if needed
        if self.use_sparse and positions is not None:
            sparse_mask = self._generate_sparse_mask(positions)
            if mask is not None:
                mask = mask * sparse_mask
            else:
                mask = sparse_mask
        
        # Apply attention with gradient checkpointing if needed
        if self.use_gradient_checkpointing and self.training:
            output, attention_weights = torch.utils.checkpoint.checkpoint(
                self.attention, x, mask
            )
        else:
            output, attention_weights = self.attention(x, mask)
        
        # Collect statistics
        info = {"attention_weights": attention_weights}
        
        if return_stats:
            info["stats"] = {
                "num_agents": num_agents,
                "memory_usage": self._estimate_memory_usage(batch_size, num_agents),
                "flops": self._estimate_flops(num_agents),
                "sparsity": self._compute_sparsity(attention_weights) if attention_weights is not None else 0
            }
        
        return output, info
    
    def _generate_sparse_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """Generate sparse attention mask based on positions
        
        Args:
            positions: Agent positions
            
        Returns:
            Sparse attention mask
        """
        batch_size, num_agents, _ = positions.size()
        
        # Compute distances
        distances = torch.cdist(positions, positions)
        
        # Keep top-k nearest neighbors
        k = max(1, int(num_agents * self.sparsity_ratio))
        _, indices = distances.topk(k, dim=-1, largest=False)
        
        # Create sparse mask
        mask = torch.zeros(batch_size, num_agents, num_agents, device=positions.device)
        
        for b in range(batch_size):
            for i in range(num_agents):
                mask[b, i, indices[b, i]] = 1.0
        
        return mask
    
    def _estimate_memory_usage(self, batch_size: int, num_agents: int) -> float:
        """Estimate memory usage in MB
        
        Args:
            batch_size: Batch size
            num_agents: Number of agents
            
        Returns:
            Estimated memory usage
        """
        if self.attention_type == "linear":
            # O(n) memory
            memory = batch_size * num_agents * self.embed_dim * 4 / 1e6
        elif self.attention_type == "sparse":
            # O(n * k) memory where k is sparsity
            k = int(num_agents * self.sparsity_ratio)
            memory = batch_size * num_agents * k * 4 / 1e6
        else:
            # O(n^2) memory
            memory = batch_size * num_agents * num_agents * 4 / 1e6
        
        return memory
    
    def _estimate_flops(self, num_agents: int) -> int:
        """Estimate FLOPs for attention computation
        
        Args:
            num_agents: Number of agents
            
        Returns:
            Estimated FLOPs
        """
        if self.attention_type == "linear":
            # O(n) complexity
            return num_agents * self.embed_dim * self.num_heads
        elif self.attention_type == "sparse":
            # O(n * k) complexity
            k = int(num_agents * self.sparsity_ratio)
            return num_agents * k * self.embed_dim * self.num_heads
        else:
            # O(n^2) complexity
            return num_agents * num_agents * self.embed_dim * self.num_heads
    
    def _compute_sparsity(self, attention_weights: torch.Tensor) -> float:
        """Compute sparsity of attention weights
        
        Args:
            attention_weights: Attention weight tensor
            
        Returns:
            Sparsity ratio
        """
        if attention_weights is None:
            return 0.0
        
        # Count non-zero elements
        non_zero = (attention_weights > 1e-6).float().sum()
        total = attention_weights.numel()
        
        return 1.0 - (non_zero / total)


class LinearAttention(nn.Module):
    """Linear attention mechanism with O(n) complexity"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feature_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """Initialize linear attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            feature_dim: Feature dimension for kernel approximation
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feature_dim = feature_dim or int(np.sqrt(self.head_dim))
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Feature mapping for kernel approximation
        self.feature_map = self._get_feature_map()
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def _get_feature_map(self):
        """Get feature mapping function"""
        def elu_feature_map(x):
            """ELU-based feature map for positive attention"""
            return F.elu(x) + 1
        
        return elu_feature_map
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with linear complexity
        
        Args:
            x: Input features [batch_size, seq_len, embed_dim]
            mask: Attention mask
            
        Returns:
            Output features and None (no attention weights in linear attention)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply feature map
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        
        # Rearrange for computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Apply mask to K and V if provided
        if mask is not None:
            # Expand mask for heads
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            K = K * mask.unsqueeze(-1)
            V = V * mask.unsqueeze(-1)
        
        # Compute KV (global context)
        KV = torch.einsum('bhsd,bhse->bhde', K, V)
        
        # Compute normalizer
        Z = 1 / (torch.einsum('bhsd,bhd->bhs', Q, K.sum(dim=2)) + 1e-6)
        
        # Compute attention output
        output = torch.einsum('bhsd,bhde,bhs->bhse', Q, KV, Z)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, None  # No attention weights in linear attention


class SparseAttention(nn.Module):
    """Sparse attention with configurable sparsity patterns"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sparsity_ratio: float = 0.1,
        use_topk: bool = True,
        dropout: float = 0.1
    ):
        """Initialize sparse attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            sparsity_ratio: Ratio of connections to keep
            use_topk: Whether to use top-k sparsity
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparsity_ratio = sparsity_ratio
        self.use_topk = use_topk
        
        # Use base attention with sparse masking
        self.attention = BaseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparse attention
        
        Args:
            x: Input features
            mask: Additional attention mask
            
        Returns:
            Output features and sparse attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Compute attention scores first to determine sparsity
        with torch.no_grad():
            # Quick score computation
            scores = torch.bmm(x, x.transpose(1, 2)) / np.sqrt(self.embed_dim)
            
            if self.use_topk:
                # Keep top-k values per row
                k = max(1, int(seq_len * self.sparsity_ratio))
                _, indices = scores.topk(k, dim=-1)
                
                # Create sparse mask
                sparse_mask = torch.zeros_like(scores)
                sparse_mask.scatter_(-1, indices, 1.0)
            else:
                # Threshold-based sparsity
                threshold = torch.quantile(scores.flatten(), 1 - self.sparsity_ratio)
                sparse_mask = (scores > threshold).float()
        
        # Convert to attention mask format
        sparse_attn_mask = torch.where(
            sparse_mask > 0,
            torch.zeros_like(sparse_mask),
            torch.full_like(sparse_mask, float('-inf'))
        )
        
        # Combine with existing mask if provided
        if mask is not None:
            final_mask = sparse_attn_mask + mask
        else:
            final_mask = sparse_attn_mask
        
        # Apply attention with sparse mask
        output, attention_weights = self.attention(
            x, x, x, attn_mask=final_mask
        )
        
        return output, attention_weights


class LocalAttention(nn.Module):
    """Local attention with sliding window"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 5,
        overlap: int = 1,
        dropout: float = 0.1
    ):
        """Initialize local attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            window_size: Size of local window
            overlap: Overlap between windows
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        
        self.attention = BaseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with local windowed attention
        
        Args:
            x: Input features
            mask: Attention mask
            
        Returns:
            Output features and attention weights
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Pad sequence for even windows
        pad_len = (self.window_size - seq_len % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len, 0, pad_len), value=float('-inf'))
        
        padded_len = x.size(1)
        num_windows = (padded_len - self.window_size) // self.stride + 1
        
        # Process each window
        outputs = []
        all_weights = []
        
        for i in range(num_windows):
            start = i * self.stride
            end = start + self.window_size
            
            # Extract window
            window_x = x[:, start:end]
            window_mask = mask[:, start:end, start:end] if mask is not None else None
            
            # Apply attention
            window_out, window_weights = self.attention(
                window_x, window_x, window_x, attn_mask=window_mask
            )
            
            outputs.append(window_out)
            all_weights.append(window_weights)
        
        # Combine windows with overlap averaging
        output = torch.zeros(batch_size, padded_len, embed_dim, device=x.device)
        counts = torch.zeros(batch_size, padded_len, 1, device=x.device)
        
        for i, window_out in enumerate(outputs):
            start = i * self.stride
            end = start + self.window_size
            output[:, start:end] += window_out
            counts[:, start:end] += 1
        
        output = output / (counts + 1e-6)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :seq_len]
        
        # Aggregate attention weights
        if all_weights[0] is not None:
            # Simple average for now
            attention_weights = sum(all_weights) / len(all_weights)
        else:
            attention_weights = None
        
        return output, attention_weights


class EfficientMultiHeadAttention(BaseMultiHeadAttention):
    """Memory-efficient multi-head attention with optimizations"""
    
    def __init__(self, *args, use_flash_attention: bool = True, **kwargs):
        """Initialize efficient attention
        
        Args:
            use_flash_attention: Whether to use flash attention if available
            *args, **kwargs: Arguments for base class
        """
        super().__init__(*args, **kwargs)
        
        self.use_flash_attention = use_flash_attention
        
        # Check if flash attention is available
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.has_flash_attn = True
        except ImportError:
            self.has_flash_attn = False
            if use_flash_attention:
                logger.warning("Flash attention requested but not available")
    
    def _compute_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention with optimizations
        
        Args:
            Q, K, V: Query, Key, Value tensors
            key_padding_mask: Padding mask
            attn_mask: Attention mask
            
        Returns:
            Attention output and weights
        """
        if self.use_flash_attention and self.has_flash_attn and attn_mask is None:
            # Use flash attention for better memory efficiency
            # Note: Flash attention doesn't return attention weights
            output = self.flash_attn_func(Q, K, V, self.dropout if self.training else 0.0)
            return output, None
        else:
            # Fall back to standard implementation
            return super()._compute_attention(Q, K, V, key_padding_mask, attn_mask)