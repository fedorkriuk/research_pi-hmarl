"""Base Multi-Head Attention Implementation

This module provides the foundation for all attention mechanisms in the system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseMultiHeadAttention(nn.Module):
    """Base multi-head attention mechanism"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize base multi-head attention
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias
            add_bias_kv: Whether to add bias to key/value projections
            add_zero_attn: Whether to add zero attention
            kdim: Key dimension (defaults to embed_dim)
            vdim: Value dimension (defaults to embed_dim)
            batch_first: Whether batch dimension is first
            device: Device for tensors
            dtype: Data type for tensors
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Optional bias for key/value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.zeros((1, 1, embed_dim)))
        else:
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)
        
        self.add_zero_attn = add_zero_attn
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
        
        logger.info(f"Initialized BaseMultiHeadAttention with {num_heads} heads")
    
    def _reset_parameters(self):
        """Initialize parameters"""
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            key_padding_mask: Mask for padded positions
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights
            
        Returns:
            Output tensor and optional attention weights
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len, _ = query.size()
        
        # Project to multi-head dimensions
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Add bias if specified
        if self.bias_k is not None and self.bias_v is not None:
            K = torch.cat([K, self.bias_k.expand(batch_size, -1, self.num_heads, self.head_dim)], dim=1)
            V = torch.cat([V, self.bias_v.expand(batch_size, -1, self.num_heads, self.head_dim)], dim=1)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention
        attn_output, attn_weights = self._compute_attention(
            Q, K, V, key_padding_mask, attn_mask
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights
        else:
            return output, None
    
    def _compute_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention
        
        Args:
            Q: Query tensor [batch, heads, seq_len, head_dim]
            K: Key tensor [batch, heads, seq_len, head_dim]
            V: Value tensor [batch, heads, seq_len, head_dim]
            key_padding_mask: Padding mask
            attn_mask: Attention mask
            
        Returns:
            Attention output and weights
        """
        batch_size, num_heads, seq_len, head_dim = Q.size()
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # Apply masks
        if attn_mask is not None:
            scores = scores + attn_mask
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Add zero attention if specified
        if self.add_zero_attn:
            batch_size = scores.size(0)
            num_heads = scores.size(1)
            src_len = scores.size(2)
            
            zero_attn_shape = (batch_size, num_heads, src_len, 1)
            zero_attn = torch.zeros(zero_attn_shape, dtype=scores.dtype, device=scores.device)
            scores = torch.cat([scores, zero_attn], dim=-1)
            
            zero_v = torch.zeros((batch_size, num_heads, 1, head_dim), 
                               dtype=V.dtype, device=V.device)
            V = torch.cat([V, zero_v], dim=2)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output, attn_weights
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get attention statistics for monitoring
        
        Returns:
            Dictionary of attention statistics
        """
        stats = {
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "head_dim": self.head_dim,
            "dropout": self.dropout
        }
        
        # Add weight statistics
        with torch.no_grad():
            stats["q_proj_norm"] = self.q_proj.weight.norm().item()
            stats["k_proj_norm"] = self.k_proj.weight.norm().item()
            stats["v_proj_norm"] = self.v_proj.weight.norm().item()
            stats["out_proj_norm"] = self.out_proj.weight.norm().item()
        
        return stats


class MultiHeadSelfAttention(BaseMultiHeadAttention):
    """Multi-head self-attention (query, key, value from same source)"""
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for self-attention
        
        Args:
            x: Input tensor
            key_padding_mask: Padding mask
            need_weights: Whether to return weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average weights
            
        Returns:
            Output and optional attention weights
        """
        return super().forward(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )


class MultiHeadCrossAttention(BaseMultiHeadAttention):
    """Multi-head cross-attention for attending to different sources"""
    
    def __init__(self, *args, **kwargs):
        """Initialize cross-attention"""
        super().__init__(*args, **kwargs)
        logger.info("Initialized MultiHeadCrossAttention")


class GroupedQueryAttention(nn.Module):
    """Grouped query attention for efficiency with large agent groups"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_groups: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """Initialize grouped query attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_groups: Number of query groups
            dropout: Dropout rate
            bias: Whether to use bias
        """
        super().__init__()
        
        assert num_heads % num_groups == 0, \
            "num_heads must be divisible by num_groups"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = embed_dim // num_heads
        
        # Grouped projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim // num_groups, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // num_groups, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with grouped queries
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Attention mask
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = query.size()
        
        # Project queries normally
        Q = self.q_proj(query).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Project keys/values with grouping
        K = self.k_proj(key).view(
            batch_size, -1, self.num_groups, self.head_dim
        ).transpose(1, 2)
        V = self.v_proj(value).view(
            batch_size, -1, self.num_groups, self.head_dim
        ).transpose(1, 2)
        
        # Repeat K, V for each head in group
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.out_proj(output)
        
        return output