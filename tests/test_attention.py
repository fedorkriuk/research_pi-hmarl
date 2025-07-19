"""Tests for Multi-Head Attention Mechanisms

This module tests the attention mechanisms with real-world constraints
and scalability requirements.
"""

import pytest
import torch
import numpy as np
import yaml
from typing import Dict, Any
import time
import os

# Import attention modules
from src.attention.base_attention import (
    BaseMultiHeadAttention, MultiHeadSelfAttention, 
    GroupedQueryAttention
)
from src.attention.hierarchical_attention import (
    HierarchicalAttention, IntraClusterAttention,
    InterClusterAttention, TemporalHierarchicalAttention
)
from src.attention.physics_aware_attention import (
    PhysicsAwareAttention, EnergyAwareAttention,
    CollisionConstraintProcessor, CommunicationConstraintProcessor
)
from src.attention.scalable_attention import (
    ScalableAttention, LinearAttention, SparseAttention, LocalAttention
)
from src.attention.adaptive_attention import (
    AdaptiveAttentionSelector, ScenarioClassifier,
    ComplexityAnalyzer, ScenarioAdaptiveAttention
)
from src.attention.attention_visualizer import (
    AttentionVisualizer, AttentionPatternAnalyzer
)
from src.attention.attention_utils import (
    compute_attention_masks, create_distance_matrix,
    get_cluster_assignments, compute_attention_entropy,
    compute_attention_sparsity
)


class TestBaseAttention:
    """Test base attention mechanisms"""
    
    @pytest.fixture
    def attention_config(self):
        """Load attention configuration"""
        config_path = "configs/attention_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default config
            return {
                "attention": {
                    "base": {
                        "embed_dim": 256,
                        "num_heads": 8,
                        "dropout": 0.1
                    }
                }
            }
    
    @pytest.fixture
    def base_attention(self, attention_config):
        """Create base attention module"""
        config = attention_config["attention"]["base"]
        return BaseMultiHeadAttention(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=config["dropout"]
        )
    
    def test_initialization(self, base_attention):
        """Test attention initialization"""
        assert base_attention.embed_dim == 256
        assert base_attention.num_heads == 8
        assert base_attention.head_dim == 32
    
    def test_forward_pass(self, base_attention):
        """Test forward pass"""
        batch_size = 4
        seq_len = 10
        embed_dim = 256
        
        # Create input
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Forward pass
        output, weights = base_attention(x, x, x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert weights.shape == (batch_size, seq_len, seq_len)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)
    
    def test_attention_mask(self, base_attention):
        """Test attention masking"""
        batch_size = 2
        seq_len = 8
        embed_dim = 256
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        output, weights = base_attention(x, x, x, attn_mask=mask)
        
        # Check that upper triangular weights are zero
        upper_triangular = torch.triu(torch.ones_like(weights[0]), diagonal=1)
        assert torch.allclose(weights[0] * upper_triangular, torch.zeros_like(weights[0]), atol=1e-5)
    
    def test_grouped_query_attention(self):
        """Test grouped query attention"""
        gqa = GroupedQueryAttention(
            embed_dim=256,
            num_heads=8,
            num_groups=2
        )
        
        x = torch.randn(2, 10, 256)
        output = gqa(x, x, x)
        
        assert output.shape == x.shape


class TestHierarchicalAttention:
    """Test hierarchical attention mechanisms"""
    
    @pytest.fixture
    def hierarchical_attention(self):
        """Create hierarchical attention"""
        return HierarchicalAttention(
            embed_dim=256,
            num_heads=8,
            num_agents=20,
            cluster_size=5
        )
    
    def test_hierarchical_forward(self, hierarchical_attention):
        """Test hierarchical attention forward pass"""
        batch_size = 2
        num_agents = 20
        embed_dim = 256
        
        x = torch.randn(batch_size, num_agents, embed_dim)
        positions = torch.randn(batch_size, num_agents, 3) * 50
        
        output, info = hierarchical_attention(x, positions)
        
        assert output.shape == x.shape
        assert "intra_cluster" in info
        assert "inter_cluster" in info
    
    def test_cluster_assignment(self):
        """Test cluster assignment"""
        positions = torch.randn(2, 20, 3) * 100
        
        # K-means clustering
        clusters_kmeans = get_cluster_assignments(
            positions, cluster_size=5, method="kmeans"
        )
        assert clusters_kmeans.shape == (2, 20)
        assert clusters_kmeans.max() < 20 // 5 + 1
        
        # Grid clustering
        clusters_grid = get_cluster_assignments(
            positions, cluster_size=5, method="grid"
        )
        assert clusters_grid.shape == (2, 20)
    
    def test_temporal_hierarchical(self):
        """Test temporal hierarchical attention"""
        temporal_attn = TemporalHierarchicalAttention(
            embed_dim=256,
            num_heads=8,
            num_agents=10,
            history_len=5
        )
        
        x = torch.randn(2, 10, 256)
        history = torch.randn(2, 5, 10, 256)
        
        output, info = temporal_attn(x, history=history)
        
        assert output.shape == x.shape
        assert "temporal" in info


class TestPhysicsAwareAttention:
    """Test physics-aware attention mechanisms"""
    
    @pytest.fixture
    def physics_attention(self):
        """Create physics-aware attention"""
        return PhysicsAwareAttention(
            embed_dim=256,
            num_heads=8,
            num_agents=10,
            communication_range=100.0,
            min_safety_distance=5.0
        )
    
    def test_physics_constraints(self, physics_attention):
        """Test physics-based constraints"""
        batch_size = 2
        num_agents = 10
        
        x = torch.randn(batch_size, num_agents, 256)
        positions = torch.randn(batch_size, num_agents, 3) * 50
        velocities = torch.randn(batch_size, num_agents, 3) * 5
        
        output, info = physics_attention(x, positions, velocities)
        
        assert output.shape == x.shape
        assert "distance_matrix" in info
        assert "communication_mask" in info
        assert "collision_mask" in info
    
    def test_collision_constraint(self):
        """Test collision constraint processor"""
        processor = CollisionConstraintProcessor(min_safety_distance=5.0)
        
        # Create scenario with two agents approaching
        positions = torch.tensor([
            [[0, 0, 0], [10, 0, 0]],
            [[0, 0, 0], [3, 0, 0]]  # Too close
        ]).float()
        
        distances = create_distance_matrix(positions)
        mask = processor(distances)
        
        # Second batch should have masked attention between agents
        assert mask[1, 0, 1] < 1.0  # Agents too close
        assert mask[1, 1, 0] < 1.0
    
    def test_communication_constraint(self):
        """Test communication constraint processor"""
        processor = CommunicationConstraintProcessor(communication_range=50.0)
        
        positions = torch.tensor([
            [[0, 0, 0], [30, 0, 0], [100, 0, 0]]  # Last agent out of range
        ]).float()
        
        distances = create_distance_matrix(positions)
        mask = processor(distances)
        
        # Check communication feasibility
        assert mask[0, 0, 1] > 0.5  # In range
        assert mask[0, 0, 2] < 0.5  # Out of range
    
    def test_energy_aware_attention(self):
        """Test energy-aware attention"""
        energy_attn = EnergyAwareAttention(
            embed_dim=256,
            num_heads=8,
            num_agents=10,
            battery_capacity=5000.0,
            critical_battery_level=0.2
        )
        
        x = torch.randn(2, 10, 256)
        positions = torch.randn(2, 10, 3) * 50
        energy_levels = torch.rand(2, 10)  # Random energy levels
        energy_levels[0, 0] = 0.1  # Low energy agent
        
        output, info = energy_attn(
            x, positions, energy_levels=energy_levels
        )
        
        assert output.shape == x.shape
        if "energy_scale" in info:
            assert info["energy_scale"][0, 0] < 1.0  # Low energy scaling


class TestScalableAttention:
    """Test scalable attention mechanisms"""
    
    def test_linear_attention(self):
        """Test linear attention with O(n) complexity"""
        linear_attn = LinearAttention(
            embed_dim=256,
            num_heads=8
        )
        
        # Test with large number of agents
        x = torch.randn(2, 50, 256)
        
        start_time = time.time()
        output, _ = linear_attn(x)
        inference_time = time.time() - start_time
        
        assert output.shape == x.shape
        # Should be fast even with 50 agents
        assert inference_time < 0.1  # 100ms
    
    def test_sparse_attention(self):
        """Test sparse attention"""
        sparse_attn = SparseAttention(
            embed_dim=256,
            num_heads=8,
            sparsity_ratio=0.1
        )
        
        x = torch.randn(2, 30, 256)
        output, weights = sparse_attn(x)
        
        assert output.shape == x.shape
        
        # Check sparsity
        if weights is not None:
            sparsity = compute_attention_sparsity(weights)
            assert sparsity["sparsity_ratio"] > 0.5  # Should be sparse
    
    def test_local_attention(self):
        """Test local windowed attention"""
        local_attn = LocalAttention(
            embed_dim=256,
            num_heads=8,
            window_size=5
        )
        
        x = torch.randn(2, 20, 256)
        output, weights = local_attn(x)
        
        assert output.shape == x.shape
    
    def test_scalability_comparison(self):
        """Compare scalability of different attention types"""
        embed_dim = 256
        num_heads = 8
        batch_size = 1
        
        results = {}
        
        for num_agents in [10, 20, 50]:
            x = torch.randn(batch_size, num_agents, embed_dim)
            
            # Standard attention
            standard = BaseMultiHeadAttention(embed_dim, num_heads)
            start = time.time()
            standard(x, x, x)
            results[f"standard_{num_agents}"] = time.time() - start
            
            # Linear attention
            linear = LinearAttention(embed_dim, num_heads)
            start = time.time()
            linear(x)
            results[f"linear_{num_agents}"] = time.time() - start
            
            # Sparse attention
            sparse = SparseAttention(embed_dim, num_heads)
            start = time.time()
            sparse(x)
            results[f"sparse_{num_agents}"] = time.time() - start
        
        # Linear should scale better
        standard_scaling = results["standard_50"] / results["standard_10"]
        linear_scaling = results["linear_50"] / results["linear_10"]
        
        assert linear_scaling < standard_scaling  # Linear scales better


class TestAdaptiveAttention:
    """Test adaptive attention selection"""
    
    @pytest.fixture
    def adaptive_selector(self):
        """Create adaptive attention selector"""
        return AdaptiveAttentionSelector(
            embed_dim=256,
            num_heads_options=[4, 8, 16],
            num_agents=20
        )
    
    def test_scenario_classification(self):
        """Test scenario classification"""
        classifier = ScenarioClassifier(embed_dim=256, num_scenarios=3)
        
        x = torch.randn(2, 10, 256)
        complexity = torch.tensor([0.2, 0.8])
        
        probs = classifier(x, complexity)
        
        assert probs.shape == (2, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
    
    def test_complexity_analysis(self):
        """Test complexity analyzer"""
        analyzer = ComplexityAnalyzer(embed_dim=256, num_agents=20)
        
        # Simple scenario - agents clustered
        positions_simple = torch.randn(2, 10, 3) * 10
        velocities_simple = torch.randn(2, 10, 3) * 1
        
        # Complex scenario - agents spread out
        positions_complex = torch.randn(2, 20, 3) * 100
        velocities_complex = torch.randn(2, 20, 3) * 10
        
        x = torch.randn(2, 10, 256)
        
        complexity_simple = analyzer(x, positions_simple, velocities_simple)
        
        x_complex = torch.randn(2, 20, 256)
        complexity_complex = analyzer(x_complex, positions_complex, velocities_complex)
        
        # Complex scenario should have higher complexity
        assert complexity_complex.mean() > complexity_simple.mean()
    
    def test_adaptive_selection(self, adaptive_selector):
        """Test adaptive attention selection"""
        # Simple scenario
        x_simple = torch.randn(1, 5, 256)
        positions_simple = torch.randn(1, 5, 3) * 10
        
        output_simple, info_simple = adaptive_selector(
            x_simple, positions_simple, force_scenario="simple"
        )
        
        # Complex scenario
        x_complex = torch.randn(1, 20, 256)
        positions_complex = torch.randn(1, 20, 3) * 100
        velocities_complex = torch.randn(1, 20, 3) * 10
        
        output_complex, info_complex = adaptive_selector(
            x_complex, positions_complex, velocities_complex, 
            force_scenario="complex"
        )
        
        assert output_simple.shape == x_simple.shape
        assert output_complex.shape == x_complex.shape
        assert info_simple["scenario_type"] == "simple"
        assert info_complex["scenario_type"] == "complex"
    
    def test_performance_adaptation(self):
        """Test performance-based adaptation"""
        adaptive_attn = ScenarioAdaptiveAttention(
            embed_dim=256,
            num_agents=20,
            min_heads=4,
            max_heads=16
        )
        
        x = torch.randn(2, 15, 256)
        positions = torch.randn(2, 15, 3) * 50
        
        # Test with target latency
        output, info = adaptive_attn(
            x, positions, target_latency=20.0  # 20ms target
        )
        
        assert output.shape == x.shape
        
        # Get performance statistics
        stats = adaptive_attn.performance_monitor.get_statistics()
        assert "scenario_distribution" in stats


class TestAttentionUtils:
    """Test attention utility functions"""
    
    def test_attention_masks(self):
        """Test attention mask computation"""
        positions = torch.randn(2, 10, 3) * 50
        velocities = torch.randn(2, 10, 3) * 5
        energy_levels = torch.rand(2, 10)
        
        masks = compute_attention_masks(
            positions,
            communication_range=100.0,
            min_safety_distance=5.0,
            velocities=velocities,
            energy_levels=energy_levels
        )
        
        assert "communication" in masks
        assert "safety" in masks
        assert "collision" in masks
        assert "energy" in masks
        assert "combined" in masks
    
    def test_attention_entropy(self):
        """Test attention entropy computation"""
        # Uniform attention - high entropy
        uniform_weights = torch.ones(10, 10) / 10
        uniform_entropy = compute_attention_entropy(uniform_weights)
        
        # Focused attention - low entropy
        focused_weights = torch.zeros(10, 10)
        focused_weights[:, 0] = 1.0
        focused_entropy = compute_attention_entropy(focused_weights)
        
        assert uniform_entropy.mean() > focused_entropy.mean()
    
    def test_attention_sparsity(self):
        """Test attention sparsity metrics"""
        # Create sparse attention pattern
        weights = torch.zeros(10, 10)
        weights[range(10), range(10)] = 1.0  # Diagonal only
        
        sparsity = compute_attention_sparsity(weights)
        
        assert sparsity["sparsity_ratio"] > 0.8
        assert sparsity["effective_connections"] == 10


class TestAttentionVisualization:
    """Test attention visualization"""
    
    @pytest.fixture
    def visualizer(self):
        """Create attention visualizer"""
        return AttentionVisualizer()
    
    def test_pattern_analysis(self):
        """Test attention pattern analysis"""
        analyzer = AttentionPatternAnalyzer()
        
        # Hub pattern
        hub_weights = torch.zeros(10, 10)
        hub_weights[:, 0] = 0.8  # All attend to agent 0
        hub_weights[0, :] = 0.2  # Agent 0 attends to all
        
        patterns = analyzer.analyze_patterns(hub_weights)
        assert patterns["hub"]["score"] > 0.5
        
        # Uniform pattern
        uniform_weights = torch.ones(10, 10) / 10
        patterns = analyzer.analyze_patterns(uniform_weights)
        assert patterns["uniform"]["score"] > 0.5
    
    def test_visualization_creation(self, visualizer):
        """Test visualization creation"""
        weights = torch.rand(10, 10)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Create summary (without saving)
        summary = visualizer.create_attention_summary(weights)
        
        assert "statistics" in summary
        assert "agent_statistics" in summary
        assert "figures" in summary
        
        # Close figures
        visualizer.close_all()


class TestIntegration:
    """Integration tests for complete attention system"""
    
    def test_end_to_end_attention(self):
        """Test complete attention pipeline"""
        # Create multi-agent scenario
        num_agents = 15
        embed_dim = 256
        batch_size = 2
        
        # Agent features and positions
        features = torch.randn(batch_size, num_agents, embed_dim)
        positions = torch.randn(batch_size, num_agents, 3) * 75
        velocities = torch.randn(batch_size, num_agents, 3) * 5
        energy_levels = torch.rand(batch_size, num_agents)
        
        # Create adaptive attention system
        attention_system = AdaptiveAttentionSelector(
            embed_dim=embed_dim,
            num_heads_options=[4, 8, 16],
            num_agents=20
        )
        
        # Forward pass
        output, info = attention_system(
            features, positions, velocities
        )
        
        assert output.shape == features.shape
        assert "scenario_type" in info
        assert "complexity_score" in info
        
        # Analyze attention patterns
        if "attention_weights" in info:
            analyzer = AttentionPatternAnalyzer()
            patterns = analyzer.analyze_patterns(info["attention_weights"])
            assert "dominant_pattern" in patterns
    
    def test_real_time_performance(self):
        """Test real-time performance constraints"""
        # Configuration for real-time operation
        num_agents = 20
        embed_dim = 256
        target_latency = 20.0  # 20ms for 50Hz operation
        
        # Create scalable attention
        attention = ScalableAttention(
            embed_dim=embed_dim,
            num_heads=8,
            max_agents=num_agents,
            attention_type="linear"
        )
        
        # Test inference time
        x = torch.randn(1, num_agents, embed_dim)
        
        # Warm up
        for _ in range(5):
            attention(x)
        
        # Measure
        times = []
        for _ in range(10):
            start = time.time()
            attention(x)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        assert avg_time < target_latency, f"Average time {avg_time}ms exceeds target {target_latency}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])