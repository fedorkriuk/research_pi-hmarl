"""Attention Visualization Tools

This module provides visualization capabilities for understanding attention
patterns and coordination behaviors in multi-agent systems.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
import logging
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import networkx as nx

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Visualizes attention patterns for interpretability"""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "viridis",
        style: str = "whitegrid"
    ):
        """Initialize attention visualizer
        
        Args:
            figsize: Figure size
            cmap: Colormap for attention weights
            style: Seaborn style
        """
        self.figsize = figsize
        self.cmap = cmap
        sns.set_style(style)
        
        # Visualization parameters
        self.node_size = 300
        self.edge_width_scale = 5.0
        self.arrow_scale = 20
        
        logger.info("Initialized AttentionVisualizer")
    
    def visualize_attention_matrix(
        self,
        attention_weights: torch.Tensor,
        agent_labels: Optional[List[str]] = None,
        title: str = "Attention Weights",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize attention weight matrix as heatmap
        
        Args:
            attention_weights: Attention weights [num_agents, num_agents]
            agent_labels: Labels for agents
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        weights = attention_weights.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        sns.heatmap(
            weights,
            annot=True,
            fmt=".2f",
            cmap=self.cmap,
            cbar_kws={"label": "Attention Weight"},
            xticklabels=agent_labels,
            yticklabels=agent_labels,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel("Attending To")
        ax.set_ylabel("Agent")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_graph(
        self,
        attention_weights: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        threshold: float = 0.1,
        title: str = "Attention Graph",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize attention as a directed graph
        
        Args:
            attention_weights: Attention weights [num_agents, num_agents]
            positions: Agent positions for layout [num_agents, 2 or 3]
            threshold: Minimum attention weight to show edge
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        weights = attention_weights.detach().cpu().numpy()
        num_agents = weights.shape[0]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(num_agents):
            G.add_node(i)
        
        # Add edges with weights above threshold
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and weights[i, j] > threshold:
                    G.add_edge(i, j, weight=weights[i, j])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Determine layout
        if positions is not None:
            pos_2d = positions.detach().cpu().numpy()[:, :2]
            pos = {i: pos_2d[i] for i in range(num_agents)}
        else:
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=self.node_size,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges with varying width based on attention weight
        edges = G.edges()
        weights_list = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            width=[w * self.edge_width_scale for w in weights_list],
            alpha=0.6,
            edge_color=weights_list,
            edge_cmap=plt.cm.get_cmap(self.cmap),
            arrows=True,
            arrowsize=self.arrow_scale,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.get_cmap(self.cmap),
            norm=plt.Normalize(vmin=0, vmax=weights.max())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Weight')
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_hierarchical_attention(
        self,
        attention_dict: Dict[str, torch.Tensor],
        positions: Optional[torch.Tensor] = None,
        cluster_assignments: Optional[torch.Tensor] = None,
        title: str = "Hierarchical Attention",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize hierarchical attention patterns
        
        Args:
            attention_dict: Dictionary with attention types
            positions: Agent positions
            cluster_assignments: Cluster assignment per agent
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0]*1.5, self.figsize[1]*1.5))
        axes = axes.flatten()
        
        # Visualize different attention levels
        for idx, (name, weights) in enumerate(attention_dict.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            if weights is not None and weights.numel() > 0:
                # Plot as heatmap
                weights_np = weights.detach().cpu().numpy()
                if weights_np.ndim > 2:
                    weights_np = weights_np.mean(axis=0)  # Average over batch
                
                im = ax.imshow(weights_np, cmap=self.cmap, aspect='auto')
                ax.set_title(f"{name} Attention")
                ax.set_xlabel("Attending To")
                ax.set_ylabel("Agent")
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, f"No {name} attention", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{name} Attention")
        
        # Hide unused subplots
        for idx in range(len(attention_dict), 4):
            axes[idx].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_flow(
        self,
        attention_weights: torch.Tensor,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        time_step: int = 0,
        title: str = "Attention Flow",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize attention flow in spatial context
        
        Args:
            attention_weights: Attention weights
            positions: Agent positions [num_agents, 3]
            velocities: Agent velocities
            time_step: Current time step
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        weights = attention_weights.detach().cpu().numpy()
        pos = positions.detach().cpu().numpy()
        num_agents = len(pos)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot agents as circles
        for i in range(num_agents):
            # Agent color based on total attention received
            attention_received = weights[:, i].sum()
            color = plt.cm.get_cmap(self.cmap)(attention_received)
            
            circle = Circle(
                pos[i, :2], 
                radius=0.5,
                color=color,
                alpha=0.7,
                zorder=2
            )
            ax.add_patch(circle)
            
            # Add agent label
            ax.text(pos[i, 0], pos[i, 1], str(i), 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add velocity arrow if available
            if velocities is not None:
                vel = velocities[i].detach().cpu().numpy()
                ax.arrow(
                    pos[i, 0], pos[i, 1],
                    vel[0] * 0.5, vel[1] * 0.5,
                    head_width=0.2, head_length=0.1,
                    fc='black', ec='black', alpha=0.5
                )
        
        # Plot attention connections
        lines = []
        colors = []
        linewidths = []
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and weights[i, j] > 0.05:
                    lines.append([pos[i, :2], pos[j, :2]])
                    colors.append(weights[i, j])
                    linewidths.append(weights[i, j] * 3)
        
        if lines:
            lc = LineCollection(
                lines,
                colors=colors,
                linewidths=linewidths,
                alpha=0.6,
                cmap=self.cmap,
                zorder=1
            )
            ax.add_collection(lc)
        
        # Set limits
        margin = 2
        ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
        ax.set_ylim(pos[:, 1].min() - margin, pos[:, 1].max() + margin)
        
        ax.set_aspect('equal')
        ax.set_title(f"{title} (t={time_step})")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for attention strength
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_evolution(
        self,
        attention_history: List[torch.Tensor],
        agent_idx: int = 0,
        title: str = "Attention Evolution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize how attention patterns evolve over time
        
        Args:
            attention_history: List of attention weights over time
            agent_idx: Agent to focus on
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Convert history to numpy
        history = [w.detach().cpu().numpy() for w in attention_history]
        time_steps = len(history)
        num_agents = history[0].shape[0]
        
        # Extract attention from/to specific agent
        attention_from = np.array([h[agent_idx, :] for h in history])
        attention_to = np.array([h[:, agent_idx] for h in history])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Plot attention from agent
        im1 = ax1.imshow(
            attention_from.T,
            aspect='auto',
            cmap=self.cmap,
            origin='lower'
        )
        ax1.set_ylabel("Target Agent")
        ax1.set_title(f"Attention FROM Agent {agent_idx}")
        plt.colorbar(im1, ax=ax1)
        
        # Plot attention to agent
        im2 = ax2.imshow(
            attention_to.T,
            aspect='auto',
            cmap=self.cmap,
            origin='lower'
        )
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Source Agent")
        ax2.set_title(f"Attention TO Agent {agent_idx}")
        plt.colorbar(im2, ax=ax2)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_attention_summary(
        self,
        attention_weights: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create comprehensive attention summary
        
        Args:
            attention_weights: Attention weights
            positions: Agent positions
            save_path: Base path for saving figures
            
        Returns:
            Dictionary with summary statistics and figures
        """
        weights = attention_weights.detach().cpu().numpy()
        num_agents = weights.shape[0]
        
        summary = {
            "num_agents": num_agents,
            "statistics": {},
            "figures": {}
        }
        
        # Compute statistics
        summary["statistics"]["mean_attention"] = weights.mean()
        summary["statistics"]["std_attention"] = weights.std()
        summary["statistics"]["max_attention"] = weights.max()
        summary["statistics"]["sparsity"] = (weights < 0.01).sum() / weights.size
        
        # Attention concentration (Gini coefficient)
        flat_weights = weights.flatten()
        sorted_weights = np.sort(flat_weights)
        n = len(sorted_weights)
        index = np.arange(1, n + 1)
        gini = (2 * index - n - 1).dot(sorted_weights) / (n * sorted_weights.sum())
        summary["statistics"]["gini_coefficient"] = gini
        
        # Per-agent statistics
        agent_stats = []
        for i in range(num_agents):
            agent_stat = {
                "agent_id": i,
                "attention_given": weights[i, :].sum(),
                "attention_received": weights[:, i].sum(),
                "self_attention": weights[i, i],
                "top_attended": weights[i, :].argsort()[-3:][::-1].tolist()
            }
            agent_stats.append(agent_stat)
        summary["agent_statistics"] = agent_stats
        
        # Create visualizations
        base_save = save_path.rsplit('.', 1)[0] if save_path else None
        
        # Matrix visualization
        fig1 = self.visualize_attention_matrix(
            attention_weights,
            title="Attention Weight Matrix",
            save_path=f"{base_save}_matrix.png" if base_save else None
        )
        summary["figures"]["matrix"] = fig1
        
        # Graph visualization
        if positions is not None:
            fig2 = self.visualize_attention_graph(
                attention_weights,
                positions=positions,
                title="Attention Network",
                save_path=f"{base_save}_graph.png" if base_save else None
            )
            summary["figures"]["graph"] = fig2
        
        return summary
    
    def close_all(self):
        """Close all matplotlib figures"""
        plt.close('all')


class AttentionPatternAnalyzer:
    """Analyzes attention patterns for insights"""
    
    def __init__(self):
        """Initialize pattern analyzer"""
        self.patterns = {
            "hub": self._detect_hub_pattern,
            "clustering": self._detect_clustering_pattern,
            "uniform": self._detect_uniform_pattern,
            "sparse": self._detect_sparse_pattern
        }
    
    def analyze_patterns(
        self,
        attention_weights: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze attention patterns
        
        Args:
            attention_weights: Attention weight matrix
            threshold: Threshold for pattern detection
            
        Returns:
            Dictionary of detected patterns
        """
        weights = attention_weights.detach().cpu().numpy()
        
        results = {}
        for pattern_name, detector in self.patterns.items():
            results[pattern_name] = detector(weights, threshold)
        
        # Determine dominant pattern
        scores = {k: v["score"] for k, v in results.items() if "score" in v}
        if scores:
            dominant = max(scores, key=scores.get)
            results["dominant_pattern"] = dominant
        
        return results
    
    def _detect_hub_pattern(
        self,
        weights: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Detect hub-and-spoke pattern
        
        Args:
            weights: Attention weights
            threshold: Detection threshold
            
        Returns:
            Pattern detection results
        """
        # Check if some agents receive disproportionate attention
        attention_received = weights.sum(axis=0)
        attention_given = weights.sum(axis=1)
        
        # Normalize
        attention_received = attention_received / attention_received.sum()
        attention_given = attention_given / attention_given.sum()
        
        # Find potential hubs
        mean_attention = 1.0 / len(weights)
        hubs = np.where(attention_received > mean_attention * 2)[0]
        
        score = len(hubs) / len(weights) if len(hubs) > 0 else 0
        
        return {
            "score": score,
            "hub_agents": hubs.tolist(),
            "max_centrality": attention_received.max()
        }
    
    def _detect_clustering_pattern(
        self,
        weights: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Detect clustering pattern
        
        Args:
            weights: Attention weights
            threshold: Detection threshold
            
        Returns:
            Pattern detection results
        """
        # Use spectral clustering approach
        # Symmetrize weights
        sym_weights = (weights + weights.T) / 2
        
        # Compute Laplacian
        degree = np.diag(sym_weights.sum(axis=1))
        laplacian = degree - sym_weights
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)
        
        # Count near-zero eigenvalues (indicates clusters)
        num_clusters = np.sum(eigenvalues < 0.1)
        
        score = min(1.0, num_clusters / (len(weights) / 3))
        
        return {
            "score": score,
            "estimated_clusters": num_clusters
        }
    
    def _detect_uniform_pattern(
        self,
        weights: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Detect uniform attention pattern
        
        Args:
            weights: Attention weights
            threshold: Detection threshold
            
        Returns:
            Pattern detection results
        """
        # Check if attention is uniformly distributed
        # Remove diagonal
        off_diagonal = weights[~np.eye(weights.shape[0], dtype=bool)]
        
        # Compute coefficient of variation
        if off_diagonal.mean() > 0:
            cv = off_diagonal.std() / off_diagonal.mean()
        else:
            cv = float('inf')
        
        # Lower CV indicates more uniform
        score = max(0, 1 - cv)
        
        return {
            "score": score,
            "coefficient_of_variation": cv,
            "mean_attention": off_diagonal.mean()
        }
    
    def _detect_sparse_pattern(
        self,
        weights: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Detect sparse attention pattern
        
        Args:
            weights: Attention weights
            threshold: Detection threshold
            
        Returns:
            Pattern detection results
        """
        # Count near-zero weights
        sparse_count = np.sum(weights < threshold)
        total_count = weights.size
        
        sparsity = sparse_count / total_count
        
        return {
            "score": sparsity,
            "sparsity_ratio": sparsity,
            "num_connections": total_count - sparse_count
        }


def create_attention_animation(
    attention_history: List[torch.Tensor],
    positions_history: List[torch.Tensor],
    save_path: str = "attention_animation.gif",
    fps: int = 10
):
    """Create animation of attention evolution
    
    Args:
        attention_history: List of attention weights over time
        positions_history: List of positions over time
        save_path: Path to save animation
        fps: Frames per second
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        visualizer = AttentionVisualizer()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            visualizer.visualize_attention_flow(
                attention_history[frame],
                positions_history[frame],
                time_step=frame,
                title=f"Attention Flow (t={frame})"
            )
        
        anim = FuncAnimation(
            fig, update,
            frames=len(attention_history),
            interval=1000/fps,
            repeat=True
        )
        
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        
        plt.close()
        
        logger.info(f"Saved attention animation to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to create animation: {e}")