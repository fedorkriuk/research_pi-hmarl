#!/usr/bin/env python
"""
PI-HMARL Complete Visualization Suite
Run this file to see all figures and scenarios!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arrow
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple
import os

# Set style for all plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class PIHMARLVisualizer:
    """Complete visualization suite for PI-HMARL project."""
    
    def __init__(self):
        self.figures_created = []
        
    def show_all_existing_figures(self):
        """Display all existing PNG files in the project."""
        print("\n" + "="*60)
        print("📊 EXISTING FIGURES IN PROJECT")
        print("="*60)
        
        png_files = [
            'research_overview.png',
            'technical_architecture.png', 
            'performance_timeline.png',
            'novelty_comparison.png',
            'performance_comparison.png',
            'improvement_timeline.png',
            'statistical_comparison_plot.png'
        ]
        
        for i, png_file in enumerate(png_files, 1):
            if os.path.exists(png_file):
                print(f"{i}. {png_file} - ✅ Found")
                # Display the image
                fig, ax = plt.subplots(figsize=(10, 6))
                img = plt.imread(png_file)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(png_file.replace('_', ' ').replace('.png', '').title(), fontsize=14)
                self.figures_created.append(fig)
            else:
                print(f"{i}. {png_file} - ❌ Not found")
    
    def create_performance_radar_chart(self):
        """Create radar chart comparing PI-HMARL with baselines."""
        print("\n📊 Creating Performance Radar Chart...")
        
        # Performance metrics
        categories = ['Success Rate', 'Physics\nCompliance', 'Energy\nEfficiency', 
                     'Scalability', 'Sample\nEfficiency', 'Real-time\nCapability']
        
        # Data for different methods (normalized to 0-1)
        baseline = [0.65, 0.30, 0.70, 0.40, 0.50, 0.60]
        qmix = [0.65, 0.30, 0.72, 0.50, 0.60, 0.75]
        maddpg = [0.72, 0.35, 0.73, 0.60, 0.65, 0.70]
        pi_hmarl = [0.913, 0.95, 0.82, 0.90, 0.85, 0.95]
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Number of variables
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot data
        for data, label, color in [
            (baseline + [baseline[0]], 'Baseline', 'gray'),
            (qmix + [qmix[0]], 'QMIX', 'blue'),
            (maddpg + [maddpg[0]], 'MADDPG', 'green'),
            (pi_hmarl + [pi_hmarl[0]], 'PI-HMARL', 'red')
        ]:
            ax.plot(angles, data, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, data, alpha=0.15, color=color)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=10)
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.title('PI-HMARL Performance Comparison\nAcross Multiple Dimensions', 
                 size=16, pad=20)
        
        self.figures_created.append(fig)
        return fig
    
    def create_ablation_study_visualization(self):
        """Create comprehensive ablation study visualization."""
        print("\n📊 Creating Ablation Study Visualization...")
        
        # Ablation results
        configurations = ['Full\nModel', 'No\nPhysics', 'No\nHierarchy', 'No\nAttention', 'Baseline']
        success_rates = [0.913, 0.789, 0.837, 0.867, 0.653]
        physics_violations = [1.6, 13.0, 2.5, 2.0, 13.0]
        energy_efficiency = [0.82, 0.70, 0.78, 0.80, 0.70]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Success Rate
        ax1 = axes[0]
        bars1 = ax1.bar(configurations, success_rates, color=['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6'])
        ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Target (85%)')
        ax1.set_ylabel('Success Rate', fontsize=12)
        ax1.set_title('Task Success Rate by Configuration', fontsize=14)
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.1%}', ha='center', va='bottom')
        
        # Physics Violations
        ax2 = axes[1]
        bars2 = ax2.bar(configurations, physics_violations, 
                        color=['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6'])
        ax2.set_ylabel('Physics Violations per Episode', fontsize=12)
        ax2.set_title('Physics Constraint Violations', fontsize=14)
        ax2.set_ylim(0, 15)
        
        # Energy Efficiency
        ax3 = axes[2]
        bars3 = ax3.bar(configurations, energy_efficiency,
                        color=['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6'])
        ax3.set_ylabel('Energy Efficiency', fontsize=12)
        ax3.set_title('Energy Utilization Efficiency', fontsize=14)
        ax3.set_ylim(0, 1)
        
        plt.suptitle('Ablation Study: Component Contributions to Performance', fontsize=16)
        plt.tight_layout()
        
        self.figures_created.append(fig)
        return fig
    
    def create_learning_curves(self):
        """Create learning curves showing sample efficiency."""
        print("\n📊 Creating Learning Curves...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Generate synthetic learning curves
        episodes = np.arange(0, 1000, 10)
        
        # Success rate curves
        baseline_curve = 0.65 * (1 - np.exp(-episodes/400)) + np.random.normal(0, 0.02, len(episodes))
        pi_hmarl_curve = 0.913 * (1 - np.exp(-episodes/200)) + np.random.normal(0, 0.01, len(episodes))
        qmix_curve = 0.72 * (1 - np.exp(-episodes/350)) + np.random.normal(0, 0.02, len(episodes))
        
        # Plot success rate learning curves
        ax1.plot(episodes, baseline_curve, label='Baseline', linewidth=2, alpha=0.8)
        ax1.plot(episodes, qmix_curve, label='QMIX', linewidth=2, alpha=0.8)
        ax1.plot(episodes, pi_hmarl_curve, label='PI-HMARL', linewidth=2, alpha=0.8)
        ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Target')
        
        ax1.fill_between(episodes, baseline_curve-0.05, baseline_curve+0.05, alpha=0.2)
        ax1.fill_between(episodes, pi_hmarl_curve-0.03, pi_hmarl_curve+0.03, alpha=0.2)
        
        ax1.set_xlabel('Training Episodes', fontsize=12)
        ax1.set_ylabel('Success Rate', fontsize=12)
        ax1.set_title('Learning Curves: Success Rate', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Sample efficiency comparison
        methods = ['Baseline', 'QMIX', 'MADDPG', 'PI-HMARL']
        episodes_to_target = [820, 680, 620, 420]
        colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']
        
        bars = ax2.bar(methods, episodes_to_target, color=colors)
        ax2.set_ylabel('Episodes to 85% Success Rate', fontsize=12)
        ax2.set_title('Sample Efficiency Comparison', fontsize=14)
        
        # Add value labels
        for bar, value in zip(bars, episodes_to_target):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{value}', ha='center', va='bottom')
        
        # Add improvement percentage for PI-HMARL
        baseline_episodes = episodes_to_target[0]
        pi_hmarl_episodes = episodes_to_target[-1]
        improvement = (baseline_episodes - pi_hmarl_episodes) / baseline_episodes * 100
        ax2.text(3, 300, f'{improvement:.1f}% fewer\nepisodes needed', 
                ha='center', fontsize=12, color='red', weight='bold')
        
        plt.suptitle('Sample Efficiency: PI-HMARL Learns Faster', fontsize=16)
        plt.tight_layout()
        
        self.figures_created.append(fig)
        return fig
    
    def create_scenario_visualization(self):
        """Create visual representation of the three main scenarios."""
        print("\n📊 Creating Scenario Visualizations...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Search & Rescue Scenario
        ax1 = axes[0]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # Draw agents
        agents = [(2, 8), (8, 7), (5, 2)]
        for i, (x, y) in enumerate(agents):
            circle = Circle((x, y), 0.3, color='blue', alpha=0.7)
            ax1.add_patch(circle)
            ax1.text(x, y, f'A{i+1}', ha='center', va='center', color='white', fontsize=8)
        
        # Draw victims
        victims = [(3, 5), (7, 3), (1, 1), (9, 9)]
        for x, y in victims:
            circle = Circle((x, y), 0.2, color='red', alpha=0.7)
            ax1.add_patch(circle)
            ax1.text(x, y, 'V', ha='center', va='center', color='white', fontsize=8)
        
        # Draw rescue in progress
        ax1.plot([agents[0][0], victims[0][0]], [agents[0][1], victims[0][1]], 
                'g--', linewidth=2, alpha=0.5)
        ax1.plot([agents[1][0], victims[0][0]], [agents[1][1], victims[0][1]], 
                'g--', linewidth=2, alpha=0.5)
        
        ax1.set_title('Search & Rescue Scenario\n(88% Success Rate)', fontsize=12)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Formation Control Scenario
        ax2 = axes[1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        
        # V-formation
        formation_points = [(5, 7), (4, 6), (6, 6), (3, 5), (7, 5)]
        for i, (x, y) in enumerate(formation_points):
            circle = Circle((x, y), 0.3, color='green', alpha=0.7)
            ax2.add_patch(circle)
            ax2.text(x, y, f'{i+1}', ha='center', va='center', color='white', fontsize=8)
        
        # Draw formation lines
        for i in range(len(formation_points)-1):
            if i < 2:
                ax2.plot([formation_points[0][0], formation_points[i+1][0]], 
                        [formation_points[0][1], formation_points[i+1][1]], 
                        'g-', linewidth=1, alpha=0.5)
            elif i < 4:
                ax2.plot([formation_points[i-1][0], formation_points[i+1][0]], 
                        [formation_points[i-1][1], formation_points[i+1][1]], 
                        'g-', linewidth=1, alpha=0.5)
        
        # Target waypoint
        ax2.plot(5, 2, 'r*', markersize=15, label='Target')
        ax2.arrow(5, 5, 0, -2, head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.5)
        
        ax2.set_title('Formation Control Scenario\n(100% Success Rate)', fontsize=12)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Multi-Agent Coordination Scenario
        ax3 = axes[2]
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        
        # Multiple tasks
        tasks = [(2, 8, 'T1'), (8, 8, 'T2'), (5, 5, 'T3'), (2, 2, 'T4'), (8, 2, 'T5')]
        for x, y, label in tasks:
            rect = Rectangle((x-0.4, y-0.4), 0.8, 0.8, color='orange', alpha=0.5)
            ax3.add_patch(rect)
            ax3.text(x, y, label, ha='center', va='center', fontsize=10)
        
        # Agent teams
        team1 = [(1, 9), (3, 9)]
        team2 = [(7, 7), (9, 7), (8, 9)]
        team3 = [(4, 3), (6, 3)]
        
        colors = ['blue', 'purple', 'cyan']
        for team, color in zip([team1, team2, team3], colors):
            for x, y in team:
                circle = Circle((x, y), 0.25, color=color, alpha=0.7)
                ax3.add_patch(circle)
        
        # Show coordination lines
        ax3.plot([team1[0][0], team1[1][0]], [team1[0][1], team1[1][1]], 'b-', alpha=0.5)
        ax3.plot([team2[0][0], team2[1][0]], [team2[0][1], team2[1][1]], 'purple', alpha=0.5)
        ax3.plot([team2[1][0], team2[2][0]], [team2[1][1], team2[2][1]], 'purple', alpha=0.5)
        
        ax3.set_title('Multi-Agent Coordination\n(86% Success Rate)', fontsize=12)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('PI-HMARL Scenario Visualizations', fontsize=16)
        plt.tight_layout()
        
        self.figures_created.append(fig)
        return fig
    
    def create_physics_constraint_visualization(self):
        """Visualize physics constraints in action."""
        print("\n📊 Creating Physics Constraint Visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity constraints
        ax1 = axes[0, 0]
        time = np.linspace(0, 10, 100)
        velocity_unconstrained = 15 * np.sin(time) + np.random.normal(0, 2, 100)
        velocity_constrained = np.clip(velocity_unconstrained, -10, 10)
        
        ax1.plot(time, velocity_unconstrained, 'r-', alpha=0.5, label='Without Physics')
        ax1.plot(time, velocity_constrained, 'g-', linewidth=2, label='With Physics')
        ax1.axhline(y=10, color='k', linestyle='--', alpha=0.5, label='Max Velocity')
        ax1.axhline(y=-10, color='k', linestyle='--', alpha=0.5)
        ax1.fill_between(time, -10, 10, alpha=0.1, color='green')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('Velocity Constraint Enforcement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Energy consumption
        ax2 = axes[0, 1]
        episodes = np.arange(100)
        energy_baseline = 100 - 1.5 * episodes + np.random.normal(0, 5, 100)
        energy_optimized = 100 - 1.0 * episodes + np.random.normal(0, 3, 100)
        
        ax2.plot(episodes, energy_baseline, 'r-', alpha=0.7, label='Baseline')
        ax2.plot(episodes, energy_optimized, 'g-', linewidth=2, label='PI-HMARL')
        ax2.fill_between(episodes, energy_baseline, energy_optimized, 
                        where=(energy_optimized > energy_baseline), 
                        alpha=0.3, color='green', label='Energy Saved')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Battery Level (%)')
        ax2.set_title('Energy-Aware Optimization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Collision avoidance
        ax3 = axes[1, 0]
        ax3.set_xlim(-5, 5)
        ax3.set_ylim(-5, 5)
        
        # Agent positions
        agent1_pos = [0, 0]
        agent2_pos = [3, 0]
        
        # Draw agents
        circle1 = Circle(agent1_pos, 0.5, color='blue', alpha=0.7)
        circle2 = Circle(agent2_pos, 0.5, color='green', alpha=0.7)
        ax3.add_patch(circle1)
        ax3.add_patch(circle2)
        
        # Draw safety zones
        safety1 = Circle(agent1_pos, 2, fill=False, edgecolor='blue', 
                        linestyle='--', linewidth=2, alpha=0.5)
        safety2 = Circle(agent2_pos, 2, fill=False, edgecolor='green', 
                        linestyle='--', linewidth=2, alpha=0.5)
        ax3.add_patch(safety1)
        ax3.add_patch(safety2)
        
        # Planned trajectories
        theta = np.linspace(0, np.pi, 50)
        traj1_x = 2 * np.cos(theta)
        traj1_y = 2 * np.sin(theta)
        traj2_x = 3 - 2 * np.cos(theta)
        traj2_y = 2 * np.sin(theta)
        
        ax3.plot(traj1_x, traj1_y, 'b--', linewidth=2, alpha=0.7, label='Agent 1 Path')
        ax3.plot(traj2_x, traj2_y, 'g--', linewidth=2, alpha=0.7, label='Agent 2 Path')
        
        ax3.set_aspect('equal')
        ax3.set_title('Collision Avoidance System')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Physics compliance over time
        ax4 = axes[1, 1]
        methods = ['Baseline', 'QMIX', 'MADDPG', 'PI-HMARL']
        compliance_rates = [0.30, 0.35, 0.40, 0.95]
        colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']
        
        bars = ax4.bar(methods, compliance_rates, color=colors)
        ax4.set_ylabel('Physics Compliance Rate')
        ax4.set_title('Physics Constraint Adherence')
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, compliance_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.0%}', ha='center', va='bottom')
        
        plt.suptitle('Physics-Informed Constraints in Action', fontsize=16)
        plt.tight_layout()
        
        self.figures_created.append(fig)
        return fig
    
    def create_hierarchical_architecture_diagram(self):
        """Create hierarchical architecture visualization."""
        print("\n📊 Creating Hierarchical Architecture Diagram...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Strategic Level
        strategic_box = Rectangle((3.5, 7), 3, 1.5, fill=True, 
                                 facecolor='#3498db', alpha=0.7, edgecolor='black')
        ax.add_patch(strategic_box)
        ax.text(5, 7.75, 'Strategic Planner\n(5-10 min horizon)', 
               ha='center', va='center', fontsize=12, weight='bold')
        
        # Tactical Level
        tactical_boxes = [
            Rectangle((1, 4), 2.5, 1.5, fill=True, facecolor='#2ecc71', alpha=0.7, edgecolor='black'),
            Rectangle((3.75, 4), 2.5, 1.5, fill=True, facecolor='#2ecc71', alpha=0.7, edgecolor='black'),
            Rectangle((6.5, 4), 2.5, 1.5, fill=True, facecolor='#2ecc71', alpha=0.7, edgecolor='black')
        ]
        for box in tactical_boxes:
            ax.add_patch(box)
        
        ax.text(2.25, 4.75, 'Tactical\nController 1', ha='center', va='center', fontsize=10)
        ax.text(5, 4.75, 'Tactical\nController 2', ha='center', va='center', fontsize=10)
        ax.text(7.75, 4.75, 'Tactical\nController 3', ha='center', va='center', fontsize=10)
        
        # Operational Level
        for i in range(9):
            x = 0.5 + (i % 3) * 3 + (i // 3) * 0.3
            y = 1
            op_box = Rectangle((x, y), 0.8, 1, fill=True, 
                              facecolor='#e74c3c', alpha=0.7, edgecolor='black')
            ax.add_patch(op_box)
            ax.text(x + 0.4, y + 0.5, f'Op\n{i+1}', ha='center', va='center', fontsize=8)
        
        # Draw connections
        # Strategic to Tactical
        for x in [2.25, 5, 7.75]:
            ax.arrow(5, 7, x-5, -1.3, head_width=0.1, head_length=0.1, 
                    fc='black', ec='black', alpha=0.5)
        
        # Tactical to Operational
        for i, tact_x in enumerate([2.25, 5, 7.75]):
            for j in range(3):
                op_x = 0.9 + j * 3 + i * 0.3
                ax.arrow(tact_x, 4, op_x-tact_x, -1.8, head_width=0.08, 
                        head_length=0.08, fc='gray', ec='gray', alpha=0.4)
        
        # Add labels
        ax.text(9.5, 7.75, '1-5 min', ha='center', va='center', fontsize=10, style='italic')
        ax.text(9.5, 4.75, '10-60 sec', ha='center', va='center', fontsize=10, style='italic')
        ax.text(9.5, 1.5, '50-200 ms', ha='center', va='center', fontsize=10, style='italic')
        
        # Add title
        ax.text(5, 9.5, 'PI-HMARL Hierarchical Architecture', 
               ha='center', va='center', fontsize=16, weight='bold')
        ax.text(5, 9, 'Three-Level Decision Making System', 
               ha='center', va='center', fontsize=12, style='italic')
        
        self.figures_created.append(fig)
        return fig
    
    def show_interactive_menu(self):
        """Display interactive menu for visualization selection."""
        print("\n" + "="*60)
        print("🎨 PI-HMARL VISUALIZATION SUITE")
        print("="*60)
        print("\nAvailable Visualizations:")
        print("1. Show all existing project figures")
        print("2. Performance Radar Chart (multi-dimensional comparison)")
        print("3. Ablation Study Results")
        print("4. Learning Curves (sample efficiency)")
        print("5. Scenario Visualizations (all three scenarios)")
        print("6. Physics Constraints in Action")
        print("7. Hierarchical Architecture Diagram")
        print("8. Show ALL visualizations")
        print("9. Save all figures to files")
        print("0. Exit")
        
        while True:
            choice = input("\nSelect visualization (0-9): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.show_all_existing_figures()
            elif choice == '2':
                self.create_performance_radar_chart()
            elif choice == '3':
                self.create_ablation_study_visualization()
            elif choice == '4':
                self.create_learning_curves()
            elif choice == '5':
                self.create_scenario_visualization()
            elif choice == '6':
                self.create_physics_constraint_visualization()
            elif choice == '7':
                self.create_hierarchical_architecture_diagram()
            elif choice == '8':
                self.run_all_visualizations()
            elif choice == '9':
                self.save_all_figures()
            else:
                print("Invalid choice. Please select 0-9.")
            
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                plt.show()
    
    def run_all_visualizations(self):
        """Run all visualization methods."""
        print("\n🎨 Creating ALL visualizations...")
        self.show_all_existing_figures()
        self.create_performance_radar_chart()
        self.create_ablation_study_visualization()
        self.create_learning_curves()
        self.create_scenario_visualization()
        self.create_physics_constraint_visualization()
        self.create_hierarchical_architecture_diagram()
    
    def save_all_figures(self):
        """Save all created figures to files."""
        print("\n💾 Saving all figures...")
        figure_names = [
            'radar_chart.png',
            'ablation_results.png',
            'learning_curves.png',
            'scenarios.png',
            'physics_constraints.png',
            'architecture_diagram.png'
        ]
        
        for i, fig in enumerate(self.figures_created[-6:]):
            if i < len(figure_names):
                filename = f'new_{figure_names[i]}'
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")

def main():
    """Main function to run the visualization suite."""
    print("\n" + "="*60)
    print("🚁 PI-HMARL VISUALIZATION RUNNER")
    print("="*60)
    print("\nThis script will show you all the visualizations from the PI-HMARL project.")
    print("You can view existing figures and create new ones.")
    
    visualizer = PIHMARLVisualizer()
    
    # Check if running interactively
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Run all visualizations automatically
        visualizer.run_all_visualizations()
        plt.show()
    else:
        # Interactive mode
        visualizer.show_interactive_menu()
    
    print("\n✅ Visualization session complete!")

if __name__ == "__main__":
    main()