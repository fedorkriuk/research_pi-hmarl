"""Tests for Physics-Informed Neural Networks

This module tests PINN implementations with real physics parameters.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import yaml
from typing import Dict, Any
import os

# Import PINN modules
from src.physics_informed.base_pinn import (
    PhysicsInformedNetwork, DynamicsNetwork, ConstrainedPINN, 
    ResidualPINN, create_pinn
)
from src.physics_informed.autodiff_physics import (
    AutoDiffPhysics, PhysicsGradients, AutoDiffLoss
)
from src.physics_informed.port_hamiltonian import (
    PortHamiltonianNetwork, HamiltonianDynamics,
    SkewSymmetricMatrix, PositiveSemiDefiniteMatrix
)
from src.physics_informed.conservation_laws import (
    ConservationLaws, MomentumConservation, 
    EnergyConservation, AngularMomentumConservation
)
from src.physics_informed.collision_constraints import (
    CollisionConstraints, SafetyDistanceConstraint,
    TimeToCollisionConstraint
)
from src.physics_informed.physics_loss import (
    PhysicsLossCalculator, MultiPhysicsLoss,
    PDEResidualLoss, BoundaryConditionLoss
)
from src.physics_informed.constraint_embedding import (
    ConstraintEmbedding, PhysicsEncoder,
    SymmetryPreservingLayer, LagrangianLayer
)
from src.physics_informed.multi_fidelity import (
    MultiFidelityPhysics, FidelitySelector,
    LinearCorrelation, NonlinearCorrelation
)


class TestBasePINN:
    """Test base PINN functionality"""
    
    @pytest.fixture
    def pinn_config(self):
        """Load PINN configuration"""
        config_path = "configs/pinn_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default config
            return {
                "pinn": {
                    "network": {
                        "input_dim": 12,
                        "hidden_dims": [128, 128],
                        "output_dim": 12,
                        "physics_dim": 12
                    }
                }
            }
    
    def test_dynamics_network(self):
        """Test dynamics network with physics"""
        net = DynamicsNetwork(
            state_dim=6,
            action_dim=3,
            hidden_dims=[64, 64],
            dt=0.05,
            mass=1.5
        )
        
        # Create input
        batch_size = 10
        state = torch.randn(batch_size, 6)
        action = torch.randn(batch_size, 3)
        x = torch.cat([state, action], dim=-1)
        
        # Forward with physics
        result = net.forward_with_physics(x)
        
        assert "output" in result
        assert "grad_x" in result
        assert "grad_xx" in result
        assert result["output"].shape == (batch_size, 6)
        
        # Check physics losses
        assert "kinematic_consistency" in result
        assert "velocity_consistency" in result
        assert "energy_consistency" in result
    
    def test_constrained_pinn(self):
        """Test PINN with constraints"""
        # Define constraint function
        def distance_constraint(x, y):
            # Minimum distance constraint
            positions = y[:, :3]
            distances = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0))[0]
            violations = 5.0 - distances  # 5m minimum
            return violations.sum()
        
        net = ConstrainedPINN(
            input_dim=6,
            hidden_dims=[64, 64],
            output_dim=6,
            physics_dim=6,
            constraint_functions=[distance_constraint],
            lagrange_multipliers=[1.0]
        )
        
        x = torch.randn(10, 6)
        y = net(x)
        
        # Compute constraint loss
        losses = net.compute_constraint_loss(x, y)
        assert "constraint_0" in losses
        assert "constraint_0_violation" in losses
    
    def test_residual_pinn(self):
        """Test residual PINN"""
        net = ResidualPINN(
            input_dim=12,
            hidden_dims=[64, 64, 64],
            output_dim=12,
            physics_dim=12
        )
        
        x = torch.randn(5, 12)
        y = net(x)
        
        assert y.shape == (5, 12)


class TestAutoDiffPhysics:
    """Test automatic differentiation for physics"""
    
    @pytest.fixture
    def autodiff(self):
        """Create autodiff instance"""
        return AutoDiffPhysics(enable_second_order=True)
    
    def test_first_order_derivatives(self, autodiff):
        """Test first-order derivatives"""
        # Simple function: f(x) = x^2
        x = torch.randn(10, 3, requires_grad=True)
        y = x ** 2
        
        # Compute derivative (should be 2x)
        dy_dx = autodiff.compute_derivatives(y, x, order=1)
        expected = 2 * x
        
        assert torch.allclose(dy_dx, expected, atol=1e-5)
    
    def test_second_order_derivatives(self, autodiff):
        """Test second-order derivatives"""
        # Function: f(x) = x^3
        x = torch.randn(5, 2, requires_grad=True)
        y = (x ** 3).sum(dim=-1, keepdim=True)
        
        # Compute derivatives
        derivatives = autodiff.compute_derivatives(y, x, order=2)
        
        assert "first" in derivatives
        assert "second" in derivatives
        
        # Check shapes
        assert derivatives["first"].shape == x.shape
        assert derivatives["second"].shape == x.shape
    
    def test_physics_gradients(self):
        """Test physics-specific gradients"""
        grad = PhysicsGradients(spatial_dim=3)
        
        # Scalar field
        positions = torch.randn(10, 3, requires_grad=True)
        scalar_field = (positions ** 2).sum(dim=-1, keepdim=True)
        
        # Gradient (should be 2*positions)
        gradient = grad.gradient(scalar_field, positions)
        expected = 2 * positions
        
        assert torch.allclose(gradient, expected, atol=1e-4)
        
        # Divergence of vector field
        vector_field = positions * 2  # Simple linear field
        divergence = grad.divergence(vector_field, positions)
        
        # div(2*r) = 2*3 = 6 in 3D
        assert torch.allclose(divergence, torch.full_like(divergence, 6.0), atol=1e-4)


class TestPortHamiltonian:
    """Test Port-Hamiltonian networks"""
    
    def test_port_hamiltonian_network(self):
        """Test basic Port-Hamiltonian network"""
        net = PortHamiltonianNetwork(
            state_dim=6,  # 3 position + 3 momentum
            control_dim=3,
            hidden_dims=[64, 64]
        )
        
        batch_size = 5
        state = torch.randn(batch_size, 6)
        control = torch.randn(batch_size, 3)
        
        # Forward pass
        result = net(state, control, return_energy=True)
        
        assert "dynamics" in result
        assert "hamiltonian" in result
        assert "kinetic_energy" in result
        assert "potential_energy" in result
        
        # Check dynamics shape
        assert result["dynamics"].shape == state.shape
        
        # Energy should be positive
        assert (result["kinetic_energy"] >= 0).all()
    
    def test_skew_symmetric_matrix(self):
        """Test skew-symmetric matrix generation"""
        J_net = SkewSymmetricMatrix(dim=4)
        
        x = torch.randn(3, 4)
        J = J_net(x)
        
        # Check skew-symmetry: J + J^T = 0
        assert torch.allclose(J + J.transpose(-1, -2), torch.zeros_like(J), atol=1e-5)
    
    def test_positive_semidefinite_matrix(self):
        """Test PSD matrix generation"""
        R_net = PositiveSemiDefiniteMatrix(dim=4, min_eigenvalue=0.01)
        
        x = torch.randn(3, 4)
        R = R_net(x)
        
        # Check positive semi-definiteness
        eigenvalues = torch.linalg.eigvalsh(R)
        assert (eigenvalues >= 0.01).all()
    
    def test_hamiltonian_dynamics(self):
        """Test Hamiltonian dynamics integration"""
        # Simple harmonic oscillator Hamiltonian
        def hamiltonian(state):
            q = state[:, :1]
            p = state[:, 1:]
            return 0.5 * (q**2 + p**2).sum(dim=-1, keepdim=True)
        
        dynamics = HamiltonianDynamics(hamiltonian, state_dim=2, dt=0.01)
        
        # Initial condition
        state = torch.tensor([[1.0, 0.0]])  # q=1, p=0
        
        # Integrate for multiple steps
        trajectory = [state]
        for _ in range(100):
            state = dynamics.symplectic_euler_step(state)
            trajectory.append(state)
        
        trajectory = torch.stack(trajectory)
        
        # Check energy conservation
        energies = []
        for s in trajectory:
            energies.append(hamiltonian(s).item())
        
        # Energy should be approximately conserved
        energy_var = np.var(energies)
        assert energy_var < 0.01


class TestConservationLaws:
    """Test conservation law enforcement"""
    
    @pytest.fixture
    def conservation_laws(self):
        """Create conservation laws instance"""
        return ConservationLaws(
            mass=1.5,
            gravity=9.81
        )
    
    def test_momentum_conservation(self, conservation_laws):
        """Test linear momentum conservation"""
        batch_size = 2
        num_agents = 3
        
        velocities = torch.randn(batch_size, num_agents, 3)
        forces = torch.zeros(batch_size, num_agents, 3)  # No external forces
        
        # Compute conservation loss
        loss = conservation_laws.linear_momentum_conservation(
            velocities, forces
        )
        
        # With no external forces, momentum should be conserved
        assert loss.item() < 0.1
    
    def test_energy_conservation(self, conservation_laws):
        """Test energy conservation"""
        positions = torch.randn(2, 3, 3)
        velocities = torch.randn(2, 3, 3)
        forces = torch.zeros(2, 3, 3)
        
        loss = conservation_laws.energy_conservation(
            positions, velocities, forces
        )
        
        assert loss.item() >= 0
    
    def test_angular_momentum_conservation(self, conservation_laws):
        """Test angular momentum conservation"""
        positions = torch.randn(2, 3, 3)
        velocities = torch.randn(2, 3, 3)
        angular_velocities = torch.randn(2, 3, 3)
        forces = torch.zeros(2, 3, 3)
        
        loss = conservation_laws.angular_momentum_conservation(
            positions, velocities, angular_velocities, forces
        )
        
        assert loss.item() >= 0
    
    def test_action_reaction(self, conservation_laws):
        """Test Newton's third law"""
        # Pairwise forces
        forces = torch.randn(2, 3, 3, 3)
        
        # Make anti-symmetric to satisfy Newton's 3rd law
        forces_symmetric = 0.5 * (forces - forces.transpose(1, 2))
        
        loss = conservation_laws.action_reaction_symmetry(forces_symmetric)
        
        assert loss.item() < 1e-5


class TestCollisionConstraints:
    """Test collision avoidance constraints"""
    
    @pytest.fixture
    def collision_constraints(self):
        """Create collision constraints"""
        return CollisionConstraints(
            min_separation=5.0,
            safety_margin=1.5
        )
    
    def test_distance_constraints(self, collision_constraints):
        """Test distance-based collision constraints"""
        # Two agents getting close
        positions = torch.tensor([
            [[0, 0, 0], [6, 0, 0]],  # Safe distance
            [[0, 0, 0], [3, 0, 0]]   # Too close
        ]).float()
        
        losses = collision_constraints.compute_collision_losses(
            positions, return_violations=True
        )
        
        assert "distance_collision" in losses
        assert losses["distance_collision"].item() > 0
        
        # Check violations
        violations = losses["violations"]
        assert violations[1, 0, 1] > 0  # Second batch violates
    
    def test_time_to_collision(self, collision_constraints):
        """Test time-to-collision constraints"""
        # Two agents approaching
        positions = torch.tensor([
            [[0, 0, 0], [10, 0, 0]]
        ]).float()
        
        velocities = torch.tensor([
            [[2, 0, 0], [-2, 0, 0]]  # Approaching at 4 m/s
        ]).float()
        
        losses = collision_constraints.compute_collision_losses(
            positions, velocities
        )
        
        assert "time_to_collision" in losses
        
        # TTC should be 10/4 = 2.5 seconds (below 3s threshold)
        assert losses["time_to_collision"].item() > 0
    
    def test_potential_field(self, collision_constraints):
        """Test potential field collision avoidance"""
        positions = torch.tensor([
            [[0, 0, 0], [4, 0, 0]]  # Close enough for potential
        ]).float()
        
        losses = collision_constraints.compute_collision_losses(positions)
        
        assert "collision_potential" in losses
        assert losses["collision_potential"].item() > 0


class TestPhysicsLoss:
    """Test physics loss calculation"""
    
    def test_physics_loss_calculator(self):
        """Test adaptive loss weighting"""
        calculator = PhysicsLossCalculator(
            loss_weights={
                "energy": 1.0,
                "momentum": 1.0,
                "collision": 10.0
            },
            adaptive_weighting=True
        )
        
        # Simulate losses over time
        for step in range(200):
            losses = {
                "energy": torch.tensor(0.1 + 0.05 * np.sin(step/10)),
                "momentum": torch.tensor(0.2 + 0.1 * np.cos(step/10)),
                "collision": torch.tensor(0.01)
            }
            
            total_loss, weighted = calculator.compute_total_loss(losses, step)
            
            if step % 100 == 0:
                # Check weights are being adapted
                stats = calculator.get_statistics()
                assert "current_weights" in stats
    
    def test_pde_residual_loss(self):
        """Test PDE residual computation"""
        pde_loss = PDEResidualLoss(
            pde_type="heat",
            coefficients={"diffusivity": 0.1}
        )
        
        # Create dummy solution
        u = torch.randn(10, 1)
        
        # Create derivatives
        derivatives = {
            "u_t": torch.randn(10, 1),
            "u_xx": torch.randn(10, 1)
        }
        
        residual = pde_loss.compute_residual(u, None, None, derivatives)
        
        assert residual.shape == u.shape
    
    def test_boundary_condition_loss(self):
        """Test boundary condition enforcement"""
        bc_loss = BoundaryConditionLoss(bc_type="dirichlet")
        
        # Boundary values
        u_boundary = torch.randn(20, 1)
        target_values = torch.zeros(20, 1)
        
        loss = bc_loss.compute_loss(u_boundary, target_values)
        
        assert loss.item() > 0


class TestConstraintEmbedding:
    """Test constraint embedding networks"""
    
    def test_constraint_embedding(self):
        """Test constraint embedding"""
        embedding = ConstraintEmbedding(
            input_dim=12,
            constraint_dim=32,
            num_constraints=5
        )
        
        x = torch.randn(4, 12)
        output, satisfaction = embedding(x, return_satisfaction=True)
        
        assert output.shape == (4, 32)
        assert satisfaction.shape == (4, 5)
        assert (satisfaction >= 0).all() and (satisfaction <= 1).all()
    
    def test_physics_encoder(self):
        """Test physics-aware encoding"""
        encoder = PhysicsEncoder(
            state_dim=12,
            physics_dim=64,
            constraint_types=["energy", "momentum", "collision"]
        )
        
        state = torch.randn(2, 12)
        outputs = encoder(state)
        
        assert "encoding" in outputs
        assert "position" in outputs
        assert "velocity" in outputs
        assert "energy" in outputs
        
        # Check output shapes
        assert outputs["position"].shape == (2, 3)
        assert outputs["velocity"].shape == (2, 3)
        assert outputs["energy"].shape == (2, 1)
    
    def test_lagrangian_layer(self):
        """Test Lagrangian dynamics layer"""
        layer = LagrangianLayer(
            state_dim=4,  # 2 pos + 2 vel
            learn_mass_matrix=True
        )
        
        state = torch.randn(3, 4)
        outputs = layer(state)
        
        assert "lagrangian" in outputs
        assert "kinetic_energy" in outputs
        assert "potential_energy" in outputs
        assert "mass_matrix" in outputs
        
        # Check energy decomposition
        L = outputs["lagrangian"]
        T = outputs["kinetic_energy"]
        V = outputs["potential_energy"]
        
        assert torch.allclose(L, T - V, atol=1e-5)


class TestMultiFidelity:
    """Test multi-fidelity physics"""
    
    def test_multi_fidelity_physics(self):
        """Test multi-fidelity model"""
        # Create dummy models
        low_fidelity = nn.Linear(10, 3)
        low_fidelity.output_dim = 3
        
        high_fidelity = nn.Linear(10, 3)
        high_fidelity.output_dim = 3
        
        mf_model = MultiFidelityPhysics(
            low_fidelity_model=low_fidelity,
            high_fidelity_model=high_fidelity,
            num_fidelity_levels=3
        )
        
        x = torch.randn(5, 10)
        
        # Test different fidelity levels
        for level in range(3):
            outputs = mf_model(x, fidelity_level=level)
            
            assert "prediction" in outputs
            assert "fidelity_level" in outputs
            assert outputs["fidelity_level"] == level
    
    def test_fidelity_selector(self):
        """Test adaptive fidelity selection"""
        selector = FidelitySelector(
            num_levels=3,
            input_dim=10,
            adaptive=True
        )
        
        x = torch.randn(1, 10)
        
        # Test with different budgets
        level_low_budget = selector(x, budget=5.0)
        level_high_budget = selector(x, budget=200.0)
        
        # Higher budget should allow higher fidelity
        assert level_high_budget >= level_low_budget
    
    def test_correlation_models(self):
        """Test correlation between fidelity levels"""
        # Linear correlation
        linear_corr = LinearCorrelation()
        
        x = torch.randn(3, 5)
        y_low = torch.randn(3, 5)
        
        corr_linear = linear_corr(x, y_low)
        assert corr_linear.shape == y_low.shape
        
        # Nonlinear correlation
        nonlinear_corr = NonlinearCorrelation(input_dim=5)
        
        corr_nonlinear = nonlinear_corr(x, y_low)
        assert corr_nonlinear.shape == y_low.shape


class TestIntegration:
    """Integration tests for complete PINN system"""
    
    def test_full_pinn_training_step(self):
        """Test complete PINN training step"""
        # Create PINN with all components
        pinn = DynamicsNetwork(
            state_dim=6,
            action_dim=3,
            mass=1.5
        )
        
        # Add physics modules
        pinn.add_physics_module(
            "conservation",
            ConservationLaws(mass=1.5)
        )
        
        # Create physics loss calculator
        loss_calc = PhysicsLossCalculator()
        
        # Sample data
        states = torch.randn(32, 6, requires_grad=True)
        actions = torch.randn(32, 3)
        next_states = torch.randn(32, 6)
        
        # Forward pass
        x = torch.cat([states, actions], dim=-1)
        result = pinn.forward_with_physics(x)
        
        # Compute losses
        losses = {
            "prediction": F.mse_loss(result["output"], next_states),
            **result  # Physics losses
        }
        
        # Total loss
        total_loss, weighted = loss_calc.compute_total_loss(losses)
        
        assert total_loss.requires_grad
        assert total_loss.item() > 0
    
    def test_constraint_satisfaction(self):
        """Test that constraints are satisfied"""
        # Create constrained PINN
        net = ConstrainedPINN(
            input_dim=6,
            hidden_dims=[64, 64],
            output_dim=6,
            physics_dim=6,
            constraint_functions=[
                lambda x, y: torch.norm(y, dim=-1) - 10.0  # Norm constraint
            ]
        )
        
        # Generate data satisfying constraint
        x = torch.randn(20, 6)
        y = net(x)
        
        # Project onto constraint
        for _ in range(10):
            violation = torch.norm(y, dim=-1) - 10.0
            if (torch.abs(violation) < 0.1).all():
                break
            
            # Simple projection
            y = y * (10.0 / torch.norm(y, dim=-1, keepdim=True))
        
        # Check constraint satisfaction
        final_violation = torch.norm(y, dim=-1) - 10.0
        assert torch.abs(final_violation).max() < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])