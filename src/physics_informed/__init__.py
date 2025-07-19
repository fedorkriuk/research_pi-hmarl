"""Physics-Informed Neural Network (PINN) Integration

This module implements physics-informed neural networks that enforce physics
constraints in multi-agent learning using real physics parameters.
"""

from .base_pinn import PhysicsInformedNetwork, PINNLayer
from .autodiff_physics import AutoDiffPhysics, PhysicsGradients
from .port_hamiltonian import PortHamiltonianNetwork, HamiltonianDynamics
from .conservation_laws import (
    ConservationLaws,
    MomentumConservation,
    EnergyConservation,
    AngularMomentumConservation
)
from .collision_constraints import (
    CollisionConstraints,
    SafetyDistanceConstraint,
    TimeToCollisionConstraint
)
from .physics_loss import PhysicsLossCalculator, MultiPhysicsLoss
from .constraint_embedding import ConstraintEmbedding, PhysicsEncoder
from .multi_fidelity import MultiFidelityPhysics, FidelitySelector

__all__ = [
    "PhysicsInformedNetwork",
    "PINNLayer",
    "AutoDiffPhysics",
    "PhysicsGradients",
    "PortHamiltonianNetwork",
    "HamiltonianDynamics",
    "ConservationLaws",
    "MomentumConservation",
    "EnergyConservation",
    "AngularMomentumConservation",
    "CollisionConstraints",
    "SafetyDistanceConstraint",
    "TimeToCollisionConstraint",
    "PhysicsLossCalculator",
    "MultiPhysicsLoss",
    "ConstraintEmbedding",
    "PhysicsEncoder",
    "MultiFidelityPhysics",
    "FidelitySelector"
]