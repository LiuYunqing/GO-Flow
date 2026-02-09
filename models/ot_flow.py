"""
Optimal Transport Flow for Molecular Conformational Space
Implements OT and Schrödinger Bridge approximations for internal coordinate dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class OptimalTransportFlow(nn.Module):
    """
    Implements Optimal Transport (OT) flow for conformational space.
    Uses entropic regularization and neural network approximations for tractability.
    """
    
    def __init__(self, dim_internal: int, hidden_dim: int = 128, 
                 epsilon: float = 0.1, num_iterations: int = 5):
        """
        Args:
            dim_internal: Dimension of internal coordinate space
            hidden_dim: Hidden dimension for neural networks
            epsilon: Entropic regularization parameter
            num_iterations: Number of Sinkhorn iterations for OT
        """
        super().__init__()
        
        self.dim_internal = dim_internal
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        
        # Dual potential networks for OT (Kantorovich potentials)
        self.potential_f = self._build_potential_net(dim_internal, hidden_dim)
        self.potential_g = self._build_potential_net(dim_internal, hidden_dim)
        
        # Velocity field network
        self.velocity_net = nn.Sequential(
            nn.Linear(dim_internal + 1, hidden_dim),  # +1 for time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim_internal)
        )
        
        # Cost function network (learned transport cost)
        self.cost_net = nn.Sequential(
            nn.Linear(2 * dim_internal, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive cost
        )
        
        # Schrödinger Bridge correction network
        self.bridge_net = nn.Sequential(
            nn.Linear(dim_internal + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim_internal)
        )
    
    def _build_potential_net(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Build a network for learning OT dual potentials."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def compute_transport_cost(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """
        Compute transport cost between internal coordinate configurations.
        
        Args:
            z0: Source internal coordinates [batch, dim_internal]
            z1: Target internal coordinates [batch, dim_internal]
            
        Returns:
            Transport cost matrix [batch, batch]
        """
        batch_size = z0.shape[0]
        
        # Expand for pairwise computation
        z0_exp = z0.unsqueeze(1).expand(-1, batch_size, -1)
        z1_exp = z1.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate pairs
        z_pairs = torch.cat([z0_exp, z1_exp], dim=-1)
        
        # Compute learned cost
        cost = self.cost_net(z_pairs).squeeze(-1)
        
        # Add L2 cost as baseline
        l2_cost = torch.sum((z0_exp - z1_exp) ** 2, dim=-1)
        
        return cost + 0.1 * l2_cost
    
    def sinkhorn_algorithm(self, cost_matrix: torch.Tensor, 
                          a: Optional[torch.Tensor] = None,
                          b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Entropic regularized OT using Sinkhorn-Knopp algorithm.
        
        Args:
            cost_matrix: Cost matrix [n, m]
            a: Source distribution weights [n]
            b: Target distribution weights [m]
            
        Returns:
            Transport plan [n, m]
        """
        n, m = cost_matrix.shape
        
        # Default to uniform distributions
        if a is None:
            a = torch.ones(n, device=cost_matrix.device) / n
        if b is None:
            b = torch.ones(m, device=cost_matrix.device) / m
        
        # Initialize dual variables
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        # Gibbs kernel
        K = torch.exp(-cost_matrix / self.epsilon)
        
        # Sinkhorn iterations
        for _ in range(self.num_iterations):
            u = a / (K @ v + 1e-8)
            v = b / (K.T @ u + 1e-8)
        
        # Transport plan
        pi = torch.diag(u) @ K @ torch.diag(v)
        
        return pi
    
    def compute_ot_path(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal transport path in conformational space.
        
        Args:
            z0: Starting internal coordinates [batch, dim_internal]
            z1: Target internal coordinates [batch, dim_internal]
            t: Time parameter [batch, 1]
            
        Returns:
            Interpolated internal coordinates at time t [batch, dim_internal]
        """
        batch_size = z0.shape[0]
        
        # Compute transport plan
        cost_matrix = self.compute_transport_cost(z0, z1)
        pi = self.sinkhorn_algorithm(cost_matrix)
        
        # Displacement interpolation (McCann interpolation)
        # This is a simplified version - full implementation would use geodesic
        # For now, use weighted barycentric interpolation based on transport plan
        
        # Normalize transport plan rows
        pi_normalized = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute barycenter at time t
        z_t = torch.zeros_like(z0)
        for i in range(batch_size):
            # Weighted average of targets based on transport from source i
            weights = pi_normalized[i, :]
            z_t[i] = (1 - t[i]) * z0[i] + t[i] * (weights @ z1)
        
        return z_t
    
    def compute_ot_velocity(self, z0: torch.Tensor, z1: torch.Tensor, 
                           z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field for optimal transport.
        
        Args:
            z0: Starting internal coordinates [batch, dim_internal]
            z1: Target internal coordinates [batch, dim_internal]
            z_t: Current internal coordinates [batch, dim_internal]
            t: Time parameter [batch, 1]
            
        Returns:
            Velocity field [batch, dim_internal]
        """
        # Concatenate current state with time
        z_t_with_time = torch.cat([z_t, t], dim=-1)
        
        # Predict velocity using neural network
        v_neural = self.velocity_net(z_t_with_time)
        
        # Only compute OT gradients if in training mode or gradients are enabled
        if z_t.requires_grad:
            try:
                # Compute gradient of dual potentials (from OT theory)
                f_grad = torch.autograd.grad(
                    self.potential_f(z_t).sum(), z_t, 
                    create_graph=True, retain_graph=True
                )[0]
                # Combine neural prediction with OT gradient
                velocity = v_neural + 0.1 * f_grad
            except RuntimeError:
                # Fallback if gradient computation fails
                velocity = v_neural
        else:
            # During validation or when gradients are disabled
            # Use only the neural network prediction
            velocity = v_neural
        
        return velocity
    
    def schrodinger_bridge_correction(self, z_t: torch.Tensor, t: torch.Tensor,
                                     sigma: float = 0.01) -> torch.Tensor:
        """
        Apply Schrödinger Bridge correction for stochastic optimal transport.
        This accounts for diffusion in addition to transport.
        
        Args:
            z_t: Current internal coordinates [batch, dim_internal]
            t: Time parameter [batch, 1]
            sigma: Diffusion coefficient
            
        Returns:
            Corrected velocity [batch, dim_internal]
        """
        # Concatenate state with time
        z_t_with_time = torch.cat([z_t, t], dim=-1)
        
        # Predict drift correction
        drift_correction = self.bridge_net(z_t_with_time)
        
        # Add stochastic component
        noise = torch.randn_like(z_t) * np.sqrt(2 * sigma)
        
        return drift_correction + noise
    
    def forward(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing OT interpolation and velocity.
        
        Args:
            z0: Starting internal coordinates [batch, dim_internal]
            z1: Target internal coordinates [batch, dim_internal]
            t: Time parameter [batch, 1]
            features: Optional molecular features [batch, feature_dim]
            
        Returns:
            Dictionary containing:
            - z_t: Interpolated coordinates
            - velocity: OT velocity field
            - transport_cost: OT cost
        """
        # Compute OT path
        z_t = self.compute_ot_path(z0, z1, t)
        
        # Compute velocity
        velocity = self.compute_ot_velocity(z0, z1, z_t, t)
        
        # Apply Schrödinger Bridge correction
        velocity_corrected = velocity + self.schrodinger_bridge_correction(z_t, t)
        
        # Compute transport cost
        cost_matrix = self.compute_transport_cost(z0, z1)
        pi = self.sinkhorn_algorithm(cost_matrix)
        transport_cost = (pi * cost_matrix).sum()
        
        return {
            'z_t': z_t,
            'velocity': velocity_corrected,
            'transport_cost': transport_cost,
            'transport_plan': pi
        }


class WassersteinLoss(nn.Module):
    """
    Wasserstein distance loss for optimal transport in conformational space.
    """
    
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.ot_flow = OptimalTransportFlow(dim_internal=10, epsilon=epsilon)
    
    def forward(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        """
        Compute Wasserstein distance between predicted and target distributions.
        
        Args:
            pred_z: Predicted internal coordinates [batch, dim]
            target_z: Target internal coordinates [batch, dim]
            
        Returns:
            Wasserstein distance (scalar)
        """
        # Compute cost matrix
        cost_matrix = self.ot_flow.compute_transport_cost(pred_z, target_z)
        
        # Solve OT problem
        pi = self.ot_flow.sinkhorn_algorithm(cost_matrix)
        
        # Compute Wasserstein distance
        wasserstein = (pi * cost_matrix).sum()
        
        return wasserstein


class ConformationalEnergyRegularizer(nn.Module):
    """
    Energy-based regularization for conformational space.
    Ensures generated conformations are energetically plausible.
    """
    
    def __init__(self, temperature: float = 300.0):
        """
        Args:
            temperature: Temperature in Kelvin for Boltzmann weighting
        """
        super().__init__()
        self.kT = 8.314e-3 * temperature  # kJ/mol
        
        # Professional force field parameters (AMBER/CHARMM-like)
        # Convert units: kcal/mol to kJ/mol (1 kcal = 4.184 kJ)
        
        # Bond stretching: Harmonic potential V = 0.5 * k * (r - r0)^2
        self.bond_params = {
            'default': {'r0': 1.50, 'k': 1255.2},  # r0 in Å, k in kJ/mol/Å^2
            'CC_single': {'r0': 1.526, 'k': 1297.0},
            'CC_double': {'r0': 1.335, 'k': 2297.0},
            'CC_aromatic': {'r0': 1.400, 'k': 1962.0},
            'CN_single': {'r0': 1.470, 'k': 1410.0},
            'CO_single': {'r0': 1.430, 'k': 1339.0},
            'CO_double': {'r0': 1.229, 'k': 2385.0},
        }
        
        # Angle bending: Harmonic potential V = 0.5 * k * (θ - θ0)^2
        self.angle_params = {
            'default': {'theta0': 1.911, 'k': 146.4},  # θ0 in radians, k in kJ/mol/rad^2
            'CCC': {'theta0': 1.911, 'k': 167.4},  # 109.5 degrees
            'CCN': {'theta0': 1.920, 'k': 209.2},  # 110.0 degrees
            'CCO': {'theta0': 1.911, 'k': 209.2},
            'HCH': {'theta0': 1.881, 'k': 138.1},  # 107.8 degrees
        }
        
        # Dihedral torsion: Fourier series V = Σ Vn * (1 + cos(n*φ - δn))
        self.dihedral_params = {
            'default': {'V': [0.628, 1.046, 0.418], 'n': [1, 2, 3], 'delta': [0, 3.14159, 0]},  # kJ/mol
            'CCCC': {'V': [0.753, 1.046, 0.837], 'n': [1, 2, 3], 'delta': [0, 3.14159, 0]},
            'OCCO': {'V': [0.0, 0.0, 1.883], 'n': [1, 2, 3], 'delta': [0, 0, 0]},
        }
        
    def compute_energy(self, internal_coords: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute molecular mechanical energy from internal coordinates.
        
        Args:
            internal_coords: Dictionary with bond_lengths, bond_angles, dihedrals
            
        Returns:
            Energy values [batch]
        """
        energy = 0.0
        
        # Bond stretching energy (harmonic)
        if 'bond_lengths' in internal_coords:
            r = internal_coords['bond_lengths']
            r0 = 0.15  # Reference bond length in nm
            energy += 0.5 * self.bond_k * ((r - r0) ** 2).sum(dim=-1)
        
        # Angle bending energy (harmonic)
        if 'bond_angles' in internal_coords:
            theta = internal_coords['bond_angles']
            theta0 = 109.5 * np.pi / 180  # Reference angle in radians
            energy += 0.5 * self.angle_k * ((theta - theta0) ** 2).sum(dim=-1)
        
        # Dihedral torsion energy (cosine series)
        if 'dihedrals' in internal_coords:
            phi = internal_coords['dihedrals']
            energy += self.dihedral_k * (1 + torch.cos(3 * phi)).sum(dim=-1)
        
        return energy
    
    def forward(self, internal_coords: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute energy-based regularization loss.
        
        Returns:
            Regularization loss encouraging low-energy conformations
        """
        energy = self.compute_energy(internal_coords)
        
        # Boltzmann factor
        boltzmann = torch.exp(-energy / self.kT)
        
        # Negative log-likelihood of Boltzmann distribution
        loss = -torch.log(boltzmann + 1e-8).mean()
        
        return loss