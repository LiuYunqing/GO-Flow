"""
Geodesic Flow on SO(3) Manifold for Molecular Rotations
Implements true rotational diffusion using quaternion SLERP and geodesic paths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from utils.quat_utils import (
    quaternion_slerp_exp, quaternion_mul_batch, quaternion_conjugate_batch,
    quaternion_log, quat_to_rotmat, rotmat_to_quat
)


class GeodesicRotationFlow(nn.Module):
    """
    Implements geodesic flow on SO(3) manifold for molecular rotations.
    Uses quaternion representation for numerical stability and avoiding gimbal lock.
    """
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.epsilon = 1e-8
        
        # Network to predict rotational velocity on SO(3)
        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim + 1 if i == 0 else hidden_dim  # +1 for time
            out_dim = 3 if i == num_layers - 1 else hidden_dim  # Output: so(3) algebra element
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.SiLU() if i < num_layers - 1 else nn.Identity()
            ])
        
        self.velocity_net = nn.Sequential(*layers)
        
        # Network for quaternion correction/refinement
        self.quat_refiner = nn.Sequential(
            nn.Linear(4 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
    
    def compute_so3_geodesic(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic path on SO(3) manifold between two quaternions.
        
        Args:
            q0: Starting quaternion [batch, 4]
            q1: Target quaternion [batch, 4]
            t: Time parameter [batch, 1] or scalar in [0, 1]
            
        Returns:
            Interpolated quaternion at time t [batch, 4]
        """
        # Ensure unit quaternions
        q0 = F.normalize(q0, p=2, dim=-1)
        q1 = F.normalize(q1, p=2, dim=-1)
        
        # Handle batch time
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=q0.device)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(q0.shape[0])
        elif t.dim() == 1:
            t = t.view(-1)
        
        # Use quaternion SLERP for geodesic interpolation
        q_t = quaternion_slerp_exp(q0, q1, t)
        
        return F.normalize(q_t, p=2, dim=-1)
    
    def geodesic_velocity(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute tangent vector (velocity) on SO(3) at time t along geodesic.
        
        Args:
            q0: Starting quaternion [batch, 4]
            q1: Target quaternion [batch, 4]
            t: Time parameter [batch, 1]
            
        Returns:
            Velocity in so(3) algebra (3D vector) [batch, 3]
        """
        # Compute relative quaternion
        q0_inv = quaternion_conjugate_batch(q0)
        q_rel = quaternion_mul_batch(q1, q0_inv)
        
        # Convert to axis-angle (logarithm map)
        # For unit quaternion q = [w, x, y, z], the logarithm is [0, theta*v]
        # where theta is rotation angle and v is unit axis
        w = q_rel[:, 0:1]
        xyz = q_rel[:, 1:]
        
        # Compute rotation angle
        theta = 2 * torch.acos(torch.clamp(w, -1 + self.epsilon, 1 - self.epsilon))
        
        # Compute rotation axis (handle small angles)
        sin_half_theta = torch.sqrt(1 - w**2 + self.epsilon)
        axis = xyz / (sin_half_theta + self.epsilon)
        
        # Angular velocity in so(3) 
        omega = theta * axis / (1 - t + self.epsilon)  # Constant angular velocity
        
        return omega.squeeze(1) if omega.shape[1] == 1 else omega
    
    def exponential_map(self, q: torch.Tensor, v: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Exponential map on SO(3): move from q in direction v by amount dt.
        
        Args:
            q: Current quaternion [batch, 4]
            v: Velocity in so(3) algebra [batch, 3]
            dt: Time step
            
        Returns:
            Updated quaternion [batch, 4]
        """
        # Convert velocity to quaternion increment
        theta = torch.norm(v, dim=-1, keepdim=True) * dt
        
        # Handle small rotations
        small_angle = theta < self.epsilon
        
        # Compute quaternion exponential
        cos_half = torch.cos(theta / 2)
        sin_half = torch.sin(theta / 2)
        
        # Normalize axis
        axis = F.normalize(v, p=2, dim=-1)
        
        # Build quaternion [w, x, y, z]
        dq = torch.cat([cos_half, sin_half * axis], dim=-1)
        
        # Handle small angles (use Taylor expansion)
        dq = torch.where(small_angle.expand_as(dq),
                        torch.cat([torch.ones_like(cos_half), v * dt / 2], dim=-1),
                        dq)
        
        # Apply rotation
        q_new = quaternion_mul_batch(dq, q)
        
        return F.normalize(q_new, p=2, dim=-1)
    
    def logarithm_map(self, q: torch.Tensor) -> torch.Tensor:
        """
        Logarithm map from SO(3) to so(3) algebra.
        
        Args:
            q: Quaternion [batch, 4]
            
        Returns:
            Element of so(3) algebra [batch, 3]
        """
        # Ensure unit quaternion
        q = F.normalize(q, p=2, dim=-1)
        
        w = q[:, 0:1]
        xyz = q[:, 1:]
        
        # Compute rotation angle
        theta = 2 * torch.acos(torch.clamp(w, -1 + self.epsilon, 1 - self.epsilon))
        
        # Compute rotation axis
        sin_half_theta = torch.sqrt(1 - w**2 + self.epsilon)
        axis = xyz / (sin_half_theta + self.epsilon)
        
        # Return so(3) element
        return theta * axis
    
    def parallel_transport(self, q_t: torch.Tensor, v: torch.Tensor, 
                          q_s: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of tangent vector v from q_t to q_s along geodesic.
        Essential for maintaining consistency of vector fields on manifold.
        
        Args:
            q_t: Source point on SO(3) [batch, 4]
            v: Tangent vector at q_t [batch, 3]
            q_s: Target point on SO(3) [batch, 4]
            
        Returns:
            Transported vector at q_s [batch, 3]
        """
        # Compute relative rotation from q_t to q_s
        q_t_inv = quaternion_conjugate_batch(q_t)
        q_rel = quaternion_mul_batch(q_s, q_t_inv)
        
        # Convert to rotation matrix
        R = quat_to_rotmat(q_rel)
        
        # Transport vector (rotate it)
        v_transported = torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)
        
        return v_transported
    
    def sample_brownian_motion(self, q: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """
        Sample Brownian motion on SO(3) manifold.
        Used for adding noise in rotational diffusion.
        
        Args:
            q: Current quaternion [batch, 4]
            sigma: Noise scale
            
        Returns:
            Noisy quaternion [batch, 4]
        """
        batch_size = q.shape[0]
        
        # Sample tangent vector from isotropic Gaussian in so(3)
        v = torch.randn(batch_size, 3, device=q.device) * sigma
        
        # Apply exponential map
        q_noisy = self.exponential_map(q, v, dt=1.0)
        
        return q_noisy
    
    def geodesic_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance on SO(3) between two quaternions.
        
        Args:
            q1, q2: Quaternions [batch, 4]
            
        Returns:
            Geodesic distances [batch]
        """
        # Ensure unit quaternions
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)
        
        # Compute dot product
        dot = torch.sum(q1 * q2, dim=-1)
        
        # Handle quaternion double cover (q and -q represent same rotation)
        dot = torch.abs(dot)
        
        # Geodesic distance
        dist = 2 * torch.acos(torch.clamp(dot, -1 + self.epsilon, 1 - self.epsilon))
        
        return dist
    
    def forward(self, features: torch.Tensor, q_current: torch.Tensor, 
                q_target: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict rotational flow velocity and refined quaternion.
        
        Args:
            features: Molecular features [batch, hidden_dim]
            q_current: Current quaternion [batch, 4]
            q_target: Target quaternion [batch, 4]
            t: Time parameter [batch, 1]
            
        Returns:
            velocity: Predicted velocity in so(3) [batch, 3]
            q_refined: Refined quaternion prediction [batch, 4]
        """
        # Concatenate features with time
        feat_t = torch.cat([features, t], dim=-1)
        
        # Predict velocity on tangent space
        velocity = self.velocity_net(feat_t)
        
        # Optionally refine quaternion prediction
        q_concat = torch.cat([q_current, features], dim=-1)
        q_delta = self.quat_refiner(q_concat)
        q_refined = F.normalize(q_current + 0.1 * q_delta, p=2, dim=-1)
        
        return velocity, q_refined


class SO3FlowMatching(nn.Module):
    """
    Flow matching loss for SO(3) manifold.
    Matches predicted flow to geodesic flow between noise and data.
    """
    
    def __init__(self):
        super().__init__()
        self.geodesic_flow = GeodesicRotationFlow()
    
    def compute_loss(self, pred_velocity: torch.Tensor, q_0: torch.Tensor,
                    q_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss on SO(3).
        
        Args:
            pred_velocity: Predicted velocity [batch, 3]
            q_0: Starting quaternion (noise) [batch, 4]
            q_1: Target quaternion (data) [batch, 4]
            t: Time parameter [batch, 1]
            
        Returns:
            Flow matching loss (scalar)
        """
        # Compute true geodesic velocity
        true_velocity = self.geodesic_flow.geodesic_velocity(q_0, q_1, t)
        
        # MSE loss in tangent space
        loss = F.mse_loss(pred_velocity, true_velocity)
        
        # Additional regularization: velocity should be smooth
        velocity_norm = torch.norm(pred_velocity, dim=-1)
        smoothness_loss = torch.var(velocity_norm)
        
        return loss + 0.01 * smoothness_loss
    
    def sample_trajectory(self, q_0: torch.Tensor, model: nn.Module, 
                         num_steps: int = 50) -> torch.Tensor:
        """
        Sample trajectory on SO(3) using learned flow.
        
        Args:
            q_0: Initial quaternion [batch, 4]
            model: Flow model that predicts velocity
            num_steps: Number of integration steps
            
        Returns:
            Final quaternion [batch, 4]
        """
        q = q_0
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=q.device)
            
            # Get velocity prediction
            with torch.no_grad():
                velocity = model(q, t)
            
            # Update using exponential map
            q = self.geodesic_flow.exponential_map(q, velocity, dt)
        
        return q