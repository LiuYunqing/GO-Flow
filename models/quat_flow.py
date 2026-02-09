import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import numpy as np
from utils.quat_utils import (
    quaternion_mul_batch, quat_to_rotmat, rotmat_to_quat, 
    quaternion_slerp_exp, quaternion_conjugate_batch, quaternion_log
)


class QuaternionFlow(nn.Module):
    """Quaternion flow module for SE(3) equivariant diffusion."""
    
    def __init__(self, input_dim=128, hidden_dim=128, flow_schedule='linear', num_steps=50, min_t=0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.flow_schedule = flow_schedule
        self.num_steps = num_steps
        self.min_t = min_t
        
        # Quaternion prediction network
        self.quat_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Output quaternion (w, x, y, z)
        )
        
        # Flow velocity network
        self.flow_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 2),  # features + time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Output 3D flow vector
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def get_schedule(self, t):
        """Get flow schedule value at time t."""
        if self.flow_schedule == 'linear':
            return t
        elif self.flow_schedule == 'cosine':
            return 0.5 * (1 - torch.cos(np.pi * t))
        elif self.flow_schedule == 'sigmoid':
            return torch.sigmoid(10 * (t - 0.5))
        else:
            raise ValueError(f"Unknown flow schedule: {self.flow_schedule}")
    
    def compute_loss(self, features, x_target, x_current, t, batch):
        """Compute quaternion flow loss."""
        
        # Get time embedding and expand to match features
        t_embed = self.time_embed(t.unsqueeze(-1))  # Shape: [batch_size, hidden_dim]
        
        # Expand time embedding to match number of atoms
        n_nodes_per_g = batch.num_nodes_per_graph
        t_embed_expanded = t_embed.repeat_interleave(n_nodes_per_g, dim=0)  # Shape: [n_atoms, hidden_dim]
        
        # Predict quaternions
        quat_pred = self.quat_net(features)
        quat_pred = F.normalize(quat_pred, dim=-1)  # Normalize to unit quaternion
        
        # Convert positions to quaternion representation
        # Group by molecules
        n_nodes_per_g = batch.num_nodes_per_graph
        x_target_split = torch.split(x_target, n_nodes_per_g.tolist(), dim=0)
        x_current_split = torch.split(x_current, n_nodes_per_g.tolist(), dim=0)
        
        # Compute target quaternions (simplified - in practice would use proper rotation extraction)
        quat_target_list = []
        for i in range(len(n_nodes_per_g)):
            # Compute rotation between current and target
            if n_nodes_per_g[i] >= 3:
                # Use first 3 atoms to define orientation (simplified)
                R = self._compute_rotation_matrix(x_current_split[i][:3], x_target_split[i][:3])
                quat = rotmat_to_quat(R)
            else:
                quat = torch.tensor([1., 0., 0., 0.], device=x_target.device)
            quat_target_list.extend([quat] * n_nodes_per_g[i])
        
        quat_target = torch.stack(quat_target_list)
        
        # Quaternion geodesic loss
        quat_loss = 1 - torch.abs(torch.sum(quat_pred * quat_target, dim=-1))
        quat_loss = scatter_mean(quat_loss, batch.batch, dim=0)
        quat_loss = torch.mean(quat_loss)
        
        # Flow loss
        feat_with_time = torch.cat([features, t_embed_expanded], dim=-1)
        flow_pred = self.flow_net(feat_with_time)
        
        # Target flow (simplified - actual implementation would use ODE solver)
        # Expand t to match atoms
        t_expanded = t.repeat_interleave(n_nodes_per_g, dim=0).unsqueeze(-1)
        flow_target = (x_target - x_current) / (1 - t_expanded + self.min_t)
        
        flow_loss = F.mse_loss(flow_pred, flow_target, reduction='none')
        flow_loss = scatter_mean(flow_loss.sum(dim=-1), batch.batch, dim=0)
        flow_loss = torch.mean(flow_loss)
        
        return quat_loss, flow_loss
    
    def correct_trajectory(self, features, x_current, t, batch):
        """Apply quaternion flow correction to trajectory."""
        # Get time embedding
        t_embed = self.time_embed(t.unsqueeze(-1))
        
        # Predict flow velocity
        feat_with_time = torch.cat([features, t_embed], dim=-1)
        flow_velocity = self.flow_net(feat_with_time)
        
        # Predict quaternions for rotation correction
        quat_pred = self.quat_net(features)
        quat_pred = F.normalize(quat_pred, dim=-1)
        
        # Apply rotation correction (simplified)
        n_nodes_per_g = batch.num_nodes_per_graph
        x_corrected = []
        
        idx = 0
        for i in range(len(n_nodes_per_g)):
            n = n_nodes_per_g[i]
            mol_coords = x_current[idx:idx+n]
            mol_quat = quat_pred[idx:idx+n]
            
            # Average quaternion for the molecule
            avg_quat = F.normalize(torch.mean(mol_quat, dim=0), dim=-1)
            
            # Apply rotation
            R = quat_to_rotmat(avg_quat)
            center = torch.mean(mol_coords, dim=0)
            mol_coords_centered = mol_coords - center
            mol_coords_rotated = torch.matmul(mol_coords_centered, R.T)
            mol_coords_final = mol_coords_rotated + center
            
            x_corrected.append(mol_coords_final)
            idx += n
        
        x_corrected = torch.cat(x_corrected, dim=0)
        
        # Add flow velocity
        correction = flow_velocity * (1 - t + self.min_t)
        
        return correction
    
    def _compute_rotation_matrix(self, X, Y):
        """Compute rotation matrix between two point sets using SVD."""
        # Center the points
        X_center = X.mean(dim=0)
        Y_center = Y.mean(dim=0)
        X_c = X - X_center
        Y_c = Y - Y_center
        
        # Compute cross-covariance matrix
        H = torch.matmul(X_c.T, Y_c)
        
        # SVD
        U, S, Vt = torch.linalg.svd(H)
        R = torch.matmul(Vt.T, U.T)
        
        # Ensure proper rotation (det(R) = 1)
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = torch.matmul(Vt.T, U.T)
        
        return R