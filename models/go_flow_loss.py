"""
Hybrid Coordinate Loss Functions for GO-Flow
Combines losses from translation, rotation, and conformational spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
from models.geodesic_flow import GeodesicRotationFlow
from models.ot_flow import WassersteinLoss
from utils.quat_utils import quaternion_geodesic_distance


class TranslationLoss(nn.Module):
    """
    Loss function for center of mass translation in R^3.
    Uses mean squared error with physical regularization and momentum conservation.
    """
    
    def __init__(self, velocity_weight: float = 0.1):
        super().__init__()
        self.velocity_weight = velocity_weight
    
    def forward(self, pred_centers: torch.Tensor, target_centers: torch.Tensor,
                pred_velocity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute translation loss.
        
        Args:
            pred_centers: Predicted center of mass [batch, 3]
            target_centers: Target center of mass [batch, 3]
            pred_velocity: Predicted velocity [batch, 3]
            
        Returns:
            Dictionary with loss components
        """
        # Position loss (MSE)
        position_loss = F.mse_loss(pred_centers, target_centers)
        
        # Velocity regularization (encourage smooth trajectories)
        velocity_loss = torch.tensor(0.0, device=pred_centers.device)
        if pred_velocity is not None:
            velocity_norm = torch.norm(pred_velocity, dim=-1)
            velocity_loss = torch.mean(velocity_norm ** 2)
        
        total_loss = position_loss + self.velocity_weight * velocity_loss
        
        return {
            'total': total_loss,
            'position': position_loss,
            'velocity': velocity_loss
        }


class RotationLoss(nn.Module):
    """
    Loss function for rotation on SO(3) manifold.
    Uses quaternion geodesic distance and angular velocity regularization.
    """
    
    def __init__(self, geodesic_weight: float = 1.0, velocity_weight: float = 0.1):
        super().__init__()
        self.geodesic_weight = geodesic_weight
        self.velocity_weight = velocity_weight
        self.geodesic_flow = GeodesicRotationFlow()
    
    def forward(self, pred_quat: torch.Tensor, target_quat: torch.Tensor,
                pred_velocity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute rotation loss on SO(3).
        
        Args:
            pred_quat: Predicted quaternions [batch, 4]
            target_quat: Target quaternions [batch, 4]
            pred_velocity: Predicted angular velocity in so(3) [batch, 3]
            
        Returns:
            Dictionary with loss components
        """
        # Ensure unit quaternions
        pred_quat = F.normalize(pred_quat, p=2, dim=-1)
        target_quat = F.normalize(target_quat, p=2, dim=-1)
        
        # Geodesic distance on SO(3)
        geodesic_dist = self.geodesic_flow.geodesic_distance(pred_quat, target_quat)
        geodesic_loss = torch.mean(geodesic_dist ** 2)
        
        # Quaternion MSE (for local accuracy)
        # Handle quaternion double cover
        dot = torch.sum(pred_quat * target_quat, dim=-1, keepdim=True)
        target_quat_aligned = torch.where(dot < 0, -target_quat, target_quat)
        quat_mse = F.mse_loss(pred_quat, target_quat_aligned)
        
        # Angular velocity regularization
        velocity_loss = torch.tensor(0.0, device=pred_quat.device)
        if pred_velocity is not None:
            velocity_norm = torch.norm(pred_velocity, dim=-1)
            velocity_loss = torch.mean(velocity_norm ** 2)
        
        total_loss = (self.geodesic_weight * geodesic_loss + 
                     quat_mse + 
                     self.velocity_weight * velocity_loss)
        
        return {
            'total': total_loss,
            'geodesic': geodesic_loss,
            'quat_mse': quat_mse,
            'velocity': velocity_loss
        }


class ConformationLoss(nn.Module):
    """
    Loss function for conformational space using internal coordinates.
    Combines Wasserstein distance, energy regularization, and internal coordinate accuracy.
    """
    
    def __init__(self, wasserstein_weight: float = 1.0, 
                 energy_weight: float = 0.1,
                 coordinate_weight: float = 0.5):
        super().__init__()
        self.wasserstein_weight = wasserstein_weight
        self.energy_weight = energy_weight
        self.coordinate_weight = coordinate_weight
        
        self.wasserstein_loss = WassersteinLoss(epsilon=0.1)
        
    def forward(self, pred_internal: Union[torch.Tensor, Dict[str, torch.Tensor]],
                target_internal: Union[torch.Tensor, Dict[str, torch.Tensor]],
                energy: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute conformational loss in internal coordinate space.
        
        Args:
            pred_internal: Predicted internal coordinates (tensor or dict)
            target_internal: Target internal coordinates (tensor or dict)
            energy: Optional molecular energy
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Handle both tensor and dictionary inputs
        if isinstance(pred_internal, torch.Tensor) and isinstance(target_internal, torch.Tensor):
            # Direct tensor comparison - treat as flattened internal coordinates
            coord_loss = F.mse_loss(pred_internal, target_internal)
        else:
            # Dictionary with specific coordinate types
            if isinstance(pred_internal, dict) and isinstance(target_internal, dict):
                # Bond length loss (critical for molecular structure)
                if 'bond_lengths' in pred_internal and 'bond_lengths' in target_internal:
                    bond_loss = F.mse_loss(pred_internal['bond_lengths'], 
                                           target_internal['bond_lengths'])
                    losses['bonds'] = bond_loss
                
                # Bond angle loss (important for molecular geometry)
                if 'bond_angles' in pred_internal and 'bond_angles' in target_internal:
                    # Use cosine similarity for angles
                    pred_cos = torch.cos(pred_internal['bond_angles'])
                    target_cos = torch.cos(target_internal['bond_angles'])
                    angle_loss = F.mse_loss(pred_cos, target_cos)
                    losses['angles'] = angle_loss
                
                # Dihedral angle loss (controls conformational flexibility)
                if 'dihedrals' in pred_internal and 'dihedrals' in target_internal:
                    # Circular loss for dihedrals
                    dihedral_diff = pred_internal['dihedrals'] - target_internal['dihedrals']
                    dihedral_loss = torch.mean(1 - torch.cos(dihedral_diff))
                    losses['dihedrals'] = dihedral_loss
                
                # Combine internal coordinate losses
                coord_loss = sum(losses.values()) / len(losses) if losses else torch.tensor(0.0, device=pred_internal['bond_lengths'].device if 'bond_lengths' in pred_internal else torch.device('cpu'))
            else:
                # Type mismatch - return zero loss
                device = pred_internal.device if isinstance(pred_internal, torch.Tensor) else torch.device('cpu')
                coord_loss = torch.tensor(0.0, device=device)
        
        # Energy regularization (encourage low-energy conformations)
        energy_loss = torch.tensor(0.0, device=coord_loss.device)
        if energy is not None:
            energy_loss = torch.mean(energy)
        
        # Total conformational loss
        total_loss = (self.coordinate_weight * coord_loss +
                     self.energy_weight * energy_loss)
        
        return {
            'total': total_loss,
            'coordinates': coord_loss,
            'energy': energy_loss,
            **losses
        }


class GOFlowLoss(nn.Module):
    """
    Main loss function for GO-Flow model.
    Combines translation, rotation, and conformational losses with flow matching.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Loss weights from config
        self.translation_weight = config.get('translation_weight', 1.0)
        self.rotation_weight = config.get('rotation_weight', 1.0)
        self.conformation_weight = config.get('conformation_weight', 1.0)
        self.flow_matching_weight = config.get('flow_matching_weight', 1.0)
        
        # Initialize component losses
        self.translation_loss = TranslationLoss(
            velocity_weight=config.get('translation_velocity_weight', 0.1)
        )
        self.rotation_loss = RotationLoss(
            geodesic_weight=config.get('geodesic_weight', 1.0),
            velocity_weight=config.get('rotation_velocity_weight', 0.1)
        )
        self.conformation_loss = ConformationLoss(
            wasserstein_weight=config.get('wasserstein_weight', 1.0),
            energy_weight=config.get('energy_weight', 0.1),
            coordinate_weight=config.get('coordinate_weight', 0.5)
        )
        
        # Adaptive loss weighting
        self.use_adaptive_weighting = config.get('use_adaptive_weighting', False)
        if self.use_adaptive_weighting:
            # Learnable loss weights (log-space for stability)
            self.log_translation_weight = nn.Parameter(torch.tensor(0.0))
            self.log_rotation_weight = nn.Parameter(torch.tensor(0.0))
            self.log_conformation_weight = nn.Parameter(torch.tensor(0.0))
    
    def compute_flow_matching_loss(self, pred_velocity: torch.Tensor,
                                  target_velocity: torch.Tensor,
                                  t: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss for optimal transport.
        
        Args:
            pred_velocity: Predicted velocity field [N_atoms, 3]
            target_velocity: Target velocity from OT [N_atoms, 3]
            t: Time parameter [batch]
            
        Returns:
            Flow matching loss
        """
        # Weight by time (emphasize accuracy at intermediate times)
        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=pred_velocity.device)
        
        time_weight = 4 * t * (1 - t)  # Beta(2,2) weighting
        
        # Handle both scalar and batch time tensors
        if time_weight.dim() > 0:
            time_weight = time_weight.mean()
        else:
            time_weight = time_weight.item() if time_weight.numel() == 1 else time_weight
        
        # MSE loss for velocity matching
        flow_loss = F.mse_loss(pred_velocity, target_velocity)
        
        return time_weight * flow_loss
    
    def forward(self, output: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total GO-Flow loss.
        
        Args:
            output: Model predictions with translation, rotation, conformation
            target: Ground truth values
            t: Time parameter
            
        Returns:
            Total loss and dictionary of component losses
        """
        losses = {}
        
        # Translation loss
        if 'centers' in output and 'centers' in target:
            trans_losses = self.translation_loss(
                output['centers'], target['centers'],
                output.get('v_translation')
            )
            losses['translation'] = trans_losses['total']
            losses.update({f'trans_{k}': v for k, v in trans_losses.items() if k != 'total'})
        
        # Rotation loss
        if 'quaternions' in output and 'quaternions' in target:
            rot_losses = self.rotation_loss(
                output['quaternions'], target['quaternions'],
                output.get('v_rotation')
            )
            losses['rotation'] = rot_losses['total']
            losses.update({f'rot_{k}': v for k, v in rot_losses.items() if k != 'total'})
        
        # Conformation loss
        if 'internal_coords' in output and 'internal_coords' in target:
            conf_losses = self.conformation_loss(
                output['internal_coords'], target['internal_coords'],
                output.get('energy')
            )
            losses['conformation'] = conf_losses['total']
            losses.update({f'conf_{k}': v for k, v in conf_losses.items() if k != 'total'})
        
        # Flow matching loss
        if 'velocity' in output and 'target_velocity' in target:
            flow_loss = self.compute_flow_matching_loss(
                output['velocity'], target['target_velocity'], t
            )
            losses['flow_matching'] = flow_loss
        
        # Combine losses with adaptive or fixed weighting
        if self.use_adaptive_weighting:
            # Uncertainty weighting (Kendall et al.)
            trans_weight = torch.exp(-self.log_translation_weight)
            rot_weight = torch.exp(-self.log_rotation_weight)
            conf_weight = torch.exp(-self.log_conformation_weight)
            
            device = next(iter(output.values())).device if output else torch.device('cpu')
            zero_tensor = torch.tensor(0.0, device=device, requires_grad=True)
            
            total_loss = (
                trans_weight * losses.get('translation', zero_tensor) + self.log_translation_weight +
                rot_weight * losses.get('rotation', zero_tensor) + self.log_rotation_weight +
                conf_weight * losses.get('conformation', zero_tensor) + self.log_conformation_weight +
                self.flow_matching_weight * losses.get('flow_matching', zero_tensor)
            )
            
            # Log adaptive weights
            losses['adaptive_trans_weight'] = trans_weight.item()
            losses['adaptive_rot_weight'] = rot_weight.item()
            losses['adaptive_conf_weight'] = conf_weight.item()
        else:
            # Fixed weighting - ensure we return a tensor with gradient
            device = next(iter(output.values())).device if output else torch.device('cpu')
            zero_tensor = torch.tensor(0.0, device=device, requires_grad=True)
            
            total_loss = (
                self.translation_weight * losses.get('translation', zero_tensor) +
                self.rotation_weight * losses.get('rotation', zero_tensor) +
                self.conformation_weight * losses.get('conformation', zero_tensor) +
                self.flow_matching_weight * losses.get('flow_matching', zero_tensor)
            )
        
        losses['total'] = total_loss
        
        return total_loss, losses


class AdaptiveParallelLossWeights(nn.Module):
    """
    Adaptive loss weighting for parallel Stage 1 training.
    Uses uncertainty-based weighting (Kendall et al.) to balance the three coordinate spaces.
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Initialize log-scale weights for numerical stability
        if initial_weights is None:
            initial_weights = {
                'translation': 1.0,
                'rotation': 1.0,
                'conformation': 1.0
            }
        
        # Learnable parameters in log space
        self.log_translation_weight = nn.Parameter(torch.log(torch.tensor(initial_weights['translation'])))
        self.log_rotation_weight = nn.Parameter(torch.log(torch.tensor(initial_weights['rotation'])))
        self.log_conformation_weight = nn.Parameter(torch.log(torch.tensor(initial_weights['conformation'])))
        
        # Track loss statistics for adaptive adjustment
        self.loss_history = {
            'translation': [],
            'rotation': [],
            'conformation': []
        }
        self.history_window = 100  # Number of steps to consider for statistics
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine losses with adaptive weighting.
        
        Args:
            losses: Dictionary with 'translation', 'rotation', 'conformation' losses
            
        Returns:
            Weighted total loss
        """
        # Compute weights from log parameters (ensures positive weights)
        trans_weight = torch.exp(-self.log_translation_weight)
        rot_weight = torch.exp(-self.log_rotation_weight)
        conf_weight = torch.exp(-self.log_conformation_weight)
        
        # Get individual losses with defaults
        device = next(iter(losses.values())).device if losses else torch.device('cpu')
        zero_tensor = torch.tensor(0.0, device=device, requires_grad=True)
        
        trans_loss = losses.get('translation', zero_tensor)
        rot_loss = losses.get('rotation', zero_tensor)
        conf_loss = losses.get('conformation', zero_tensor)
        
        # Uncertainty-based weighting (Kendall et al.)
        # L = 1/(2σ²) * L_i + log(σ)
        total_loss = (
            trans_weight * trans_loss + self.log_translation_weight +
            rot_weight * rot_loss + self.log_rotation_weight +
            conf_weight * conf_loss + self.log_conformation_weight
        )
        
        # Update loss history for monitoring
        self.update_history(losses)
        
        return total_loss
    
    def update_history(self, losses: Dict[str, torch.Tensor]):
        """Update loss history for adaptive adjustment."""
        for key in ['translation', 'rotation', 'conformation']:
            if key in losses:
                self.loss_history[key].append(losses[key].detach().cpu().item())
                # Keep only recent history
                if len(self.loss_history[key]) > self.history_window:
                    self.loss_history[key].pop(0)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weight values."""
        with torch.no_grad():
            return {
                'translation': torch.exp(-self.log_translation_weight).item(),
                'rotation': torch.exp(-self.log_rotation_weight).item(),
                'conformation': torch.exp(-self.log_conformation_weight).item()
            }
    
    def get_loss_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics of recent losses."""
        stats = {}
        for key, history in self.loss_history.items():
            if history:
                stats[key] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history)
                }
            else:
                stats[key] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return stats


class Stage1ParallelLoss(nn.Module):
    """
    Combined loss function for parallel Stage 1 training.
    Handles all three coordinate spaces simultaneously with adaptive weighting.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Component losses
        self.translation_loss = TranslationLoss(
            velocity_weight=config.get('translation_velocity_weight', 0.1)
        )
        self.rotation_loss = RotationLoss(
            geodesic_weight=config.get('geodesic_weight', 1.0),
            velocity_weight=config.get('rotation_velocity_weight', 0.1)
        )
        self.conformation_loss = ConformationLoss(
            wasserstein_weight=config.get('wasserstein_weight', 1.0),
            energy_weight=config.get('energy_weight', 0.1),
            coordinate_weight=config.get('coordinate_weight', 0.5)
        )
        
        # Adaptive weighting
        self.use_adaptive = config.get('use_adaptive_weights', True)
        if self.use_adaptive:
            initial_weights = {
                'translation': config.get('init_translation_weight', 1.0),
                'rotation': config.get('init_rotation_weight', 1.0),
                'conformation': config.get('init_conformation_weight', 1.0)
            }
            self.adaptive_weights = AdaptiveParallelLossWeights(initial_weights)
        else:
            # Fixed weights
            self.translation_weight = config.get('translation_weight', 1.0)
            self.rotation_weight = config.get('rotation_weight', 1.0)
            self.conformation_weight = config.get('conformation_weight', 1.0)
    
    def forward(self, parallel_outputs: Dict[str, Dict], t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss for parallel Stage 1 training.
        
        Args:
            parallel_outputs: Dictionary with 'translation', 'rotation', 'conformation' outputs
            t: Time parameter
            
        Returns:
            Total loss and dictionary of component losses
        """
        losses = {}
        
        # Translation loss
        if 'translation' in parallel_outputs:
            trans_output = parallel_outputs['translation']
            if 'v_translation' in trans_output and 'v_target' in trans_output:
                trans_losses = self.translation_loss(
                    trans_output['v_translation'],
                    trans_output['v_target'],
                    trans_output.get('velocity')
                )
                losses['translation'] = trans_losses['total']
                losses.update({f'trans_{k}': v for k, v in trans_losses.items() if k != 'total'})
        
        # Rotation loss
        if 'rotation' in parallel_outputs:
            rot_output = parallel_outputs['rotation']
            if 'quaternions' in rot_output and 'q1' in rot_output:
                rot_losses = self.rotation_loss(
                    rot_output['quaternions'],
                    rot_output['q1'],
                    rot_output.get('v_rotation')
                )
                losses['rotation'] = rot_losses['total']
                losses.update({f'rot_{k}': v for k, v in rot_losses.items() if k != 'total'})
        
        # Conformation loss
        if 'conformation' in parallel_outputs:
            conf_output = parallel_outputs['conformation']
            if 'v_conformation' in conf_output and 'v_target' in conf_output:
                # Velocity matching loss with proper normalization
                conf_loss = F.mse_loss(conf_output['v_conformation'], conf_output['v_target'])
                losses['conformation'] = conf_loss
        
        # Combine with adaptive or fixed weighting
        if self.use_adaptive:
            total_loss = self.adaptive_weights(losses)
            # Log current weights
            current_weights = self.adaptive_weights.get_current_weights()
            losses.update({f'weight_{k}': v for k, v in current_weights.items()})
        else:
            # Ensure all losses are tensors and get device
            if losses:
                # Convert any float values to tensors
                for key, value in losses.items():
                    if not isinstance(value, torch.Tensor):
                        # Get device from any existing tensor in losses
                        device = None
                        for v in losses.values():
                            if isinstance(v, torch.Tensor):
                                device = v.device
                                break
                        if device is None:
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        losses[key] = torch.tensor(value, device=device, requires_grad=True)
                
                device = next(iter(losses.values())).device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            zero_tensor = torch.tensor(0.0, device=device, requires_grad=True)
            
            total_loss = (
                self.translation_weight * losses.get('translation', zero_tensor) +
                self.rotation_weight * losses.get('rotation', zero_tensor) +
                self.conformation_weight * losses.get('conformation', zero_tensor)
            )
        
        losses['total'] = total_loss
        
        return total_loss, losses


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss combining atom-level and fragment-level losses.
    Ensures consistency across hierarchical representations.
    """
    
    def __init__(self, atom_weight: float = 1.0, fragment_weight: float = 0.5):
        super().__init__()
        self.atom_weight = atom_weight
        self.fragment_weight = fragment_weight
        
    def forward(self, atom_pred: torch.Tensor, atom_target: torch.Tensor,
                frag_pred: torch.Tensor, frag_target: torch.Tensor,
                backmapping_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale consistency loss.
        
        Args:
            atom_pred: Atom-level predictions [N_atoms, 3]
            atom_target: Atom-level targets [N_atoms, 3]
            frag_pred: Fragment-level predictions [N_frags, 3]
            frag_target: Fragment-level targets [N_frags, 3]
            backmapping_matrix: Matrix to map fragments to atoms
            
        Returns:
            Dictionary with loss components
        """
        # Atom-level loss
        atom_loss = F.mse_loss(atom_pred, atom_target)
        
        # Fragment-level loss
        frag_loss = F.mse_loss(frag_pred, frag_target)
        
        # Consistency loss (atoms should match backmapped fragments)
        atom_from_frag = backmapping_matrix @ frag_pred
        consistency_loss = F.mse_loss(atom_pred, atom_from_frag)
        
        total_loss = (self.atom_weight * atom_loss + 
                     self.fragment_weight * frag_loss +
                     consistency_loss)
        
        return {
            'total': total_loss,
            'atom': atom_loss,
            'fragment': frag_loss,
            'consistency': consistency_loss
        }