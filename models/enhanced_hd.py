import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_add
from utils.atom_utils import remove_mean, backmapping_matrix, mapping_matrix, kabsch_torch_batched
from .hd import EquivariantHeatDissipation
from .quat_flow import QuaternionFlow


class EnhancedEquivariantHeatDissipation(EquivariantHeatDissipation):
    """Enhanced Equivariant Heat Dissipation model with quaternion flow integration."""
    
    def __init__(self, config, args, encoder):
        super().__init__(config, args, encoder)
        
        # Quaternion flow specific parameters
        self.use_quaternion_flow = getattr(config, 'use_quaternion_flow', True)
        self.quat_loss_weight = getattr(config, 'quat_loss_weight', 0.1)
        self.coord_loss_weight = getattr(config, 'coord_loss_weight', 1.0)
        self.flow_loss_weight = getattr(config, 'flow_loss_weight', 0.05)
        
        if self.use_quaternion_flow:
            self.quat_flow = QuaternionFlow(
                input_dim=args.nf,
                hidden_dim=getattr(config, 'quat_hidden_dim', 128),
                flow_schedule=getattr(config, 'flow_schedule', 'linear'),
                num_steps=args.num_steps,
                min_t=getattr(config, 'min_t', 0.01)
            )
    
    def get_loss(self, batch):
        """Enhanced loss computation with quaternion flow."""
        # prepare coordinates
        m_mat = mapping_matrix(batch.fg2cg, batch.num_nodes_per_graph)
        bm_mat = backmapping_matrix(batch.fg2cg, batch.num_nodes_per_graph)
        x_a_ref = remove_mean(batch.ref_pos, batch.batch)
        x_a_gt = remove_mean(batch.alg_gt_pos, batch.batch)
        x_f_gt, x_f_ref = m_mat@x_a_gt, m_mat@x_a_ref
        x_f_gt_ext = bm_mat@x_f_gt
        x_r_gt = x_a_gt - x_f_gt_ext

        # perturb data using scheduler
        blur_t = self.sigma_blur(self.T)
        t_steps = self.t_sample(batch.batch[-1].item()+1, batch.batch.device, self.T)
        x_a_b, x_a_lb = self.fwd_blur(batch, t_steps, blur_t, x_a_gt, x_f_ref, bm_mat)
        n_ = torch.randn_like(x_a_b)
        n = remove_mean(n_, batch.batch)
        x_a_b, x_a_lb = remove_mean(x_a_b, batch.batch), remove_mean(x_a_lb, batch.batch)
        x_a_b = x_a_b + n * self.sigma

        # prediction
        t = t_steps / self.T
        pred_a, feat = self.encoder._forward(t, batch, x_a_b, m_mat@x_a_b, m_mat, bm_mat)

        # align prediction to gt atom position
        n_nodes_per_g = batch.num_nodes_per_graph
        pred__ = list(torch.split(pred_a, n_nodes_per_g.tolist(), 0))
        pred_batch = pad_sequence(pred__, batch_first=True, padding_value=0.)
        x_a_gt__ = list(torch.split(x_a_gt, n_nodes_per_g.tolist(), 0))
        x_a_gt_batch = pad_sequence(x_a_gt__, batch_first=True, padding_value=0.)
        alg_pred_batch_to_gt = kabsch_torch_batched(pred_batch, x_a_gt_batch)
        alg_pred_to_gt = torch.cat([alg_pred_batch_to_gt[i][:n_nodes_per_g[i]] for i in range(len(n_nodes_per_g))], dim=0)

        # standard coordinate loss
        loss_a_alg_ = (alg_pred_to_gt - x_a_gt) ** 2
        loss_a_alg_ = torch.sum(scatter_add(loss_a_alg_.T, batch.batch).T, dim=-1)
        loss_a_alg = torch.mean(loss_a_alg_)
        
        total_loss = self.coord_loss_weight * loss_a_alg
        
        # quaternion flow loss if enabled
        if self.use_quaternion_flow and feat is not None:
            # Extract quaternion representations from features
            quat_loss, flow_loss = self.quat_flow.compute_loss(
                feat, x_a_gt, x_a_b, t, batch
            )
            total_loss += self.quat_loss_weight * quat_loss
            total_loss += self.flow_loss_weight * flow_loss
            
            return total_loss, t_steps, {
                'coord_loss': loss_a_alg.item(),
                'quat_loss': quat_loss.item() if self.use_quaternion_flow else 0.0,
                'flow_loss': flow_loss.item() if self.use_quaternion_flow else 0.0
            }
        
        return total_loss, t_steps
    
    def sample(self, sample_init, batch, delta):
        """Enhanced sampling with quaternion flow."""
        with torch.no_grad():
            K = self.T
            m_mat = mapping_matrix(batch.fg2cg, batch.num_nodes_per_graph)
            bm_mat = backmapping_matrix(batch.fg2cg, batch.num_nodes_per_graph)
            
            sample_gen_traj = []
            sample_init = remove_mean(sample_init, batch.batch)

            sample_init__ = list(torch.split(sample_init, batch.num_nodes_per_graph.tolist(), 0))
            sample_init_batch = pad_sequence(sample_init__, batch_first=True, padding_value=0.)
            
            x_f = m_mat@sample_init
            u = bm_mat@x_f
            u = remove_mean(u, batch.batch)
            noise = torch.randn_like(u)
            noise = remove_mean(noise, batch.batch)
            u = u + noise*delta

            sample_gen_traj.append(u.clone())
            
            # Reverse diffusion process
            for t_idx in reversed(range(1, K)):
                # Create time tensor for each molecule in the batch
                t = torch.full((batch.num_graphs,), t_idx / K, device=u.device)
                
                # Get prediction from encoder
                pred_a, feat = self.encoder._forward(t, batch, u, m_mat@u, m_mat, bm_mat)
                
                # Apply quaternion flow correction if enabled and weights are loaded
                if self.use_quaternion_flow and feat is not None and hasattr(self.quat_flow, '_weights_loaded'):
                    quat_correction = self.quat_flow.correct_trajectory(
                        feat, u, t, batch
                    )
                    pred_a = pred_a + quat_correction
                
                # Update position
                u = pred_a
                u = remove_mean(u, batch.batch)
                
                # Add noise for non-final steps
                if t_idx > 1:
                    noise = torch.randn_like(u) * delta * (t_idx / K)
                    noise = remove_mean(noise, batch.batch)
                    u = u + noise
                
                sample_gen_traj.append(u.clone())
            
            return u, sample_gen_traj
    
    def get_model_info(self):
        """Return model configuration information."""
        info = {
            'model_type': 'enhanced_hd',
            'use_quaternion_flow': self.use_quaternion_flow,
            'coord_loss_weight': self.coord_loss_weight,
            'quat_loss_weight': self.quat_loss_weight,
            'flow_loss_weight': self.flow_loss_weight,
            'num_steps': self.T,
            'sigma': self.sigma,
            'hidden_dim': getattr(self.encoder, 'hidden_nf', 'unknown'),
            'n_layers': getattr(self.encoder, 'n_layers', 'unknown')
        }
        
        if self.use_quaternion_flow and hasattr(self, 'quat_flow'):
            info.update({
                'quat_flow_schedule': getattr(self.quat_flow, 'flow_schedule', 'unknown'),
                'quat_flow_num_steps': getattr(self.quat_flow, 'num_steps', 'unknown'),
                'quat_flow_min_t': getattr(self.quat_flow, 'min_t', 'unknown')
            })
        
        return info