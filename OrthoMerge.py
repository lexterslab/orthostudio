from sd_mecha import merge_method, Parameter, Return
import torch
from torch import Tensor

@merge_method
def orthomergev2(
    c: Parameter(Tensor),
    *deltas: Parameter(Tensor, merge_space="delta"),
    alpha: Parameter(float) = 1.0,
    conflict_aware: Parameter(bool) = False,
    theta_agg: Parameter(str) = "mean",
    direction_weight: Parameter(str) = "theta",
    **kwargs,
) -> Return(Tensor):
    """
    Orthogonal Model Merging v2 (OrthoMergeV2)
    Implements Orthogonal-Residual Decoupling strategy for N deltas.
    Based on OrthoMerge_G_TA implementation mapping Cayley values to skew-symmetric space,
    and averaging them separately by magnitude and angle directions.
    """
    # Filter out all-zero deltas
    deltas_non_zero = []
    for d in deltas:
        if not torch.allclose(d, torch.zeros_like(d)):
            deltas_non_zero.append(d)
            
    if len(deltas_non_zero) == 0:
        return c

    if c.ndim < 2:
        merged_delta = sum(deltas_non_zero) / len(deltas_non_zero)
        return c + alpha * merged_delta

    orig_shape = c.shape
    W0 = c.flatten(1).float()

    out_dim, in_dim = W0.shape
    transpose_mode = False

    # Optimize SVD: apply Procrustes on the smaller dimension
    if out_dim > in_dim:
        W0 = W0.T
        transpose_mode = True

    # Compute average task vector (tau_mean) if conflict-aware
    d_2d_list = []
    for d in deltas_non_zero:
        d_2d = d.flatten(1).float()
        if transpose_mode:
            d_2d = d_2d.T
        d_2d_list.append(d_2d)

    if conflict_aware:
        tau_mean = sum(d_2d_list) / len(d_2d_list)

    def extract_orthogonal_and_residual(d_2d_i, W_base):
        W_i = W_base + d_2d_i
        
        # Determine target matrix for orthogonal extraction
        if conflict_aware:
            # Strategy 2: Conflict-Aware Decoupling
            dot_products = torch.sum(d_2d_i * tau_mean, dim=0) # shape: (dim2,)
            conflicts = dot_products < 0 # boolean mask of length dim2
            
            tau_i_conf = torch.zeros_like(d_2d_i)
            tau_i_conf[:, conflicts] = d_2d_i[:, conflicts]
            
            W_target_i = W_base + tau_i_conf
        else:
            # Strategy 1: Global Decoupling
            W_target_i = W_i
        
        target_prod = W_target_i @ W_base.T
        if not torch.isfinite(target_prod).all():
            target_prod = torch.nan_to_num(target_prod, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            U, S, Vh = torch.linalg.svd(target_prod, full_matrices=False)
        except RuntimeError:
            U, S, Vh = torch.linalg.svd(target_prod.cpu(), full_matrices=False)
            U, Vh = U.to(target_prod.device), Vh.to(target_prod.device)
        
        R_i = U @ Vh # Orthogonal matrix
        
        rho_i = W_i - R_i @ W_base
        return R_i, rho_i

    def cayley_to_skew(R):
        # A = (R + I)^-1 (R - I)
        I = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
        r_p = R.clone()
        r_p.diagonal()[:] += 1
        r_n = R.clone()
        r_n.diagonal()[:] -= 1
        try:
            A_k = torch.linalg.solve(r_p, r_n, left=True)
        except RuntimeError:
            r_p.diagonal()[:] += 1e-5
            A_k = torch.linalg.solve(r_p, r_n, left=True)
            
        A_k = 0.5 * (A_k - A_k.transpose(-1, -2))
        return A_k

    def cayley_from_skew(A_k):
        # R = (I - A)^-1 (I + A)
        I = torch.eye(A_k.shape[-1], device=A_k.device, dtype=A_k.dtype)
        m_left = -A_k.clone()
        m_left.diagonal()[:] += 1
        m_right = A_k.clone()
        m_right.diagonal()[:] += 1
        try:
            R = torch.linalg.solve(m_left, m_right, left=True)
        except RuntimeError:
            m_left.diagonal()[:] += 1e-5
            R = torch.linalg.solve(m_left, m_right, left=True)
        return R

    A_list = []
    rho_list = []
    
    for d_2d_i in d_2d_list:
        R_i, rho_i = extract_orthogonal_and_residual(d_2d_i, W0)
        A_list.append(cayley_to_skew(R_i))
        rho_list.append(rho_i)

    # Merge Skew matrices A_list using direction_weight and theta_agg
    base_shape_A = A_list[0].shape
    A_stack = torch.stack(A_list, dim=0)  # [T, d, d]
    T_dim = A_stack.shape[0]

    A_flat = A_stack.reshape(T_dim, -1)  # [T, N]

    theta = torch.linalg.vector_norm(A_flat, dim=1)  # [T]
    theta_clamped = torch.clamp(theta, min=1e-8)
    u = A_flat / theta_clamped.unsqueeze(1)  # [T, N]

    if direction_weight == "theta":
        w = theta.clone()
    elif direction_weight == "uniform":
        w = torch.ones_like(theta)
    else:
        # Fallback to theta if something else is passed
        w = theta.clone()

    u_weighted = u * w.unsqueeze(1)  # [T, N]
    u_sum = u_weighted.sum(dim=0)    # [N]
    u_sum_norm = torch.linalg.vector_norm(u_sum)
    
    if u_sum_norm < 1e-8:
        A_merged = torch.zeros(base_shape_A, device=A_flat.device, dtype=A_flat.dtype)
    else:
        u_merge = u_sum / u_sum_norm  

        if theta_agg == "mean":
            theta_merge = theta.mean()
        elif theta_agg == "median":
            theta_merge = theta.median()
        elif theta_agg == "max":
            theta_merge = theta.max()
        else:
            theta_merge = theta.mean()

        merged_flat = u_merge * theta_merge    # [N]
        A_merged = merged_flat.reshape(base_shape_A)

    R_merged = cayley_from_skew(A_merged)

    # Residual Component Merging
    num_deltas = len(deltas_non_zero)
    rho_merged = sum(rho_list) / num_deltas

    # Hybrid Merging computations
    merged_delta_2d = (R_merged @ W0 + rho_merged) - W0

    if transpose_mode:
        merged_delta_2d = merged_delta_2d.T

    merged_delta = merged_delta_2d.view(orig_shape).to(c.dtype)
    return c + alpha * merged_delta