import torch

def fit_trunk_axis_and_radius(points_xyz, part_labels, ignore_mask,
                              trunk_class=0, min_points=10, radius_quantile=0.9):
    """
    points_xyz : [B, N, 3] float tensor on any device
    part_labels: [B, N] long tensor (0 trunk, 1 root_ball, 2 pit, 3 background)
    ignore_mask: [B, N] 0 = normal, 1 = 'foreign' points (we will penalize if misclassified)
    
    Returns:
        centers : list of length B, each is tensor [3] or None
        axes    : list of length B, each is tensor [3] or None (unit vector)
        radii   : list of length B, each is scalar tensor or None
    """
    B, N, _ = points_xyz.shape

    centers, axes, radii = [], [], []

    with torch.no_grad():
        for b in range(B):
            # trunk points for the *current* tree instance:
            mask_trunk = (part_labels[b] == trunk_class) & (ignore_mask[b] == 0)
            num_trunk = mask_trunk.sum()

            if num_trunk < min_points:
                centers.append(None)
                axes.append(None)
                radii.append(None)
                continue

            P = points_xyz[b, mask_trunk]  # [M, 3]
            M = P.shape[0]

            center = P.mean(dim=0, keepdim=True)  # [1,3]
            P_centered = P - center

            cov = (P_centered.T @ P_centered) / max(M - 1, 1)  # [3,3]
            eigvals, eigvecs = torch.linalg.eigh(cov)

            v = eigvecs[:, -1]
            v = v / (v.norm() + 1e-8)

            proj_len = (P_centered @ v)
            proj     = proj_len.unsqueeze(1) * v.unsqueeze(0)
            perp     = P_centered - proj
            d_perp   = perp.norm(dim=1)

            R = torch.quantile(d_perp, radius_quantile)

            centers.append(center.squeeze(0))
            axes.append(v)
            radii.append(R)

    return centers, axes, radii


def shape_prior_loss(points_xyz, logits, part_labels, ignore_mask,
                     centers, axes, radii,
                     trunk_class=0,
                     part_classes=(0, 1, 2),
                     lambda_sup=1.0,
                     lambda_unsup=0.5,
                    from_log_softmax=True):
    """
    points_xyz : [B, N, 3]
    logits     : [B, N, num_parts]
    part_labels: [B, N]
    ignore_mask: [B, N] (1 = foreign points; 0 = current-tree points)
    centers/axes/radii: from fit_trunk_axis_and_radius
    trunk_class: index of trunk class (usually 0)
    part_classes: which classes count as 'tree parts' for supervised penalty
    lambda_sup : weight for supervised term (ignore_point == 1)
    lambda_unsup: weight for unsupervised trunk-shape term
    
    Returns:
        scalar tensor (total shape prior loss)
    """
    device = logits.device
    B, N, num_parts = logits.shape
    
    if from_log_softmax:
        probs = logits.exp() # model output is log softmax and we convert it to logits (probabilities)

#     probs = torch.softmax(logits, dim=-1)  # [B, N, num_parts]
    sup_losses = []
    unsup_losses = []

    for b in range(B):
        c = centers[b]
        v = axes[b]
        R = radii[b]

        if c is None or v is None or R is None or R <= 0:
            continue

        # --- Geometry: distance to trunk axis (no grad wrt geometry) ---
        with torch.no_grad():
            P = points_xyz[b]  # [N,3]
            P_centered = P - c.unsqueeze(0)       # [N,3]
            proj_len   = (P_centered @ v)         # [N]
            proj       = proj_len.unsqueeze(1) * v.unsqueeze(0)
            perp       = P_centered - proj
            d_perp     = perp.norm(dim=1)        # [N]

            margin = 0.1 * R                     # tune margin
            overreach = (d_perp - (R + margin)).clamp(min=0.0) / (R + margin + 1e-6)
            # 0 inside cylinder, >0 outside

        # --- 1) Supervised term: only for foreign points (ignore_point == 1) ---
        # penalize if model calls them any part 0/1/2 AND they are far
        p_parts = probs[b, :, list(part_classes)].sum(dim=-1)    # [N]
        foreign_mask = (ignore_mask[b] == 1).float()             # [N]

        if foreign_mask.sum() > 0:
            L_sup_b = (p_parts * overreach * foreign_mask).sum() / (foreign_mask.sum() + 1e-6)
            sup_losses.append(L_sup_b)

        # --- 2) Unsupervised term: trunk shape prior ---
        # penalize points with high trunk prob far from axis
        # (this does NOT touch root_ball/pit if model predicts them,
        # because we only look at p_trunk)
        p_trunk = probs[b, :, trunk_class]                       # [N]

        # You can optionally mask out foreign points here to avoid double penalty:
        # mask_unsup = (ignore_mask[b] == 0).float()
        # L_unsup_b = (p_trunk * overreach * mask_unsup).mean()
        L_unsup_b = (p_trunk * overreach).mean()
        unsup_losses.append(L_unsup_b)

    if not sup_losses and not unsup_losses:
        return torch.tensor(0.0, device=device)

    sup_term   = torch.stack(sup_losses).mean()   if sup_losses   else torch.tensor(0.0, device=device)
    unsup_term = torch.stack(unsup_losses).mean() if unsup_losses else torch.tensor(0.0, device=device)

    return lambda_sup * sup_term + lambda_unsup * unsup_term