import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminativeEmbeddingLoss(nn.Module):
    """
    De Brabandere et al. (CVPR 2017) discriminative loss:
      L = alpha * L_var + beta * L_dist + gamma * L_reg

    Works for:
      embeddings: (N, D) or (B, N, D)
      instance_labels: (N,) or (B, N)
    """
    def __init__(
        self,
        delta_var=0.5,      # δ_v  : intra-cluster margin
        delta_dist=1.5,     # δ_d  : inter-cluster margin (centers separated by >= 2*δ_d)
        alpha=1.0,
        beta=1.0,
        gamma=0.001,
        norm=2,             # use L2 by default (paper uses L2; ASIS uses L1)
        ignore_index=1, # 0: FG, 1: BG, 2: Crossover tree
        eps=1e-8,
    ):
        super().__init__()
        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.norm = norm
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, embeddings, instance_labels):
        # ---- unify shapes to (B, N, D) ----
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # (1, N, D)
        if instance_labels.dim() == 1:
            instance_labels = instance_labels.unsqueeze(0)  # (1, N)

        B, N, D = embeddings.shape
        device = embeddings.device

        total_var = embeddings.new_zeros(())
        total_dist = embeddings.new_zeros(())
        total_reg = embeddings.new_zeros(())
        valid_batches = 0

        for b in range(B):
            emb = embeddings[b]         # (N, D)
            lbl = instance_labels[b]    # (N,)

            # ---- keep only non-ignored points ----
            if self.ignore_index is not None:
                keep = (lbl != self.ignore_index)
                if keep.sum() == 0:
                    continue
                emb = emb[keep]
                lbl = lbl[keep]

            unique_ids = torch.unique(lbl)
            C = unique_ids.numel()
            if C == 0:
                continue

            # ---- compute cluster means μ_c ----
            # means: (C, D)
            means = []
            counts = []
            for cid in unique_ids:
                m = (lbl == cid)
                means.append(emb[m].mean(dim=0))
                counts.append(m.sum())
            means = torch.stack(means, dim=0)
            counts = torch.stack(counts, dim=0).to(embeddings.dtype)

            # =========================
            # 1) Variance term L_var
            # =========================
            # For each point: distance to its cluster mean
            var_sum = emb.new_zeros(())
            for k, cid in enumerate(unique_ids):
                m = (lbl == cid)
                if m.sum() == 0:
                    continue
                dist = torch.linalg.norm(emb[m] - means[k].unsqueeze(0), ord=self.norm, dim=1)
                # hinge: [dist - δv]_+^2
                hinge = F.relu(dist - self.delta_var) ** 2
                var_sum = var_sum + hinge.mean()

            L_var = var_sum / (C + self.eps)

            # =========================
            # 2) Distance term L_dist
            # =========================
            if C > 1:
                # pairwise center distances: (C, C)
                # Use torch.cdist for simplicity
                center_dists = torch.cdist(means, means, p=self.norm)

                # exclude diagonal
                eye = torch.eye(C, device=device, dtype=torch.bool)
                d = center_dists[~eye].view(C, C - 1)

                # hinge: [2δd - dist]_+^2
                hinge = F.relu(2.0 * self.delta_dist - d) ** 2
                L_dist = hinge.mean()
            else:
                L_dist = emb.new_zeros(())

            # =========================
            # 3) Regularizer L_reg
            # =========================
            # mean norm of centers
            L_reg = torch.linalg.norm(means, ord=self.norm, dim=1).mean()

            total_var = total_var + L_var
            total_dist = total_dist + L_dist
            total_reg = total_reg + L_reg
            valid_batches += 1

        if valid_batches == 0:
            # nothing to compute (all ignored)
            return embeddings.new_zeros(())

        total_var = total_var / valid_batches
        total_dist = total_dist / valid_batches
        total_reg = total_reg / valid_batches

        loss = self.alpha * total_var + self.beta * total_dist + self.gamma * total_reg
        return loss
