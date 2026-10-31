import torch

from collections import defaultdict


def class_weights(seg, eps=1e-6):
    """
    seg: [N], values {0, 1}
    returns: per-point class weight [N]
    """
    N = seg.numel()
    fg_count = (seg == 0).sum().float()
    bg_count = (seg == 1).sum().float()

    w_fg = N / (2.0 * (fg_count + eps))
    w_bg = N / (2.0 * (bg_count + eps))

    w = torch.where(seg == 0, w_fg, w_bg)
    return w

def knn_indices(x, k):
    with torch.no_grad():
        dist = torch.cdist(x, x)        # [N, N]
        _, idx = dist.topk(k + 1, largest=False)
        return idx[:, 1:]




def approx_knn_indices_voxel(x, k, voxel_size):
    """
    Approximate k-NN using a voxel grid to avoid full O(N^2) cdist.

    x: [N, 3] (or [N, D]) tensor (can be on CPU or CUDA)
    k: number of neighbors to return
    voxel_size: size of spatial grid cell (same units as x coords)

    Returns:
        idx: LongTensor [N, k] with neighbor indices for each point
    """
    device = x.device
    x_cpu = x.detach().cpu()
    N, D = x_cpu.shape

    # 1) integer voxel coordinates
    vox = torch.floor(x_cpu / voxel_size).to(torch.int64)   # [N, D]
    if D > 3:
        vox = vox[:, :3]  # use XYZ only

    # 2) build voxel -> list of point indices
    voxel_dict = defaultdict(list)
    for i in range(N):
        key = tuple(vox[i].tolist())
        voxel_dict[key].append(i)

    # 3) precompute neighbor voxel offsets (3x3x3 neighborhood)
    offsets = torch.tensor(
        [[dx, dy, dz]
         for dx in (-1, 0, 1)
         for dy in (-1, 0, 1)
         for dz in (-1, 0, 1)],
        dtype=torch.int64,
    )

    # output neighbor indices
    idx_all = torch.empty((N, k), dtype=torch.long)

    for i in range(N):
        center_vox = vox[i]
        cand = []

        # gather candidate points from neighboring voxels
        for off in offsets:
            key = tuple((center_vox + off).tolist())
            if key in voxel_dict:
                cand.extend(voxel_dict[key])

        if not cand:
            # fallback: if somehow isolated, just use self as all neighbors
            idx_all[i] = i
            continue

        cand = torch.tensor(cand, dtype=torch.long)
        cand_pts = x_cpu[cand]  # [M, D]

        # 4) compute distances to candidates only
        diff = cand_pts - x_cpu[i].unsqueeze(0)  # [M, D]
        dist2 = (diff * diff).sum(dim=1)         # [M]

        # 5) pick up to k nearest from candidates
        if cand.shape[0] >= k + 1:
            knn_local = dist2.topk(k + 1, largest=False).indices
            # drop self if it's among them
            knn_global = cand[knn_local]
            # remove i if present, then pad back to k
            mask_not_self = (knn_global != i)
            knn_global = knn_global[mask_not_self]
            if knn_global.numel() < k:
                knn_global = torch.nn.functional.pad(
                    knn_global,
                    (0, k - knn_global.numel()),
                    value=i,
                )
            else:
                knn_global = knn_global[:k]
        else:
            # fewer than k+1 candidates: sort all and pad
            num_cand = cand.shape[0]
            knn_local = dist2.topk(num_cand, largest=False).indices
            knn_global = cand[knn_local]
            # remove self if present
            knn_global = knn_global[knn_global != i]
            if knn_global.numel() == 0:
                knn_global = torch.tensor([i], dtype=torch.long)
            if knn_global.numel() < k:
                knn_global = torch.nn.functional.pad(
                    knn_global,
                    (0, k - knn_global.numel()),
                    value=knn_global[-1].item(),
                )
            else:
                knn_global = knn_global[:k]

        idx_all[i] = knn_global

    return idx_all.to(device)

    
def gaussian_boundary_weights(
    point_set,
    seg,
    k=16,
    sigma=0.05,
    boundary_boost=2.0,
    eps=1e-6,
    voxel_size=None,
):
    """
    point_set: [N, 3]
    seg: [N]
    returns: [N] boundary-aware weights
    """
    if voxel_size is None:
        # heuristic: a bit larger than sigma
        voxel_size = sigma * 3.0

#     print("before approx knn")
    idx = approx_knn_indices_voxel(point_set, k, voxel_size)   # [N, k]
#     print("after approx knn")

    neighbors = point_set[idx]                    # [N, k, 3]
    center = point_set.unsqueeze(1)               # [N, 1, 3]

    dist2 = ((neighbors - center) ** 2).sum(-1)   # [N, k]

    # Gaussian kernel
    gaussian = torch.exp(-dist2 / (2 * sigma**2)) # [N, k]

    neighbor_labels = seg[idx]                    # [N, k]
    diff = (neighbor_labels != seg.unsqueeze(1)).float()

    # weighted boundary response
    boundary_response = (gaussian * diff).sum(dim=1)

    # normalize to [0, 1]-ish
    norm = gaussian.sum(dim=1) + eps
    boundary_response = boundary_response / norm

    w_boundary = 1.0 + boundary_boost * boundary_response
    return w_boundary



def compute_point_weights_gaussian(
    point_set,
    seg,
    k=16,
    sigma=0.05,
    voxel_size=None,
):
    w_class = class_weights(seg)
#     print("before gaussian")
    w_boundary = gaussian_boundary_weights(
        point_set,
        seg,
        k=k,
        sigma=sigma,
        boundary_boost=2.0,
        voxel_size=voxel_size,
    )
#     print("after gaussian")

    w = w_class * w_boundary
    w = w / w.mean()   # stabilize training
    return w


