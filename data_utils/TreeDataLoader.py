# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
from copy import deepcopy
import torch
from collections import defaultdict


warnings.filterwarnings('ignore')



# Grouping and batching idea

def group_and_batch(root_folder, json_path, batch_size):
    with open(os.path.join(root_folder, 'train_test_split', json_path), 'r') as f:
        train_ids = set([str(d.split('/')[-1]) for d in json.load(f)])
        
    instances = []
    for train_id in train_ids:
        filepath = os.path.join(root_folder, "0", "numpy", train_id)
        arr = np.load(filepath)
        instances.append([train_id, arr.shape[0]])
        
    instances.sort(key = lambda x: x[1])
    sorted_instances_ids = [x[0] for x in instances]
    
    num_groups = int(len(instances) / batch_size)
    
    groups = defaultdict(list)
    for gid in range(num_groups):
        start = gid * batch_size
        end = start + batch_size
        idxs = range(start, end)
        groups[gid].extend(idxs)
    
    group_min_n = {}
    for gid, idxs in groups.items():
        if len(idxs) == 0:
            continue
            
        group_lengths = [instances[i][1] for i in idxs]
        group_min_n[gid] = min(group_lengths)
        
#         for idx in idxs:
#             instance_filename = instances[idx][0]
#             filepath = os.path.join(root_folder, "0", "numpy", instance_filename)
#             point_set = np.load(filepath).astype(np.float32)
            
#             point_set_exp = np.expand_dims(point_set, axis=0)
            
#             fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=group_min_n[gid], 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
#             new_data = index_points(point_set_exp, fps_idx)
#             new_filepath = os.path.join(root_folder, "0", "numpy_after_fps", instance_filename)
#             np.save(new_filepath, new_data[0, :, :])
            
#         print(f"Group {gid}: {len(idxs)} instances, min_N = {group_min_n[gid]}")

    return instances, groups, group_min_n
       

class GroupedFPSDataset(Dataset):
    def __init__(self, root, instances, idxs, batch_size, npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        self.instances = instances
        self.idxs = idxs
        self.batch_size = batch_size

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
                
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        
        # self.cat = {"fallen_trees": 0}

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
#         print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg, filename = self.cache[index]
#         else:
        inst_idx = self.idxs[index]
        filename = self.instances[inst_idx][0]

#             cat = self.datapath[index][0]
#             cls = self.classes[cat]
#             cls = np.array([cls]).astype(np.int32)

        cat = "fallen_trees"
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)

#             data = np.loadtxt(fn[1]).astype(np.float32)
        data = np.load(os.path.join(self.root, "0", "numpy_after_fps", filename)).astype(np.float32)

        point_set = deepcopy(data)

        seg = data[:, -1].astype(np.int32) # for all point clouds last index is for parts
        ignore = data[:, -2].astype(np.int32) # 2nd last index is for ignore point
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls, seg, filename)
        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # N, C
        
        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])
        
        point_set_all_pts = deepcopy(point_set)
        seg_all_pts = deepcopy(seg)
        
#         point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)
#         fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
#         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=self.npoints, 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
#         new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
#         seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))

#         point_set = new_data[0, :, :]
#         print(point_set.shape, cls.shape, seg.shape, point_set.shape, filename)
        
#         if self.is_test:
#             return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return self.batch_size
#         return len(self.datapath)





def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def rgb_normalize(points):
    min_vals = points.min(axis=0)      # shape: (,3)
    max_vals = points.max(axis=0)      # shape: (,3)
    
    normalized = (points - min_vals) / (max_vals - min_vals + 1e-8)  # avoid div by zero
    
    return normalized

def intensity_normalize(points):
    min_vals = points.min()      # shape: (,3)
    max_vals = points.max()      # shape: (,3)
    
    normalized = (points - min_vals) / (max_vals - min_vals + 1e-8)  # avoid div by zero
    
    return normalized

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
#     device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
#     device = xyz.device
    xyz = xyz[:, :, :3]
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



import torch

def _fps_basic(xyz, npoint):
    """
    Standard FPS on coords only.
    xyz: [B, N, 3]
    returns: [B, npoint] indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance  = torch.ones(B, N, device=device) * 1e10
    farthest  = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def _fps_on_mask(xyz_coords, mask, npoint):
    """
    FPS restricted to a boolean mask of valid points.

    xyz_coords: [B, N, 3]
    mask      : [B, N] bool
    npoint    : int

    returns: [B, npoint] indices in the ORIGINAL [0..N-1] space
    """
    device = xyz_coords.device
    B, N, _ = xyz_coords.shape
    out_idx = torch.zeros(B, npoint, dtype=torch.long, device=device)

    for b in range(B):
        valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(1)  # [Nv]
        Nv = valid_idx.numel()
        if Nv == 0:
            raise ValueError(f"No valid points in batch {b} for FPS.")
        # If fewer valid points than requested, just repeat some
        n_b = min(npoint, Nv)

        xyz_valid = xyz_coords[b:b+1, valid_idx, :]  # [1, Nv, 3]
        fps_idx_local = _fps_basic(xyz_valid, n_b)[0]  # [n_b] in [0..Nv-1]
        chosen = valid_idx[fps_idx_local]              # [n_b] in [0..N-1]

        # fill output; if n_b < npoint, pad by repeating
        out_idx[b, :n_b] = chosen
        if n_b < npoint:
            # simple pad: repeat some indices
            repeat_idx = chosen[torch.randint(0, n_b, (npoint - n_b,), device=device)]
            out_idx[b, n_b:] = repeat_idx

    return out_idx


def farthest_point_sample_with_bck_and_ignore(
    xyz,
    npoint,
    fg_ratio=0.7,
    bg_label=3,
    ignore_channel=3,
    label_channel=4,
):
    """
    Stratified farthest point sampling with ignore + background control.

    Input:
        xyz : [B, N, 5]
              channels = [X, Y, Z, ignore_point, part_label]
        npoint : total #samples per batch
        fg_ratio : fraction of samples reserved for foreground
        bg_label : label value used for background (here 3)
        ignore_channel : index of ignore_point channel (3)
        label_channel  : index of part_label channel (4)

    Return:
        centroids: sampled indices, [B, npoint]
    """
    device = xyz.device
    coords = xyz[:, :, :3]                  # [B, N, 3]
    ignore = xyz[:, :, ignore_channel] > 0  # [B, N] bool
    parts  = xyz[:, :, label_channel].long()

    # valid = not ignored
    valid = ~ignore                         # [B, N]

    # foreground = valid & part in {0,1,2}
    fg_mask = valid & torch.isin(parts, torch.tensor([0,1,2], device=device))
    # background = valid & part == bg_label
    bg_mask = valid & (parts == bg_label)

    B, N, _ = coords.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)

    n_fg_target = int(npoint * fg_ratio)
    n_bg_target = npoint - n_fg_target

    for b in range(B):
        fg_count = fg_mask[b].sum().item()
        bg_count = bg_mask[b].sum().item()

        # adapt per-batch in case some batch has little FG/BG
        n_fg = min(n_fg_target, max(fg_count, 0))
        n_bg = npoint - n_fg
        n_bg = min(n_bg, max(bg_count, 0))

        # if still underfull, steal from the other set or fallback to all valid
        if n_fg == 0 and n_bg == 0:
            # fallback: FPS on all valid points
            if valid[b].sum() == 0:
                # nothing usable; just sample from all points
                all_idx = _fps_basic(coords[b:b+1], npoint)[0]
                centroids[b] = all_idx
                continue
            all_idx = _fps_on_mask(coords[b:b+1], valid[b:b+1], npoint)[0]
            centroids[b] = all_idx
            continue

        # ensure total = npoint, if one side lacks enough, fill with other
        needed = npoint
        n_fg = min(n_fg, fg_count)
        needed -= n_fg
        n_bg = min(max(needed, 0), bg_count)
        needed -= n_bg

        # If still need more, use all valid points to fill
        extra_idx = None
        if needed > 0 and valid[b].sum().item() > (n_fg + n_bg):
            extra_idx = _fps_on_mask(coords[b:b+1], valid[b:b+1], needed)[0]

        # sample FG
        idx_fg = torch.zeros(0, dtype=torch.long, device=device)
        if n_fg > 0 and fg_count > 0:
            idx_fg = _fps_on_mask(coords[b:b+1], fg_mask[b:b+1], n_fg)[0]

        # sample BG
        idx_bg = torch.zeros(0, dtype=torch.long, device=device)
        if n_bg > 0 and bg_count > 0:
            idx_bg = _fps_on_mask(coords[b:b+1], bg_mask[b:b+1], n_bg)[0]

        # concat (optionally shuffle if you want)
        chosen = torch.cat([idx_fg, idx_bg], dim=0)
        if extra_idx is not None:
            chosen = torch.cat([chosen, extra_idx], dim=0)

        # If we accidentally overshoot (rounding), trim
        if chosen.numel() > npoint:
            chosen = chosen[:npoint]

        centroids[b] = chosen

    return centroids



class TreePartNormalDatasetAllPoints(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    print(pc_data.shape)
                    if pc_data.shape[0] > 512:
                        train_ids.append(fn)
                        
                    else:
                        print(pc_data.shape)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    print(pc_data.shape)
                    if pc_data.shape[0] > 512:
                        test_ids.append(fn)
                    else:
                        print(pc_data.shape)
            

            if split == 'trainval':
                fns = [fn for fn in fns if (fn in train_ids)]
#                 print("train: ", fns)
            elif split == 'train':
                fns = [fn for fn in fns if fn in train_ids]
#             elif split == 'val':
#                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn in test_ids]
#                 print("test: ", fns)
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
#             print(fns)
            for fn in fns:
#                 token = (os.path.splitext(os.path.basename(fn))[0])
#                 print("token: ", token)
                self.meta[item].append(os.path.join(dir_point, fn))
        
#         print("cat: ", self.cat)
        self.datapath = []
        for item in self.cat:
#             print(item)
#             print(self.meta[item])
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
#         print("datapath: ", self.datapath)


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg, filename = self.cache[index]
        else:
            fn = self.datapath[index]
            filename = fn[1]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
#             data = np.loadtxt(fn[1]).astype(np.float32)
            data = np.load(fn[1]).astype(np.float32)
    
#             point_set_all_pts = deepcopy(data)
        
#             if not self.normal_channel:
#                 point_set = deepcopy(data[:, 0:3])
#             else:
#                 point_set = deepcopy(data[:, 0:6])

#             if not self.use_rgbi:
#                 point_set = deepcopy(data[:, 0:3])
#             else:
#                 point_set = deepcopy(data[:, 0:7]) # x,y,z,r,g,b,intensity
            point_set = deepcopy(data)
                
            seg = data[:, -1].astype(np.int32) # for all point clouds last index is for parts
            ignore = data[:, -2].astype(np.int32) # 2nd last index is for ignore point
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, fn[1])
        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # N, C
        
        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])
        
        point_set_all_pts = deepcopy(point_set)
        seg_all_pts = deepcopy(seg)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)
#         fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
#         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=self.npoints, 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
#         new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
#         seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)


    def __len__(self):
        return len(self.datapath)
    

    
class TreeSemDataset60kPointsFGvsBG(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        print("use_rgbi: ", self.use_rgbi)


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_60k_pts.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_clas_fps_6000_split.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_60k_pts.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_clas_fps_6000_split.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_60k_repeat_and_overlap") # 8192 pts
#             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
       
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    train_ids.append(fn)
#                     if pc_data.shape[0] >= 2048:
#                         train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    test_ids.append(fn)
#                     if pc_data.shape[0] >= 2048:
#                         test_ids.append(fn)
            

            if split == 'trainval':
                fns = [fn for fn in fns if (fn in train_ids)]
#                 print("train: ", fns)
            elif split == 'train':
                fns = [fn for fn in fns if fn in train_ids]
#             elif split == 'val':
#                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn in test_ids]
#                 print("test: ", fns)
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
#             print(fns)
            for fn in fns:
#                 token = (os.path.splitext(os.path.basename(fn))[0])
#                 print("token: ", token)
                self.meta[item].append(os.path.join(dir_point, fn))
        
#         print("cat: ", self.cat)
        self.datapath = []
        for item in self.cat:
#             print(item)
#             print(self.meta[item])
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
#         self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
#         print("datapath: ", self.datapath)


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg, filename = self.cache[index]
        else:
            fn = self.datapath[index]
            filename = fn[1]
            
            file_name = os.path.basename(filename)
            instance_id = file_name.split("_")[2]
            
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
#             data = np.loadtxt(fn[1]).astype(np.float32)
            data = np.load(fn[1]).astype(np.float32)
    
#             point_set_all_pts = deepcopy(data)
        
#             if not self.normal_channel:
#                 point_set = deepcopy(data[:, 0:3])
#             else:
#                 point_set = deepcopy(data[:, 0:6])

#             if not self.use_rgbi:
#                 point_set = deepcopy(data[:, 0:3])
#             else:
#                 point_set = deepcopy(data[:, 0:7]) # x,y,z,r,g,b,intensity
            point_set = deepcopy(data)
                
            seg = data[:, -1].astype(np.int32) # for all point clouds last index is for parts
            ignore = data[:, -2].astype(np.int32) # 2nd last index is for ignore point
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, fn[1])
        
        point_set_orig = deepcopy(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # N, C
        
        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])
        
        point_set_all_pts = deepcopy(point_set)
        seg_all_pts = deepcopy(seg)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)
#         fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
#         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=self.npoints, 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
#         new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
#         seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))

        # NEW: only FG vs BG
        seg = np.where(seg == 3, 1, 0) # 3 becomes 1 (background); 0,1,2 becomse 0 (foreground)
        
        point_set = point_set_exp[0, :, :]
        
        # TODO: compute sample weight !!!!!!!!!!!!!!!!!!!!
        #
        #
        #
        #
        #
        #################################
#         print("before sample wt")
#         sample_weight = compute_point_weights_gaussian(torch.tensor(point_set_orig[:, :3]), torch.tensor(seg), k=5, sigma=0.05)
        
        gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights", f"fallen_trees_{instance_id}.npy")
        sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        
        sample_weight = np.ones((60000))
#         print("sample_weight: ", sample_weight.shape)
        

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)

#         print("point_set_all_pts_SHAPE_1: ", point_set_all_pts.shape)
#         print("filename: ", filename)
        if self.is_test:
            # NEW: get all points
            file_name = os.path.basename(filename)
            instance_id = file_name.split("_")[2]
            
            # fallen_trees_{instance_id}_fps_block_{fps_block_id}.npy
            all_pts_file_path = os.path.join(self.root, "0", "numpy", f"fallen_trees_{instance_id}.npy")
            
            point_set_all_pts = np.load(all_pts_file_path).astype(np.float32)
            seg_all_pts = point_set_all_pts[:, -1].astype(np.int32)
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
#             print("point_set_all_pts_SHAPE_2: ", point_set_all_pts.shape)
            
            # NOTE: -3 index of point_set_all_pts is is_repeated field
            
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, sample_weight
        
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
