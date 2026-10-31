# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
from copy import deepcopy
import torch
from .utils import compute_point_weights_gaussian

import numpy as np


warnings.filterwarnings('ignore')

def _random_yaw_rotation(points_xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return points_xyz @ R.T

def _xyz_jitter(points_xyz: np.ndarray, rng: np.random.Generator, sigma=0.005, clip=0.02) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=points_xyz.shape).astype(np.float32)
    noise = np.clip(noise, -clip, clip)
    return points_xyz + noise

def _color_jitter_rgb(rgb: np.ndarray, rng: np.random.Generator,
                      brightness=0.10, contrast=0.10, saturation=0.10) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    orig_min = float(np.min(rgb))
    orig_max = float(np.max(rgb))

    # brightness
    b = rng.uniform(-brightness, brightness)
    rgb = rgb + b * (orig_max - orig_min + 1e-6)

    # contrast
    c = rng.uniform(1.0 - contrast, 1.0 + contrast)
    mean = np.mean(rgb, axis=0, keepdims=True)
    rgb = (rgb - mean) * c + mean

    # saturation
    s = rng.uniform(1.0 - saturation, 1.0 + saturation)
    gray = np.mean(rgb, axis=1, keepdims=True)
    rgb = (rgb - gray) * s + gray

    return np.clip(rgb, orig_min, orig_max)

def _intensity_jitter(intensity: np.ndarray, rng: np.random.Generator,
                      mul_range=0.10, add_sigma=0.01) -> np.ndarray:
    intensity = intensity.astype(np.float32)
    orig_min = float(np.min(intensity))
    orig_max = float(np.max(intensity))

    m = rng.uniform(1.0 - mul_range, 1.0 + mul_range)
    intensity = intensity * m
    intensity = intensity + rng.normal(0.0, add_sigma, size=intensity.shape).astype(np.float32)

    return np.clip(intensity, orig_min, orig_max)



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def pc_normalize_new(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc = pc / m
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


class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


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
            train_ids = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[-1]) for d in json.load(f)])
            
#         print("train_ids: ", train_ids)
#         print("test_ids: ", test_ids)
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "xyz_labels")
            fns = sorted(os.listdir(dir_point))
#             print(fns[0][0:-4])
#             if split == 'trainval':
#                 fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
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

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
#         print("datapath: ", self.datapath)


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
#             data = np.loadtxt(fn[1]).astype(np.float32)
            data = np.load(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = deepcopy(data[:, 0:3])
            else:
                point_set = deepcopy(data[:, 0:6])
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi


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
                    if pc_data.shape[0] >= 512:
                        train_ids.append(fn)
                    else:
                        print(pc_data.shape)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= 512:
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
        
#         print("classes: ", self.classes)
        
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
        
#             if not self.normal_channel:
#                 point_set = deepcopy(data[:, 0:3])
#             else:
#                 point_set = deepcopy(data[:, 0:6])
                
            if not self.use_rgbi:
                point_set = deepcopy(data[:, 0:3])
            else:
                point_set = deepcopy(data[:, 0:7]) # x,y,z,r,g,b,intensity
            
            print(data.shape)
            seg = data[:, -2].astype(np.int32) # second last index is for parts, last index is to indicate repeated point cloud
            print(np.unique(seg))
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, fn[1])
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])
    

        return point_set, cls, seg, data, filename

    def __len__(self):
        return len(self.datapath)

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

    
class TreePartNormalDatasetAllPointsFGBG(Dataset):
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
#                     print(pc_data.shape)
                    if pc_data.shape[0] > 512:
                        train_ids.append(fn)
                    else:
                        print(pc_data.shape)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
#                     print(pc_data.shape)
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

        seg = np.where(seg == 3, 1, 0)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)

    
class TreePartNormalDatasetLargeNumPoints(Dataset):
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
        
#             if not self.normal_channel:
#                 point_set = deepcopy(data[:, 0:3])
#             else:
#                 point_set = deepcopy(data[:, 0:6])

            if not self.use_rgbi:
                point_set = deepcopy(data[:, 0:3])
            else:
                point_set = deepcopy(data[:, 0:7]) # x,y,z,r,g,b,intensity
#             point_set = deepcopy(data)
                
            seg = data[:, -1].astype(np.int32) # for all point clouds last index is for parts
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, fn[1])
        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # N, C
        
        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])
        
        point_set_all_pts = deepcopy(point_set)
        seg_all_pts = deepcopy(seg)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)

class TreePartNormalDatasetLargeNumPointsWithBackground(Dataset):
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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, fn[1])
        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # N, C
        
        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])
        
        point_set_all_pts = deepcopy(point_set)
        seg_all_pts = deepcopy(seg)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
#         point_set = new_data[0, :, :]
        
        if not self.use_rgbi:
            point_set = deepcopy(new_data[0, :, 0:3])
        else:
            point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset2048PointsWithBackgroundWithIgnore(Dataset):
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
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
#         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=self.npoints, 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
    
class TreePartNormalDataset6000PointsFGBG(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
#             train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
#             test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            
        with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            if npoints == 4000:
#                 dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k_with_crossover")
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k")
            elif npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_6k")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    train_ids.append(fn)
#                     if pc_data.shape[0] >= self.npoints:
#                         train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    test_ids.append(fn)
#                     if pc_data.shape[0] >= self.npoints:
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
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
#         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=self.npoints, 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        seg = np.where(seg==3, 1, 0)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsFGBGFrom2D(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        
        with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            if npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k_with_crossover")
            elif npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_6k_with_crossover")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    test_ids.append(fn)
#                     if pc_data.shape[0] >= self.npoints:
#                         test_ids.append(fn)
            

            
            fns = [fn for fn in fns if fn in test_ids]

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
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
#         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=self.npoints, 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        seg = np.where(seg==3, 1, 0)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class SyntheticDatasetPostTest(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        
        with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            if npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k_with_crossover")
            elif npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_6k_with_crossover")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    test_ids.append(fn)
#                     if pc_data.shape[0] >= self.npoints:
#                         test_ids.append(fn)
            

            
            fns = [fn for fn in fns if fn in test_ids]

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
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
#         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=self.npoints, 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        seg = np.where(seg==3, 1, 0)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsFGBGFrom2D_FINAL_INFERENCE(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        
        with open(os.path.join(self.root, 'train_test_split', f'test.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            if npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k")
            elif npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_6k")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    test_ids.append(fn)
#                     if pc_data.shape[0] >= self.npoints:
#                         test_ids.append(fn)
            

            
            fns = [fn for fn in fns if fn in test_ids]

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
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
#         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
#                                                             npoint=self.npoints, 
#                                                            fg_ratio=0.7,
#                                                             bg_label=3,
#                                                             ignore_channel=-2,
#                                                             label_channel=-1,)
         
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        seg = np.where(seg==3, 1, 0)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
    
# class TreePartNormalDataset6000PointsFGBGFrom2D_NO_FPS(Dataset):
#     def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0):
#         self.npoints = npoints
#         self.root = root
#         self.catfile = os.path.join(self.root, 'category.txt')
#         self.cat = {}
#         self.normal_channel = normal_channel
#         self.use_rgbi = use_rgbi
#         self.is_test = is_test
#         self.fold_id = fold_id
        
#         print("use_rgbi: ", self.use_rgbi)


#         with open(self.catfile, 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = ls[1]
#         self.cat = {k: v for k, v in self.cat.items()}
#         self.classes_original = dict(zip(self.cat, range(len(self.cat))))

#         if not class_choice is  None:
#             self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
#         # print(self.cat)

#         self.meta = {}
        
#         with open(os.path.join(self.root, 'train_test_split', f'test.json'), 'r') as f:
#             test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
#         for item in self.cat:
#             # print('category', item)
#             self.meta[item] = []
#             if npoints == 4000:
#                 dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k_with_crossover")
#             elif npoints == 6000:
#                 dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_6k_with_crossover")
#             fns = sorted(os.listdir(dir_point))
            
#             # filter point cloud that has less than 2048 points
#             train_ids = []
#             test_ids = []
            
#             for fn in fns:
#                 if fn in test_ids_orig:
#                     fn_path = os.path.join(dir_point, fn)
#                     pc_data = np.load(fn_path)
#                     test_ids.append(fn)
# #                     if pc_data.shape[0] >= self.npoints:
# #                         test_ids.append(fn)
            

            
#             fns = [fn for fn in fns if fn in test_ids]

#             # print(os.path.basename(fns))
# #             print(fns)
#             for fn in fns:
# #                 token = (os.path.splitext(os.path.basename(fn))[0])
# #                 print("token: ", token)
#                 self.meta[item].append(os.path.join(dir_point, fn))
        
# #         print("cat: ", self.cat)
#         self.datapath = []
#         for item in self.cat:
# #             print(item)
# #             print(self.meta[item])
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))

#         self.classes = {}
#         for i in self.cat.keys():
#             self.classes[i] = self.classes_original[i]
        
#         print("classes: ", self.classes)
        
#         # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
# #         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
#         self.seg_classes = {'fallen_trees': [0, 1]}

#         # for cat in sorted(self.seg_classes.keys()):
#         #     print(cat, self.seg_classes[cat])

#         self.cache = {}  # from index to (point_set, cls, seg) tuple
#         self.cache_size = 20000
# #         print("datapath: ", self.datapath)


#     def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg, filename = self.cache[index]
#         else:
#             fn = self.datapath[index]
#             filename = fn[1]
            
#             test_filename = os.path.basename(filename).split(".")[0] # fallen_trees_298.npy
#             fps_file_path = os.path.join(self.root, "0", f"numpy_split_fps_4000_2_blocks_fold_{self.fold_id}", f"{test_filename}_fps_block_single.npy")
            
#             point_set_fps = np.load(fps_file_path)
            
#             cat = self.datapath[index][0]
#             cls = self.classes[cat]
#             cls = np.array([cls]).astype(np.int32)
# #             data = np.loadtxt(fn[1]).astype(np.float32)
#             data = np.load(fn[1]).astype(np.float32)
    
# #             point_set_all_pts = deepcopy(data)
        
# #             if not self.normal_channel:
# #                 point_set = deepcopy(data[:, 0:3])
# #             else:
# #                 point_set = deepcopy(data[:, 0:6])

# #             if not self.use_rgbi:
# #                 point_set = deepcopy(data[:, 0:3])
# #             else:
# #                 point_set = deepcopy(data[:, 0:7]) # x,y,z,r,g,b,intensity
#             point_set = deepcopy(data)
                
#             seg = data[:, -1].astype(np.int32) # for all point clouds last index is for parts
#             ignore = data[:, -2].astype(np.int32) # 2nd last index is for ignore point
#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls, seg, fn[1])
        
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # N, C
#         point_set_fps[:, 0:3] = pc_normalize(point_set_fps[:, 0:3])
        
#         if self.use_rgbi:
#             point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
#             point_set[:, 6] = intensity_normalize(point_set[:, 6])
            
#             point_set_fps[:, 3:6] = rgb_normalize(point_set_fps[:, 3:6])
#             point_set_fps[:, 6] = intensity_normalize(point_set_fps[:, 6])
        
#         point_set_all_pts = deepcopy(point_set)
#         seg_all_pts = deepcopy(seg)
        
#         point_set_exp = np.expand_dims(point_set, axis=0)
# #         print("point_set_exp: ", point_set_exp.shape)
#         fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        
# #         fps_idx = farthest_point_sample_with_bck_and_ignore(torch.tensor(point_set_exp), 
# #                                                             npoint=self.npoints, 
# #                                                            fg_ratio=0.7,
# #                                                             bg_label=3,
# #                                                             ignore_channel=-2,
# #                                                             label_channel=-1,)
         
#         new_data = index_points(point_set_exp, fps_idx)
        
        
#         seg_new = np.expand_dims(seg, axis=0) # 1, N
#         seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
#         seg_new = torch.tensor(seg_new)
#         seg_new = index_points(seg_new, fps_idx)
#         seg = seg_new[0, :, :]
#         seg = np.array(seg)
#         seg = seg.reshape(-1)
        
#         seg = np.where(seg==3, 1, 0)
# #         print(seg[0, 0])

# #         any_greater = np.any(seg > 3)
# #         if any_greater:
# #             print(np.unique(seg))
        
#         point_set = new_data[0, :, :]

# #         if not self.use_rgbi:
# #             point_set = deepcopy(new_data[0, :, 0:3])
# #         else:
# #             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
# #         print(point_set.shape, seg.shape)
        
#         if self.is_test:
#             seg_all_pts = np.where(seg_all_pts==3, 1, 0)
#             return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
#         return point_set, cls, seg, point_set, filename


#     def __len__(self):
#         return len(self.datapath)
    

class TreePartNormalDataset6000PointsFGBGStage2(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_all_pts.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_all_pts.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "for_stage_2_numpy_full")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        seg = np.where(seg==3, 1, 0)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsFGBGStage2From2D(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        
        with open(os.path.join(self.root, 'train_test_split', f'test.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
#                 if fn in train_ids_orig:
#                     fn_path = os.path.join(dir_point, fn)
#                     pc_data = np.load(fn_path)
# #                     train_ids.append(fn)
#                     if pc_data.shape[0] >= self.npoints:
#                         train_ids.append(fn)
                        
                if fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

#             if split == 'trainval':
#                 fns = [fn for fn in fns if (fn in train_ids)]
# #                 print("train: ", fns)
#             elif split == 'train':
#                 fns = [fn for fn in fns if fn in train_ids]
# #             elif split == 'val':
# #                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
#             elif split == 'test':
#                 fns = [fn for fn in fns if fn in test_ids]
# #                 print("test: ", fns)
#             else:
#                 print('Unknown split: %s. Exiting..' % (split))
#                 exit(-1)
            fns = [fn for fn in fns if fn in test_ids]

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


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        seg = np.where(seg==3, 1, 0)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsFGBGStage2From2DNoStage1(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        
        with open(os.path.join(self.root, 'train_test_split', f'test.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
#                 if fn in train_ids_orig:
#                     fn_path = os.path.join(dir_point, fn)
#                     pc_data = np.load(fn_path)
# #                     train_ids.append(fn)
#                     if pc_data.shape[0] >= self.npoints:
#                         train_ids.append(fn)
                        
                if fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

#             if split == 'trainval':
#                 fns = [fn for fn in fns if (fn in train_ids)]
# #                 print("train: ", fns)
#             elif split == 'train':
#                 fns = [fn for fn in fns if fn in train_ids]
# #             elif split == 'val':
# #                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
#             elif split == 'test':
#                 fns = [fn for fn in fns if fn in test_ids]
# #                 print("test: ", fns)
#             else:
#                 print('Unknown split: %s. Exiting..' % (split))
#                 exit(-1)
            fns = [fn for fn in fns if fn in test_ids]

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


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        seg = np.where(seg==3, 1, 0)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetTestStage3(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_all_pts.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_all_pts.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "for_stage_3_numpy_full")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
        ignore_all_pts = deepcopy(ignore)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        ignore_new = np.expand_dims(ignore, axis=0) # 1, N
        ignore_new = np.expand_dims(ignore_new, axis=-1) # 1, N, 1
        ignore_new = torch.tensor(ignore_new)
        ignore_new = index_points(ignore_new, fps_idx)
        ignore = ignore_new[0, :, :]
        ignore = np.array(ignore)
        ignore = ignore.reshape(-1)
        
        seg = np.where(seg==4, 3, seg)
        seg = np.where(ignore==1, 3, seg)
        
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore_all_pts==1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetTestStage3FINAL(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k_with_crossover")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
        ignore_all_pts = deepcopy(ignore)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        ignore_new = np.expand_dims(ignore, axis=0) # 1, N
        ignore_new = np.expand_dims(ignore_new, axis=-1) # 1, N, 1
        ignore_new = torch.tensor(ignore_new)
        ignore_new = index_points(ignore_new, fps_idx)
        ignore = ignore_new[0, :, :]
        ignore = np.array(ignore)
        ignore = ignore.reshape(-1)
        
        seg = np.where(seg==4, 3, seg)
        seg = np.where(ignore==1, 3, seg)
        
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore_all_pts==1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetTestStage3From2D(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        with open(os.path.join(self.root, 'train_test_split', f'test.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
#                 if fn in train_ids_orig:
#                     fn_path = os.path.join(dir_point, fn)
#                     pc_data = np.load(fn_path)
# #                     train_ids.append(fn)
#                     if pc_data.shape[0] >= self.npoints:
#                         train_ids.append(fn)
                        
                if fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

#             if split == 'trainval':
#                 fns = [fn for fn in fns if (fn in train_ids)]
# #                 print("train: ", fns)
#             elif split == 'train':
#                 fns = [fn for fn in fns if fn in train_ids]
# #             elif split == 'val':
# #                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
#             elif split == 'test':
#                 fns = [fn for fn in fns if fn in test_ids]
# #                 print("test: ", fns)
#             else:
#                 print('Unknown split: %s. Exiting..' % (split))
#                 exit(-1)

            fns = [fn for fn in fns if fn in test_ids]

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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
        ignore_all_pts = deepcopy(ignore)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        ignore_new = np.expand_dims(ignore, axis=0) # 1, N
        ignore_new = np.expand_dims(ignore_new, axis=-1) # 1, N, 1
        ignore_new = torch.tensor(ignore_new)
        ignore_new = index_points(ignore_new, fps_idx)
        ignore = ignore_new[0, :, :]
        ignore = np.array(ignore)
        ignore = ignore.reshape(-1)
        
        seg = np.where(seg==4, 3, seg)
        seg = np.where(ignore==1, 3, seg)
        
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore_all_pts==1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
    
class TreePartNormalDatasetTestSingleStage(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
        ignore_all_pts = deepcopy(ignore)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        ignore_new = np.expand_dims(ignore, axis=0) # 1, N
        ignore_new = np.expand_dims(ignore_new, axis=-1) # 1, N, 1
        ignore_new = torch.tensor(ignore_new)
        ignore_new = index_points(ignore_new, fps_idx)
        ignore = ignore_new[0, :, :]
        ignore = np.array(ignore)
        ignore = ignore.reshape(-1)
        
        seg = np.where(seg==4, 3, seg)
        seg = np.where(ignore==1, 3, seg)
        
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore_all_pts==1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetTestSingleStageFINAL(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_4k_with_crossover")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
        ignore_all_pts = deepcopy(ignore)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        ignore_new = np.expand_dims(ignore, axis=0) # 1, N
        ignore_new = np.expand_dims(ignore_new, axis=-1) # 1, N, 1
        ignore_new = torch.tensor(ignore_new)
        ignore_new = index_points(ignore_new, fps_idx)
        ignore = ignore_new[0, :, :]
        ignore = np.array(ignore)
        ignore = ignore.reshape(-1)
        
        seg = np.where(seg==4, 3, seg)
        seg = np.where(ignore==1, 3, seg)
        
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore_all_pts==1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetTestStandingTrees(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        with open(os.path.join(self.root, 'train_test_split', f'train_all_pts.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', f'test_all_pts.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            
            if self.npoints == 15000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_15k")
            elif self.npoints == 30000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_30k")
                
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    
                    if pc_data.shape[0] >= 150000:
                        continue
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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
        self.seg_classes = {'standing_trees': [0, 1]}

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
        ignore_all_pts = deepcopy(ignore)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        ignore_new = np.expand_dims(ignore, axis=0) # 1, N
        ignore_new = np.expand_dims(ignore_new, axis=-1) # 1, N, 1
        ignore_new = torch.tensor(ignore_new)
        ignore_new = index_points(ignore_new, fps_idx)
        ignore = ignore_new[0, :, :]
        ignore = np.array(ignore)
        ignore = ignore.reshape(-1)
        
#         seg = np.where(seg==4, 3, seg)
        seg = np.where(ignore==1, 1, seg)
        
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
#             seg_all_pts = np.where(seg_all_pts==4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore_all_pts==1, 1, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetTestStandingTreesAllPts(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
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
        with open(os.path.join(self.root, 'train_test_split', f'train_all_pts.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', f'test_all_pts.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            
            if self.npoints == 15000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_15k")
            elif self.npoints == 30000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_padded_30k")
                
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    
                    if pc_data.shape[0] >= 150000:
                        continue
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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
        self.seg_classes = {'standing_trees': [0, 1]}

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
        ignore_all_pts = deepcopy(ignore)
        
        point_set_exp = np.expand_dims(point_set, axis=0)
#         print("point_set_exp: ", point_set_exp.shape)


        # TODO: repeat pts if less than 4000 ....
#         fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
#         new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
#         seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        ignore_new = np.expand_dims(ignore, axis=0) # 1, N
        ignore_new = np.expand_dims(ignore_new, axis=-1) # 1, N, 1
        ignore_new = torch.tensor(ignore_new)
#         ignore_new = index_points(ignore_new, fps_idx)
        ignore = ignore_new[0, :, :]
        ignore = np.array(ignore)
        ignore = ignore.reshape(-1)
        
#         seg = np.where(seg==4, 3, seg)
        seg = np.where(ignore==1, 1, seg)
        
        
        point_set = point_set_exp[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
#             seg_all_pts = np.where(seg_all_pts==4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore_all_pts==1, 1, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsFGBGStage2Thr30(Dataset):
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
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "test_numpy_thr30")
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     train_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= self.npoints:
                        test_ids.append(fn)
            

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


        # TODO: repeat pts if less than 4000 ....
        fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
        new_data = index_points(point_set_exp, fps_idx)
        
        
        seg_new = np.expand_dims(seg, axis=0) # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        
        seg = np.where(seg==3, 1, 0)
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = new_data[0, :, :]

#         if not self.use_rgbi:
#             point_set = deepcopy(new_data[0, :, 0:3])
#         else:
#             point_set = deepcopy(new_data[0, :, 0:7]) # x,y,z,r,g,b,intensity
            
#         print(point_set.shape, seg.shape)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts==3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, fps_idx
        
        return point_set, cls, seg, point_set, filename


    def __len__(self):
        return len(self.datapath)
    
    
    
class TreePartNormalDatasetALLPointsWithBackgroundWithIgnore(Dataset):
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
                    if pc_data.shape[0] >= 2048:
                        train_ids.append(fn)
                        
                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
#                     test_ids.append(fn)
                    if pc_data.shape[0] >= 2048:
                        test_ids.append(fn)
            

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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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

    
class TreePartNormalDataset8192PointsWithBackgroundWithIgnore(Dataset):
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
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_clas_fps_6000_split.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_clas_fps_6000_split.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps") # 8192 pts
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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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

class TreePartNormalDataset6000PointsWithOverlapPadding(Dataset):
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
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_clas_fps_6000_split.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_clas_fps_6000_split.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
    
    
class TreePartNormalDataset6000PointsWithOverlapPaddingFGBG(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
#             if self.fold_id == 0:
#                 with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_4000_pts.json'), 'r') as f:
#                     train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#                 with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_4000_pts.json'), 'r') as f:
#                     test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             elif self.fold_id == 1:
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            if self.npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 12000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
            elif self.npoints == 2048:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
            elif self.npoints == 3000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_4000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 5000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
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
        
        seg = np.where(seg == 3, 1, 0)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_4000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsWithOverlapPaddingFGBGIgnoreHandled(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, post_train=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        self.sigma = sigma
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        print("alpha_factor: ", self.alpha_factor)
        print("alpha_fg: ", self.alpha_fg)


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
        
        if self.npoints == 6000:
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_6k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_6k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_12k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_12k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_2k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_2k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_3k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_3k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            if post_train:
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            else:
                with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_2_blocks.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_2_blocks.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_5k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_5k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            if self.npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_6000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 12000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_12000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 2048:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_2048_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 3000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_3000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 5000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_5000_2_blocks_fold_{self.fold_id}") # 8192 pts
#             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
       
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                tree_id = int(fn.split("_")[2])
                 
                if fn in train_ids_orig:
                    if tree_id >= 790:
                        continue
                        
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
        
# #         print("cat: ", self.cat)
#         self.datapath = []
#         for item in self.cat:
# #             print(item)
# #             print(self.meta[item])
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))
            
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))

        # optional: further repeat everything
#         self.datapath = self.datapath * self.repeat_factor

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
#         print("datapath: ", self.datapath)


    def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg, filename = self.cache[index]
#         else:
        fn = self.datapath[index]
#             filename = fn[1]
        cat, filename, aug_id = self.datapath[index]

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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 3, 1, 0)
        seg = np.where(ignore == 1, 1, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                if self.alpha_factor is not None:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_6000_fold_{self.fold_id}_sigma_{self.sigma}_factor_{self.alpha_factor}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                else:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_6000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                if self.alpha_factor is not None:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}_factor_{self.alpha_factor}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                else:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                
#                 sample_weight_alpha = np.ones((self.npoints,))
#                 if self.alpha_fg is not None:
#                     crossover_alpha = np.where(ignore == 1, self.alpha_fg, 1) # crossover point get same alpha as alpha_fg
#                     sample_weight = sample_weight * crossover_alpha
                
                # set BG wt = 1 for test only ...
#                 sample_weight = np.where(seg == 1, 1, sample_weight) # running on tmux 4
        
#         sample_weight = np.ones((self.npoints,)) # running on tmux 3
        
        if self.alpha_fg is not None:
            if int(self.alpha_fg) == 0:
                # set BG wt = 1 for test only ...
                sample_weight = np.ones((self.npoints,))
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            seg_all_pts = np.where(ignore == 1, 1, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, ignore
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight, ignore


    def __len__(self):
        return len(self.datapath)
    
class SyntheticDatasetPostTrain(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, post_train=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        self.sigma = sigma
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        print("alpha_factor: ", self.alpha_factor)
        print("alpha_fg: ", self.alpha_fg)


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
        
        if self.npoints == 6000:
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_6k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_6k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_12k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_12k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_2k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_2k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_3k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_3k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            if post_train:
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            else:
                with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_5_blocks.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_5_blocks.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_5k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_5k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            if self.npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_6000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 12000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_12000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 2048:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_2048_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 3000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_3000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_5_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 5000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_5000_2_blocks_fold_{self.fold_id}") # 8192 pts
#             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
       
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
#             print(f"test_ids_orig: {test_ids_orig}")
            
            for fn in fns:
                tree_id = int(fn.split("_")[2])
                 
                if fn in train_ids_orig:
#                     if tree_id >= 790:
#                         continue
                        
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
#             print(f"test_ids: {test_ids}")

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
        
# #         print("cat: ", self.cat)
#         self.datapath = []
#         for item in self.cat:
# #             print(item)
# #             print(self.meta[item])
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))
            
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))

        # optional: further repeat everything
#         self.datapath = self.datapath * self.repeat_factor

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
#         print("datapath: ", self.datapath)


    def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg, filename = self.cache[index]
#         else:
        fn = self.datapath[index]
#             filename = fn[1]
        cat, filename, aug_id = self.datapath[index]

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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 3, 1, 0)
        seg = np.where(ignore == 1, 1, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                if self.alpha_factor is not None:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_6000_fold_{self.fold_id}_sigma_{self.sigma}_factor_{self.alpha_factor}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                else:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_6000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                if self.alpha_factor is not None:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}_factor_{self.alpha_factor}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                else:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                
#                 sample_weight_alpha = np.ones((self.npoints,))
#                 if self.alpha_fg is not None:
#                     crossover_alpha = np.where(ignore == 1, self.alpha_fg, 1) # crossover point get same alpha as alpha_fg
#                     sample_weight = sample_weight * crossover_alpha
                
                # set BG wt = 1 for test only ...
#                 sample_weight = np.where(seg == 1, 1, sample_weight) # running on tmux 4
        
#         sample_weight = np.ones((self.npoints,)) # running on tmux 3
        
        if self.alpha_fg is not None:
            if int(self.alpha_fg) == 0:
                # set BG wt = 1 for test only ...
                sample_weight = np.ones((self.npoints,))
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            seg_all_pts = np.where(ignore == 1, 1, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, ignore
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight, ignore


    def __len__(self):
        return len(self.datapath)
    
class SyntheticDatasetPostTrainStage2(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, post_train=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        self.sigma = sigma
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        print("alpha_factor: ", self.alpha_factor)
        print("alpha_fg: ", self.alpha_fg)


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
        
        if self.npoints == 6000:
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_6k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_6k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_12k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_12k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_2k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_2k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_3k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_3k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            if post_train:
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            else:
                with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_single_block.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_single_block.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_5k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_5k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            if self.npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_6000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 12000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_12000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 2048:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_2048_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 3000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_3000_2_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_single_block_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 5000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_5000_2_blocks_fold_{self.fold_id}") # 8192 pts
#             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
       
            fns = sorted(os.listdir(dir_point))
            
            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []
            
            for fn in fns:
                tree_id = int(fn.split("_")[2])
                 
                if fn in train_ids_orig:
                    if tree_id >= 790:
                        continue
                        
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
        
# #         print("cat: ", self.cat)
#         self.datapath = []
#         for item in self.cat:
# #             print(item)
# #             print(self.meta[item])
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))
            
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))

        # optional: further repeat everything
#         self.datapath = self.datapath * self.repeat_factor

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
#         print("datapath: ", self.datapath)


    def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg, filename = self.cache[index]
#         else:
        fn = self.datapath[index]
#             filename = fn[1]
        cat, filename, aug_id = self.datapath[index]

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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 3, 1, 0)
        seg = np.where(ignore == 1, 1, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                if self.alpha_factor is not None:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_6000_fold_{self.fold_id}_sigma_{self.sigma}_factor_{self.alpha_factor}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                else:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_6000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                if self.alpha_factor is not None:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}_factor_{self.alpha_factor}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                else:
                    gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                    sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                
#                 sample_weight_alpha = np.ones((self.npoints,))
#                 if self.alpha_fg is not None:
#                     crossover_alpha = np.where(ignore == 1, self.alpha_fg, 1) # crossover point get same alpha as alpha_fg
#                     sample_weight = sample_weight * crossover_alpha
                
                # set BG wt = 1 for test only ...
#                 sample_weight = np.where(seg == 1, 1, sample_weight) # running on tmux 4
        
#         sample_weight = np.ones((self.npoints,)) # running on tmux 3
        
        if self.alpha_fg is not None:
            if int(self.alpha_fg) == 0:
                # set BG wt = 1 for test only ...
                sample_weight = np.ones((self.npoints,))
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            seg_all_pts = np.where(ignore == 1, 1, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, ignore
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight, ignore


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsWithOverlapPaddingFGBGIgnoreHandledMinWt0(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, post_train=False, num_aug_copies=0):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        self.sigma = sigma
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_6k_pts_5_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_6k_pts_5_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            # with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_4000_pts.json'), 'r') as f:
            #     train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            # with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_4000_pts.json'), 'r') as f:
            #     test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            
            if post_train:
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            else:
                with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_5_blocks.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_5_blocks.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            if self.npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_6000_5_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 12000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
            elif self.npoints == 2048:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
            elif self.npoints == 3000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_5_blocks_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 5000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
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
        
# #         print("cat: ", self.cat)
#         self.datapath = []
#         for item in self.cat:
# #             print(item)
# #             print(self.meta[item])
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))
            
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))

        # optional: further repeat everything
#         self.datapath = self.datapath * self.repeat_factor

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
#         print("datapath: ", self.datapath)


    def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg, filename = self.cache[index]
#         else:
        fn = self.datapath[index]
#             filename = fn[1]
        cat, filename, aug_id = self.datapath[index]

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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 3, 1, 0)
        seg = np.where(ignore == 1, 1, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_6000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
                sample_weight = sample_weight - 1
                
                # set BG wt = 1 for test only ...
#                 sample_weight = np.where(seg == 1, 1, sample_weight) # running on tmux 4
        
#         sample_weight = np.ones((self.npoints,)) # running on tmux 3
        
        # set BG wt = 1 for test only ...
#         sample_weight = np.where(seg == 1, 1, sample_weight) # running on tmux 4
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            seg_all_pts = np.where(ignore == 1, 1, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, ignore
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight, ignore


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsWithOverlapPaddingFGBGIgnoreHandledSingleBlock(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, repeat_factor=1, subtraction=0, post_train=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        self.sigma = sigma
        self.repeat_factor = repeat_factor
        self.subtraction = subtraction
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            # with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_4000_pts.json'), 'r') as f:
            #     train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            # with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_4000_pts.json'), 'r') as f:
            #     test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            
            if post_train:
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            else:
                with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_1_blocks.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_1_blocks.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            if self.npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 12000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
            elif self.npoints == 2048:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
            elif self.npoints == 3000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_1_blocks_fold_{self.fold_id}")
            elif self.npoints == 5000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
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
            
        self.datapath = self.datapath * self.repeat_factor

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
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
        
        seg = np.where(seg == 3, 1, 0)
        seg = np.where(ignore == 1, 1, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
#             if not self.is_test:
            gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_1_blocks_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
            sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000: 
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        sample_weight = sample_weight - self.subtraction
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, sample_weight, ignore
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight, ignore


    def __len__(self):
        return len(self.datapath)
    

class TreePartNormalDataset6000PointsWithOverlapPaddingFGBGIgnoreHandledSingleBlockWithAugmentation(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, repeat_factor=1, subtraction=0, post_train=False, num_aug_copies=0):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        self.sigma = sigma
        self.repeat_factor = repeat_factor
        self.subtraction = subtraction
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            # with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_4000_pts.json'), 'r') as f:
            #     train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            # with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_4000_pts.json'), 'r') as f:
            #     test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            
            if post_train:
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            else:
                with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_1_blocks.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_1_blocks.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            if self.npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 12000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
            elif self.npoints == 2048:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
            elif self.npoints == 3000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_1_blocks_fold_{self.fold_id}")
            elif self.npoints == 5000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
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
#         self.datapath = []
#         for item in self.cat:
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))
            
#         self.datapath = self.datapath * self.repeat_factor
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))

        # optional: further repeat everything
        self.datapath = self.datapath * self.repeat_factor


        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
#         print("datapath: ", self.datapath)


    def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg, filename = self.cache[index]
#         else:
        fn = self.datapath[index]
        filename = fn[1]
#             cat = self.datapath[index][0]
        cat, filename, aug_id = self.datapath[index]

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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------

        
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
        
        seg = np.where(seg == 3, 1, 0)
        seg = np.where(ignore == 1, 1, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
#             if not self.is_test:
            gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_1_blocks_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
            sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000: 
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        sample_weight = sample_weight - self.subtraction
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, sample_weight, ignore
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight, ignore


    def __len__(self):
        return len(self.datapath)

    
class TreePartNormalDataset6000PointsWithOverlapPaddingFGBGIgnoreHandledNoIntensity(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, post_train=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        self.fold_id = fold_id
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            # with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_4000_pts.json'), 'r') as f:
            #     train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            # with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_4000_pts.json'), 'r') as f:
            #     test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            
            if post_train:
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', 'difficult_cases', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            else:
                with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                    test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            if self.npoints == 6000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 12000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
            elif self.npoints == 2048:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
            elif self.npoints == 3000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
            elif self.npoints == 4000:
                dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_repeat_and_overlap_fold_{self.fold_id}") # 8192 pts
            elif self.npoints == 5000:
                dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
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
        
        seg = np.where(seg == 3, 1, 0)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts, ignore
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight, ignore


    def __len__(self):
        return len(self.datapath)
    
    
class TreePartNormalDataset6000PointsWithOverlapPaddingFGBGStage2Thr30(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        
        print("split: ", split)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_4000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_4000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            
            if split == 'test':
                if self.npoints == 6000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 12000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
                elif self.npoints == 2048:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
                elif self.npoints == 3000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_4000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 5000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
    #             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
            elif split == 'train' or split == 'trainval': 
                if self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_4000_repeat_and_overlap_thr30") # 8192 pts
                
       
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
        
        seg = np.where(seg == 3, 1, 0)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_4000_thr30", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsWithOverlapPaddingFGBGStage2Thr68(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, same_weight=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        self.fold_id = fold_id
        self.same_weight = same_weight
        self.sigma = sigma
        
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        
        print("split: ", split)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            
            if split == 'test':
                if self.npoints == 6000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 12000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
                elif self.npoints == 2048:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
                elif self.npoints == 3000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"for_stage_2_numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts
                elif self.npoints == 5000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
    #             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
            elif split == 'train' or split == 'trainval': 
                if self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"for_stage_2_numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts
                
       
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
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))
                    
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
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
            cat, filename, aug_id = self.datapath[index]
#             cat = self.datapath[index][0]
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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 3, 1, 0)
        seg = np.where(ignore == 1, 1, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#                 sample_weight = np.ones((self.npoints,))
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        if self.same_weight:
            sample_weight = np.ones((self.npoints,))
    
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            seg_all_pts = np.where(ignore == 1, 1, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetStage3(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, same_weight=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        self.fold_id = fold_id
        self.same_weight = same_weight
        self.sigma = sigma
        
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        
        print("split: ", split)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            
            if split == 'test':
                if self.npoints == 6000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 12000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
                elif self.npoints == 2048:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
                elif self.npoints == 3000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"for_stage_3_numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts
                elif self.npoints == 5000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
    #             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
            elif split == 'train' or split == 'trainval': 
                if self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"for_stage_3_numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts
                
       
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
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))
                    
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
            cat, filename, aug_id = self.datapath[index]
#             cat = self.datapath[index][0]
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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 4, 3, seg)
        seg = np.where(ignore == 1, 3, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#                 sample_weight = np.ones((self.npoints,))
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        if self.same_weight:
            sample_weight = np.ones((self.npoints,))
    
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore == 1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
    
class SyntheticDatasetPostTrainStage3(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, same_weight=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        self.fold_id = fold_id
        self.same_weight = same_weight
        self.sigma = sigma
        
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        
        print("split: ", split)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000:  # train_fold_0_4k_pts_single_block.json
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_single_block.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_single_block.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            
            if split == 'test':
                if self.npoints == 6000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 12000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
                elif self.npoints == 2048:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
                elif self.npoints == 3000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_single_block_fold_0") # 8192 pts
                elif self.npoints == 5000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
    #             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
            elif split == 'train' or split == 'trainval': 
                if self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_single_block_fold_0") # 8192 pts
                
       
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
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))
                    
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
            cat, filename, aug_id = self.datapath[index]
#             cat = self.datapath[index][0]
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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 4, 3, seg)
        seg = np.where(ignore == 1, 3, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_0_sigma_{self.sigma}", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#                 sample_weight = np.ones((self.npoints,))
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        if self.same_weight:
            sample_weight = np.ones((self.npoints,))
    
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore == 1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
    
class TreePartNormalDatasetStage3FINAL(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, same_weight=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        self.fold_id = fold_id
        self.same_weight = same_weight
        self.sigma = sigma
        
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        
        print("split: ", split)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_single_block.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_single_block.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            
            if split == 'test':
                if self.npoints == 6000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 12000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
                elif self.npoints == 2048:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
                elif self.npoints == 3000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_single_block_fold_0") # 8192 pts
                elif self.npoints == 5000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
    #             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
            elif split == 'train' or split == 'trainval': 
                if self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_single_block_fold_0") # 8192 pts
                
       
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
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))
                    
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
            cat, filename, aug_id = self.datapath[index]
#             cat = self.datapath[index][0]
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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 4, 3, seg)
        seg = np.where(ignore == 1, 3, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
        if self.npoints == 12000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 6000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 2048:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 3000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        if self.npoints == 4000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_0_sigma_{self.sigma}", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#                 sample_weight = np.ones((self.npoints,))
        if self.npoints == 5000:
            if not self.is_test:
                gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
                sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         sample_weight = np.ones((self.npoints,))
        
        if self.same_weight:
            sample_weight = np.ones((self.npoints,))
    
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore == 1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetSingleStage(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, same_weight=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        self.fold_id = fold_id
        self.same_weight = same_weight
        self.sigma = sigma
        
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        
        print("split: ", split)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_5_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_5_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            
            if split == 'test':
                if self.npoints == 6000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 12000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
                elif self.npoints == 2048:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
                elif self.npoints == 3000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_5_blocks_fold_{self.fold_id}") # 8192 pts 
                elif self.npoints == 5000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
    #             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
            elif split == 'train' or split == 'trainval': 
                if self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_5_blocks_fold_{self.fold_id}") # 8192 pts
                
       
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
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))
                    
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
            cat, filename, aug_id = self.datapath[index]
#             cat = self.datapath[index][0]
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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 4, 3, seg)
        seg = np.where(ignore == 1, 3, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
#         if self.npoints == 12000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 6000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 2048:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 3000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 4000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
# #                 sample_weight = np.ones((self.npoints,))
#         if self.npoints == 5000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        sample_weight = np.ones((self.npoints,))
        
#         if self.same_weight:
#             sample_weight = np.ones((self.npoints,))
    
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore == 1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
    
class TreePartNormalDatasetSingleStageFINAL(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, same_weight=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        self.fold_id = fold_id
        self.same_weight = same_weight
        self.sigma = sigma
        
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        
        print("split: ", split)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts_2_blocks.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts_2_blocks.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            
            if split == 'test':
                if self.npoints == 6000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 12000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
                elif self.npoints == 2048:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
                elif self.npoints == 3000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts 
                elif self.npoints == 5000: 
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
    #             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
            elif split == 'train' or split == 'trainval': 
                if self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts
                
       
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
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))
                    
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
            cat, filename, aug_id = self.datapath[index]
#             cat = self.datapath[index][0]
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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
        seg = np.where(seg == 4, 3, seg)
        seg = np.where(ignore == 1, 3, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
#         if self.npoints == 12000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 6000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 2048:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 3000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 4000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
# #                 sample_weight = np.ones((self.npoints,))
#         if self.npoints == 5000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        sample_weight = np.ones((self.npoints,))
        
#         if self.same_weight:
#             sample_weight = np.ones((self.npoints,))
    
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore == 1, 3, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDatasetStandingTrees(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False, is_test=False, fold_id=0, sigma=None, same_weight=False, num_aug_copies=0, alpha_factor=None, alpha_fg=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test
        
        self.fold_id = fold_id
        self.same_weight = same_weight
        self.sigma = sigma
        
        self.alpha_factor = alpha_factor
        self.alpha_fg = alpha_fg
        
        self.num_aug_copies = num_aug_copies   # how many augmented copies to add per original
        self.aug_seed = 1234               # base seed for deterministic aug
        
        print("use_rgbi: ", self.use_rgbi)
        print("npoints: ", self.npoints)
        
        print("split: ", split)


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
        
        if self.npoints == 6000:
            
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#             with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:
#                 test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
                
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_6k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_6k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 12000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_12k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_12k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 2048: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_2048_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_2048_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 3000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_3000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_3000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 4000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_4k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_4k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 5000: 
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5000_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_5000_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 30000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_30k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_30k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
        elif self.npoints == 15000: 
            with open(os.path.join(self.root, 'train_test_split', f'train_fold_{self.fold_id}_15k_pts.json'), 'r') as f:
                train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', f'test_fold_{self.fold_id}_15k_pts.json'), 'r') as f:
                test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            
            if split == 'test':
                if self.npoints == 6000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 12000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_12k_repeat_and_overlap") # 8192 pts
                elif self.npoints == 2048:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_2048_repeat_and_overlap") # 8192 pts
                elif self.npoints == 3000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_3000_repeat_and_overlap") # 8192 pts
                elif self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts 
                elif self.npoints == 5000:
                    dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_5000_repeat_and_overlap") # 8192 pts
    #             dir_point = os.path.join(self.root, self.cat[item], "numpy_split_classfps_6000")
                elif self.npoints == 30000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_30000_2_blocks_fold_{self.fold_id}")
                elif self.npoints == 15000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_15000_2_blocks_fold_{self.fold_id}")
            elif split == 'train' or split == 'trainval': 
                if self.npoints == 4000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_4000_2_blocks_fold_{self.fold_id}") # 8192 pts
                elif self.npoints == 30000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_30000_2_blocks_fold_{self.fold_id}")
                elif self.npoints == 15000:
                    dir_point = os.path.join(self.root, self.cat[item], f"numpy_split_fps_15000_2_blocks_fold_{self.fold_id}")
                
       
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
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # aug_id=0 => original (no aug)
                self.datapath.append((item, fn, 0))
                # aug_id=1..K => augmented copies (added as new samples)
                for aug_id in range(1, self.num_aug_copies + 1):
                    self.datapath.append((item, fn, aug_id))
                    
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        print("classes: ", self.classes)
        
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
        self.seg_classes = {'standing_trees': [0, 1]}

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
            cat, filename, aug_id = self.datapath[index]
#             cat = self.datapath[index][0]
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
            
        # ----- deterministic augmentation for augmented copies only -----
        if (not self.is_test) and aug_id > 0:
            # make aug deterministic per (file, aug_id)
            # filename is full path; stable hashing across runs if you avoid Python hash()
            stable = abs(sum(bytearray(os.path.basename(filename).encode("utf-8"))))
            seed = (self.aug_seed * 1000003 + stable * 9176 + aug_id * 101) % (2**32 - 1)
            rng = np.random.default_rng(seed)

            # 1) yaw rotation
            point_set[:, 0:3] = _random_yaw_rotation(point_set[:, 0:3], rng)
            # 2) xyz jitter
            point_set[:, 0:3] = _xyz_jitter(point_set[:, 0:3], rng, sigma=0.005, clip=0.02)
            # 3) RGB jitter
            point_set[:, 3:6] = _color_jitter_rgb(point_set[:, 3:6], rng,
                                                  brightness=0.10, contrast=0.10, saturation=0.10)
            # 4) intensity jitter
            point_set[:, 6] = _intensity_jitter(point_set[:, 6], rng,
                                                mul_range=0.10, add_sigma=0.01)
        # ---------------------------------------------------------------
        
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
        
#         seg = np.where(seg == 4, 3, seg)
        seg = np.where(ignore == 1, 1, seg)
        
#         print(seg[0, 0])

#         any_greater = np.any(seg > 3)
#         if any_greater:
#             print(np.unique(seg))
        
        point_set = point_set_exp[0, :, :]
        
#         if self.npoints == 12000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 6000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_6k", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 2048:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_2048", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 3000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_3000", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
#         if self.npoints == 4000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", f"gaussian_weights_4000_fold_{self.fold_id}_sigma_{self.sigma}", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
# #                 sample_weight = np.ones((self.npoints,))
#         if self.npoints == 5000:
#             if not self.is_test:
#                 gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_5000", os.path.basename(filename))
#                 sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        sample_weight = np.ones((self.npoints,))
        
#         if self.same_weight:
#             sample_weight = np.ones((self.npoints,))
    
        if self.is_test:
#             seg_all_pts = np.where(seg_all_pts == 4, 3, seg_all_pts)
            seg_all_pts = np.where(ignore == 1, 1, seg_all_pts)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
#         return point_set, cls, seg, point_set, filename, sample_weight
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsWithOverlapPaddingFGBGGaussianWt(Dataset):
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
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_clas_fps_6000_split.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_clas_fps_6000_split.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
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
        
        seg = np.where(seg == 3, 1, 0)
        
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

        gaussian_wt_file_path = os.path.join(self.root, "0", "gaussian_weights_12k", f"fallen_trees_{instance_id}.npy")
        sample_weight = np.load(gaussian_wt_file_path).astype(np.float32)
        
        if self.is_test:
            seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)
    
class TreePartNormalDataset6000PointsWithOverlapPaddingShapePrior(Dataset):
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
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_fps_split_overlap_with_padding.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_clas_fps_6000_split.json'), 'r') as f:
            train_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_fps_split_overlap_with_padding.json'), 'r') as f:  # 8192 pts
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list_clas_fps_6000_split.json'), 'r') as f:
            test_ids_orig = set([str(d.split('/')[-1]) for d in json.load(f)])
  
            
        for item in self.cat:
            # print('category', item)
            self.meta[item] = [] 
            dir_point = os.path.join(self.root, self.cat[item], "numpy_split_fps_6000_repeat_and_overlap") # 8192 pts
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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}

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
            return point_set, cls, seg, point_set, ignore, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, ignore, filename


    def __len__(self):
        return len(self.datapath)

    
class TreePartNormalDataset60kPointsFGvsBG(Dataset):
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
            
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)    
    
    
class TreePartNormalDataset60kPoints(Dataset):
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
        self.seg_classes = {'fallen_trees': [0, 1, 2, 3]}
#         self.seg_classes = {'fallen_trees': [0, 1]}

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
#         seg = np.where(seg == 3, 1, 0) # 3 becomes 1 (background); 0,1,2 becomse 0 (foreground)
        
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
#             seg_all_pts = np.where(seg_all_pts == 3, 1, 0)
#             print("point_set_all_pts_SHAPE_2: ", point_set_all_pts.shape)
            
            # NOTE: -3 index of point_set_all_pts is is_repeated field
            
            return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, point_set, filename, sample_weight


    def __len__(self):
        return len(self.datapath)    
    
    
# for new method where we train mini pointnet in decoder to replace interpolation from 2048 to N points
class TreePartNormalDatasetLargeNumPointsV2(Dataset):
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
        
#             if not self.normal_channel:
#                 point_set = deepcopy(data[:, 0:3])
#             else:
#                 point_set = deepcopy(data[:, 0:6])

            if not self.use_rgbi:
                point_set = deepcopy(data[:, 0:3])
            else:
                point_set = deepcopy(data[:, 0:7]) # x,y,z,r,g,b,intensity
                
            seg = data[:, -2].astype(np.int32) # for all point clouds second last index is for parts, last index is to indicate repeated point cloud
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, fn[1])
        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # N, C
        
        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])
        
#         point_set_all_pts = deepcopy(point_set)
#         seg_all_pts = deepcopy(seg)
        
#         point_set_exp = np.expand_dims(point_set, axis=0)
#         fps_idx = farthest_point_sample(torch.tensor(point_set_exp), npoint=self.npoints)
#         new_data = index_points(point_set_exp, fps_idx)
        
        
#         seg_new = np.expand_dims(seg, axis=0) # 1, N
#         seg_new = np.expand_dims(seg_new, axis=-1) # 1, N, 1
#         seg_new = torch.tensor(seg_new)
#         seg_new = index_points(seg_new, fps_idx)
#         seg = seg_new[0, :, :]
#         seg = np.array(seg)
#         seg = seg.reshape(-1)
        
#         point_set = new_data[0, :, :]
        
#         if self.is_test:
#             return point_set, cls, seg, point_set, filename, point_set_all_pts, seg_all_pts
        
        return point_set, cls, seg, data, filename


    def __len__(self):
        return len(self.datapath)
    

class TreePartNormalDatasetAllPointsPointNet(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, use_rgbi=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi


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
            train_ids = set([str(d.split('/')[-1]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[-1]) for d in json.load(f)])
            
#         print("train_ids: ", train_ids)
#         print("test_ids: ", test_ids)
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy")
            fns = sorted(os.listdir(dir_point))
#             print(fns[0][0:-4])
#             if split == 'trainval':
#                 fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
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
        
#             if not self.normal_channel:
#                 point_set = deepcopy(data[:, 0:3])
#             else:
#                 point_set = deepcopy(data[:, 0:6])
                
            if not self.use_rgbi:
                point_set = deepcopy(data[:, 0:3])
            else:
                point_set = deepcopy(data[:, 0:7]) # x,y,z,r,g,b,intensity
            seg = data[:, -2].astype(np.int32) # second last index is for parts, last index is to indicate repeated point cloud
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, fn[1])
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])
    

        return point_set, cls, seg, data, filename

    def __len__(self):
        return len(self.datapath)


