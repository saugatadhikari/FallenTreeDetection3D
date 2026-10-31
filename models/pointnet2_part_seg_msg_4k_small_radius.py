import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation

# from segmentation_models_pytorch.losses import DiceLoss

from models.dice import DiceLoss
from models.focal_loss import FocalLoss, FocalLossPtWt
from models.discriminative_loss import DiscriminativeEmbeddingLoss



class get_model(nn.Module):
    def __init__(self, part_num=50, num_classes=16, normal_channel=False, use_rgbi=False, use_intensity=1):
        super(get_model, self).__init__()
#         if normal_channel:
#             additional_channel = 3
#         else:
#             additional_channel = 0
        
        if use_rgbi:
            if use_intensity:
                additional_channel = 4
            else:
                additional_channel = 3
        else:
            additional_channel = 0
            
        self.part_num = part_num
        self.num_classes = num_classes
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.05, 0.1, 0.2], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.2,0.4], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
#         self.fp1 = PointNetFeaturePropagation(in_channel=136+additional_channel, mlp=[128, 128]) # 150 changed to 136
        self.fp1 = PointNetFeaturePropagation(
            in_channel=self.num_classes + 6 + additional_channel + 128,  # +128 from l1_points features
            mlp=[128, 128]
        )

#         print("self.fp1.shape: ", self.fp1.shape)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, part_num, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
#         print("xyz: ", xyz.shape)
        B,C,N = xyz.shape
#         if self.normal_channel:
#             l0_points = xyz
#             l0_xyz = xyz[:,:3,:]
#         else:
#             l0_points = xyz
#             l0_xyz = xyz
        if self.use_rgbi:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
            
            
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,self.num_classes,1).repeat(1,1,N)
#         print("self.num_classes: ", self.num_classes)
#         print(cls_label_one_hot.shape)
#         print(l0_xyz.shape, l1_xyz.shape, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1).shape, l1_points.shape)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss



# class DiceLoss(nn.Module):
#     def __init__(self, num_classes, class_weights=None, ignore_index=None,
#                  smooth=1e-6, from_log_softmax=True):
#         super().__init__()
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
#         self.smooth = smooth
#         self.from_log_softmax = from_log_softmax

#         if class_weights is None:
#             self.register_buffer("class_weights",
#                                  torch.ones(num_classes, dtype=torch.float))
#         else:
#             self.register_buffer("class_weights",
#                                  torch.as_tensor(class_weights, dtype=torch.float))
            
#     def forward(self, pred, target):
#         """
#         pred:   (P, C)  where P = B * N  (log-softmax or logits)
#         target: (P,)    int64 labels in [0, C-1]
#         """
#         P, C = pred.shape
#         assert C == self.num_classes, "num_classes mismatch"

#         # Convert to probabilities
#         if self.from_log_softmax:
#             probs = pred.exp()          # pred was log_softmax
#         else:
#             probs = F.softmax(pred, dim=1)  # if you pass raw logits

#         # One-hot encode target: (P,) -> (P, C)
#         target_clamped = torch.clamp(target, min=0)  # in case ignore_index is -1, we’ll mask it
#         target_onehot = F.one_hot(target_clamped, num_classes=C).float()

#         # Per-class sums over P
#         intersection = (probs * target_onehot).sum(dim=0)    # (C,)
#         pred_sum     = probs.sum(dim=0)                      # (C,)
#         target_sum   = target_onehot.sum(dim=0)              # (C,)

#         dice_score = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
#         dice_loss_per_class = 1.0 - dice_score              # (C,)

#         # Class weighting
#         weights = self.class_weights.to(pred.device)
#         weighted = weights * dice_loss_per_class
#         loss = weighted.sum()

#         return loss




class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weights=None, from_log_softmax=True, alpha=0.5):
        super().__init__()
        self.alpha = alpha

        # weighted CE on all classes
        self.ce = nn.CrossEntropyLoss(weight=ce_weights)
        
        self.from_log_softmax = from_log_softmax
        
        if num_classes == 4:
            classes = [0, 1, 2]
            
            self.dice = DiceLoss(
                mode="multiclass",
                from_logits=True,
                classes=classes,
            )
        elif num_classes == 2:
            classes = [0]
            
            self.dice = DiceLoss(
                mode="binary",
                from_logits=True,
                classes=classes,
            )

        

    def forward(self, pred, target):
        if self.from_log_softmax:
            probs = pred.exp() # model output is log softmax and we convert it to logits (probabilities)
            
        ce_loss = self.ce(probs, target) # need to pass logits (probabilities)
        dice_loss = self.dice(probs, target) # need to pass logits (probabilities)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss

    


    
    
class get_loss(nn.Module):
    def __init__(self, num_classes, ce_weights=None, ignore_index=None, loss_fn="focal", alpha_fg=None, hardneg_thresh=0.5):
        super(get_loss, self).__init__()
        
        self.hardneg_thresh = hardneg_thresh
        
#         self.combined_loss = CombinedLoss(
#             num_classes=num_classes,
#             ce_weights=ce_weights,
#             from_log_softmax=True,
#         )
        
#         self.dice = DiceLoss(
#             num_classes=num_classes, # 0,1,2,3
#             class_weights=class_weights, # 2, 2,,2,0.08
#             ignore_index=ignore_index,
#             from_log_softmax=True,
#         )

        self.loss_fn = loss_fn

#         if num_classes == 4:
#             classes = [0, 1, 2]
            
#             self.dice = DiceLoss(
#                 mode="multiclass",
#                 from_logits=True,
#                 classes=classes,
#                 log_cosh=True,
#             )
#         elif num_classes == 2:
#             classes = [0]
            
#             self.dice = DiceLoss(
#                 mode="binary",
#                 from_logits=True,
#                 classes=classes,
#                 log_cosh=True,
#             )

        self.disc_loss_fn = DiscriminativeEmbeddingLoss(
            delta_var=0.5,
            delta_dist=1.5,
            alpha=1.0,
            beta=1.0,
            gamma=0.001,
            norm=2,
            ignore_index=1,
        )
    
        
        
        if num_classes == 4:
            self.focal_loss = FocalLossPtWt(
                num_classes=num_classes,
                alpha=[2.0, 2.0, 2.0, 1.0],  # tweak these
#                 alpha=None,  # tweak these
                gamma=2.0,
                ignore_index=None,  # or your ignore label if you have one
                from_log_softmax=True
            )
        elif num_classes == 2:
            if alpha_fg is not None:
                alpha = [alpha_fg*10, 1]
            else:
                alpha = None
                
            print("alpha: ", alpha)
            
            if alpha_fg is None:
                hardneg_weight = None
            else:
                hardneg_weight=(alpha_fg*10)/2

            self.focal_loss = FocalLossPtWt(
                num_classes=num_classes,
                alpha=alpha,
                gamma=2.0,
                ignore_index=None,  # or your ignore label if you have one
                from_log_softmax=True,
                hardneg_thresh=self.hardneg_thresh,
                hardneg_weight=hardneg_weight
            )

    def forward(self, pred, target, trans_feat, sample_weight=None, ignore_point=None, embeddings=None, lambda_1=0.1):
#         loss = self.combined_loss(pred, target)
#         loss = self.dice(pred, target)
        if self.loss_fn == "focal":
            loss = self.focal_loss(pred, target, sample_weight=sample_weight, ignore_point=ignore_point)
#             disc_loss = self.disc_loss_fn(embeddings, target)
#             loss = loss + lambda_1*disc_loss
            
        elif self.loss_fn == "logcoshdice":
            loss = self.dice(pred, target)
            
        return loss