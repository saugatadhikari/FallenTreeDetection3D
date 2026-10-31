import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from models.focal_loss import FocalLoss, FocalLossPtWt

from models.dice import DiceLoss
from models.focal_loss import FocalLoss, FocalLossPtWt
from models.discriminative_loss import DiscriminativeEmbeddingLoss



class get_model(nn.Module):
    def __init__(self, num_classes, use_rgbi=False):
        super(get_model, self).__init__()
        self.use_rgbi = use_rgbi
        
        if use_rgbi:
            additional_channel = 4
        else:
            additional_channel = 0

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3+additional_channel, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        if self.use_rgbi:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
            
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        
#         print("l0_points: ", l0_points.shape)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self, num_classes, ce_weights=None, ignore_index=None, loss_fn="focal"):
        super(get_loss, self).__init__()
        
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
            self.focal_loss = FocalLossPtWt(
                num_classes=num_classes,
#                 alpha=[2.0, 0.05],  # tweak these
#                 alpha=[2.0, 0.1],  # tweak these
                alpha=None,  # tweak these
                gamma=2.0,
                ignore_index=None,  # or your ignore label if you have one
                from_log_softmax=True
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