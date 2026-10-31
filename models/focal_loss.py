import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean', ignore_index=None, from_log_softmax=True):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.from_log_softmax = from_log_softmax

        if alpha is None:
            self.alpha = None
        else:
            # alpha: 1D tensor of shape [num_classes]
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, logits, target):
        """
        logits: [B, C, N] or [B, C]
        target: [B, N] or [B]
        """
        if logits.dim() == 3:
            B, C, N = logits.shape
            logits = logits.permute(0, 2, 1).reshape(-1, C)  # -> [B*N, C]
            target = target.view(-1)                         # -> [B*N]
        else:
            C = logits.size(1)
            logits = logits.reshape(-1, C)
            target = target.view(-1)

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            logits = logits[valid_mask]
            target = target[valid_mask]
            if target.numel() == 0:
                return torch.tensor(0.0, device=logits.device)
        
        if not self.from_log_softmax:
            log_prob = F.log_softmax(logits, dim=1)          # [N', C]
            prob = log_prob.exp()                            # [N', C]
        else:
            log_prob = logits
            prob = log_prob.exp()

        # gather log(p_t) and p_t
        target = target.long()
        log_pt = log_prob[torch.arange(log_prob.size(0)), target]  # [N']
        pt = prob[torch.arange(prob.size(0)), target]               # [N']

        # alpha factor
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)[target]  # [N']
        else:
            alpha = 1.0

        focal_factor = (1 - pt) ** self.gamma
        loss = -alpha * focal_factor * log_pt            # [N']

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class FocalLossPtWt(nn.Module): # different weight for different point
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean', ignore_index=None, from_log_softmax=True, hardneg_thresh=None, hardneg_weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.from_log_softmax = from_log_softmax
        
        self.hardneg_thresh = hardneg_thresh   # e.g., 0.7; None disables
        self.hardneg_weight = hardneg_weight
        

        if alpha is None:
            self.alpha = None
        else:
            # alpha: 1D tensor of shape [num_classes]
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, logits, target, sample_weight=None, ignore_point=None):
        """
        logits: [B, C, N] or [B, C]
        target: [B, N] or [B]
        sample_weight: [B, N] or [B] or None
        """
        if logits.dim() == 3:
            B, C, N = logits.shape
            logits = logits.permute(0, 2, 1).reshape(-1, C)  # [B*N, C]
            target = target.view(-1)                         # [B*N]
            if sample_weight is not None:
                sample_weight = sample_weight.view(-1)       # [B*N]
        else:
            C = logits.size(1)
            logits = logits.reshape(-1, C)
            target = target.view(-1)
            if sample_weight is not None:
                sample_weight = sample_weight.view(-1)
                
            if ignore_point is not None:
                ignore_point = ignore_point.view(-1)

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            logits = logits[valid_mask]
            target = target[valid_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[valid_mask]

            if target.numel() == 0:
                return torch.tensor(0.0, device=logits.device)

        if not self.from_log_softmax:
            log_prob = F.log_softmax(logits, dim=1)
            prob = log_prob.exp()
        else:
            log_prob = logits
            prob = log_prob.exp()

        target = target.long()
        idx = torch.arange(log_prob.size(0), device=logits.device)
        log_pt = log_prob[idx, target]
        pt = prob[idx, target]
        

        # class-wise alpha (optional)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)[target]
        else:
            alpha = 1.0
        

        focal_factor = (1 - pt) ** self.gamma
        loss = -alpha * focal_factor * log_pt  # [N']
           
        if sample_weight is not None:
            loss = loss * sample_weight.to(logits.device)
            
        # ---------- hard-negative mining (binary or multi-class via fg_class) ----------
        if self.hardneg_thresh is not None and self.hardneg_weight is not None and self.hardneg_weight != 1.0:
            # compute p_fg for each point
#             if self.num_classes == 2:
            # prob for class FG (0)
            p_fg = prob[:, 0]

            # hard negatives = background points predicted as FG with high confidence
            hard_neg_mask = (target == 1) & (p_fg >= self.hardneg_thresh)   # [N']
            if hard_neg_mask.any():
                loss = loss * torch.where(
                    hard_neg_mask,
                    torch.tensor(self.hardneg_weight, device=loss.device, dtype=loss.dtype),
                    torch.tensor(1.0, device=loss.device, dtype=loss.dtype)
                )
        # ---------------------------------------------------------------------------
            
#         print("ignore_point: ", ignore_point.shape)
#         print("loss: ", loss.shape)
        if ignore_point is not None:
            # mask of valid points (True = keep)
            non_ignore_mask = (ignore_point == 0)

            # ensure same device & dtype for multiplication
            non_ignore_mask = non_ignore_mask.to(loss.device).float()

            # zero out ignored points
            loss = loss * non_ignore_mask

            # avoid dividing by zero if all are ignored
            valid_count = non_ignore_mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.sum() * 0.0   # or torch.tensor(0., device=loss.device)
        else:
            loss = loss.mean()  # or whatever reduction you use
        
        return loss

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

