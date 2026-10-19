import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, label_smoothing=0.0, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.ls = label_smoothing

    def forward(self, logit, target):
        return focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight, label_smoothing=self.ls), self.gamma)

class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, label_smoothing=0.0, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau
        self.ls = label_smoothing

    def forward(self, logit, target):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target, label_smoothing=self.ls)
    
class MarginMetricSoftmaxLoss(nn.Module):
    def __init__(self, text_features, label_smoothing=0.0, lam=0.5, scale=30.0):
        super().__init__()

        # Precompute class–class cosine similarity and D(T_y, T_c) = 1 - S(T_y, T_c)
        with torch.no_grad():
            # S(T_y, T_c): (C x C)
            text_sim = text_features @ text_features.t()
            # D(T_y, T_c) = 1 - S(T_y, T_c)
            D = 1.0 - text_sim
            # λ * D(T_y, T_c)
            margin_matrix = lam * D

        # Register as buffers so they move with .to(device) and are saved in state_dict
        self.register_buffer("text_features", text_features)
        self.register_buffer("margin_matrix", margin_matrix)
        self.ls = label_smoothing
        self.scale = scale

    def forward(self, logit, target):
        # Get per-sample margin vector λ D(T_y, T_c) for each y in the batch
        # shape: [B, C]
        margins = self.margin_matrix[target]
        logits_adjusted = (logit + margins) * self.scale
        return F.cross_entropy(logits_adjusted, target, label_smoothing=self.ls)