import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class VSELoss(nn.Module):
    def __init__(self, scale_factor=1.0):
        super(VSELoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2 * (1 + self.scale_factor * (y_pred - y_true) ** 2))