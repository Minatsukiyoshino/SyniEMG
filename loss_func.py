import torch
from torch import nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class PinballLoss(nn.Module):
    """
    Calculates the quantile loss function.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """

    def __init__(self, quantiles):
        super(PinballLoss, self).__init__()
        self.pred = None
        self.targes = None
        self.quantiles = quantiles

    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        upper = self.quantiles * error
        lower = (self.quantiles - 1) * error

        losses = torch.max(lower, upper)
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss


class DifferentiableSDNNLoss(nn.Module):
    def __init__(self, alpha=10):
        super(DifferentiableSDNNLoss, self).__init__()
        self.alpha = alpha

    def differentiable_peak_detection(self, signal):
        """Approximate peak detection using softplus function."""
        diff_signal = signal[:, :, 1:] - signal[:, :, :-1]
        peaks = F.softplus(self.alpha * diff_signal)
        return peaks

    def compute_differentiable_sdnn(self, signal):
        """Compute an approximate differentiable SDNN."""
        peaks = self.differentiable_peak_detection(signal)

        # Instead of exact RR intervals, use the soft peaks as weights
        weighted_diffs = peaks * (signal[:, :, 1:] - signal[:, :, :-1])

        # Compute "soft" SDNN
        sdnn = torch.std(weighted_diffs, dim=2)

        return sdnn

    def forward(self, fake_ppg, GT_ppg):
        """Compute the loss between fake PPG and ground truth PPG."""
        fake_sdnn = self.compute_differentiable_sdnn(fake_ppg)
        gt_sdnn = self.compute_differentiable_sdnn(GT_ppg)

        # Mean squared error between the SDNN values
        loss = F.mse_loss(fake_sdnn, gt_sdnn)
        return loss
