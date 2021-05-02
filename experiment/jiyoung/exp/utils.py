# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_lr(optimizer):
    """
    현재 learning rate 리턴
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def add_hist(hist, label_trues, label_preds, n_class):
    """
    stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist


# 출처: 강민용_T1001 캠퍼 http://boostcamp.stages.ai/competitions/28/discussion/post/256
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
    - overall accuracy
    - mean accuracy
    - mean IU
    - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()  # diag - 대각선만 구한다. (예측 = 타겟)

    with np.errstate(divide="ignore", invalid="ignore"):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc


def get_miou(label_trues, label_preds, n_class):
    miou_list = []
    for lt, lp in zip(label_trues, label_preds):
        hist = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        with np.errstate(divide="ignore", invalid="ignore"):
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        miou_list.append(np.nanmean(iu))
    return miou_list


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        """
        input_tensor : 2-d tensor (N, C, ...), outputs
        target_tensor : 1-d tensor (N, ...) , label 값
        """
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


def get_mean_std(loader):
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for imgs, _ in loader:
        data = torch.stack(imgs)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)

        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(f"num_samples: {nb_samples}")

    return mean, std


# PyTorch
# 일단 0도 동일하게 계산해봄
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(
        self, inputs, targets, smooth=0.00001, alpha=0.2, beta=0.8, gamma=4 / 3
    ):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        p0_g0 = None
        p0_g1 = None
        p1_g0 = None

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        info_list = np.array([TP, FP, FN, Tversky], dtype=np.float)

        return FocalTversky, info_list


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc
