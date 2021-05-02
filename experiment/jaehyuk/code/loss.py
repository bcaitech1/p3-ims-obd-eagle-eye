import torch
import torch.nn as nn
import torch.nn.functional as F
from SegLoss.losses_pytorch import ND_Crossentropy as NDCE
from SegLoss.losses_pytorch import dice_loss as dice
from SegLoss.losses_pytorch import focal_loss as focal
from typing import Optional
EPS = 1e-10
# 제공받은 코드

class return_1(nn.Module):
    """return 1
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input_tensor, target_tensor):
        return 1

# https://arxiv.org/pdf/2104.08717.pdf
class CompoundLoss(nn.Module):
    """
    The base class for implementing a compound loss:
        l = l_1 + alpha * l_2
    """
    def __init__(self, 
                 alpha: float = 1.,
                 factor: float = 1.,
                 step_size: int = 0,
                 max_alpha: float = 100.,
                 temp: float = 1.,
                 ignore_index: int = 255,
                 background_index: int = 0,
                 weight: Optional[torch.Tensor] = None) -> None:
        
        super().__init__()
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size
        self.temp = temp
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight

    def cross_entropy(self, inputs: torch.Tensor, labels: torch.Tensor):
        
        loss = F.cross_entropy(
                inputs, labels, weight=self.weight, ignore_index=self.ignore_index)
        return loss

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.factor, self.max_alpha)

    def get_gt_proportion(self,
                          labels: torch.Tensor,
                          target_shape,
                          ignore_index: int = 255):
        
        bin_labels, valid_mask = expand_onehot_labels(labels, target_shape, ignore_index)
        
        gt_proportion = get_region_proportion(bin_labels, valid_mask)
        return gt_proportion, valid_mask

    def get_pred_proportion(self, 
                            logits: torch.Tensor,
                            temp: float = 1.0,
                            valid_mask=None):
        
        preds = F.log_softmax(temp * logits, dim=1).exp()
        pred_proportion = get_region_proportion(preds, valid_mask)
        return pred_proportion
def expand_onehot_labels(labels, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels, valid_mask
def get_region_proportion(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Get region proportion
    Args:
        x : one-hot label map/mask
        valid_mask : indicate the considered elements
    """
    if valid_mask is not None:
        if valid_mask.dim() == 4:
            x = torch.einsum("bcwh, bcwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bcwh->bc", valid_mask)
        else:
            x = torch.einsum("bcwh,bwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bwh->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3]

    region_proportion = (torch.einsum("bcwh->bc", x) + EPS) / (cardinality + EPS)

    return region_proportion
class CrossEntropyWithL1(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * |gt_region - prob_region|
    """
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        # ce term
        loss_ce = self.cross_entropy(inputs, labels)
        # regularization
        gt_proportion, valid_mask = self.get_gt_proportion(labels, inputs.shape)
        pred_proportion = self.get_pred_proportion(inputs, temp=self.temp, valid_mask=valid_mask)
        loss_reg = (pred_proportion - gt_proportion).abs().mean()

        loss = loss_ce + self.alpha * loss_reg

        # return loss, loss_ce, loss_reg
        return loss

# 더 많은 loss를 사용하려면 아래 git hub or SegLoss/losses_pyorch에 들어가시면 됩니다.
# https://github.com/JunMa11/SegLoss
_criterion_entrypoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "cross_entropy_ND":NDCE.CrossentropyND,
    "focal": focal.FocalLoss,
    "1":return_1,
    "CEWithL1":CrossEntropyWithL1,
    "l1":nn.L1Loss,
    # GDiceLoss를 사용하면 에러 발생
    "dice" : dice.GDiceLossV2,
    # "iou" : IOU_loss,
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion


if __name__ == "__main__":
    pass
