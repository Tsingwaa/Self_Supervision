import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


def nll_loss(output, target):
    return F.nll_loss(output, target)


class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input_, target):
        return F.cross_entropy(input_, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
