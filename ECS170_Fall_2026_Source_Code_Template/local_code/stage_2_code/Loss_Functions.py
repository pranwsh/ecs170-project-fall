"""Loss functions used in the Stage 2 experiments (2-4 asks for different losses).

Method_MLP.forward() returns raw logits and Method_MLP calls
`self.loss_function(logits, targets)`, so every loss here takes (logits, targets)
where targets are integer class labels. Cross-entropy is the default; the others
are wrappers that make a non-cross-entropy loss usable in the same training loop
without touching the model code.
"""

import torch
from torch import nn


class SoftmaxMSE(nn.Module):
    """Mean squared error between predicted probabilities and one-hot targets.

    MSE needs probability targets, so the logits are passed through softmax and
    the integer labels are expanded to one-hot vectors. This is the classic
    "MSE with softmax output" variant, slower to converge than cross-entropy
    because the softmax saturates, but useful as a comparison point.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probabilities = torch.softmax(logits, dim=1)
        one_hot = nn.functional.one_hot(targets, num_classes=self.num_classes).to(probabilities.dtype)
        return nn.functional.mse_loss(probabilities, one_hot)


class LogSoftmaxNLL(nn.Module):
    """Negative log-likelihood on top of log-softmax.

    Mathematically the same value as cross-entropy, but implemented as two
    separate steps (log_softmax + NLLLoss) like the original PyTorch tutorial.
    Useful to show in the report that the two implementations agree.
    """

    def __init__(self, weight=None):
        super().__init__()
        self.nll = nn.NLLLoss(weight=weight)

    def forward(self, logits, targets):
        return self.nll(torch.log_softmax(logits, dim=1), targets)


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing, a cheap regulariser."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        return nn.functional.cross_entropy(
            logits, targets, label_smoothing=self.smoothing)


# name -> builder; 'mse' additionally needs the number of classes
LOSSES = {
    'ce': lambda **kw: nn.CrossEntropyLoss(**kw),
    'mse': lambda num_classes=10, **kw: SoftmaxMSE(num_classes=num_classes),
    'nll': lambda **kw: LogSoftmaxNLL(**kw),
    'ce_ls': lambda smoothing=0.1, **kw: LabelSmoothingCE(smoothing=smoothing),
}

LOSS_DESCRIPTIONS = {
    'ce': 'cross-entropy (default)',
    'mse': 'mean squared error on softmax outputs',
    'nll': 'log-softmax + negative log-likelihood',
    'ce_ls': 'cross-entropy with label smoothing 0.1',
}


def build_loss(name, num_classes=10, **kwargs):
    """Create a loss module by name, e.g. build_loss('mse', num_classes=10)."""
    if name not in LOSSES:
        raise ValueError(f'unknown loss {name!r}; choose from {sorted(LOSSES)}')
    if name == 'mse':
        return LOSSES[name](num_classes=num_classes, **kwargs)
    return LOSSES[name](**kwargs)
