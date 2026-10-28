"""Small layers shared by methods without changing their network structure."""

from torch import nn
from torch.nn import functional as F


class SingletonBatchNorm(nn.BatchNorm1d):
    """Keep a one-example tail batch using saved training-side statistics."""

    def forward(self, values):
        if self.training and len(values) == 1:
            return F.batch_norm(
                values,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )
        return super().forward(values)
