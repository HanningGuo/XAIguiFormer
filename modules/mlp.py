import torch
import torch.nn as nn
from modules.activation import GeGLU


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class MLP(nn.Module):
    """
    A flexible MLP model that can set number of layers,
    include or not include norm, dropout and final activation,
    and change activation function
    """

    def __init__(self, in_features, out_features, hidden_features=None, act_func=nn.GELU, num_layers=2,
                 isFinalActivation=False, norm=None, bias=True, dropout=0.):
        super().__init__()
        hidden_features1 = hidden_features2 = hidden_features or in_features
        if isinstance(act_func(), GeGLU):
            hidden_features2 *= 2

        self.fcs = nn.ModuleList([nn.Linear(in_features if i == 0 else hidden_features1,
                                            hidden_features2 if i < num_layers - 1 else out_features,
                                            bias=bias)
                                  for i in range(num_layers)])

        self.norms = nn.ModuleList(
            [norm(hidden_features if i < num_layers - 1 else out_features) if norm is not None else Identity()
             for i in range(num_layers)])

        self.act_funcs = nn.ModuleList([act_func() for __ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for __ in range(num_layers)])
        self.num_layers = num_layers
        self.isFinalActivation = isFinalActivation

    def reset_parameters(self):
        for fc, norm, act_func, dropout in zip(self.fcs, self.norms, self.act_funcs, self.dropouts):
            fc.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        for i, (fc, norm, act_func, dropout) in enumerate(zip(self.fcs, self.norms, self.act_funcs, self.dropouts)):
            x = fc(x)
            if i < self.num_layers - 1 or self.isFinalActivation:
                x = norm(x)
                x = act_func(x)
            x = dropout(x)
        return x


class AvgPool(nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)
