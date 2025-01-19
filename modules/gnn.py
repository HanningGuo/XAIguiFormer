import torch.nn as nn
from modules.mlp import Identity
from modules.activation import GeGLU
import modules.gnn_wrapper as gnn_wrapper


class GNN(nn.Module):
    def __init__(self, in_features, out_features, num_layers, hidden_features=None,
                 gnn_type='GINEConv', act_funcs=nn.GELU, norm=nn.LayerNorm, dropout=0., res=True):
        super().__init__()

        hidden_features = hidden_features or in_features
        self.res = res

        self.convs = nn.ModuleList(
            [getattr(gnn_wrapper, gnn_type)(in_features if i == 0 else hidden_features, hidden_features, dropout=dropout)
             for i in range(num_layers)])
        self.norms = nn.ModuleList(
            [norm(hidden_features) if norm else Identity()
             for __ in range(num_layers)])
        self.act_funcs = nn.ModuleList(
            [act_funcs() if act_funcs else Identity()
             for __ in range(num_layers)])

        self.projs = nn.ModuleList([nn.Linear(hidden_features, hidden_features * 2) for __ in range(num_layers)])

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for __ in range(num_layers)])
        self.output_embedding = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index, edge_attr):
        previous_x = x
        for conv, norm, proj, act_func, dropout in zip(self.convs, self.norms, self.projs, self.act_funcs, self.dropouts):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            if isinstance(act_func, GeGLU):
                x = proj(x)
            x = act_func(x)
            x = dropout(x)
            if self.res:
                x = x + previous_x
                previous_x = x

        x = self.output_embedding(x)
        return x
