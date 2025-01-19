import torch.nn as nn
from modules.mlp import MLP
import torch_geometric.nn as gnn


#################################################################
# A low-level gnn wrapper that includes many gnn basic blocks   #
#################################################################
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.layer = gnn.GCNConv(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)  # the shape of edge_attr must be (n,) or (n,1)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr="add", bias=True):
        super().__init__()
        self.layer = gnn.GraphConv(in_channels, out_channels, aggr=aggr, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class ResGatedGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.layer = gnn.ResGatedGraphConv(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


class GINConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0., bias=True):
        super().__init__()
        self.MLP = MLP(in_channels, out_channels, bias=bias, dropout=dropout)
        self.layer = gnn.GINConv(self.MLP, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


class GINEConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0., bias=True):
        super().__init__()
        self.MLP = MLP(in_channels, out_channels, bias=bias, dropout=dropout)
        self.layer = gnn.GINEConv(self.MLP, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # Node and edge feature dimension must match,
        # setting them by node_hidden_features, edge_hidden_features in ConnectomeEncoder
        return self.layer(x, edge_index, edge_attr)


class TransformerConv(nn.Module):
    def __init__(self, in_channles, out_channles, bias=True, num_heads=8):
        super().__init__()
        self.layer = gnn.TransformerConv(in_channles, out_channles//num_heads, num_heads, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class GATConv(nn.Module):
    def __init__(self, in_channels, out_channles, bias=True, num_heads=1):
        super().__init__()
        self.layer = gnn.GATConv(in_channels, out_channles//num_heads, num_heads, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class GatedGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, bias=True):
        super().__init__()
        self.layer = gnn.GatedGraphConv(out_channels, num_layers, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.MLP = MLP(2 * in_channels, out_channels, bias=bias)
        self.layer = gnn.EdgeConv(self.MLP)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


class DynamicEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=6, bias=True):
        super().__init__()
        self.MLP = MLP(2 * in_channels, out_channels, bias=bias)
        self.layer = gnn.DynamicEdgeConv(self.MLP, k)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)