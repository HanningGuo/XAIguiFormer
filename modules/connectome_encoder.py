import torch.nn as nn
from modules.gnn import GNN
from torch_scatter import scatter


class ConnectomeEncoder(nn.Module):
    def __init__(
            self,
            node_in_features,
            edge_in_features,
            node_hidden_features,
            edge_hidden_features,
            out_features,
            num_gnn_layers,
            dropout,
            gnn_type='GCNConv',
            gnn_hidden_features=None,
            pooling='mean',
            num_freqband=9,
            act_funcs=nn.GELU,
            norm=nn.LayerNorm
    ):
        super().__init__()

        self.pooling = pooling
        self.num_freqband = num_freqband

        self.node_embedding = nn.Linear(node_in_features, node_hidden_features)
        self.edge_embedding = nn.Linear(edge_in_features, edge_hidden_features)
        self.gnns = GNN(node_hidden_features, out_features, num_gnn_layers,
                        gnn_hidden_features, gnn_type, act_funcs, norm, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # map their dimension to specified dimension in order to cater different kind of GNN's requirement
        x = self.node_embedding(data.x)
        edge_attr = self.edge_embedding(data.edge_attr)

        freqband_repr = self.gnns(x, data.edge_index, edge_attr)
        freqband_repr = scatter(freqband_repr, data.freqband_order, dim=0, reduce=self.pooling)

        input_repr = freqband_repr.reshape(-1, self.num_freqband, freqband_repr.shape[1])

        return self.dropout(input_repr)
