import torch.nn as nn
from utils.visualizer import get_local
from modules.explainer import Explainer
from modules.connectome_encoder import ConnectomeEncoder
from modules.transformer import XAIguiTransformerEncoder


class XAIguiFormer(nn.Module):
    def __init__(
            self,
            node_in_features,
            edge_in_features,
            node_hidden_features,
            edge_hidden_features,
            num_classes,
            num_gnn_layers,
            num_heads,
            num_transformer_layers,
            freqband,
            gnn_type='GINEConv',
            gnn_hidden_features=None,
            pooling='mean',
            explainer_type='DeepLift',
            dim_feedforward=None,
            mlp_ratio=4.,
            act_func=nn.GELU,
            norm=nn.LayerNorm,
            layer_norm_eps=1e-05,
            bias=True,
            dropout=0.1,
            init_values=None,
            attn_drop=0.,
            droppath=0.,
    ):
        """
            A transformer with XAI guided improvement

            :param node_in_features: int, the dimension of node input features
            :param edge_in_features: int, the dimension of edge input features
            :param node_hidden_features: int, the dimension of node embedding
            :param edge_hidden_features: int, the dimension of edge embedding
            :param num_classes: int, the number of class
            :param num_gnn_layers: int, the number of gnn layer
            :param num_heads: int, the number of head in multi head self-attention
            :param num_transformer_layers: int, the number of transformer layer
            :param freqband: tensor, the range of frequency band for 2D rotary position encoding
            :param gnn_type: str (optional), specify which GNN algorithm is used to generate embedding
            (default: GINEConv)
            :param gnn_hidden_features: int (optional), the dimension of gnn hidden layer
            (default: None)
            :param pooling: str (optional), specify pooling method for GNN
            (default: mean)
            :param explainer_type: str (optional), specify which XAI algorithm is used to interpret model
            (default: DeepLift)
            :param dim_feedforward: int (optional), the dimension of feed forward layer
            (default: None)
            :param mlp_ratio: float (optional), mlp ratio is used to calculate the dimension of feed forward layer, as same effect as dim_feedforward
            (default: 4.)
            :param act_func: activation (optional), activation layer
            (default: nn.GELU)
            :param norm: normalization (optional), norm layer
            (default: nn.LayerNorm)
            :param layer_norm_eps: float (optional), the eps value in layer normalization components
            (default: 1e-5)
            :param bias: bool (optional), If set to False, Linear and LayerNorm layers will not learn an additive bias.
            (default: True)
            :param dropout: float (optional), drop out rate
            (default: 0.1)
            :param init_values: float (optional), the initial value of layer scale, if set to None, the layer scale isn't employed
             (default: None)
            :param attn_drop: float (optional), the attention drop out rate
            (default: 0.)
            :param droppath: float (optional), the drop path rate
            (default: 0.)
        """
        super().__init__()
        num_freqband = len(freqband)

        self.ConnectomeEncoder = ConnectomeEncoder(
            node_in_features, edge_in_features, node_hidden_features, edge_hidden_features,
            node_hidden_features, num_gnn_layers, dropout, gnn_type, gnn_hidden_features,
            pooling, num_freqband, act_func, norm
        )

        self.TransformerEncoder = XAIguiTransformerEncoder(
            node_hidden_features, num_heads, num_transformer_layers, freqband, num_classes,
            dim_feedforward, dropout, act_func, layer_norm_eps, mlp_ratio, bias,
            init_values, attn_drop, droppath, norm
        )

        self.explainer = Explainer(self.TransformerEncoder, self.TransformerEncoder.TransformerEncoder, explainer_type)

    @get_local('contribution')
    def forward(self, data):
        out = []
        x = self.ConnectomeEncoder(data)
        pred_y = self.TransformerEncoder(x, data.demographic_info)

        # the output without XAI-based enhance
        out.append(pred_y)
        # generate the explanation and enhance the model
        explanation = self.explainer((x, data.demographic_info), target=pred_y.argmax(dim=1))
        # obtain the frequency band contribution for visualization
        contribution = explanation[0][0]
        pred_y = self.TransformerEncoder(x, data.demographic_info, explanation)

        out.append(pred_y)

        return out
