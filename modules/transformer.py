import torch
import torch.nn as nn
from utils.visualizer import get_local
from timm.models.layers import DropPath
from modules.mlp import MLP, Identity, AvgPool
from modules.positional_encoding_wrapper import dRoFE


class XAIguiAttention(nn.Module):
    def __init__(self, in_features, num_heads, freqband, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert in_features % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = in_features // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(in_features, in_features * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_features, in_features)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope2D = dRoFE(in_features // num_heads, freqband, freqs_for='pixel')

    @get_local('attn')
    def forward(self, x, demographic_info, explanation=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if explanation is not None:
            qkv_explanation = self.qkv(explanation).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv_explanation[0]
            k = qkv_explanation[1]

        # rotate the query and key to add the 2D frequency encoding
        q, k = self.rope2D(q, k, demographic_info)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, in_features, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(in_features))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class XAIguiTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            in_features,
            num_heads,
            freqband,
            dim_feedforward=None,
            mlp_ratio=4.,
            bias=False,
            dropout=0.,
            attn_drop=0.,
            init_values=None,
            droppath=0.,
            act_func=nn.GELU,
            norm=nn.LayerNorm,
            layer_norm_eps=1e-05,
    ):
        super().__init__()

        dim_feedforward = dim_feedforward or int(mlp_ratio * in_features)

        self.norm1 = norm(in_features, eps=layer_norm_eps)
        self.attn = XAIguiAttention(in_features, num_heads=num_heads, freqband=freqband, qkv_bias=bias, attn_drop=attn_drop, proj_drop=dropout)
        self.ls1 = LayerScale(in_features, init_values=init_values) if init_values else Identity()
        self.drop_path1 = DropPath(droppath) if droppath > 0. else Identity()

        self.norm2 = norm(in_features, eps=layer_norm_eps)
        self.mlp = MLP(in_features, in_features, hidden_features=dim_feedforward, act_func=act_func, dropout=dropout)
        self.ls2 = LayerScale(in_features, init_values=init_values) if init_values else Identity()
        self.drop_path2 = DropPath(droppath) if droppath > 0. else Identity()

    def forward(self, x, demographic_info, explanation=None):
        x = self.norm1(x + self.drop_path1(self.ls1(self.attn(x, demographic_info, explanation))))
        x = self.norm2(x + self.drop_path2(self.ls2(self.mlp(x))))
        return x


class XAIguiTransformerEncoder(nn.Module):
    def __init__(
            self,
            in_features,
            num_heads,
            num_layers,
            freqband,
            num_classes,
            dim_feedforward=None,
            dropout=0.1,
            act_func=nn.GELU,
            layer_norm_eps=1e-05,
            mlp_ratio=4.,
            bias=True,
            init_values=None,
            attn_drop=0.,
            droppath=0.,
            norm=nn.LayerNorm,
    ):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, droppath, num_layers)]  # stochastic depth decay rule

        self.TransformerEncoder = nn.ModuleList([XAIguiTransformerEncoderLayer(
            in_features=in_features, num_heads=num_heads, freqband=freqband,
            dim_feedforward=dim_feedforward, mlp_ratio=mlp_ratio, bias=bias,
            dropout=dropout, attn_drop=attn_drop, init_values=init_values,
            droppath=dpr[i], act_func=act_func, norm=norm, layer_norm_eps=layer_norm_eps
        ) for i in range(num_layers)])
        self.avgpool = AvgPool(dim=1)
        self.head = MLP(in_features, num_classes, act_func=act_func, dropout=dropout)

    def forward(self, freq_series, demographic_info, explanation=None):
        for i, layer in enumerate(self.TransformerEncoder):
            if explanation is not None:
                freq_series = layer(freq_series, demographic_info, explanation[i][0].to(torch.float32))
            else:
                freq_series = layer(freq_series, demographic_info)
        y = self.head(self.avgpool(freq_series))
        return y
