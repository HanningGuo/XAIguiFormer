# -----------------------------------------------------
# steal 1D rotary embeddings from the github repository:
# https://github.com/lucidrains/rotary-embedding-torch
# and further modify it by integrating demographics to rotary position encoding
# -----------------------------------------------------

import math
import torch
import torch.nn as nn
from torch import einsum, Tensor
from torch.cuda.amp import autocast
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    """
    positional encoding
    """
    def __init__(self, d_model, num_freqband, n=10000):
        super().__init__()

        position = torch.arange(num_freqband).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(n) / d_model))

        pe = torch.zeros(num_freqband, d_model)
        pe.require_grad = False
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self):
        return self.pe


# rotary embedding helper functions
def rotate_every_two(x):
    # convert to [-q1, q0, -q3, q2, ..., -qd-1, qd-2]
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


@autocast(enabled=False)
def apply_rotary_emb(freqs, to_be_rotated, demographic_info=None, start_index=0, scale=1., seq_dim=-2):
    if to_be_rotated.ndim == 3:
        seq_len = to_be_rotated.shape[seq_dim]
        freqs = freqs[-seq_len:].to(to_be_rotated)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    age, gender = demographic_info[:, 0], demographic_info[:, 1]
    freqs_cos = age.view(len(age), 1, 1) * freqs.cos() + gender.view(len(gender), 1, 1)
    freqs_sin = age.view(len(age), 1, 1) * freqs.sin() + gender.view(len(gender), 1, 1)
    freqs_cos = freqs_cos.unsqueeze(1).repeat(1, to_be_rotated.shape[1], 1, 1)
    freqs_sin = freqs_sin.unsqueeze(1).repeat(1, to_be_rotated.shape[1], 1, 1)

    assert rot_dim <= to_be_rotated.shape[-1], f'feature dimension {to_be_rotated.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = to_be_rotated[..., :start_index], to_be_rotated[..., start_index:end_index], to_be_rotated[..., end_index:]
    t = (t * freqs_cos * scale) + (rotate_every_two(t) * freqs_sin * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


# learned rotation helpers
def apply_learned_rotations(rotations, to_be_rotated, start_index=0, freq_ranges=None):
    if freq_ranges is not None:
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, to_be_rotated, start_index=start_index)


class dRoFE(nn.Module):
    """
    two-dimensional rotary frequency encoding with demographics
    """
    def __init__(
            self,
            dim,
            freq_band_pos,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
            use_xpos=False,
            xpos_scale_base=512,
            interpolate_factor=1.,
            theta_rescale_factor=1.,
            seq_before_head_dim=False,
            cache_if_possible=True
    ):
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freq_band_pos = freq_band_pos
        self.freqs_for = freqs_for
        assert self.freqs_for in ['lang', 'pixel', 'constant']

        assert dim % 4 == 0, 'The head dimension must be divided by 4 in 2D rotary position encoding!'
        rotary_dim = dim // 2
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, rotary_dim, 2)[:(rotary_dim // 2)].float() / rotary_dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, rotary_dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device
        self.tmp_store('dummy', torch.tensor(0))
        # default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors
        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos
        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_scale(
        self,
        t: Tensor,
        seq_len=None,
        offset=0
    ):
        assert self.use_xpos

        should_cache = self.cache_if_possible and seq_len is not None

        if should_cache and self.cached_scales is not None and (seq_len + offset) <= self.cached_scales.shape[0]:
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.tmp_store('cached_scales', scale)

        return scale

    @autocast(enabled=False)
    def forward(
            self,
            to_be_rotated_q,
            to_be_rotated_k,
            demographic_info,
            seq_len=None,
            seq_dim=None,
            freq_seq_len=None
    ):
        seq_dim = seq_dim if seq_dim is not None else self.default_seq_dim

        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        if seq_len is None:
            seq_len = to_be_rotated_q.shape[seq_dim]

        if freq_seq_len is not None:
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        should_cache = self.cache_if_possible and not self.learned_freq and seq_len is not None and self.freqs_for != 'pixel'

        lower_boundary, upper_boundary = self.freq_band_pos[..., 0].to(self.freqs.device), self.freq_band_pos[..., 1].to(self.freqs.device)

        # rotary frequency encoding
        freqs = self.freqs
        # lower_boundary * [theta0, theta1, theta2, ..., theta(d/4-1)]
        lower_freqs = einsum('..., f -> ... f', lower_boundary.type(freqs.dtype), freqs)
        # upper_boundary * [theta0, theta1, theta2, ..., theta(d/4-1)]
        upper_freqs = einsum('..., f -> ... f', upper_boundary.type(freqs.dtype), freqs)

        # [[lower_boundary * theta0, upper_boundary * theta0],
        # [lower_boundary * theta1, upper_boundary * theta1],
        # ...,
        # [lower_boundary * theta(d/4-1), upper_boundary * theta(d/4-1)]]
        freqs_2D = torch.cat([lower_freqs.unsqueeze(-1), upper_freqs.unsqueeze(-1)], dim=-1)

        # [lower_boundary * theta0, lower_boundary * theta0, upper_boundary * theta0, upper_boundary * theta0,
        # lower_boundary * theta1, lower_boundary * theta1, upper_boundary * theta1, upper_boundary * theta1,
        # ...,
        # lower_boundary * theta(d/4-1), lower_boundary * theta(d/4-1), upper_boundary * theta(d/4-1), upper_boundary * theta(d/4-1)]
        freqs_2D = repeat(freqs_2D.flatten(-2), '... n -> ... (n r)', r=2)

        if seq_dim == -3:
            freqs_2D = rearrange(freqs_2D, 'n d -> n 1 d')

        if should_cache:
            self.tmp_store('cached_freqs', freqs_2D.detach())

        rotated_q = apply_rotary_emb(freqs_2D, to_be_rotated_q, demographic_info, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs_2D, to_be_rotated_k, demographic_info, seq_dim=seq_dim)

        return rotated_q, rotated_k
