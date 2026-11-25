"""Frequency-aware graph encoding and radial state-space building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GINConv, MessagePassing

import kafnets as kaf


GRAPH_ATTENTION_LAYERS = {
    'gat': GATConv,
    'gatv2': GATv2Conv,
    'gin': GINConv,
}


FREQUENCY_FUSION_MODES = {
    'gated',
    'low_only',
    'high_only',
    'direct_sum',
}


FREQUENCY_ACTIVATIONS = {
    'kaf',
    'silu',
}


class IdentityResetSiLU(nn.SiLU):
    def reset_parameters(self):
        return None


def build_frequency_backbone_layer(attention_type, in_channels, out_channels):
    if attention_type == 'gin':
        return GINConv(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
        )
    attention_layer = GRAPH_ATTENTION_LAYERS[attention_type]
    return attention_layer(in_channels, out_channels, heads=4, concat=False)


class SemanticAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super(SemanticAttention, self).__init__()
        reduction_dim = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.fc(x)
        return x * attention_weights


class FrequencyGatedGraphAttention(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        attention_type='gat',
        frequency_fusion_mode='gated',
        frequency_activation='kaf',
    ):
        super(FrequencyGatedGraphAttention, self).__init__(aggr='add')
        if attention_type not in GRAPH_ATTENTION_LAYERS:
            raise ValueError(
                f"Unsupported attention_type '{attention_type}'. "
                f"Expected one of {sorted(GRAPH_ATTENTION_LAYERS)}."
            )
        if frequency_fusion_mode not in FREQUENCY_FUSION_MODES:
            raise ValueError(
                f"Unsupported frequency_fusion_mode '{frequency_fusion_mode}'. "
                f"Expected one of {sorted(FREQUENCY_FUSION_MODES)}."
            )
        if frequency_activation not in FREQUENCY_ACTIVATIONS:
            raise ValueError(
                f"Unsupported frequency_activation '{frequency_activation}'. "
                f"Expected one of {sorted(FREQUENCY_ACTIVATIONS)}."
            )
        self.frequency_fusion_mode = frequency_fusion_mode
        self.frequency_activation = frequency_activation
        self.low_pass = build_frequency_backbone_layer(
            attention_type,
            in_channels,
            out_channels,
        )
        self.skip_proj = nn.Linear(in_channels, out_channels)
        if frequency_activation == 'kaf':
            self.kaf_low = kaf.KAF(out_channels, D=8, kernel='gaussian')
            self.kaf_high = kaf.KAF(out_channels, D=8, kernel='gaussian')
        else:
            self.kaf_low = IdentityResetSiLU()
            self.kaf_high = IdentityResetSiLU()

        gate_input_dim = out_channels * 4
        hidden_dim = out_channels * 2
        self.freq_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels)
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        base = self.skip_proj(x)
        h_low_raw = self.low_pass(x, edge_index)
        h_low = self.kaf_low(h_low_raw)

        h_high = self.kaf_high(base - h_low_raw)
        low_delta = h_low - base

        if self.frequency_fusion_mode == 'low_only':
            return self.bn(h_low)
        if self.frequency_fusion_mode == 'high_only':
            return self.bn(base + h_high)
        if self.frequency_fusion_mode == 'direct_sum':
            return self.bn(h_low + h_high)

        gate_input = torch.cat(
            [base, low_delta, h_high, torch.abs(low_delta - h_high)],
            dim=-1
        )
        gate = torch.sigmoid(self.freq_gate(gate_input))
        out = base + gate * low_delta + (1.0 - gate) * h_high
        return self.bn(out)

    def reset_parameters(self):
        self.low_pass.reset_parameters()
        self.skip_proj.reset_parameters()
        self.kaf_low.reset_parameters()
        self.kaf_high.reset_parameters()
        for layer in self.freq_gate:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.bn.reset_parameters()


class RadialStateSpaceBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(RadialStateSpaceBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.delta_proj = nn.Linear(d_model, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.a_log = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.delta_proj.reset_parameters()
        self.input_proj.reset_parameters()
        self.gate_proj.reset_parameters()
        self.out_proj.reset_parameters()
        nn.init.zeros_(self.a_log)

    def forward(self, x, mask=None):
        state = x.new_zeros(x.size(0), x.size(-1))
        outputs = []
        decay_base = F.softplus(self.a_log).unsqueeze(0)

        for step in range(x.size(1) - 1, -1, -1):
            x_step = self.norm(x[:, step, :])
            delta = F.softplus(self.delta_proj(x_step))
            input_state = torch.tanh(self.input_proj(x_step))
            gate = torch.sigmoid(self.gate_proj(x_step))
            decay = torch.exp(-delta * decay_base)
            updated_state = decay * state + (1.0 - decay) * input_state

            if mask is not None:
                step_mask = mask[:, step].unsqueeze(-1)
                state = torch.where(step_mask, updated_state, state)
            else:
                state = updated_state

            outputs.append(self.out_proj(state * gate))

        outputs.reverse()
        return torch.stack(outputs, dim=1)
