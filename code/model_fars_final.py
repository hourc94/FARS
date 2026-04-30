import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from model_fars_blocks import (
    FrequencyGatedGraphAttention,
    RadialStateSpaceBlock,
    SemanticAttention,
)


class FARSFinalBackboneMixin:
    """Backbone utilities for FARS final model."""

    def _init_backbone(
        self,
        dataset,
        latent_dim,
        dropout_e,
        force_undirected,
        attention_type='gat',
    ):
        self.dropout_e = dropout_e
        self.force_undirected = force_undirected

        self.conv1 = FrequencyGatedGraphAttention(
            dataset.num_features,
            latent_dim[0],
            attention_type=attention_type,
        )
        self.conv2 = FrequencyGatedGraphAttention(
            latent_dim[0],
            latent_dim[1],
            attention_type=attention_type,
        )
        self.conv3 = FrequencyGatedGraphAttention(
            latent_dim[1],
            latent_dim[2],
            attention_type=attention_type,
        )

        self.semantic_attn1 = SemanticAttention(latent_dim[0])
        self.semantic_attn2 = SemanticAttention(latent_dim[1])
        self.semantic_attn3 = SemanticAttention(latent_dim[2])
        self.total_latent_dim = sum(latent_dim)

    def _reset_backbone(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

        for module in (self.semantic_attn1, self.semantic_attn2, self.semantic_attn3):
            for layer in module.fc:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def _encode_backbone(self, x, edge_index, batch):
        del batch
        edge_index, _ = dropout_edge(
            edge_index,
            p=self.dropout_e,
            force_undirected=self.force_undirected,
            training=self.training,
        )

        x1 = self.semantic_attn1(self.conv1(x, edge_index))
        x2 = self.semantic_attn2(self.conv2(x1, edge_index))
        x3 = self.semantic_attn3(self.conv3(x2, edge_index))
        return x1, x2, x3


class FARSRadialFrequencyFinal(torch.nn.Module, FARSFinalBackboneMixin):
    """FARS final model."""

    def __init__(
        self,
        dataset,
        latent_dim=None,
        dropout_n=0.4,
        dropout_e=0.1,
        force_undirected=False,
        radial_layers=2,
        attention_type='gat',
    ):
        super(FARSRadialFrequencyFinal, self).__init__()

        if latent_dim is None:
            latent_dim = [256, 128, 64]

        self.dropout_n = dropout_n
        self.max_hop = int(getattr(dataset, 'hop', 2))
        self._init_backbone(
            dataset,
            latent_dim,
            dropout_e,
            force_undirected,
            attention_type=attention_type,
        )

        self.shell_input_dim = self.total_latent_dim * 4 + 2
        self.shell_proj = nn.Sequential(
            nn.LayerNorm(self.shell_input_dim),
            nn.Linear(self.shell_input_dim, self.total_latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_n),
        )
        self.radial_blocks = nn.ModuleList(
            [RadialStateSpaceBlock(self.total_latent_dim, dropout=dropout_n) for _ in range(radial_layers)]
        )
        self.readout_norm = nn.LayerNorm(self.total_latent_dim)
        self.lin1 = nn.Linear(self.total_latent_dim, 128)
        self.lin2 = nn.Linear(128, 1)

    def _hop_wise_pool(self, x, batch, shell_index, side_index):
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        num_shells = self.max_hop + 1
        clamped_shell = shell_index.clamp(max=self.max_hop)
        flat_index = batch * (num_shells * 2) + clamped_shell * 2 + side_index

        num_slots = num_graphs * num_shells * 2
        pooled_sum = x.new_zeros(num_slots, x.size(-1))
        pooled_sum.index_add_(0, flat_index, x)

        counts = x.new_zeros(num_slots)
        counts.index_add_(0, flat_index, torch.ones_like(shell_index, dtype=x.dtype))
        slot_mask = counts > 0
        counts_clamped = counts.clamp_min(1.0)

        pooled_mean = pooled_sum / counts_clamped.unsqueeze(-1)
        pooled_max = x.new_full((num_slots, x.size(-1)), float('-inf'))
        pooled_max.scatter_reduce_(
            0,
            flat_index.unsqueeze(-1).expand(-1, x.size(-1)),
            x,
            reduce='amax',
            include_self=True,
        )
        pooled_max = torch.where(slot_mask.unsqueeze(-1), pooled_max, torch.zeros_like(pooled_max))

        pooled_mean = pooled_mean.view(num_graphs, num_shells, 2, x.size(-1))
        pooled_max = pooled_max.view(num_graphs, num_shells, 2, x.size(-1))
        counts = counts.view(num_graphs, num_shells, 2)
        shell_mask = counts.sum(dim=-1) > 0
        shell_counts = torch.log1p(counts)

        shell_features = torch.cat(
            [
                pooled_mean[:, :, 0, :],
                pooled_max[:, :, 0, :],
                pooled_mean[:, :, 1, :],
                pooled_max[:, :, 1, :],
                shell_counts,
            ],
            dim=-1,
        )
        return shell_features, shell_mask

    def reset_parameters(self):
        self._reset_backbone()

        for layer in self.shell_proj:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for block in self.radial_blocks:
            block.reset_parameters()

        self.readout_norm.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        raw_x, edge_index, batch = data.x, data.edge_index, data.batch
        node_labels = torch.argmax(raw_x, dim=-1)
        shell_index = node_labels // 2
        side_index = node_labels % 2

        x1, x2, x3 = self._encode_backbone(raw_x, edge_index, batch)
        node_states = torch.cat([x1, x2, x3], dim=-1)

        shell_states, shell_mask = self._hop_wise_pool(node_states, batch, shell_index, side_index)
        shell_states = self.shell_proj(shell_states)

        radial_states = shell_states
        for block in self.radial_blocks:
            radial_states = radial_states + block(radial_states, shell_mask)

        graph_state = self.readout_norm(radial_states[:, 0, :])
        graph_state = F.relu(self.lin1(graph_state))
        graph_state = F.dropout(graph_state, p=self.dropout_n, training=self.training)
        graph_state = self.lin2(graph_state)
        return graph_state[:, 0]

