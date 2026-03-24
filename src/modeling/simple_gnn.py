"""
Simple GCN and GIN encoders for multi-architecture comparison.

Drop-in replacements for SharedMolecularEncoder with the same interface:
    input:  PyG Data with data.x, data.edge_index, data.batch
    output: [batch_size, node_dim] molecular embedding

No edge features used (GCNConv/GINConv don't support them natively).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

try:
    from src.modeling.causal_aop_gnn import MoleculeFeaturizer
except ImportError:
    from causal_aop_gnn import MoleculeFeaturizer

ATOM_DIM = sum(len(v) for v in MoleculeFeaturizer.ATOM_FEATURES.values())


class SimpleGCNEncoder(nn.Module):
    """GCN encoder: atom_encoder -> GCNConv layers -> global_mean_pool -> projection."""

    def __init__(self, hidden_dim=256, node_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.atom_encoder = nn.Linear(ATOM_DIM, hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        x = self.atom_encoder(data.x)
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        mol = global_mean_pool(x, batch)
        return self.projection(mol)


class SimpleGINEncoder(nn.Module):
    """GIN encoder: atom_encoder -> GINConv(MLP) layers -> global_mean_pool -> projection."""

    def __init__(self, hidden_dim=256, node_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.atom_encoder = nn.Linear(ATOM_DIM, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        x = self.atom_encoder(data.x)
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        mol = global_mean_pool(x, batch)
        return self.projection(mol)
