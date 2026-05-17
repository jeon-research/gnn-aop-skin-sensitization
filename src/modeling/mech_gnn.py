"""
MechGNN: Atom-level mechanistic supervision for interpretable GNN toxicity prediction.

Extends the plain AblationGNN (AttentiveFP) with a per-atom MIE prediction head
that intercepts node features after message passing but before graph-level pooling.
This teaches the GNN to produce mechanistically meaningful atom-level representations.

Architecture:
    Input graph → atom/bond encoders → AttentiveFP node pass (GATEConv + GRUs)
    → node_features [N, 256]
        ├── AttentiveFP graph pass → projection → sensitization_head → sens_logit
        └── NodeMIEHead (Linear→ReLU→DO→Linear) → atom_mie_logits [N, 1]

    Loss = L_sens + λ * L_atom_mie
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool import global_add_pool

from src.modeling.ablation_model import SharedMolecularEncoder
from src.modeling.causal_aop_gnn import CausalPredictionHead


class NodeMIEHead(nn.Module):
    """Per-atom MIE prediction head.

    Predicts whether each atom is part of a Molecular Initiating Event
    reactive center based on its post-message-passing representation.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [N, input_dim] node representations after message passing.

        Returns:
            logits: [N] per-atom MIE logits.
        """
        return self.head(node_features).squeeze(-1)


class MechGNN(nn.Module):
    """GNN with atom-level mechanistic supervision.

    Reuses SharedMolecularEncoder (AttentiveFP backbone) but intercepts
    node features after the atom-level message passing to add a per-atom
    MIE prediction head alongside the standard graph-level sensitization head.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        node_dim: int = 64,
        num_gnn_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim

        # Shared molecular encoder (AttentiveFP backbone)
        self.encoder = SharedMolecularEncoder(
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # Graph-level sensitization prediction head (same as AblationGNN plain)
        self.sensitization_head = CausalPredictionHead(node_dim, hidden_dim // 2, dropout)

        # NEW: Per-atom MIE prediction head
        self.node_mie_head = NodeMIEHead(
            input_dim=hidden_dim,
            hidden_dim=64,
            dropout=dropout,
        )

    def _attentive_fp_node_pass(self, x: torch.Tensor, edge_index: torch.Tensor,
                                edge_attr: torch.Tensor) -> torch.Tensor:
        """Run only the atom-level message passing of AttentiveFP.

        Reproduces the first half of AttentiveFP.forward() to get
        node features after all GATEConv/GRU layers but before pooling.

        Args:
            x: [N, hidden_dim] encoded atom features.
            edge_index: [2, E] edge indices.
            edge_attr: [E, edge_dim] encoded edge features.

        Returns:
            node_features: [N, hidden_dim] post-message-passing node features.
        """
        gnn = self.encoder.gnn

        # First layer: GATEConv + GRU
        x = F.leaky_relu_(gnn.lin1(x))
        h = F.elu_(gnn.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=gnn.dropout, training=self.training)
        x = gnn.gru(h, x).relu_()

        # Additional atom convolution layers
        for conv, gru in zip(gnn.atom_convs, gnn.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=gnn.dropout, training=self.training)
            x = gru(h, x).relu()

        return x

    def _attentive_fp_graph_pass(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Run only the graph-level pooling of AttentiveFP.

        Reproduces the second half of AttentiveFP.forward():
        global_add_pool → mol_conv/mol_gru iterations → lin2.

        Args:
            x: [N, hidden_dim] post-message-passing node features.
            batch: [N] batch assignment vector.

        Returns:
            graph_embedding: [B, hidden_dim] graph-level representations.
        """
        gnn = self.encoder.gnn

        # Molecule embedding (same as AttentiveFP)
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for _ in range(gnn.num_timesteps):
            h = F.elu_(gnn.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=gnn.dropout, training=self.training)
            out = gnn.mol_gru(h, out).relu_()

        out = F.dropout(out, p=gnn.dropout, training=self.training)
        return gnn.lin2(out)

    def forward(self, data: Union[Data, Batch]) -> Dict[str, torch.Tensor]:
        """Forward pass with both graph-level and atom-level predictions.

        Returns:
            Dict with:
                'sensitization': [B] graph-level sensitization logits
                'atom_mie_logits': [N] per-atom MIE logits
                'atom_mie_batch': [N] batch assignment for atom_mie_logits
        """
        # Encode atom and bond features
        x = self.encoder.atom_encoder(data.x)
        edge_attr = (self.encoder.bond_encoder(data.edge_attr)
                     if data.edge_attr.numel() > 0 else None)

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )

        # Atom-level message passing (intercept node features)
        node_features = self._attentive_fp_node_pass(x, data.edge_index, edge_attr)

        # Branch 1: Graph-level prediction
        graph_embedding = self._attentive_fp_graph_pass(node_features, batch)
        graph_embedding = self.encoder.projection(graph_embedding)
        sens_logit, _ = self.sensitization_head(graph_embedding)

        # Branch 2: Atom-level MIE prediction
        atom_mie_logits = self.node_mie_head(node_features)

        return {
            'sensitization': sens_logit,
            'atom_mie_logits': atom_mie_logits,
            'atom_mie_batch': batch,
        }

    def load_pretrained(self, checkpoint_path: str, device: torch.device = None):
        """Initialize encoder + sensitization_head from a trained AblationGNN checkpoint.

        The node_mie_head remains randomly initialized.

        Args:
            checkpoint_path: Path to AblationGNN best_model.pt checkpoint.
            device: Device for loading weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # Load encoder weights (strip leading 'encoder.' prefix only)
        prefix = 'encoder.'
        encoder_keys = {k[len(prefix):]: v for k, v in state_dict.items()
                        if k.startswith(prefix)}
        missing, unexpected = self.encoder.load_state_dict(encoder_keys, strict=False)
        if missing:
            print(f"  Warning: missing encoder keys: {missing}")

        # Load sensitization_head weights (strip leading prefix only)
        prefix = 'sensitization_head.'
        head_keys = {k[len(prefix):]: v for k, v in state_dict.items()
                     if k.startswith(prefix)}
        missing, unexpected = self.sensitization_head.load_state_dict(head_keys, strict=False)
        if missing:
            print(f"  Warning: missing sensitization_head keys: {missing}")

        print(f"  Loaded pretrained weights from {checkpoint_path}")
        print(f"  node_mie_head initialized randomly (will be trained)")
