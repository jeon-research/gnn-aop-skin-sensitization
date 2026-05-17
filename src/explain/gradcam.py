"""
GradCAM for graph neural networks.

Gradient-weighted Class Activation Mapping adapted for GNNs (Pope et al., 2019).
Uses a modified forward pass to capture intermediate node representations
and their gradients for computing per-atom importance.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool


class GradCAM:
    """GradCAM for atom-level attribution in molecular GNNs.

    Captures node-level activations from the GNN message-passing layers
    and weights them by gradients to produce per-atom importance scores.
    Uses a modified forward through the encoder to capture pre-pooling
    node embeddings.
    """

    def __init__(self, model: nn.Module, target_key: str = 'sensitization'):
        self.model = model
        self.target_key = target_key

    def _forward_with_node_features(self, data: Data, device: torch.device):
        """Modified forward that returns both predictions and per-atom node features.

        We replicate the encoder's forward pass but intercept the node-level
        features before global pooling.
        """
        encoder = self.model.encoder
        gnn = encoder.gnn

        x = encoder.atom_encoder(data.x)
        edge_attr = encoder.bond_encoder(data.edge_attr) if data.edge_attr.numel() > 0 else None

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
            torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # Replicate AttentiveFP forward but capture node features before pooling
        # Atom embedding phase
        x = F.leaky_relu(gnn.lin1(x))  # Use non-inplace version
        h = F.elu(gnn.gate_conv(x, data.edge_index, edge_attr))
        h = F.dropout(h, p=gnn.dropout, training=gnn.training)
        x = gnn.gru(h, x).relu()

        for conv, gru in zip(gnn.atom_convs, gnn.atom_grus):
            h = conv(x, data.edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=gnn.dropout, training=gnn.training)
            x = gru(h, x).relu()

        # x is now the per-atom representation [n_atoms, hidden_dim]
        # This is the node-level feature we want for GradCAM
        node_features = x
        node_features.retain_grad()

        # Molecule embedding phase (pooling)
        row = torch.arange(batch.size(0), device=device)
        edge_index_mol = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu()
        for _ in range(gnn.num_timesteps):
            h = F.elu(gnn.mol_conv((x, out), edge_index_mol))
            h = F.dropout(h, p=gnn.dropout, training=gnn.training)
            out = gnn.mol_gru(h, out).relu()

        out = F.dropout(out, p=gnn.dropout, training=gnn.training)
        mol_embedding = gnn.lin2(out)

        # Apply projection
        embedding = encoder.projection(mol_embedding)

        # Continue with the rest of the model (plain condition: shared embedding)
        sens_logit, _ = self.model.sensitization_head(embedding)

        return sens_logit, node_features, batch

    def attribute(self, data: Data, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute atom-level GradCAM attributions for a single molecule.

        Args:
            data: PyG Data object (single molecule).
            device: Device to run on.

        Returns:
            [n_atoms] importance scores (non-negative).
        """
        if device is None:
            device = next(self.model.parameters()).device

        data = data.clone().to(device)
        n_atoms = data.x.size(0)

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

        self.model.eval()
        data.x = data.x.detach().requires_grad_(True)

        # Forward pass capturing node features
        logit, node_features, batch = self._forward_with_node_features(data, device)

        if logit.dim() == 0:
            logit = logit.unsqueeze(0)

        # Compute gradients of target w.r.t. node features
        self.model.zero_grad()
        logit.sum().backward(retain_graph=False)

        # Use Gradient x Activation approach (adapted for GNNs with global pooling)
        # Standard GradCAM uses global-average-pooled gradients, but with sum pooling
        # the gradients are nearly uniform across atoms. Instead, use per-node
        # gradient-activation products (Pope et al., 2019, Section 3.2).
        if node_features.grad is not None:
            gradients = node_features.grad.detach()
        elif data.x.grad is not None:
            gradients = data.x.grad.detach()
            node_features = data.x.detach()
        else:
            return torch.zeros(n_atoms)

        activations = node_features.detach()

        # Per-node: absolute sum of (gradient * activation) across features
        cam = (gradients * activations).sum(dim=-1).abs()  # [n_atoms]

        return cam.detach().cpu()

    def attribute_batch(self, batch: Batch, device: Optional[torch.device] = None) -> list:
        """Compute GradCAM attributions for a batch of molecules."""
        data_list = batch.to_data_list()
        results = []
        for data in data_list:
            importance = self.attribute(data, device=device)
            results.append(importance)
        return results
