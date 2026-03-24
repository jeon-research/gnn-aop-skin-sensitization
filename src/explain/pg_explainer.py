"""
PGExplainer for molecular GNNs.

PGExplainer (Luo et al., 2020) learns a parameterized edge mask predictor.
Instead of optimizing masks per instance (like GNNExplainer), it trains a
small MLP that predicts edge importance from edge embeddings.

This is a standalone implementation adapted for AblationGNN, since PyG's
Explainer API has compatibility issues with AttentiveFP's internal indexing.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class PGExplainerWrapper:
    """PGExplainer for atom-level attribution in molecular GNNs.

    Learns an MLP that predicts edge importance masks from edge features.
    After training on a set of graphs, it can explain new graphs without
    re-optimization. Edge importances are aggregated to node-level.
    """

    def __init__(
        self,
        model: nn.Module,
        target_key: str = 'sensitization',
        epochs: int = 30,
        lr: float = 0.003,
        temp: float = 1.0,
        reg_coef: float = 0.01,
    ):
        self.model = model
        self.target_key = target_key
        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coef = reg_coef
        self._edge_mlp = None
        self._trained = False

    def _build_edge_mlp(self, edge_dim: int, device: torch.device):
        """Build the edge importance MLP."""
        self._edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

    def _get_node_embeddings(self, data: Data, device: torch.device) -> torch.Tensor:
        """Extract per-node embeddings from the model's encoder."""
        data = data.clone().to(device)
        if data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # Use the encoder to get node-level representations
        x = self.model.encoder.atom_encoder(data.x)
        edge_attr = self.model.encoder.bond_encoder(data.edge_attr) if data.edge_attr.numel() > 0 else None

        # Get GNN intermediate representations (before global pooling)
        # AttentiveFP stores intermediate node features
        self.model.eval()
        with torch.no_grad():
            # Run through GNN layers to get node embeddings
            gnn = self.model.encoder.gnn
            # AttentiveFP forward: x -> layers -> readout
            # We need pre-pooling node features
            from torch_geometric.nn.models.attentive_fp import AttentiveFP
            # Access internal layers
            x_out = x
            for layer in gnn.atom_convs:
                x_out = layer(x_out, data.edge_index, edge_attr)
            for layer in gnn.atom_grus:
                x_out = layer(x_out)

        return x_out.detach()

    def _get_edge_embeddings(self, data: Data, device: torch.device) -> torch.Tensor:
        """Create edge embeddings by concatenating source and target node embeddings."""
        node_emb = self._get_node_embeddings(data, device)
        edge_index = data.edge_index.to(device)

        src_emb = node_emb[edge_index[0]]
        dst_emb = node_emb[edge_index[1]]
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        return edge_emb

    def train_on_loader(self, graphs: list, device: torch.device):
        """Train the edge MLP on a set of graphs.

        Args:
            graphs: List of PyG Data objects.
            device: Training device.
        """
        if not graphs:
            return

        # Get edge embedding dimension from first graph
        sample_emb = self._get_edge_embeddings(graphs[0], device)
        edge_dim = sample_emb.size(-1)

        if self._edge_mlp is None:
            self._build_edge_mlp(edge_dim, device)

        optimizer = torch.optim.Adam(self._edge_mlp.parameters(), lr=self.lr)

        # Get original predictions
        self.model.eval()
        orig_preds = []
        for g in graphs:
            data = g.clone().to(device)
            if data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            with torch.no_grad():
                out = self.model(data)
                orig_preds.append(torch.sigmoid(out[self.target_key]).item())

        for epoch in range(self.epochs):
            total_loss = 0
            for i, g in enumerate(graphs):
                data = g.clone().to(device)
                if data.batch is None:
                    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

                # Get edge embeddings
                edge_emb = self._get_edge_embeddings(data, device)

                # Predict edge importance
                edge_logits = self._edge_mlp(edge_emb).squeeze(-1)

                # Sample edge mask using Gumbel-softmax reparameterization
                edge_mask = torch.sigmoid(edge_logits / self.temp)

                # Apply mask to node features via message passing effect
                # Mask edge_attr: scale edge features by importance
                masked_data = data.clone()
                if masked_data.edge_attr is not None and masked_data.edge_attr.numel() > 0:
                    masked_data.edge_attr = masked_data.edge_attr * edge_mask.unsqueeze(-1)

                # Forward pass with masked edges
                try:
                    out = self.model(masked_data)
                    pred = out[self.target_key]
                    if pred.dim() == 0:
                        pred = pred.unsqueeze(0)

                    # Prediction preservation loss
                    target_prob = torch.tensor([orig_preds[i]], device=device)
                    pred_loss = F.binary_cross_entropy_with_logits(pred, target_prob)

                    # Sparsity regularization
                    reg_loss = self.reg_coef * edge_mask.mean()

                    # Entropy regularization
                    ent = -edge_mask * torch.log(edge_mask + 1e-8) - \
                          (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
                    ent_loss = 0.1 * ent.mean()

                    loss = pred_loss + reg_loss + ent_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except Exception:
                    continue

        self._trained = True

    def attribute(self, data: Data, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute atom-level PGExplainer attributions.

        Args:
            data: PyG Data object (single molecule).
            device: Device to run on.

        Returns:
            [n_atoms] importance scores in [0, 1].
        """
        if device is None:
            device = next(self.model.parameters()).device

        data = data.clone().to(device)
        n_atoms = data.x.size(0)

        if data.batch is None:
            data.batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

        if self._edge_mlp is None or not self._trained:
            # Fallback: return uniform importance if not trained
            return torch.ones(n_atoms) / n_atoms

        # Get edge embeddings and predict importance
        edge_emb = self._get_edge_embeddings(data, device)
        with torch.no_grad():
            edge_logits = self._edge_mlp(edge_emb).squeeze(-1)
            edge_importance = torch.sigmoid(edge_logits)

        # Aggregate edge importance to nodes
        node_importance = torch.zeros(n_atoms, device=device)
        edge_index = data.edge_index
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            node_importance[src] += edge_importance[i]
            node_importance[dst] += edge_importance[i]

        # Normalize to [0, 1]
        if node_importance.max() > node_importance.min():
            node_importance = (node_importance - node_importance.min()) / \
                             (node_importance.max() - node_importance.min())

        return node_importance.detach().cpu()
