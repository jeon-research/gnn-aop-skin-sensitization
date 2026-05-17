"""
GraphMask-style explainer for molecular GNNs.

Inspired by GraphMask (Schlichtkrull et al., 2021), learns soft binary
gates on edges to determine which message-passing connections are critical.

Standalone implementation that works with AblationGNN, since PyG's
Explainer API has compatibility issues with AttentiveFP's internal indexing.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class GraphMaskExplainerWrapper:
    """GraphMask-style explainer for atom-level attribution.

    For each molecule, optimizes binary edge gates that determine which
    edges are necessary for the model's prediction. Like GNNExplainer
    but uses a different loss formulation inspired by GraphMask.
    """

    def __init__(
        self,
        model: nn.Module,
        target_key: str = 'sensitization',
        n_steps: int = 100,
        lr: float = 0.01,
        penalty_scaling: float = 5.0,
        allowance: float = 0.03,
    ):
        self.model = model
        self.target_key = target_key
        self.n_steps = n_steps
        self.lr = lr
        self.penalty_scaling = penalty_scaling
        self.allowance = allowance

    def attribute(self, data: Data, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute atom-level GraphMask attributions for a single molecule.

        Learns binary edge gates that preserve the prediction while being
        as sparse as possible.

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
        n_edges = data.edge_index.size(1)

        if n_edges == 0:
            return torch.zeros(n_atoms)

        if data.batch is None:
            data.batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            orig_out = self.model(data)
            orig_logit = orig_out[self.target_key]
            if orig_logit.dim() == 0:
                orig_logit = orig_logit.unsqueeze(0)
            orig_prob = torch.sigmoid(orig_logit).item()

        # Initialize learnable edge gate logits
        gate_logits = (torch.randn(n_edges, device=device) * 0.1).requires_grad_(True)

        # Lagrange multiplier for the constraint
        lambda_param = torch.tensor(self.penalty_scaling, device=device, requires_grad=False)

        optimizer = torch.optim.Adam([gate_logits], lr=self.lr)

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # Hard sigmoid gates (approximate binary)
            gates = torch.sigmoid(gate_logits)

            # Apply gates to edge features
            masked_data = data.clone()
            if masked_data.edge_attr is not None and masked_data.edge_attr.numel() > 0:
                masked_data.edge_attr = masked_data.edge_attr * gates.unsqueeze(-1)

            # Also apply gates to edge index effect via node feature masking
            # Weight node features by average incoming edge gate
            node_gate_weight = torch.ones(n_atoms, device=device)
            edge_index = data.edge_index
            counts = torch.zeros(n_atoms, device=device)
            gate_sums = torch.zeros(n_atoms, device=device)
            for i in range(n_edges):
                dst = edge_index[1, i].item()
                gate_sums[dst] += gates[i]
                counts[dst] += 1
            for j in range(n_atoms):
                if counts[j] > 0:
                    node_gate_weight[j] = gate_sums[j] / counts[j]

            masked_data.x = data.x * node_gate_weight.unsqueeze(-1)

            # Forward pass
            out = self.model(masked_data)
            pred = out[self.target_key]
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)

            # Faithfulness loss: prediction should match original
            target = torch.tensor([orig_prob], device=device)
            faith_loss = F.binary_cross_entropy_with_logits(pred, target)

            # Gate penalty: encourage sparse binary gates
            # GraphMask-style: penalize gates being open, with allowance
            gate_mean = gates.mean()
            penalty = lambda_param * F.relu(gate_mean - self.allowance)

            # Entropy regularization: push gates toward 0 or 1
            ent = -gates * torch.log(gates + 1e-8) - \
                  (1 - gates) * torch.log(1 - gates + 1e-8)
            ent_loss = 0.1 * ent.mean()

            loss = faith_loss + penalty + ent_loss
            loss.backward()
            optimizer.step()

        # Final gates
        with torch.no_grad():
            final_gates = torch.sigmoid(gate_logits)

            # Aggregate edge gates to node importance
            node_importance = torch.zeros(n_atoms, device=device)
            edge_index = data.edge_index
            for i in range(n_edges):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                node_importance[src] += final_gates[i]
                node_importance[dst] += final_gates[i]

            # Normalize to [0, 1]
            if node_importance.max() > node_importance.min():
                node_importance = (node_importance - node_importance.min()) / \
                                 (node_importance.max() - node_importance.min())

        return node_importance.detach().cpu()
