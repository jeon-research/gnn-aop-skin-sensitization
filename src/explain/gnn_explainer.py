"""
GNNExplainer wrapper for molecular GNNs.

Learns soft node and edge masks that preserve the model's prediction
while encouraging sparsity (Ying et al., 2019). Adapted for the
AblationGNN architecture.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


class GNNExplainerWrapper:
    """GNNExplainer for atom-level attribution in molecular GNNs.

    Optimizes a soft node mask to identify important atoms for the
    model's prediction. Uses entropy and sparsity regularization.
    """

    def __init__(
        self,
        model: nn.Module,
        target_key: str = 'sensitization',
        n_steps: int = 200,
        lr: float = 0.01,
        node_mask_coef: float = 0.05,
        entropy_coef: float = 0.1,
    ):
        self.model = model
        self.target_key = target_key
        self.n_steps = n_steps
        self.lr = lr
        self.node_mask_coef = node_mask_coef
        self.entropy_coef = entropy_coef

    def attribute(self, data: Data, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute atom-level GNNExplainer attributions for a single molecule.

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

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

        # Get original prediction (target to preserve)
        self.model.eval()
        with torch.no_grad():
            orig_outputs = self.model(data)
            orig_pred = orig_outputs[self.target_key]
            if orig_pred.dim() == 0:
                orig_pred = orig_pred.unsqueeze(0)
            orig_prob = torch.sigmoid(orig_pred).item()

        # Initialize learnable node mask (logits) with small random noise
        # to break symmetry (all-zeros leads to uniform gradients)
        node_mask_logits = (torch.randn(n_atoms, device=device) * 0.1).requires_grad_(True)
        optimizer = torch.optim.Adam([node_mask_logits], lr=self.lr)

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # Sigmoid to get mask values in (0, 1)
            node_mask = torch.sigmoid(node_mask_logits)

            # Apply mask to node features
            masked_data = data.clone()
            masked_data.x = data.x * node_mask.unsqueeze(-1)

            # Forward pass with masked features
            outputs = self.model(masked_data)
            pred = outputs[self.target_key]
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)

            # Prediction loss: preserve original prediction
            pred_loss = F.binary_cross_entropy_with_logits(
                pred, torch.tensor([orig_prob], device=device)
            )

            # Sparsity loss: encourage sparse masks (L1)
            sparsity_loss = self.node_mask_coef * node_mask.mean()

            # Entropy loss: encourage binary masks (0 or 1)
            entropy_loss = self.entropy_coef * (
                -node_mask * torch.log(node_mask + 1e-8)
                - (1 - node_mask) * torch.log(1 - node_mask + 1e-8)
            ).mean()

            loss = pred_loss + sparsity_loss + entropy_loss
            loss.backward()
            optimizer.step()

        # Final mask
        with torch.no_grad():
            importance = torch.sigmoid(node_mask_logits)

        return importance.detach().cpu()

    def attribute_batch(self, batch: Batch, device: Optional[torch.device] = None) -> list:
        """Compute GNNExplainer attributions for a batch of molecules."""
        data_list = batch.to_data_list()
        results = []
        for data in data_list:
            importance = self.attribute(data, device=device)
            results.append(importance)
        return results
