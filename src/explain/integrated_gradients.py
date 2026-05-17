"""
Integrated Gradients attribution for molecular GNNs.

Axiomatic attribution method (Sundararajan et al., 2017) adapted for
graph-structured molecular data. Computes path-integrated gradients
from a zero baseline to the actual node features.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


class IntegratedGradients:
    """Integrated Gradients for atom-level attribution in molecular GNNs."""

    def __init__(self, model: nn.Module, target_key: str = 'sensitization', n_steps: int = 50):
        self.model = model
        self.target_key = target_key
        self.n_steps = n_steps

    @torch.no_grad()
    def _get_baseline(self, data: Data) -> torch.Tensor:
        """Zero baseline (no atom information)."""
        return torch.zeros_like(data.x)

    def attribute(self, data: Data, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute atom-level IG attributions for a single molecule.

        Args:
            data: PyG Data object (single molecule, no batch dim).
            device: Device to run on.

        Returns:
            [n_atoms] importance scores (non-negative).
        """
        if device is None:
            device = next(self.model.parameters()).device

        data = data.clone().to(device)
        baseline = self._get_baseline(data)
        input_features = data.x.clone()

        # Linear interpolation from baseline to input
        scaled_inputs = []
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            scaled = baseline + alpha * (input_features - baseline)
            scaled_inputs.append(scaled)

        # Accumulate gradients along the path
        total_gradients = torch.zeros_like(input_features)

        self.model.eval()
        for scaled_x in scaled_inputs:
            scaled_x = scaled_x.detach().requires_grad_(True)
            data_copy = data.clone()
            data_copy.x = scaled_x

            # Ensure batch attribute exists for single molecule
            if not hasattr(data_copy, 'batch') or data_copy.batch is None:
                data_copy.batch = torch.zeros(data_copy.x.size(0), dtype=torch.long, device=device)

            outputs = self.model(data_copy)
            target = outputs[self.target_key]

            # For binary classification, use the logit directly
            if target.dim() == 0:
                target = target.unsqueeze(0)
            score = target.sum()

            score.backward(retain_graph=False)
            if scaled_x.grad is not None:
                total_gradients += scaled_x.grad.detach()

        # Trapezoidal rule approximation
        avg_gradients = total_gradients / (self.n_steps + 1)

        # Attribution = (input - baseline) * avg_gradients
        attributions = (input_features - baseline) * avg_gradients

        # Aggregate across feature dimensions: sum of absolute values per atom
        atom_importance = attributions.abs().sum(dim=-1)

        return atom_importance.detach().cpu()

    def attribute_batch(self, batch: Batch, device: Optional[torch.device] = None) -> list:
        """Compute IG attributions for a batch of molecules.

        Args:
            batch: PyG Batch object.

        Returns:
            List of [n_atoms_i] tensors, one per molecule in the batch.
        """
        if device is None:
            device = next(self.model.parameters()).device

        # Unbatch and process individually (IG requires per-molecule gradients)
        data_list = batch.to_data_list()
        results = []
        for data in data_list:
            importance = self.attribute(data, device=device)
            results.append(importance)
        return results
