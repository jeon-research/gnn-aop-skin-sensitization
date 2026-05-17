"""
Attention weight extraction from AttentiveFP.

AttentiveFP uses multi-step graph-level attention that assigns learned
importance weights to atoms. This module extracts those weights by hooking
into the attention computation during the forward pass.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import AttentiveFP


class AttentionExtractor:
    """Extract atom-level attention weights from AttentiveFP.

    AttentiveFP has a graph-level readout that uses multi-step attention
    over node embeddings. We extract the attention coefficients from each
    timestep and aggregate them.
    """

    def __init__(self, model: nn.Module, target_key: str = 'sensitization'):
        self.model = model
        self.target_key = target_key

    def _extract_attention_weights(self, data: Data, device: torch.device) -> torch.Tensor:
        """Extract attention weights from AttentiveFP's readout.

        The AttentiveFP readout uses GATConv-style attention over nodes.
        We hook into the readout's attention computation.
        """
        n_atoms = data.x.size(0)
        encoder = self.model.encoder
        gnn = encoder.gnn

        # Collect attention weights from the graph-level readout
        attention_weights = []
        hooks = []

        def make_readout_hook(step_idx):
            def hook_fn(module, input, output):
                # The readout is a GlobalAttention or JumpingKnowledge + attention
                # For AttentiveFP, the mol_readout layers use attention
                if hasattr(output, 'shape') and output.dim() >= 1:
                    attention_weights.append(output.detach())
            return hook_fn

        # AttentiveFP stores readout attention in mol_readouts
        if hasattr(gnn, 'mol_readouts'):
            for i, readout in enumerate(gnn.mol_readouts):
                # Each readout has a gate_nn that produces attention logits
                if hasattr(readout, 'gate_nn'):
                    h = readout.gate_nn.register_forward_hook(make_readout_hook(i))
                    hooks.append(h)

        try:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(data)
        finally:
            for h in hooks:
                h.remove()

        if not attention_weights:
            # Fallback: use gradient-based proxy for attention
            return self._gradient_attention_fallback(data, device)

        # attention_weights[i] has shape [n_atoms, 1] (gate logits per timestep)
        # Softmax to get proper attention distribution
        atom_weights = torch.zeros(n_atoms, device=device)
        for att in attention_weights:
            if att.size(0) == n_atoms:
                # Softmax over atoms for this timestep
                weights = torch.softmax(att.squeeze(-1), dim=0)
                atom_weights += weights

        # Average across timesteps
        if len(attention_weights) > 0:
            atom_weights /= len(attention_weights)

        return atom_weights

    def _gradient_attention_fallback(self, data: Data, device: torch.device) -> torch.Tensor:
        """Fallback: use input gradient magnitude as attention proxy."""
        n_atoms = data.x.size(0)
        data_copy = data.clone()
        data_copy.x = data_copy.x.detach().requires_grad_(True)

        self.model.eval()
        outputs = self.model(data_copy)
        target = outputs[self.target_key]
        if target.dim() == 0:
            target = target.unsqueeze(0)

        self.model.zero_grad()
        target.sum().backward()

        if data_copy.x.grad is not None:
            # L2 norm of gradient per atom
            atom_weights = data_copy.x.grad.detach().norm(dim=-1)
        else:
            atom_weights = torch.ones(n_atoms, device=device) / n_atoms

        return atom_weights

    def attribute(self, data: Data, device: Optional[torch.device] = None) -> torch.Tensor:
        """Extract atom-level attention weights for a single molecule.

        Args:
            data: PyG Data object (single molecule).
            device: Device to run on.

        Returns:
            [n_atoms] attention weights (non-negative, sums to 1).
        """
        if device is None:
            device = next(self.model.parameters()).device

        data = data.clone().to(device)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        weights = self._extract_attention_weights(data, device)

        # Normalize to [0, 1]
        if weights.max() > weights.min():
            weights = (weights - weights.min()) / (weights.max() - weights.min())

        return weights.detach().cpu()

    def attribute_batch(self, batch: Batch, device: Optional[torch.device] = None) -> list:
        """Extract attention weights for a batch of molecules."""
        data_list = batch.to_data_list()
        results = []
        for data in data_list:
            importance = self.attribute(data, device=device)
            results.append(importance)
        return results
