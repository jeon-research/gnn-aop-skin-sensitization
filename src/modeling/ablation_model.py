"""
Ablation model for skin sensitization prediction.

Uses a shared GNN encoder (AttentiveFP) with per-endpoint prediction heads
and optional continuous assay feature fusion. The shared embedding is passed
directly to CausalPredictionHead instances for each endpoint and auxiliary
Key Event target.

Imports reusable components from causal_aop_gnn.py.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP
from typing import Dict, Optional, Union

try:
    from src.modeling.causal_aop_gnn import (
        CausalPredictionHead,
        AssayFeatureEncoder,
        MoleculeFeaturizer,
    )
except ImportError:
    from causal_aop_gnn import (
        CausalPredictionHead,
        AssayFeatureEncoder,
        MoleculeFeaturizer,
    )

# Valid ablation conditions
ABLATION_CONDITIONS = ['plain']

# Valid GNN architectures
VALID_ARCHITECTURES = ['attentivefp', 'gcn', 'gin']


class SharedMolecularEncoder(nn.Module):
    """
    Shared GNN encoder for the ablation model.

    Uses AttentiveFP (same backbone as CausalMolecularEncoder) with
    a single linear projection to node_dim.
    """

    # Reuse feature definitions from the original encoder
    ATOM_FEATURES = MoleculeFeaturizer.ATOM_FEATURES
    BOND_FEATURES = MoleculeFeaturizer.BOND_FEATURES

    def __init__(
        self,
        hidden_dim: int = 256,
        node_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim

        atom_dim = sum(len(v) for v in self.ATOM_FEATURES.values())
        bond_dim = sum(len(v) for v in self.BOND_FEATURES.values())

        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        self.bond_encoder = nn.Linear(bond_dim, hidden_dim // 4)

        self.gnn = AttentiveFP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=hidden_dim // 4,
            num_layers=num_layers,
            num_timesteps=2,
            dropout=dropout,
        )

        # Single linear projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Returns:
            embedding: [batch_size, node_dim]
        """
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr) if data.edge_attr.numel() > 0 else None

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )

        mol_embedding = self.gnn(x, data.edge_index, edge_attr, batch)
        return self.projection(mol_embedding)


class AblationGNN(nn.Module):
    """
    Ablation model with shared encoder and per-endpoint prediction heads.

    Architecture:
        SharedMolecularEncoder (AttentiveFP -> Linear projection)
        -> [optional] AssayFeatureEncoder + fusion
        -> CausalPredictionHead per endpoint
        -> CausalPredictionHead per Key Event (auxiliary)
    """

    def __init__(
        self,
        condition: str = 'plain',
        hidden_dim: int = 256,
        node_dim: int = 64,
        num_gnn_layers: int = 3,
        dropout: float = 0.3,
        use_continuous_features: bool = True,
        architecture: str = 'attentivefp',
        **kwargs,
    ):
        super().__init__()

        if condition not in ABLATION_CONDITIONS:
            raise ValueError(f"condition must be one of {ABLATION_CONDITIONS}, got '{condition}'")
        if architecture not in VALID_ARCHITECTURES:
            raise ValueError(f"architecture must be one of {VALID_ARCHITECTURES}, got '{architecture}'")

        self.condition = condition
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.use_continuous_features = use_continuous_features
        self.architecture = architecture

        # === Shared encoder (select by architecture) ===
        if architecture == 'attentivefp':
            self.encoder = SharedMolecularEncoder(
                hidden_dim=hidden_dim,
                node_dim=node_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
            )
        else:
            try:
                from src.modeling.simple_gnn import SimpleGCNEncoder, SimpleGINEncoder
            except ImportError:
                from simple_gnn import SimpleGCNEncoder, SimpleGINEncoder
            encoder_cls = SimpleGCNEncoder if architecture == 'gcn' else SimpleGINEncoder
            self.encoder = encoder_cls(
                hidden_dim=hidden_dim,
                node_dim=node_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
            )

        # === Optional continuous assay feature encoder ===
        if use_continuous_features:
            self.assay_encoder = AssayFeatureEncoder(
                hidden_dim=node_dim,
                dropout=dropout,
            )
            self.assay_fusion = nn.Sequential(
                nn.Linear(node_dim * 2, node_dim),
                nn.LayerNorm(node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # === Prediction heads ===
        self.sensitization_head = CausalPredictionHead(node_dim, hidden_dim // 2, dropout)
        self.irritation_head = CausalPredictionHead(node_dim, hidden_dim // 2, dropout)
        self.corrosion_head = CausalPredictionHead(node_dim, hidden_dim // 2, dropout)

        # === Key Event auxiliary heads ===
        self.mie_head = CausalPredictionHead(node_dim, hidden_dim // 4, dropout)
        self.ke1_head = CausalPredictionHead(node_dim, hidden_dim // 4, dropout)
        self.ke2_head = CausalPredictionHead(node_dim, hidden_dim // 4, dropout)
        self.ke3_head = CausalPredictionHead(node_dim, hidden_dim // 4, dropout)

        # LLNA auxiliary head
        self.sensitization_llna_head = CausalPredictionHead(node_dim, hidden_dim // 2, dropout)

    def forward(
        self,
        data: Union[Data, Batch],
        return_intermediates: bool = False,
        return_uncertainty: bool = False,
        assay_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Returns dict with at minimum:
            sensitization, irritation, corrosion (logits)
        Optionally:
            mie, ke1, ke2, ke3, sensitization_llna (KE intermediates)
            *_uncertainty (uncertainty estimates)
        """
        # Encode molecule
        embedding = self.encoder(data)

        # Fuse continuous assay features if available
        if self.use_continuous_features and assay_features is not None:
            assay_emb = self.assay_encoder(assay_features)
            embedding = self.assay_fusion(
                torch.cat([embedding, assay_emb], dim=-1)
            )

        # Shared embedding for all heads
        sens_emb = irr_emb = corr_emb = embedding
        mie_emb = ke1_emb = ke2_emb = ke3_emb = llna_emb = embedding

        # Predict outcomes
        sens_logit, sens_unc = self.sensitization_head(sens_emb)
        irr_logit, irr_unc = self.irritation_head(irr_emb)
        corr_logit, corr_unc = self.corrosion_head(corr_emb)

        outputs = {
            'sensitization': sens_logit,
            'irritation': irr_logit,
            'corrosion': corr_logit,
        }

        if return_uncertainty:
            outputs['sensitization_uncertainty'] = sens_unc
            outputs['irritation_uncertainty'] = irr_unc
            outputs['corrosion_uncertainty'] = corr_unc

        if return_intermediates:
            mie_logit, _ = self.mie_head(mie_emb)
            ke1_logit, _ = self.ke1_head(ke1_emb)
            ke2_logit, _ = self.ke2_head(ke2_emb)
            ke3_logit, _ = self.ke3_head(ke3_emb)
            llna_logit, _ = self.sensitization_llna_head(llna_emb)

            outputs['mie'] = mie_logit
            outputs['ke1'] = ke1_logit
            outputs['ke2'] = ke2_logit
            outputs['ke3'] = ke3_logit
            outputs['sensitization_llna'] = llna_logit

        return outputs


if __name__ == '__main__':
    # Quick verification: instantiate and check output format
    print("=" * 60)
    print("AblationGNN Verification")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create a dummy batch
    from torch_geometric.data import Batch as GeoBatch
    smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
    graphs = [MoleculeFeaturizer.smiles_to_graph(s) for s in smiles_list]
    graphs = [g for g in graphs if g is not None]
    batch = GeoBatch.from_data_list(graphs).to(device)

    for condition in ABLATION_CONDITIONS:
        model = AblationGNN(
            condition=condition,
            hidden_dim=256,
            node_dim=64,
            use_continuous_features=False,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        with torch.no_grad():
            outputs = model(batch, return_intermediates=True, return_uncertainty=True)

        endpoint_keys = ['sensitization', 'irritation', 'corrosion']
        ke_keys = ['mie', 'ke1', 'ke2', 'ke3', 'sensitization_llna']
        unc_keys = [f'{e}_uncertainty' for e in endpoint_keys]

        # Verify all expected keys present
        for key in endpoint_keys + ke_keys + unc_keys:
            assert key in outputs, f"Missing key '{key}' for condition '{condition}'"

        # Verify output shapes
        batch_size = len(graphs)
        for key in endpoint_keys + ke_keys:
            assert outputs[key].shape == (batch_size,), \
                f"Wrong shape for {key}: {outputs[key].shape} (expected ({batch_size},))"

        print(f"\n  {condition:12s}: {n_params:>8,} params | "
              f"outputs OK ({len(outputs)} keys)")

    print("\n" + "=" * 60)
    print("All conditions verified successfully!")
    print("=" * 60)
