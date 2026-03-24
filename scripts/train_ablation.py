"""
Training script for GNN skin sensitization models.

Shared across all conditions:
  - Data augmentation (SMILES enumeration + graph dropout)
  - Continuous assay features
  - Asymmetric LLNA auxiliary loss
  - Scaffold split with seed control
  - Mixed precision + TF32
  - ReduceLROnPlateau + 5-epoch warmup
  - Platt scaling calibration

Usage:
    python scripts/train_ablation.py \
        --condition plain \
        --data_path data/processed/causal_aop_comprehensive_v6.csv \
        --seed 42 --epochs 100 --batch_size 32
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

import sys
# Ensure imports work from scripts/ directory
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root / 'src' / 'modeling'))

from ablation_model import AblationGNN, ABLATION_CONDITIONS, VALID_ARCHITECTURES
from causal_aop_gnn import (
    CausalLoss,
    MoleculeFeaturizer,
)

# Continuous feature columns
CONTINUOUS_FEATURE_COLS = [
    'dpra_mean', 'dpra_cys', 'dpra_lys', 'kdpra_log',
    'keratinosens_ec15',
    'hclat_mean', 'hclat_ec200', 'usens_cv70',
    'irritation_viability_mean',
    'corrosion_ghs_score', 'corrosion_ter_mean',
    'mol_weight', 'logp', 'tpsa', 'hbd', 'hba',
    'num_rotatable_bonds', 'fraction_csp3',
]


# =============================================================================
# Data augmentation
# =============================================================================

from rdkit import Chem
from typing import List


def enumerate_smiles(smiles: str, num_variants: int = 5, seed: int = None) -> List[str]:
    """Generate alternative SMILES via deterministic atom reordering."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    variants = set()
    canonical = Chem.MolToSmiles(mol, canonical=True)
    variants.add(canonical)

    if seed is None:
        seed = hash(smiles) % (2**31)
    rng = np.random.RandomState(seed)

    num_atoms = mol.GetNumAtoms()
    if num_atoms <= 1:
        return [canonical]

    for i in range(num_variants * 3):
        try:
            perm = rng.permutation(num_atoms).tolist()
            renumbered = Chem.RenumberAtoms(mol, perm)
            variant = Chem.MolToSmiles(renumbered, canonical=False)
            variants.add(variant)
            if len(variants) >= num_variants:
                break
        except Exception:
            continue

    return list(variants)[:num_variants]


def scaffold_split(df, smiles_col='smiles', seed=42, frac_train=0.7, frac_val=0.15, frac_test=0.15):
    """Bemis-Murcko scaffold-based split."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    scaffold_to_indices = defaultdict(list)
    no_scaffold_indices = []

    for idx, row in df.iterrows():
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(smi) if pd.notna(smi) else None
        if mol is not None:
            try:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                generic = MurckoScaffold.MakeScaffoldGeneric(core)
                scaffold_smi = Chem.MolToSmiles(generic)
                scaffold_to_indices[scaffold_smi].append(idx)
            except Exception:
                no_scaffold_indices.append(idx)
        else:
            no_scaffold_indices.append(idx)

    rng = np.random.RandomState(seed)
    scaffold_items = list(scaffold_to_indices.items())
    rng.shuffle(scaffold_items)
    scaffold_items.sort(key=lambda x: len(x[1]), reverse=True)
    scaffold_groups = [indices for _, indices in scaffold_items]

    n_total = len(df)
    target_train = int(n_total * frac_train)
    target_val = int(n_total * frac_val)

    train_indices, val_indices, test_indices = [], [], []

    for group in scaffold_groups:
        if len(train_indices) < target_train:
            train_indices.extend(group)
        elif len(val_indices) < target_val:
            val_indices.extend(group)
        else:
            test_indices.extend(group)

    if no_scaffold_indices:
        rng.shuffle(no_scaffold_indices)
        n_ns = len(no_scaffold_indices)
        n_ns_train = int(n_ns * frac_train)
        n_ns_val = int(n_ns * frac_val)
        train_indices.extend(no_scaffold_indices[:n_ns_train])
        val_indices.extend(no_scaffold_indices[n_ns_train:n_ns_train + n_ns_val])
        test_indices.extend(no_scaffold_indices[n_ns_train + n_ns_val:])

    n_scaffolds = len(scaffold_to_indices)
    print(f"  Scaffold split: {n_scaffolds} unique scaffolds")
    print(f"  Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    return train_indices, val_indices, test_indices


# =============================================================================
# Dataset
# =============================================================================

class AblationDataset(Dataset):
    """Dataset for ablation study with augmentation support."""

    def __init__(
        self,
        df: pd.DataFrame,
        augment: bool = False,
        augment_positives_factor: int = 3,
        graph_dropout_rate: float = 0.1,
        training: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.augment_positives_factor = augment_positives_factor
        self.graph_dropout_rate = graph_dropout_rate
        self.training = training

        self.valid_indices = []
        self.graphs = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            smiles = row.get('smiles')
            if pd.notna(smiles) and smiles:
                is_positive = row.get('sensitization_human', 0) == 1

                if augment and is_positive and training:
                    smiles_variants = enumerate_smiles(smiles, num_variants=augment_positives_factor)
                else:
                    smiles_variants = [smiles]

                for smi in smiles_variants:
                    graph = MoleculeFeaturizer.smiles_to_graph(smi)
                    if graph is not None:
                        self.valid_indices.append(idx)
                        self.graphs.append(graph)

        print(f"Dataset: {len(self.valid_indices)}/{len(self.df)} valid molecules")
        if augment and training:
            print(f"  (Augmentation: {augment_positives_factor}x SMILES, "
                  f"{graph_dropout_rate:.0%} edge dropout)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        orig_idx = self.valid_indices[idx]
        row = self.df.iloc[orig_idx]
        graph = self.graphs[idx].clone()

        # Graph augmentation during training
        if self.augment and self.training and self.graph_dropout_rate > 0:
            aug_gen = torch.Generator().manual_seed(idx)
            if graph.edge_index.size(1) > 0:
                num_edges = graph.edge_index.size(1)
                keep_mask = torch.rand(num_edges, generator=aug_gen) > self.graph_dropout_rate
                if keep_mask.sum() < 2:
                    keep_mask[:2] = True
                graph.edge_index = graph.edge_index[:, keep_mask]
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    graph.edge_attr = graph.edge_attr[keep_mask]

            if torch.rand(1, generator=aug_gen).item() < 0.5:
                noise = torch.randn(graph.x.shape, generator=aug_gen) * 0.05
                graph.x = graph.x + noise

        # Labels
        def get_label(col):
            val = row.get(col) if col in row else None
            return float(val) if pd.notna(val) else float('nan')

        graph.sensitization = torch.tensor([get_label('sensitization_human')], dtype=torch.float)
        graph.irritation = torch.tensor([get_label('irritation')], dtype=torch.float)
        graph.corrosion = torch.tensor([get_label('corrosion')], dtype=torch.float)
        graph.sensitization_llna = torch.tensor([get_label('llna_result')], dtype=torch.float)

        # Key Event labels
        graph.mie = torch.tensor([get_label('dpra_result')], dtype=torch.float)
        graph.ke1 = torch.tensor([get_label('keratinosens_result')], dtype=torch.float)
        graph.ke2 = torch.tensor([get_label('hclat_result')], dtype=torch.float)
        graph.ke3 = torch.tensor([float('nan')], dtype=torch.float)  # sparse, usually NaN

        # Continuous assay features
        for col in CONTINUOUS_FEATURE_COLS:
            val = row.get(col) if col in row.index else None
            if pd.notna(val):
                setattr(graph, col, torch.tensor([float(val)], dtype=torch.float))
            else:
                setattr(graph, col, torch.tensor([float('nan')], dtype=torch.float))

        return graph


def collate_fn(batch):
    return Batch.from_data_list(batch)


# =============================================================================
# Training and evaluation
# =============================================================================

def train_epoch(
    model: AblationGNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CausalLoss,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_continuous_features: bool = True,
    use_learnable_graph: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    component_losses = {}
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        targets = {
            'sensitization': batch.sensitization.squeeze(),
            'irritation': batch.irritation.squeeze(),
            'corrosion': batch.corrosion.squeeze(),
        }

        if hasattr(batch, 'sensitization_llna'):
            targets['sensitization_llna'] = batch.sensitization_llna.squeeze()

        ke_targets = {
            'mie': batch.mie.squeeze(),
            'ke1': batch.ke1.squeeze(),
            'ke2': batch.ke2.squeeze(),
            'ke3': batch.ke3.squeeze() if hasattr(batch, 'ke3') else None,
        }
        ke_targets = {k: v for k, v in ke_targets.items() if v is not None}

        assay_features = None
        if use_continuous_features:
            assay_features = {}
            for col in CONTINUOUS_FEATURE_COLS:
                if hasattr(batch, col):
                    assay_features[col] = getattr(batch, col).squeeze()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    batch,
                    return_intermediates=True,
                    return_uncertainty=True,
                    assay_features=assay_features,
                )
                loss, losses = loss_fn(outputs, targets, ke_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                batch,
                return_intermediates=True,
                return_uncertainty=True,
                assay_features=assay_features,
            )
            loss, losses = loss_fn(outputs, targets, ke_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        for k, v in losses.items():
            val = v.item() if hasattr(v, 'item') else float(v)
            component_losses[k] = component_losses.get(k, 0) + val
        n_batches += 1

    return {
        'total': total_loss / max(n_batches, 1),
        **{k: v / max(n_batches, 1) for k, v in component_losses.items()}
    }


@torch.no_grad()
def evaluate(
    model: AblationGNN,
    dataloader: DataLoader,
    device: torch.device,
    use_continuous_features: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on all endpoints."""
    model.eval()

    all_preds = {k: [] for k in ['sensitization', 'irritation', 'corrosion',
                                   'mie', 'ke1', 'ke2', 'ke3', 'sensitization_llna']}
    all_targets = {k: [] for k in all_preds.keys()}

    for batch in dataloader:
        batch = batch.to(device)

        assay_features = None
        if use_continuous_features:
            assay_features = {}
            for col in CONTINUOUS_FEATURE_COLS:
                if hasattr(batch, col):
                    assay_features[col] = getattr(batch, col).squeeze()

        outputs = model(batch, return_intermediates=True, assay_features=assay_features)

        for endpoint in ['sensitization', 'irritation', 'corrosion']:
            if endpoint not in outputs:
                continue
            preds = torch.sigmoid(outputs[endpoint]).cpu().numpy()
            if hasattr(batch, endpoint):
                targets = getattr(batch, endpoint).squeeze().cpu().numpy()
            else:
                continue
            valid_mask = ~np.isnan(targets)
            all_preds[endpoint].extend(preds[valid_mask])
            all_targets[endpoint].extend(targets[valid_mask])

        for ke, attr in [('mie', 'mie'), ('ke1', 'ke1'), ('ke2', 'ke2'), ('ke3', 'ke3')]:
            if ke in outputs and hasattr(batch, attr):
                preds = torch.sigmoid(outputs[ke]).cpu().numpy()
                targets = getattr(batch, attr).squeeze().cpu().numpy()
                valid_mask = ~np.isnan(targets)
                all_preds[ke].extend(preds[valid_mask])
                all_targets[ke].extend(targets[valid_mask])

        if 'sensitization_llna' in outputs and hasattr(batch, 'sensitization_llna'):
            preds = torch.sigmoid(outputs['sensitization_llna']).cpu().numpy()
            targets = batch.sensitization_llna.squeeze().cpu().numpy()
            valid_mask = ~np.isnan(targets)
            all_preds['sensitization_llna'].extend(preds[valid_mask])
            all_targets['sensitization_llna'].extend(targets[valid_mask])

    metrics = {}
    for endpoint in all_preds.keys():
        if len(all_targets[endpoint]) > 0:
            y_true = np.array(all_targets[endpoint])
            y_prob = np.array(all_preds[endpoint])

            best_thresh = 0.5
            best_bacc = 0.0
            for thresh in np.arange(0.1, 0.9, 0.05):
                y_pred_temp = (y_prob >= thresh).astype(int)
                bacc_temp = balanced_accuracy_score(y_true, y_pred_temp)
                if bacc_temp > best_bacc:
                    best_bacc = bacc_temp
                    best_thresh = thresh

            y_pred = (y_prob >= best_thresh).astype(int)

            metrics[endpoint] = {
                'n_samples': len(y_true),
                'n_positive': int(y_true.sum()),
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'optimal_threshold': best_thresh,
            }

            if len(np.unique(y_true)) > 1:
                metrics[endpoint]['auc'] = roc_auc_score(y_true, y_prob)
            else:
                metrics[endpoint]['auc'] = 0.0

    return metrics


def fit_calibrators(
    model: AblationGNN,
    dataloader: DataLoader,
    device: torch.device,
    use_continuous_features: bool = True,
) -> Dict[str, LogisticRegression]:
    """Fit Platt scaling calibrators on validation data."""
    model.eval()
    endpoints = ['sensitization', 'irritation', 'corrosion']

    logits = {ep: [] for ep in endpoints}
    labels = {ep: [] for ep in endpoints}

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            assay_features = None
            if use_continuous_features:
                feature_dict = {}
                for col in CONTINUOUS_FEATURE_COLS:
                    if hasattr(batch, col):
                        feature_dict[col] = getattr(batch, col)
                if feature_dict:
                    assay_features = feature_dict

            outputs = model(batch, assay_features=assay_features)

            for ep in endpoints:
                if ep in outputs and hasattr(batch, ep):
                    pred = outputs[ep].cpu().numpy().flatten()
                    target = getattr(batch, ep).squeeze().cpu().numpy().flatten()
                    valid_mask = ~np.isnan(target)
                    if valid_mask.any():
                        logits[ep].extend(pred[valid_mask])
                        labels[ep].extend(target[valid_mask])

    calibrators = {}
    for ep in endpoints:
        if len(logits[ep]) > 10:
            X = np.array(logits[ep]).reshape(-1, 1)
            y = np.array(labels[ep])
            calibrator = LogisticRegression(
                C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced'
            )
            try:
                calibrator.fit(X, y)
                calibrators[ep] = calibrator
            except Exception as e:
                print(f"  Warning: Could not fit calibrator for {ep}: {e}")

    return calibrators


@torch.no_grad()
def evaluate_with_calibration(
    model: AblationGNN,
    dataloader: DataLoader,
    device: torch.device,
    calibrators: Dict[str, LogisticRegression],
    use_continuous_features: bool = True,
) -> Dict[str, Dict]:
    """Evaluate with calibrated probabilities."""
    model.eval()

    predictions = {ep: {'y_true': [], 'y_logit': [], 'y_prob_raw': [], 'y_prob_cal': []}
                   for ep in ['sensitization', 'irritation', 'corrosion']}

    for batch in dataloader:
        batch = batch.to(device)

        assay_features = None
        if use_continuous_features:
            feature_dict = {}
            for col in CONTINUOUS_FEATURE_COLS:
                if hasattr(batch, col):
                    feature_dict[col] = getattr(batch, col)
            if feature_dict:
                assay_features = feature_dict

        outputs = model(batch, assay_features=assay_features)

        for endpoint in ['sensitization', 'irritation', 'corrosion']:
            if endpoint in outputs and hasattr(batch, endpoint):
                pred = outputs[endpoint].cpu().numpy().flatten()
                target = getattr(batch, endpoint).squeeze().cpu().numpy().flatten()
                valid_mask = ~np.isnan(target)
                if valid_mask.any():
                    logits_valid = pred[valid_mask]
                    labels_valid = target[valid_mask]
                    probs_raw = 1 / (1 + np.exp(-logits_valid))
                    if endpoint in calibrators:
                        probs_cal = calibrators[endpoint].predict_proba(
                            logits_valid.reshape(-1, 1)
                        )[:, 1]
                    else:
                        probs_cal = probs_raw

                    predictions[endpoint]['y_true'].extend(labels_valid)
                    predictions[endpoint]['y_logit'].extend(logits_valid)
                    predictions[endpoint]['y_prob_raw'].extend(probs_raw)
                    predictions[endpoint]['y_prob_cal'].extend(probs_cal)

    metrics = {}
    for endpoint in ['sensitization', 'irritation', 'corrosion']:
        preds = predictions[endpoint]
        y_true = np.array(preds['y_true'])
        y_prob_raw = np.array(preds['y_prob_raw'])
        y_prob_cal = np.array(preds['y_prob_cal'])

        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            best_thresh = 0.5
            best_bacc = 0.0
            for thresh in np.arange(0.1, 0.9, 0.05):
                y_pred_temp = (y_prob_cal >= thresh).astype(int)
                bacc_temp = balanced_accuracy_score(y_true, y_pred_temp)
                if bacc_temp > best_bacc:
                    best_bacc = bacc_temp
                    best_thresh = thresh

            y_pred = (y_prob_cal >= best_thresh).astype(int)

            metrics[endpoint] = {
                'n_samples': len(y_true),
                'n_positive': int(y_true.sum()),
                'auc': roc_auc_score(y_true, y_prob_cal),
                'auc_raw': roc_auc_score(y_true, y_prob_raw),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'optimal_threshold': best_thresh,
                'calibrated': endpoint in calibrators,
            }

            try:
                prob_true, prob_pred = calibration_curve(y_true, y_prob_cal, n_bins=10, strategy='uniform')
                ece = np.mean(np.abs(prob_true - prob_pred))
                metrics[endpoint]['ece'] = ece
            except Exception:
                metrics[endpoint]['ece'] = None

    return metrics


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Ablation GNN')

    # Ablation condition
    parser.add_argument('--condition', type=str, default='plain',
                        choices=ABLATION_CONDITIONS,
                        help='Ablation condition')

    # GNN architecture
    parser.add_argument('--architecture', type=str, default='attentivefp',
                        choices=VALID_ARCHITECTURES,
                        help='GNN encoder architecture')

    # Data
    parser.add_argument('--data_path', type=str,
                        default='data/processed/causal_aop_comprehensive_v6.csv')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=15)

    # Augmentation
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--no_augment', dest='augment', action='store_false')
    parser.add_argument('--augment_positives_factor', type=int, default=5)
    parser.add_argument('--graph_dropout_rate', type=float, default=0.1)

    # Features
    parser.add_argument('--use_continuous_features', action='store_true', default=True)
    parser.add_argument('--no_continuous_features', dest='use_continuous_features', action='store_false')

    # LLNA auxiliary
    parser.add_argument('--use_llna_auxiliary', action='store_true', default=True)
    parser.add_argument('--no_llna_auxiliary', dest='use_llna_auxiliary', action='store_false')
    parser.add_argument('--use_asymmetric_llna', action='store_true', default=True)
    parser.add_argument('--no_asymmetric_llna', dest='use_asymmetric_llna', action='store_false')
    parser.add_argument('--llna_pos_weight', type=float, default=0.6)
    parser.add_argument('--llna_neg_weight', type=float, default=0.0)
    # Output
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Output directory — include architecture if not default
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.architecture == 'attentivefp':
        run_dir = Path(args.output_dir) / args.condition / f'seed_{args.seed}' / f'run_{timestamp}'
    else:
        run_dir = Path(args.output_dir) / f'{args.condition}_{args.architecture}' / f'seed_{args.seed}' / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    # Scaffold split
    print(f"\nScaffold split (seed={args.seed})...")
    train_idx, val_idx, test_idx = scaffold_split(df, smiles_col='smiles', seed=args.seed)
    df['split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    # Report endpoint distribution
    for endpoint in ['sensitization_human', 'irritation', 'corrosion']:
        if endpoint in df.columns:
            for split in ['train', 'val', 'test']:
                split_df = df[df['split'] == split]
                n_pos = (split_df[endpoint] == 1).sum()
                n_total = split_df[endpoint].notna().sum()
                print(f"  {split} {endpoint}: {n_pos}/{n_total} positive")

    # Merge NAM data (deduplicate first to avoid row explosion)
    try:
        nam_path = Path(args.data_path).parent / 'aat_skin_multitask.csv'
        nam_df = pd.read_csv(nam_path)
        ke_cols = ['cas_number', 'dpra_result', 'keratinosens_result', 'hclat_result']
        ke_cols = [c for c in ke_cols if c in nam_df.columns]
        if len(ke_cols) > 1 and 'cas_number' in df.columns:
            # Deduplicate NAM by cas_number (keep first non-null)
            nam_dedup = nam_df[ke_cols].drop_duplicates(subset='cas_number', keep='first')
            n_before = len(df)
            df = df.merge(nam_dedup, on='cas_number', how='left',
                         suffixes=('', '_nam'))
            # Use NAM data where original is missing
            for col in ['dpra_result', 'keratinosens_result', 'hclat_result']:
                nam_col = f'{col}_nam'
                if nam_col in df.columns:
                    df[col] = df[col].fillna(df[nam_col])
                    df = df.drop(columns=[nam_col])
            assert len(df) == n_before, f"Merge changed row count: {n_before} -> {len(df)}"
            print("Merged NAM data for Key Event supervision")
    except Exception as e:
        print(f"Could not merge NAM data: {e}")

    # Split data
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create datasets
    train_dataset = AblationDataset(
        train_df,
        augment=args.augment,
        augment_positives_factor=args.augment_positives_factor,
        graph_dropout_rate=args.graph_dropout_rate,
        training=True,
    )
    val_dataset = AblationDataset(val_df, augment=False, training=False)
    test_dataset = AblationDataset(test_df, augment=False, training=False)

    # DataLoaders
    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
        torch.manual_seed(args.seed + worker_id)

    g = torch.Generator()
    g.manual_seed(args.seed)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': collate_fn,
        'num_workers': 0,
        'pin_memory': device.type == 'cuda',
        'worker_init_fn': worker_init_fn,
        'generator': g,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Create model
    print(f"\nCreating AblationGNN (condition={args.condition}, architecture={args.architecture})...")
    model = AblationGNN(
        condition=args.condition,
        hidden_dim=args.hidden_dim,
        node_dim=args.node_dim,
        num_gnn_layers=args.num_layers,
        dropout=args.dropout,
        use_continuous_features=args.use_continuous_features,
        architecture=args.architecture,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Condition: {args.condition}")

    # Loss function
    loss_fn = CausalLoss(
        outcome_weight=1.0,
        ke_weight=1.0,
        use_llna_auxiliary=args.use_llna_auxiliary,
        use_asymmetric_llna=args.use_asymmetric_llna,
        llna_pos_weight=args.llna_pos_weight,
        llna_neg_weight=args.llna_neg_weight,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Schedulers
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    print("\n" + "=" * 70)
    print(f"TRAINING ABLATION: {args.condition} (seed={args.seed})")
    print("=" * 70)

    best_val_score = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch <= 5:
            warmup_scheduler.step()

        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            use_continuous_features=args.use_continuous_features,
            use_learnable_graph=False,
        )

        val_metrics = evaluate(model, val_loader, device,
                               use_continuous_features=args.use_continuous_features)

        val_aucs = [val_metrics[ep]['auc'] for ep in ['sensitization', 'irritation', 'corrosion']
                    if ep in val_metrics and val_metrics[ep]['n_samples'] > 0]
        avg_val_auc = np.mean(val_aucs) if val_aucs else 0.0

        val_baccs = [val_metrics[ep]['balanced_accuracy'] for ep in ['sensitization', 'irritation', 'corrosion']
                     if ep in val_metrics and val_metrics[ep]['n_samples'] > 0]
        avg_val_bacc = np.mean(val_baccs) if val_baccs else 0.0

        combined_score = 0.5 * avg_val_auc + 0.5 * avg_val_bacc
        scheduler.step(combined_score)

        history.append({
            'epoch': epoch,
            'train_loss': train_losses['total'],
            'avg_val_auc': avg_val_auc,
            'avg_val_bacc': avg_val_bacc,
            'combined_score': combined_score,
            **{f'val_{ep}_{m}': val_metrics.get(ep, {}).get(m, 0)
               for ep in ['sensitization', 'irritation', 'corrosion']
               for m in ['auc', 'balanced_accuracy', 'f1']}
        })

        # Print progress every 10 epochs or at key events
        if epoch % 10 == 0 or epoch == 1 or combined_score > best_val_score:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            for ep in ['sensitization', 'irritation', 'corrosion']:
                if ep in val_metrics:
                    m = val_metrics[ep]
                    print(f"  Val {ep[:4].upper()}: AUC={m['auc']:.3f}, BAcc={m['balanced_accuracy']:.3f}")
            print(f"  Combined: {combined_score:.4f}")

        if combined_score > best_val_score:
            best_val_score = combined_score
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'combined_score': combined_score,
                'condition': args.condition,
                'architecture': args.architecture,
            }, run_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(run_dir / 'training_history.csv', index=False)

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    checkpoint = torch.load(run_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Fitting calibrators...")
    calibrators = fit_calibrators(model, val_loader, device,
                                  use_continuous_features=args.use_continuous_features)

    test_metrics = evaluate_with_calibration(
        model, test_loader, device, calibrators,
        use_continuous_features=args.use_continuous_features
    )

    print(f"\nBest model from epoch {best_epoch}")
    print("Test Set Results (calibrated):")

    for ep in ['sensitization', 'irritation', 'corrosion']:
        if ep in test_metrics:
            m = test_metrics[ep]
            print(f"\n  {ep.upper()}:")
            print(f"    N: {m['n_samples']} ({m['n_positive']} positive)")
            print(f"    AUC: {m['auc']:.4f} (raw: {m.get('auc_raw', 0):.4f})")
            print(f"    BAcc: {m['balanced_accuracy']:.4f}")
            print(f"    F1: {m['f1']:.4f}")

    # Save results
    results = {
        'condition': args.condition,
        'seed': args.seed,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_score': best_val_score,
        'test_metrics': test_metrics,
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to: {run_dir}")
    return results


if __name__ == '__main__':
    main()
