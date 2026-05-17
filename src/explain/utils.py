"""
Shared utilities for the explanation pipeline.

Contains model loading, dataset loading, and scaffold splitting that
don't depend on the training script's relative imports.
"""

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from config import PROJECT_ROOT, RESULTS_DIR
from src.modeling.ablation_model import AblationGNN
from src.modeling.causal_aop_gnn import MoleculeFeaturizer


ALL_SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777,
             1111, 2222, 3333, 4444, 6666, 8888, 9999, 1234, 5678, 9876]


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'causal_aop_comprehensive_v6.csv'


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = 'smiles',
    seed: int = 42,
    frac_train: float = 0.7,
    frac_val: float = 0.15,
    frac_test: float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """Bemis-Murcko scaffold-based split.

    Reproduces the exact same split as in the training script.
    """
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

    print(f"  Scaffold split: {len(scaffold_to_indices)} scaffolds, "
          f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    return train_indices, val_indices, test_indices


def find_checkpoint(seed: int, architecture: str = 'attentivefp') -> Path:
    """Find the best_model.pt checkpoint for a given seed and architecture."""
    if architecture == 'attentivefp':
        condition_dir = 'plain'
    else:
        condition_dir = f'plain_{architecture}'
    seed_dir = RESULTS_DIR / 'ablation' / condition_dir / f'seed_{seed}'
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")
    # Find the most recent run directory that has a best_model.pt
    run_dirs = sorted(
        [d for d in seed_dir.iterdir() if d.is_dir() and (d / 'best_model.pt').exists()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run with best_model.pt in {seed_dir}")
    return run_dirs[0] / 'best_model.pt'


def load_model(seed: int, device: torch.device, architecture: str = 'attentivefp') -> AblationGNN:
    """Load a trained AblationGNN from checkpoint."""
    checkpoint_path = find_checkpoint(seed, architecture=architecture)
    model = AblationGNN(condition='plain', architecture=architecture)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_dataset(seed: int = 42) -> Tuple[pd.DataFrame, list, list, list]:
    """Load dataset and create scaffold split."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['smiles', 'sensitization_human']).reset_index(drop=True)
    train_idx, val_idx, test_idx = scaffold_split(df, smiles_col='smiles', seed=seed)
    return df, train_idx, val_idx, test_idx


def featurize_molecules(smiles_list: list):
    """Convert SMILES to PyG Data objects."""
    graphs = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        graph = MoleculeFeaturizer.smiles_to_graph(smi)
        if graph is not None:
            graphs.append(graph)
            valid_indices.append(i)
    return graphs, valid_indices


def setup_device() -> torch.device:
    """Set up compute device with TF32 optimizations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def find_mechgnn_checkpoint(seed: int, lambda_mie: float) -> Path:
    """Find MechGNN best_model.pt for a given seed and lambda."""
    checkpoint = RESULTS_DIR / 'mechgnn' / f'lambda_{lambda_mie}' / f'seed_{seed}' / 'best_model.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f"MechGNN checkpoint not found: {checkpoint}")
    return checkpoint


def load_mechgnn_model(seed: int, lambda_mie: float, device: torch.device):
    """Load a trained MechGNN from checkpoint.

    Args:
        seed: Random seed used during training.
        lambda_mie: Lambda value for atom MIE loss weight.
        device: Device to load model onto.

    Returns:
        MechGNN model in eval mode.
    """
    from src.modeling.mech_gnn import MechGNN

    checkpoint_path = find_mechgnn_checkpoint(seed, lambda_mie)
    model = MechGNN(hidden_dim=256, node_dim=64, num_gnn_layers=3, dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model
