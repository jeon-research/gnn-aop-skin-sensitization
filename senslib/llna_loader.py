"""
LLNA dataset loader utilities.

Wraps src.explain.utils with LLNA-specific dataset loaders:
  - load_dataset_llna:      scaffold split using llna_result as the label column.
  - load_dataset_shuffled:  scaffold split with permuted labels (negative control).
  - persist_split:          write a seed's train/val/test molecule IDs to JSON once,
                            read it back on re-runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Locate the directory that holds src/explain/utils.py. Works whether
# senslib sits directly at the repo root or one directory below it.
SENSLIB_PARENT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = next(
    p for p in (SENSLIB_PARENT.parent, SENSLIB_PARENT)
    if (p / 'src' / 'explain' / 'utils.py').exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain.utils import scaffold_split, DATA_PATH  # type: ignore


# Per-seed splits live at results/splits/ (both public and local layouts).
SPLITS_DIR = next(
    (p for p in (SENSLIB_PARENT / 'results' / 'splits', SENSLIB_PARENT / 'data' / 'splits')
     if p.exists()),
    SENSLIB_PARENT / 'results' / 'splits',
)


def _read_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_dataset_llna(seed: int = 42) -> Tuple[pd.DataFrame, list, list, list]:
    """Load dataset filtered to rows with an LLNA label, apply scaffold split.

    Returns (df, train_idx, val_idx, test_idx). Indices are positions in the
    returned df (after the LLNA filter), not positions in the raw CSV.
    """
    df = _read_df()
    df = df.dropna(subset=['smiles', 'llna_result']).reset_index(drop=True)
    train_idx, val_idx, test_idx = scaffold_split(df, smiles_col='smiles', seed=seed)
    return df, train_idx, val_idx, test_idx


def load_dataset_shuffled(
    seed: int,
    label_col: str = 'llna_result',
) -> Tuple[pd.DataFrame, list, list, list]:
    """Load dataset with labels shuffled in-place. Shuffle uses the data seed
    so the shuffle is deterministic per run.

    The scaffold split is computed on the original (unshuffled) rows, so the
    molecule partitioning is the same as load_dataset_llna — only the labels
    differ. Used as the label-permutation negative control.
    """
    df = _read_df()
    df = df.dropna(subset=['smiles', label_col]).reset_index(drop=True)
    rng = np.random.RandomState(seed ^ 0xD157BEEF)  # distinct from scaffold rng
    shuffled = df[label_col].values.copy()
    rng.shuffle(shuffled)
    df[label_col] = shuffled
    train_idx, val_idx, test_idx = scaffold_split(df, smiles_col='smiles', seed=seed)
    return df, train_idx, val_idx, test_idx


def persist_split(
    seed: int,
    label_source: str,
    df: pd.DataFrame,
    train_idx: list,
    val_idx: list,
    test_idx: list,
) -> Path:
    """Write train/val/test SMILES + labels to a per-seed JSON. Idempotent —
    if the file exists, verifies contents match and returns the path.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f'{label_source}_seed_{seed}.json'

    payload = {
        'seed': seed,
        'label_source': label_source,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'train_smiles': df.loc[train_idx, 'smiles'].tolist(),
        'val_smiles':   df.loc[val_idx,   'smiles'].tolist(),
        'test_smiles':  df.loc[test_idx,  'smiles'].tolist(),
    }

    if path.exists():
        existing = json.loads(path.read_text())
        for key in ('train_smiles', 'val_smiles', 'test_smiles'):
            if existing.get(key) != payload[key]:
                raise RuntimeError(
                    f"Split drift detected at {path}: {key} differs from committed split."
                )
        return path

    path.write_text(json.dumps(payload, indent=2))
    return path
