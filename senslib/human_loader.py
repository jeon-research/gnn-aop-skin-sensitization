"""
Human-label dataset loader utilities.

Mirrors senslib.llna_loader but filters on the human patch-test label
(sensitization_human) instead of LLNA, used for the same-test-set
contrast against the LLNA-primary configuration.

Functions:
  - load_dataset_human(seed):           scaffold split using sensitization_human as label.
  - load_dataset_human_shuffled(seed):  same but with labels permuted per seed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

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


HUMAN_LABEL_COL = 'sensitization_human'


def _read_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_dataset_human(seed: int = 42) -> Tuple[pd.DataFrame, list, list, list]:
    df = _read_df()
    df = df.dropna(subset=['smiles', HUMAN_LABEL_COL]).reset_index(drop=True)
    train_idx, val_idx, test_idx = scaffold_split(df, smiles_col='smiles', seed=seed)
    return df, train_idx, val_idx, test_idx


def load_dataset_human_shuffled(seed: int) -> Tuple[pd.DataFrame, list, list, list]:
    df = _read_df()
    df = df.dropna(subset=['smiles', HUMAN_LABEL_COL]).reset_index(drop=True)
    rng = np.random.RandomState(seed ^ 0xD157BEEF)
    shuffled = df[HUMAN_LABEL_COL].values.copy()
    rng.shuffle(shuffled)
    df[HUMAN_LABEL_COL] = shuffled
    train_idx, val_idx, test_idx = scaffold_split(df, smiles_col='smiles', seed=seed)
    return df, train_idx, val_idx, test_idx
