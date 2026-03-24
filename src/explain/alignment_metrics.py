"""
Alignment metrics for comparing GNN explanations to AOP reference labels.

Threshold-free metrics: Atom-level ROC-AUC, Average Precision
Threshold-based metrics: HitRate@K (adaptive), Precision@K (fixed-K), IoU@K
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def hit_rate_at_k(importance: np.ndarray, reference: np.ndarray, k: int) -> float:
    """HitRate@K: fraction of MIE atoms in the top-K important atoms.

    When K = n_mie, this equals both Precision@K and Recall@K,
    so we report it as a single metric to avoid redundancy.
    """
    n_mie = reference.sum()
    if k <= 0 or n_mie == 0:
        return 0.0
    top_k_idx = np.argsort(importance)[-k:]
    return float(reference[top_k_idx].sum() / k)


def precision_at_fixed_k(importance: np.ndarray, reference: np.ndarray, k: int) -> float:
    """Precision@K with fixed K (not adaptive)."""
    if k <= 0 or reference.sum() == 0:
        return 0.0
    k = min(k, len(importance))
    top_k_idx = np.argsort(importance)[-k:]
    return float(reference[top_k_idx].sum() / k)


def iou_at_k(importance: np.ndarray, reference: np.ndarray, k: int) -> float:
    """IoU@K: |top-K intersection MIE| / |top-K union MIE|."""
    if k <= 0:
        return 0.0
    top_k_set = set(np.argsort(importance)[-k:])
    mie_set = set(np.where(reference > 0)[0])
    if len(top_k_set | mie_set) == 0:
        return 0.0
    return len(top_k_set & mie_set) / len(top_k_set | mie_set)


def atom_auc(importance: np.ndarray, reference: np.ndarray) -> float:
    """Atom-level ROC-AUC: can GNN importance distinguish MIE from non-MIE atoms?"""
    if reference.sum() == 0 or reference.sum() == len(reference):
        return float('nan')
    try:
        return float(roc_auc_score(reference, importance))
    except ValueError:
        return float('nan')


def atom_ap(importance: np.ndarray, reference: np.ndarray) -> float:
    """Average Precision: area under precision-recall curve."""
    if reference.sum() == 0:
        return float('nan')
    try:
        return float(average_precision_score(reference, importance))
    except ValueError:
        return float('nan')


def compute_alignment_metrics(
    importance: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, float]:
    """Compute all alignment metrics for a single molecule.

    Args:
        importance: [n_atoms] GNN importance scores.
        reference: [n_atoms] binary MIE reference mask.

    Returns:
        Dict with all metric values.
    """
    imp = importance.float().numpy()
    ref = reference.float().numpy()

    n_atoms = len(imp)
    n_mie = int(ref.sum())

    if n_atoms == 0:
        return {k: float('nan') for k in [
            'atom_auc', 'atom_ap', 'hit_rate_at_k',
            'precision_at_3', 'precision_at_5',
            'iou_at_k', 'k', 'n_atoms', 'n_mie_atoms',
        ]}

    # Adaptive K = number of MIE atoms
    k = max(1, n_mie)

    return {
        'atom_auc': atom_auc(imp, ref),
        'atom_ap': atom_ap(imp, ref),
        'hit_rate_at_k': hit_rate_at_k(imp, ref, k),
        'precision_at_3': precision_at_fixed_k(imp, ref, 3),
        'precision_at_5': precision_at_fixed_k(imp, ref, 5),
        'iou_at_k': iou_at_k(imp, ref, k),
        'k': k,
        'n_atoms': n_atoms,
        'n_mie_atoms': n_mie,
    }


def compute_batch_alignment(
    importances: List[torch.Tensor],
    references: List[torch.Tensor],
) -> Dict[str, float]:
    """Compute mean alignment metrics across a batch of molecules.

    Only includes molecules with at least one MIE atom (otherwise metrics
    are undefined).

    Returns:
        Dict with mean metrics and count of valid molecules.
    """
    all_metrics = []

    for imp, ref in zip(importances, references):
        if ref.sum() == 0:
            continue  # Skip molecules with no MIE atoms
        metrics = compute_alignment_metrics(imp, ref)
        if not np.isnan(metrics['atom_auc']):
            all_metrics.append(metrics)

    if not all_metrics:
        return {'n_valid': 0}

    # Aggregate
    keys = ['atom_auc', 'atom_ap', 'hit_rate_at_k',
            'precision_at_3', 'precision_at_5', 'iou_at_k']
    result = {'n_valid': len(all_metrics)}

    for key in keys:
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if values:
            result[f'mean_{key}'] = float(np.mean(values))
            result[f'std_{key}'] = float(np.std(values))
        else:
            result[f'mean_{key}'] = float('nan')
            result[f'std_{key}'] = float('nan')

    return result


def compute_stratified_alignment(
    importances: List[torch.Tensor],
    references: List[torch.Tensor],
    strata: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute alignment metrics stratified by a grouping variable.

    Args:
        importances: List of [n_atoms] GNN importance tensors.
        references: List of [n_atoms] binary MIE reference masks.
        strata: List of stratum labels (e.g., mechanism type).

    Returns:
        Dict mapping stratum label -> mean alignment metrics.
    """
    from collections import defaultdict

    groups = defaultdict(lambda: ([], []))
    for imp, ref, stratum in zip(importances, references, strata):
        groups[stratum][0].append(imp)
        groups[stratum][1].append(ref)

    result = {}
    for stratum, (imps, refs) in groups.items():
        result[stratum] = compute_batch_alignment(imps, refs)

    return result
