"""Summarise human-label shuffle-control and pos_weight sweep results
(broader cross-check, §3.5 onward).

Outputs a JSON blob keyed by:
  {
    "human_shuffle": {test_auc_mean, test_auc_ci, n_seeds,
                      atom_auc_attention_mean (if available), ...},
    "human_pw_sweep": {pw4: {...}, pw8: {...}, pw12: {...}}
  }
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import List

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REVISION_ROOT))


def _bootstrap_ci(values: List[float], n_boot: int = 5000, alpha: float = 0.05):
    rng = np.random.RandomState(42)
    arr = np.array(values)
    if len(arr) == 0:
        return (float('nan'), float('nan'))
    boots = [arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def _gather_classification(root: Path, seeds: List[int]):
    rows = []
    for s in seeds:
        f = root / f'seed_{s}' / 'metrics.json'
        if f.exists():
            d = json.loads(f.read_text())
            rows.append({
                'seed': s,
                'test_auc': d['test_auc'],
                'test_n': d['test_n'],
                'test_n_pos': d['test_n_pos'],
                'pos_weight': d.get('pos_weight'),
            })
    aucs = [r['test_auc'] for r in rows]
    return {
        'n_seeds': len(rows),
        'auc_mean': float(np.mean(aucs)) if aucs else float('nan'),
        'auc_sd':   float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float('nan'),
        'auc_min':  float(np.min(aucs)) if aucs else float('nan'),
        'auc_max':  float(np.max(aucs)) if aucs else float('nan'),
        'auc_ci95': _bootstrap_ci(aucs),
        'rows': rows,
    }


def _gather_alignment(align_root: Path, seeds: List[int], mask_mode: str = 'union'):
    """Read per-seed alignment metrics for attention/IG/GradCAM/ensemble."""
    methods = ['attention', 'ig', 'gradcam', 'ensemble']
    out = {}
    for m in methods:
        per_seed = []
        for s in seeds:
            f = align_root / f'seed_{s}_{mask_mode}' / 'metrics.json'
            if f.exists():
                d = json.loads(f.read_text())
                v = d.get(m, {}).get('atom_auc_all_molecule')
                if v is not None:
                    per_seed.append(v)
        if per_seed:
            out[m] = {
                'n': len(per_seed),
                'mean': float(np.mean(per_seed)),
                'sd':   float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float('nan'),
                'ci95': _bootstrap_ci(per_seed),
            }
    return out


def main():
    from senslib.seeds import PRIMARY
    seeds = list(PRIMARY)

    out = {}

    # Human-label shuffle classification
    out['human_shuffle_classification'] = _gather_classification(
        REVISION_ROOT / 'results' / 'shuffle_control_human', seeds
    )
    # Human-label pos_weight sweep
    out['human_pw_sweep'] = {}
    for pw_tag, label in [('pw4', 4), ('pw8', 8), ('pw12', 12)]:
        out['human_pw_sweep'][label] = _gather_classification(
            REVISION_ROOT / 'results' / f'human_training_{pw_tag}', seeds
        )

    # Atom-AUC alignment (if extracted)
    align_dir = REVISION_ROOT / 'results' / 'alignment_shuffled_human_sens'
    if align_dir.exists():
        out['human_shuffle_alignment'] = _gather_alignment(align_dir, seeds, 'union')

    summary_path = REVISION_ROOT / 'results' / 'summary_human_label.json'
    summary_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
