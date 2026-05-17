"""pos_weight sweep summary.

Reads the per-seed metrics.json from llna_training_pw{4,8,12} and reports
mean ± 95% CI of test AUC, test AP, and balanced accuracy at threshold 0.5,
plus a classification confusion summary. Compared to the primary (data-
derived pos_weight ≈ 4.45) run.

Outputs:
  results/posweight_sweep/summary.csv
  results/posweight_sweep/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REVISION_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REVISION_ROOT / 'results' / 'posweight_sweep'


def _collect(root: Path) -> List[Dict]:
    rows = []
    for seed_dir in sorted(root.glob('seed_*')):
        p = seed_dir / 'metrics.json'
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        rows.append({
            'seed':       m['seed'],
            'pos_weight': m['pos_weight'],
            'test_auc':   m['test_auc'],
            'test_ap':    m['test_ap'],
            'test_n':     m['test_n'],
            'test_n_pos': m['test_n_pos'],
        })
    return rows


def _ci(values: np.ndarray, n_boot: int = 10000) -> Dict:
    if len(values) == 0:
        return {'mean': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan')}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {'mean': float(values.mean()), 'ci_low': float(lo), 'ci_high': float(hi)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    configs = [
        ('LLNA primary (data-derived)', REVISION_ROOT / 'results' / 'llna_training'),
        ('pos_weight=4',                REVISION_ROOT / 'results' / 'llna_training_pw4'),
        ('pos_weight=8',                REVISION_ROOT / 'results' / 'llna_training_pw8'),
        ('pos_weight=12',               REVISION_ROOT / 'results' / 'llna_training_pw12'),
    ]

    per_seed: List[Dict] = []
    summary_rows: List[Dict] = []
    for label, root in configs:
        data = _collect(root)
        for d in data:
            d['config'] = label
            per_seed.append(d)
        if not data:
            summary_rows.append({'config': label, 'n_seeds': 0})
            continue
        aucs = np.array([d['test_auc'] for d in data])
        aps  = np.array([d['test_ap']  for d in data])
        ci_auc = _ci(aucs)
        ci_ap  = _ci(aps)
        summary_rows.append({
            'config':    label,
            'n_seeds':   len(data),
            'pos_weight': float(data[0]['pos_weight']),
            'test_auc_mean':   ci_auc['mean'],
            'test_auc_ci_low': ci_auc['ci_low'],
            'test_auc_ci_high':ci_auc['ci_high'],
            'test_ap_mean':    ci_ap['mean'],
            'test_ap_ci_low':  ci_ap['ci_low'],
            'test_ap_ci_high': ci_ap['ci_high'],
        })

    pd.DataFrame(per_seed).to_csv(OUT_DIR / 'per_seed.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / 'summary.csv', index=False)
    summary.to_json(OUT_DIR / 'summary.json', orient='records', indent=2)

    print('Pos_weight sweep summary:')
    if 'pos_weight' not in summary.columns:
        print('  (no pos_weight runs found; run train_llna_labels.py '
              '--pos-weight {4,8,12} --output-tag pw{4,8,12} first)')
        return
    cols = ['config', 'n_seeds', 'pos_weight',
            'test_auc_mean', 'test_auc_ci_low', 'test_auc_ci_high']
    print(summary[cols].to_string(index=False))


if __name__ == '__main__':
    main()
