"""Classification confusion matrices for every model (Table 1, Table S11).

Walks LLNA-primary, shuffle-control, pos_weight variants, and the
architecture-ablation checkpoints. Reports TP/FP/TN/FN, accuracy,
balanced_accuracy, precision, recall, specificity, F1, MCC. Threshold is
0.5 by default for comparability; optimal-threshold variants are appended
for the LLNA-primary rows.

Outputs: results/classification_confusion/summary.{csv,json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef,
    precision_score, recall_score, f1_score,
)

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE.parent, _HERE.parent.parent):
    if (_candidate / 'src' / 'explain' / 'utils.py').exists():
        PROJECT_ROOT = _candidate
        break
else:
    PROJECT_ROOT = _HERE.parent.parent  # fall back; ablation fallback path tolerates missing dirs
REVISION_ROOT = _HERE.parent
OUTPUT_DIR = REVISION_ROOT / 'results' / 'classification_confusion'


def _confusion_row(probs: np.ndarray, labels: np.ndarray,
                   threshold: float = 0.5) -> Dict:
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    n = len(labels)
    n_pos = int(labels.sum())
    out = {
        'n':        n,
        'n_pos':    n_pos,
        'tp':       int(tp),
        'fp':       int(fp),
        'tn':       int(tn),
        'fn':       int(fn),
        'accuracy': (tp + tn) / n if n else float('nan'),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)) if len(np.unique(labels)) > 1 else float('nan'),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall_sensitivity': float(recall_score(labels, preds, zero_division=0)),
        'specificity': (tn / (tn + fp)) if (tn + fp) > 0 else float('nan'),
        'f1':         float(f1_score(labels, preds, zero_division=0)),
        'mcc':        float(matthews_corrcoef(labels, preds)) if len(np.unique(labels)) > 1 else float('nan'),
        'threshold':  threshold,
    }
    return out


def _optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Threshold that maximises balanced accuracy on the test split.
    Test-set threshold tuning inflates BA by ~0.02-0.03 (Limitations §4.7).
    """
    best_ba, best_t = -1.0, 0.5
    for t in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        if len(np.unique(labels)) < 2:
            continue
        ba = balanced_accuracy_score(labels, preds)
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t


def _collect_label_native_model(model_root: Path, model_name: str) -> List[Dict]:
    rows: List[Dict] = []
    for seed_dir in sorted(model_root.glob('seed_*')):
        mpath = seed_dir / 'metrics.json'
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        probs = np.asarray(m.get('test_probs', []), dtype=float)
        labels = np.asarray(m.get('test_labels', []), dtype=float)
        if probs.size == 0 or labels.size == 0:
            continue
        row = _confusion_row(probs, labels)
        # Also compute at optimal threshold so tables are comparable with
        # the architecture-ablation rows.
        topt = _optimal_threshold(probs, labels)
        opt = _confusion_row(probs, labels, threshold=topt)
        row['balanced_accuracy_opt'] = opt['balanced_accuracy']
        row['precision_opt']         = opt['precision']
        row['recall_opt']            = opt['recall_sensitivity']
        row['specificity_opt']       = opt['specificity']
        row['f1_opt']                = opt['f1']
        row['mcc_opt']               = opt['mcc']
        row['optimal_threshold']     = topt
        row.update({
            'model': model_name,
            'seed': int(seed_dir.name.split('_')[1]),
            'source': str(mpath),
        })
        rows.append(row)
    return rows


def _collect_arch_ablation(arch_name: str, condition: str) -> List[Dict]:
    """Read raw probs from results/frozen_ablation_probs/ (if
    extract_frozen_ablation_probs.py has been run); otherwise fall back to
    final_metrics.json from results/ablation/.
    """
    rows: List[Dict] = []

    # Preferred source: re-run probs extracted by extract_frozen_ablation_probs.py
    repo_root = Path(__file__).resolve().parent.parent
    probs_root = repo_root / 'results' / 'frozen_ablation_probs' / f'{condition}_{arch_name.lower() if arch_name != "AttentiveFP" else "attentivefp"}'
    if probs_root.exists():
        for seed_dir in sorted(probs_root.glob('seed_*')):
            p = seed_dir / 'probs.json'
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            probs = np.asarray(d['probs'], dtype=float)
            labels = np.asarray(d['labels'], dtype=float)
            if probs.size == 0 or labels.size == 0:
                continue
            # At threshold=0.5 (calibration-naive)
            row = _confusion_row(probs, labels)
            # Also compute at optimal threshold (paper convention)
            topt = _optimal_threshold(probs, labels)
            opt = _confusion_row(probs, labels, threshold=topt)
            row['balanced_accuracy_opt'] = opt['balanced_accuracy']
            row['precision_opt']         = opt['precision']
            row['recall_opt']            = opt['recall_sensitivity']
            row['specificity_opt']       = opt['specificity']
            row['f1_opt']                = opt['f1']
            row['mcc_opt']               = opt['mcc']
            row['optimal_threshold']     = topt
            row.update({
                'model':  f'{condition} ({arch_name})',
                'seed':   d['seed'],
                'source': str(p),
            })
            rows.append(row)
        if rows:
            return rows
        # fall through to aggregate if probs_root empty

    root = PROJECT_ROOT / 'results' / 'ablation' / condition
    if not root.exists():
        return rows

    for seed_dir in sorted(root.glob('seed_*')):
        run_dirs = sorted([d for d in seed_dir.iterdir() if d.is_dir()],
                          key=lambda d: d.name, reverse=True)
        chosen = None
        for run in run_dirs:
            if (run / 'results.json').exists() or (run / 'final_metrics.json').exists():
                chosen = run
                break
        if not chosen:
            continue

        # Prefer results.json (the format actually used for the paper)
        res_path = chosen / 'results.json'
        if res_path.exists():
            d = json.loads(res_path.read_text())
            sens = d.get('test_metrics', {}).get('sensitization', {})
        else:
            d = json.loads((chosen / 'final_metrics.json').read_text())
            sens = d.get('sensitization', {})
        if not sens:
            continue

        rows.append({
            'model':    f'{condition} ({arch_name})',
            'seed':     int(seed_dir.name.split('_')[1]),
            'n':        int(sens.get('n_samples', 0)),
            'n_pos':    int(sens.get('n_positive', 0)),
            'tp':       None, 'fp': None, 'tn': None, 'fn': None,
            'accuracy': sens.get('accuracy'),
            'balanced_accuracy': sens.get('balanced_accuracy'),
            'precision':         None,
            'recall_sensitivity': None,
            'specificity':        None,
            'f1':                 sens.get('f1'),
            'mcc':                None,
            'auc':                sens.get('auc'),
            'threshold':          sens.get('optimal_threshold'),
            'source':             str(res_path if res_path.exists() else (chosen / 'final_metrics.json')),
            'note':               'aggregate only; raw probs unavailable',
        })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []

    # --- Revision models with raw probs available ---
    native_models = [
        (REVISION_ROOT / 'results' / 'llna_training',     'LLNA primary'),
        (REVISION_ROOT / 'results' / 'shuffle_control',   'shuffled null'),
    ]
    # Optional: pos-weight sweeps if present
    for tag in ('pw4', 'pw8', 'pw12'):
        p = REVISION_ROOT / 'results' / f'llna_training_{tag}'
        if p.exists():
            native_models.append((p, f'LLNA (pos_weight={tag[2:]})'))

    for root, name in native_models:
        rows = _collect_label_native_model(root, name)
        print(f'  {name}: {len(rows)} seeds')
        all_rows.extend(rows)

    # --- Frozen architecture ablations (may be aggregate-only) ---
    for arch, condition in (
        ('AttentiveFP', 'plain'),
        ('GCN',         'plain_gcn'),
        ('GIN',         'plain_gin'),
    ):
        rows = _collect_arch_ablation(arch, condition)
        print(f'  {condition} ({arch}): {len(rows)} seeds')
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / 'per_seed.csv', index=False)
    df.to_json(OUTPUT_DIR / 'per_seed.json', orient='records', indent=2)

    if df.empty:
        print()
        print('No per-seed metrics found. Run train_llna_labels.py / '
              'train_human_labels.py first to produce '
              'results/*_training/seed_*/metrics.json.')
        return

    # Per-model summary with mean ± sd across seeds
    key_cols = ['accuracy', 'balanced_accuracy', 'precision',
                'recall_sensitivity', 'specificity', 'f1', 'mcc',
                'balanced_accuracy_opt', 'precision_opt', 'recall_opt',
                'specificity_opt', 'f1_opt', 'mcc_opt', 'optimal_threshold']
    summary_rows: List[Dict] = []
    for model, sub in df.groupby('model'):
        stat = {'model': model, 'n_seeds': len(sub)}
        for k in key_cols:
            s = sub[k].dropna()
            if len(s) == 0:
                stat[f'{k}_mean'] = float('nan')
                stat[f'{k}_std']  = float('nan')
            else:
                stat[f'{k}_mean'] = float(s.mean())
                stat[f'{k}_std']  = float(s.std())
        if sub['tp'].notna().all():
            stat['tp_mean'] = float(sub['tp'].mean())
            stat['fp_mean'] = float(sub['fp'].mean())
            stat['tn_mean'] = float(sub['tn'].mean())
            stat['fn_mean'] = float(sub['fn'].mean())
        summary_rows.append(stat)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / 'summary.csv', index=False)
    summary.to_json(OUTPUT_DIR / 'summary.json', orient='records', indent=2)

    print()
    print('Summary (threshold = 0.5):')
    display_cols = ['model', 'n_seeds', 'balanced_accuracy_mean',
                    'precision_mean', 'recall_sensitivity_mean',
                    'specificity_mean', 'mcc_mean']
    print(summary[display_cols].to_string(index=False))

    print()
    print('Summary (per-model optimal threshold):')
    display_cols_opt = ['model', 'n_seeds', 'optimal_threshold_mean',
                        'balanced_accuracy_opt_mean', 'precision_opt_mean',
                        'recall_opt_mean', 'specificity_opt_mean', 'mcc_opt_mean']
    existing = [c for c in display_cols_opt if c in summary.columns]
    print(summary[existing].to_string(index=False))


if __name__ == '__main__':
    main()
