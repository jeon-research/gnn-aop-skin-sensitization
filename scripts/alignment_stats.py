"""Aggregate per-seed alignment results: bootstrap 95% CI of the across-seed
mean for each method x metric, plus paired Wilcoxon signed-rank between
method pairs on TP reactive-center atom-AUC.

Inputs:  results/alignment_llna_sens_7method/seed_<N>_<mode>/metrics.json
Outputs: results/alignment_llna_sens_7method/stats_<mode>.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


REVISION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REVISION_ROOT / 'results' / 'alignment_llna_sens_7method'


def _collect_seeds(mode: str, results_dir: Path) -> List[dict]:
    """Load all per-seed metrics.json under results_dir matching this mode."""
    rows: List[dict] = []
    for seed_dir in sorted(results_dir.glob(f'seed_*_{mode}')):
        path = seed_dir / 'metrics.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return rows


def _bootstrap_ci(values: List[float], n_boot: int = 10000,
                  ci: float = 0.95, rng_seed: int = 0) -> Dict[str, float]:
    if not values:
        return {'mean': float('nan'), 'boot_ci_low': float('nan'),
                'boot_ci_high': float('nan'), 'n': 0}
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        'mean':        float(arr.mean()),
        'boot_ci_low': float(lo),
        'boot_ci_high':float(hi),
        'n':           int(len(arr)),
    }


def _gather_metric(rows: List[dict], method: str, ref_kind: str,
                   metric: str) -> List[float]:
    """Collect per-seed values of mean_<metric> for one method × ref_kind."""
    key = f'mean_{metric}'
    out: List[float] = []
    for row in rows:
        slot = row['methods'].get(method, {}).get(ref_kind, {})
        v = slot.get(key)
        if v is not None and not _is_nan(v):
            out.append(float(v))
    return out


def _is_nan(x) -> bool:
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return False


def _paired_wilcoxon(a_vals: List[float], b_vals: List[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank. Only uses seeds present for BOTH methods.

    Values are expected to be aligned by seed position — caller must ensure
    that a_vals[i] and b_vals[i] come from the same seed.
    """
    if len(a_vals) != len(b_vals):
        raise ValueError("paired arrays must be equal length")
    diffs = np.asarray(a_vals) - np.asarray(b_vals)
    # Filter exact zeros (Wilcoxon drops them with zero_method='wilcox')
    if np.all(diffs == 0) or len(diffs) < 5:
        return {'W': float('nan'), 'p': float('nan'),
                'n_pairs': len(diffs), 'median_diff': float(np.median(diffs))}
    try:
        res = stats.wilcoxon(a_vals, b_vals, zero_method='wilcox', alternative='two-sided')
        return {'W': float(res.statistic), 'p': float(res.pvalue),
                'n_pairs': len(diffs), 'median_diff': float(np.median(diffs))}
    except ValueError as e:
        return {'W': float('nan'), 'p': float('nan'),
                'n_pairs': len(diffs), 'median_diff': float(np.median(diffs)),
                'note': str(e)}


def _collect_paired(rows: List[dict], method: str, ref_kind: str,
                    metric: str) -> List[tuple]:
    """Return [(seed, value), ...] in seed order for paired comparisons."""
    pairs = []
    key = f'mean_{metric}'
    for row in rows:
        seed = row['seed']
        v = row['methods'].get(method, {}).get(ref_kind, {}).get(key)
        if v is not None and not _is_nan(v):
            pairs.append((seed, float(v)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Alignment statistics.')
    parser.add_argument('--mode', choices=('union', 'intersection'),
                        default='union')
    parser.add_argument('--results-dir', type=Path,
                        default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--metrics', nargs='+',
                        default=['atom_auc', 'atom_ap',
                                 'hit_rate_at_k', 'iou_at_k'])
    args = parser.parse_args()

    rows = _collect_seeds(args.mode, args.results_dir)
    if not rows:
        print(f"No metrics found for mode={args.mode}")
        return

    print(f"Loaded {len(rows)} seed(s) for mode={args.mode}: "
          f"{[r['seed'] for r in rows]}")

    methods = sorted(rows[0]['methods'].keys())
    ref_kinds = ('tp_reactive_center', 'tp_full_substructure',
                 'all_reactive_center', 'all_full_substructure')

    out: Dict = {'mode': args.mode, 'n_seeds': len(rows), 'per_method': {}}

    for method in methods:
        out['per_method'][method] = {}
        for rk in ref_kinds:
            metric_stats: Dict = {}
            for metric in args.metrics:
                vals = _gather_metric(rows, method, rk, metric)
                metric_stats[metric] = _bootstrap_ci(vals)
            out['per_method'][method][rk] = metric_stats

    # Paired Wilcoxon on primary metric (tp_reactive_center atom_auc) between all method pairs
    primary_pairs: Dict[str, List[tuple]] = {
        m: _collect_paired(rows, m, 'tp_reactive_center', 'atom_auc')
        for m in methods
    }
    wilcoxon_out: List[Dict] = []
    for a, b in itertools.combinations(methods, 2):
        # Intersect seeds
        a_by_seed = dict(primary_pairs[a])
        b_by_seed = dict(primary_pairs[b])
        common_seeds = sorted(set(a_by_seed) & set(b_by_seed))
        if len(common_seeds) < 5:
            continue
        a_vals = [a_by_seed[s] for s in common_seeds]
        b_vals = [b_by_seed[s] for s in common_seeds]
        res = _paired_wilcoxon(a_vals, b_vals)
        res.update({'a': a, 'b': b, 'metric': 'tp_reactive_center.atom_auc',
                    'seeds': common_seeds})
        wilcoxon_out.append(res)
    out['paired_wilcoxon'] = wilcoxon_out

    path = args.results_dir / f'stats_{args.mode}.json'
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")

    # Console summary — primary metric
    print(f"\n=== Primary metric: TP reactive-center atom-AUC ({args.mode}) ===")
    for m in methods:
        s = out['per_method'][m]['tp_reactive_center']['atom_auc']
        print(f"  {m:10s}  {s['mean']:.3f}  "
              f"[{s['boot_ci_low']:.3f}, {s['boot_ci_high']:.3f}]  "
              f"(n_seeds={s['n']})")

    print(f"\n=== Paired Wilcoxon (primary metric) ===")
    for w in wilcoxon_out:
        print(f"  {w['a']:10s} vs {w['b']:10s}  "
              f"W={w['W']:.2f}  p={w['p']:.4f}  "
              f"median_diff={w['median_diff']:+.3f}  n={w['n_pairs']}")


if __name__ == '__main__':
    main()
