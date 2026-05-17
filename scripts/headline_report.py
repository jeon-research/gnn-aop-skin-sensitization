"""Console summary of LLNA-primary vs shuffled-label results across the
20-seed run: classification AUC, atom-level alignment for the seven
attribution methods on each reference mask, and the Gate A / Gate B
verdict for attention.

Reads:  results/alignment_llna_sens_7method/stats_union.json
        results/alignment_shuffled_sens_7method/stats_union.json
        results/llna_training/seed_*/metrics.json
        results/shuffle_control/seed_*/metrics.json

Usage:  python3 scripts/headline_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / 'results'


def _load_stats(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_training_metrics(train_root: Path) -> List[Dict]:
    out = []
    for p in sorted(train_root.glob('seed_*/metrics.json')):
        out.append(json.loads(p.read_text()))
    return out


def _fmt(x: float) -> str:
    if np.isnan(x):
        return '  NaN  '
    return f'{x:+.3f}' if x < 0 else f' {x:.3f}'


def main():
    llna_train = _load_training_metrics(ROOT / 'llna_training')
    shuf_train = _load_training_metrics(ROOT / 'shuffle_control')

    llna_stats = _load_stats(ROOT / 'alignment_llna_sens_7method' / 'stats_union.json')
    shuf_stats = _load_stats(ROOT / 'alignment_shuffled_sens_7method' / 'stats_union.json')

    print('=' * 72)
    print(' Headline summary — LLNA primary vs shuffled-label null')
    print('=' * 72)

    # ---- Classification ----
    print('\n[1] Classification AUC (test set, per-seed):')
    for label, rows in (('LLNA', llna_train), ('Shuffled', shuf_train)):
        if not rows:
            print(f'  {label:10s}  (no results)')
            continue
        aucs = [r['test_auc'] for r in rows]
        print(f"  {label:10s}  mean={np.mean(aucs):.3f}  "
              f"min={np.min(aucs):.3f}  max={np.max(aucs):.3f}  "
              f"n_seeds={len(aucs)}")

    # ---- Alignment ----
    print('\n[2] Atom-level alignment (TP reactive-center, atom_auc):')
    print(f"  {'Method':<10s} {'LLNA':>22s}  {'Shuffled':>22s}  {'Δ':>8s}")

    methods = sorted(set(list(llna_stats.get('per_method', {}).keys()) +
                         list(shuf_stats.get('per_method', {}).keys())))
    for m in methods:
        def _auc(stats, m):
            s = (stats.get('per_method', {}).get(m, {})
                 .get('tp_reactive_center', {}).get('atom_auc'))
            if s is None:
                return None, None, None
            return s['mean'], s['boot_ci_low'], s['boot_ci_high']

        pm, plo, phi = _auc(llna_stats, m)
        sm, slo, shi = _auc(shuf_stats, m)

        pm_s = f"{pm:.3f} [{plo:.3f},{phi:.3f}]" if pm is not None else '      —      '
        sm_s = f"{sm:.3f} [{slo:.3f},{shi:.3f}]" if sm is not None else '      —      '
        delta_s = _fmt(pm - sm) if (pm is not None and sm is not None) else '   —   '
        print(f"  {m:<10s} {pm_s:>22s}  {sm_s:>22s}  {delta_s}")

    print('\n[3] Atom-level alignment (all-molecule reactive-center, atom_auc):')
    print(f"  {'Method':<10s} {'LLNA':>22s}  {'Shuffled':>22s}  {'Δ':>8s}")
    for m in methods:
        def _auc2(stats, m):
            s = (stats.get('per_method', {}).get(m, {})
                 .get('all_reactive_center', {}).get('atom_auc'))
            if s is None:
                return None, None, None
            return s['mean'], s['boot_ci_low'], s['boot_ci_high']

        pm, plo, phi = _auc2(llna_stats, m)
        sm, slo, shi = _auc2(shuf_stats, m)

        pm_s = f"{pm:.3f} [{plo:.3f},{phi:.3f}]" if pm is not None else '      —      '
        sm_s = f"{sm:.3f} [{slo:.3f},{shi:.3f}]" if sm is not None else '      —      '
        delta_s = _fmt(pm - sm) if (pm is not None and sm is not None) else '   —   '
        print(f"  {m:<10s} {pm_s:>22s}  {sm_s:>22s}  {delta_s}")

    # ---- Gate verdict for primary method ----
    def _gate(pm_mean, pm_ci_lo, pm_ci_hi, sm_mean, sm_ci_lo, sm_ci_hi):
        thresh = sm_ci_hi + 0.05
        gate_a = pm_mean > thresh
        within = 0.45 <= sm_ci_lo and sm_ci_hi <= 0.55
        disjoint = (sm_ci_hi < pm_ci_lo) or (pm_ci_hi < sm_ci_lo)
        return {
            'gate_a': gate_a,
            'gate_a_margin': pm_mean - thresh,
            'gate_b': within and disjoint,
            'within': within,
            'disjoint': disjoint,
        }

    print('\n[4] Gate A / Gate B verdict (attention, all-molecule reactive-center):')
    for ref_label, ref_key in (('TP reactive-center', 'tp_reactive_center'),
                               ('All-mol reactive-center', 'all_reactive_center')):
        try:
            pls = llna_stats['per_method']['attention'][ref_key]['atom_auc']
            sls = shuf_stats['per_method']['attention'][ref_key]['atom_auc']
        except KeyError:
            print(f"  {ref_label}: missing data")
            continue
        g = _gate(pls['mean'], pls['boot_ci_low'], pls['boot_ci_high'],
                  sls['mean'], sls['boot_ci_low'], sls['boot_ci_high'])
        print(f"\n  {ref_label}:")
        print(f"    primary  mean={pls['mean']:.3f}  CI [{pls['boot_ci_low']:.3f}, {pls['boot_ci_high']:.3f}]")
        print(f"    shuffled mean={sls['mean']:.3f}  CI [{sls['boot_ci_low']:.3f}, {sls['boot_ci_high']:.3f}]")
        print(f"    Gate A:  {'PASS' if g['gate_a'] else 'FAIL'}  (margin {g['gate_a_margin']:+.3f})")
        print(f"    Gate B:  {'PASS' if g['gate_b'] else 'FAIL'}  "
              f"(within_window={g['within']}, disjoint={g['disjoint']})")


if __name__ == '__main__':
    main()
