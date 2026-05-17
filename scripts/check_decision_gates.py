"""Pre-registered Gate A / Gate B verdict (Limitations §4.7).
Gate A: mean(atom_auc_primary) > upper_95ci(atom_auc_shuffled) + 0.05.
Gate B: shuffled_95ci subset [0.45, 0.55] AND disjoint from primary_95ci.

Inputs:  results/alignment_llna_sens_7method/stats_union.json (primary)
         results/alignment_shuffled_sens_7method/stats_union.json (shuffle, optional;
         if omitted, the theoretical null is substituted and the verdict
         is marked indicative)
Outputs: gates.json + decision_gate_log.md entry; verdict on stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

REVISION_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REVISION_ROOT / 'decision_gate_log.md'


THEORETICAL_SHUFFLE = {
    'mean': 0.50,
    'boot_ci_low': 0.45,
    'boot_ci_high': 0.55,
    'n': 0,
    'source': 'theoretical — empirical shuffle not yet run',
}


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(REVISION_ROOT.parent), 'rev-parse', '--short', 'HEAD'],
            text=True).strip()
    except Exception:
        return 'unknown'


def _gate_a(primary_stat: Dict, shuffle_stat: Dict) -> Dict:
    pm = primary_stat['mean']
    sh = shuffle_stat['boot_ci_high']
    threshold = sh + 0.05
    passed = pm > threshold
    return {
        'passed': bool(passed),
        'primary_mean': pm,
        'shuffle_ci_upper': sh,
        'threshold': threshold,
        'margin': pm - threshold,
    }


def _gate_b(primary_stat: Dict, shuffle_stat: Dict) -> Dict:
    s_lo = shuffle_stat['boot_ci_low']
    s_hi = shuffle_stat['boot_ci_high']
    within_window = (0.45 <= s_lo) and (s_hi <= 0.55)
    p_lo = primary_stat['boot_ci_low']
    p_hi = primary_stat['boot_ci_high']
    disjoint = (s_hi < p_lo) or (p_hi < s_lo)
    return {
        'passed': bool(within_window and disjoint),
        'within_window': bool(within_window),
        'disjoint': bool(disjoint),
        'shuffle_ci': [s_lo, s_hi],
        'primary_ci': [p_lo, p_hi],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gate A and Gate B.')
    parser.add_argument('--primary-path', type=Path,
                        default=REVISION_ROOT / 'results' / 'alignment_llna_sens_7method' / 'stats_union.json')
    parser.add_argument('--shuffle-path', type=Path,
                        default=REVISION_ROOT / 'results' / 'alignment_shuffled_sens_7method' / 'stats_union.json')
    parser.add_argument('--method', type=str, default='attention',
                        help='Primary attribution method to evaluate against.')
    parser.add_argument('--ref-kind', type=str, default='tp_reactive_center')
    parser.add_argument('--metric', type=str, default='atom_auc')
    parser.add_argument('--label', type=str, default='primary_attention',
                        help='Short label for this gate check (e.g. "primary_attention").')
    args = parser.parse_args()

    if not args.primary_path.exists():
        raise SystemExit(f"Missing primary stats: {args.primary_path}")

    primary_all = json.loads(args.primary_path.read_text())
    primary = (primary_all['per_method'][args.method][args.ref_kind]
               [args.metric])

    indicative = False
    if args.shuffle_path.exists():
        shuffle_all = json.loads(args.shuffle_path.read_text())
        shuffle = (shuffle_all['per_method'][args.method][args.ref_kind]
                   [args.metric])
    else:
        shuffle = dict(THEORETICAL_SHUFFLE)
        indicative = True

    gate_a = _gate_a(primary, shuffle)
    gate_b = _gate_b(primary, shuffle)

    verdict = {
        'label': args.label,
        'indicative': indicative,
        'primary_source': str(args.primary_path),
        'shuffle_source': str(args.shuffle_path) if not indicative else 'theoretical',
        'method': args.method,
        'ref_kind': args.ref_kind,
        'metric': args.metric,
        'n_seeds_primary': primary.get('n', 0),
        'n_seeds_shuffle': shuffle.get('n', 0),
        'primary': primary,
        'shuffle': shuffle,
        'gate_a': gate_a,
        'gate_b': gate_b,
        'commit': _git_head(),
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    }

    out_path = args.primary_path.parent / 'gates.json'
    out_path.write_text(json.dumps(verdict, indent=2))
    print(f"Wrote {out_path}")

    print()
    prefix = '[INDICATIVE] ' if indicative else ''
    print(f"{prefix}Gate check: {args.label}")
    print(f"  method={args.method}  ref={args.ref_kind}  metric={args.metric}")
    print(f"  primary mean={primary['mean']:.3f}  CI [{primary['boot_ci_low']:.3f}, {primary['boot_ci_high']:.3f}]  (n={primary.get('n',0)})")
    print(f"  shuffle mean={shuffle['mean']:.3f}  CI [{shuffle['boot_ci_low']:.3f}, {shuffle['boot_ci_high']:.3f}]  (n={shuffle.get('n',0)})")
    print()
    print(f"  Gate A (primary > shuffle_ci_high + 0.05): "
          f"{'PASS' if gate_a['passed'] else 'FAIL'}  "
          f"[margin = {gate_a['margin']:+.3f}]")
    print(f"  Gate B (shuffle_ci ⊂ [0.45, 0.55] AND disjoint): "
          f"{'PASS' if gate_b['passed'] else 'FAIL'}  "
          f"[within_window={gate_b['within_window']}, disjoint={gate_b['disjoint']}]")

    # Append to decision_gate_log.md
    n_primary = primary.get('n', 0)
    n_shuffle_note = 'THEORETICAL' if indicative else f"n_seeds={shuffle.get('n', 0)}"
    pass_a = 'PASS' if gate_a['passed'] else 'FAIL'
    pass_b = 'PASS' if gate_b['passed'] else 'FAIL'
    suffix = ' (INDICATIVE)' if indicative else ''

    entry = (
        f"\n### {verdict['timestamp']} — {args.label}{suffix}\n"
        f"- **Commit:** `{verdict['commit']}`\n"
        f"- **Evidence:** method={args.method}, ref={args.ref_kind}, metric={args.metric}\n"
        f"  - primary mean={primary['mean']:.3f}, "
        f"CI [{primary['boot_ci_low']:.3f}, {primary['boot_ci_high']:.3f}], "
        f"n_seeds={n_primary}\n"
        f"  - shuffle mean={shuffle['mean']:.3f}, "
        f"CI [{shuffle['boot_ci_low']:.3f}, {shuffle['boot_ci_high']:.3f}], "
        f"{n_shuffle_note}\n"
        f"- **Gate A:** {pass_a} (margin {gate_a['margin']:+.3f})\n"
        f"- **Gate B:** {pass_b} "
        f"(within_window={gate_b['within_window']}, disjoint={gate_b['disjoint']})\n"
    )
    with LOG_PATH.open('a') as f:
        f.write(entry)
    print(f"\nAppended to {LOG_PATH}")


if __name__ == '__main__':
    main()
