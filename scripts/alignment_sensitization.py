"""Atom-level alignment under the 50-pattern sensitization-specific alert
set (Enoch 2008 + TIMES-SS + Roberts 2008). Reads existing per-seed
explanation files and recomputes atom-AUC / atom-AP / HitRate@K /
Precision@K / IoU@K against reactive-center and full-substructure masks.

Mask mode is parameterised (--mask-mode {union,intersection}).

Inputs:  results/<explanations_dir>/seed_<N>/explanations.json
Outputs: results/alignment_llna_sens_7method/seed_<N>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE.parent, _HERE.parent.parent):
    if (_candidate / 'src' / 'explain' / 'utils.py').exists():
        PROJECT_ROOT = _candidate
        break
else:
    raise RuntimeError(f"src/explain/utils.py not found from {_HERE}")
REVISION_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
if str(REVISION_ROOT) != str(PROJECT_ROOT):
    sys.path.insert(0, str(REVISION_ROOT))

from src.explain.alignment_metrics import compute_batch_alignment          # type: ignore
from senslib.aop_reference_sensitization import AOPReferenceSensitization


from senslib.seeds import PRIMARY as DEFAULT_SEEDS  # noqa: E402
DEFAULT_EXPLANATIONS_DIR = REVISION_ROOT / 'results' / 'explanations_llna_7method'
DEFAULT_OUTPUT_DIR = REVISION_ROOT / 'results' / 'alignment_llna_sens_7method'


def _load_explanations(seed: int, explanations_dir: Path) -> Dict:
    path = explanations_dir / f'seed_{seed}' / 'explanations.json'
    if not path.exists():
        raise FileNotFoundError(f"No explanations for seed {seed} at {path}")
    with open(path) as f:
        return json.load(f)


def _run_seed(seed: int, mask_mode: str, ref: AOPReferenceSensitization,
              explanations_dir: Path) -> Dict:
    data = _load_explanations(seed, explanations_dir)
    smiles_list: List[str] = data['smiles']
    labels = np.asarray(data['labels'])
    predictions = np.asarray(data['predictions'])

    # Build sensitization-specific reference masks
    ref_masks: List[torch.Tensor] = []
    rc_masks:  List[torch.Tensor] = []
    for smi in smiles_list:
        mask, _ = ref.get_atom_mask(smi, mode=mask_mode)
        ref_masks.append(mask)
        rc, _ = ref.get_reactive_center_mask(smi)
        rc_masks.append(rc)

    tp_mask = (labels == 1) & (predictions >= 0.5)
    n_sens = int((labels == 1).sum())
    n_tp = int(tp_mask.sum())

    print(f"    seed={seed} mode={mask_mode}: "
          f"{len(smiles_list)} mols, {n_sens} sensitizers, {n_tp} TP")

    per_method: Dict[str, Dict] = {}
    for method_name, importances in data['explanations'].items():
        imp_tensors = [torch.tensor(x, dtype=torch.float32) for x in importances]

        # TP-only alignment on reactive centers (primary metric)
        tp_imp = [imp_tensors[i] for i in range(len(imp_tensors)) if tp_mask[i]]
        tp_rc = [rc_masks[i] for i in range(len(rc_masks)) if tp_mask[i]]
        tp_ref = [ref_masks[i] for i in range(len(ref_masks)) if tp_mask[i]]

        per_method[method_name] = {
            'tp_reactive_center': compute_batch_alignment(tp_imp, tp_rc),
            'tp_full_substructure': compute_batch_alignment(tp_imp, tp_ref),
            'all_reactive_center': compute_batch_alignment(imp_tensors, rc_masks),
            'all_full_substructure': compute_batch_alignment(imp_tensors, ref_masks),
        }

    return {
        'seed': seed,
        'mask_mode': mask_mode,
        'n_molecules': len(smiles_list),
        'n_sensitizers': n_sens,
        'n_tp': n_tp,
        'methods': per_method,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Recompute alignment with sensitization-specific alerts.')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--mask-mode', choices=('union', 'intersection'),
                        default='union')
    parser.add_argument('--explanations-dir', type=Path,
                        default=DEFAULT_EXPLANATIONS_DIR)
    parser.add_argument('--output-dir', type=Path,
                        default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref = AOPReferenceSensitization()

    print(f"Sensitization-specific alignment recompute")
    print(f"  seeds: {args.seeds}")
    print(f"  mask_mode: {args.mask_mode}")
    print(f"  explanations_dir: {args.explanations_dir}")
    print(f"  output_dir:       {args.output_dir}")

    per_seed = []
    for seed in args.seeds:
        try:
            res = _run_seed(seed, args.mask_mode, ref, args.explanations_dir)
        except FileNotFoundError as e:
            print(f"  !! {e} — skipping")
            continue
        seed_dir = args.output_dir / f'seed_{seed}_{args.mask_mode}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / 'metrics.json').write_text(json.dumps(res, indent=2))
        per_seed.append(res)

    # Aggregate across seeds
    agg = _aggregate(per_seed, args.mask_mode)
    (args.output_dir / f'summary_{args.mask_mode}.json').write_text(json.dumps(agg, indent=2))
    print(f"\nWrote aggregate to {args.output_dir / f'summary_{args.mask_mode}.json'}")
    print(f"Primary atom-AUC (TP, reactive center) by method:")
    for method, stat in agg['methods'].items():
        entry = stat['tp_reactive_center'].get('mean_atom_auc')
        if not entry:
            print(f"  {method:10s}  no valid seeds")
            continue
        print(f"  {method:10s}  {entry['mean']:.3f} ± {entry['std']:.3f}  "
              f"(across-seed std, n_seeds={entry['n_seeds']})")


def _aggregate(per_seed: List[Dict], mask_mode: str) -> Dict:
    if not per_seed:
        return {'mask_mode': mask_mode, 'methods': {}, 'n_seeds': 0}

    methods = sorted(per_seed[0]['methods'].keys())
    out: Dict = {'mask_mode': mask_mode, 'n_seeds': len(per_seed), 'methods': {}}

    for method in methods:
        per_ref_kind: Dict = {}
        for ref_kind in ('tp_reactive_center', 'tp_full_substructure',
                         'all_reactive_center', 'all_full_substructure'):
            metric_stats: Dict = {}
            all_keys = per_seed[0]['methods'][method][ref_kind].keys()
            for k in all_keys:
                if not k.startswith('mean_'):
                    continue
                vals = []
                for s in per_seed:
                    v = s['methods'][method][ref_kind].get(k)
                    if v is not None and not _is_nan(v):
                        vals.append(v)
                if vals:
                    metric_stats[k] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'n_seeds': len(vals),
                    }
            per_ref_kind[ref_kind] = metric_stats
        out['methods'][method] = per_ref_kind

    return out


def _is_nan(x) -> bool:
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return False


if __name__ == '__main__':
    main()
