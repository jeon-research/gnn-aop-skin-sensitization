"""Per-mechanism atom-AUC stratified analysis (Table 8).

Stratifies alignment by Enoch 2008 mechanism class and reports LLNA-minus-
shuffle deltas per mechanism. Pro-hapten is split into pre-hapten (non-
enzymatic autoxidation: phenols, hydroquinones, catechols) and pro-hapten
(enzymatic activation: aromatic amines, eugenol-type).

Inputs:  results/explanations_llna/seed_<N>/explanations.json
         results/explanations_shuffled/seed_<N>/explanations.json
Outputs: results/mechanism_stratified_7method/{llna,shuffled}_stats.json
         results/mechanism_stratified_7method/comparison.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

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

from src.explain.alignment_metrics import atom_auc                           # type: ignore
from senslib.aop_reference_sensitization import (
    AOPReferenceSensitization,
    PROHAPTEN_PATTERNS,
)


from senslib.seeds import ALL as PRIMARY_SEEDS  # noqa: E402
DEFAULT_OUTPUT_DIR = REVISION_ROOT / 'results' / 'mechanism_stratified_7method'


# Split pro_hapten into pre-hapten (non-enzymatic autoxidation) vs
# pro-hapten (enzyme/metabolism activation). These are distinct mechanisms
# per Aptula & Roberts 2006 and must not be conflated.
PREHAPTEN_NAMES = {
    'prohapten_hydroquinone',    # → quinone by air oxidation
    'prohapten_catechol',        # → ortho-quinone
}
PROHAPTEN_ENZYMATIC_NAMES = {
    'prohapten_para_phenylenediamine', # PPD via peroxidase
    'prohapten_aminophenol',           # aminophenol via peroxidase
    'prohapten_allylphenol',           # eugenol via CYP/peroxidase
    'prohapten_aminoquinoline',
}


def _refined_primary_mechanism(smi: str, ref: AOPReferenceSensitization) -> str:
    """Same as ref.classify_mechanism, but splits pro_hapten into pre vs pro."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 'unknown'

    mechanism_hits: Dict[str, List[str]] = defaultdict(list)
    for name, pattern in ref.patterns.items():
        if not mol.GetSubstructMatches(pattern):
            continue
        if name in PREHAPTEN_NAMES:
            mechanism_hits['pre_hapten'].append(name)
        elif name in PROHAPTEN_ENZYMATIC_NAMES:
            mechanism_hits['pro_hapten_enzymatic'].append(name)
        elif name.startswith('prohapten_'):
            mechanism_hits['pro_hapten_other'].append(name)
        else:
            # Reuse the module-level map for non-prohapten mechanisms
            from senslib.aop_reference_sensitization import MECHANISM_MAP
            mech = MECHANISM_MAP.get(name, 'unknown')
            mechanism_hits[mech].append(name)

    if not mechanism_hits:
        return 'none'
    return max(mechanism_hits, key=lambda m: len(mechanism_hits[m]))


def gather_per_mechanism(condition: str, seeds: List[int],
                         ref: AOPReferenceSensitization,
                         expl_dirname: str = None) -> pd.DataFrame:
    """One row per (condition, seed, mechanism, method, molecule)."""
    rows: List[Dict] = []
    expl_root = REVISION_ROOT / 'results' / (expl_dirname or f'explanations_{condition}')
    for seed in seeds:
        path = expl_root / f'seed_{seed}' / 'explanations.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        methods = list(data['explanations'].keys())

        for i, smi in enumerate(data['smiles']):
            rc_mask, _ = ref.get_reactive_center_mask(smi)
            if rc_mask.numel() == 0 or int(rc_mask.sum()) == 0:
                continue  # no reactive centers -> can't compute atom-AUC
            ref_np = rc_mask.numpy()
            mech = _refined_primary_mechanism(smi, ref)

            for method in methods:
                imp = np.asarray(data['explanations'][method][i], dtype=np.float32)
                if len(imp) != len(ref_np):
                    continue
                au = atom_auc(imp, ref_np)
                if np.isnan(au):
                    continue
                rows.append({
                    'condition': condition,
                    'seed':      seed,
                    'method':    method,
                    'mechanism': mech,
                    'smiles':    smi,
                    'atom_auc':  float(au),
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean atom-AUC per (condition, method, mechanism) with bootstrap CI."""
    out_rows: List[Dict] = []
    grouped = df.groupby(['condition', 'method', 'mechanism'])
    for (cond, method, mech), grp in grouped:
        vals = grp['atom_auc'].values
        if len(vals) == 0:
            continue
        n_boot = 10000
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        means = vals[idx].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        out_rows.append({
            'condition': cond,
            'method':    method,
            'mechanism': mech,
            'n_mols':    len(vals),
            'mean_atom_auc': float(vals.mean()),
            'boot_ci_low':   float(lo),
            'boot_ci_high':  float(hi),
        })
    return pd.DataFrame(out_rows)


def comparison_table(summary: pd.DataFrame, focus_method: str = 'attention') -> pd.DataFrame:
    """Side-by-side LLNA vs shuffled for one method."""
    sub = summary[summary['method'] == focus_method].copy()
    pivot = sub.pivot_table(
        index='mechanism',
        columns='condition',
        values=['mean_atom_auc', 'boot_ci_low', 'boot_ci_high', 'n_mols'],
        aggfunc='first',
    )
    pivot.columns = [f'{cond}_{stat}' for stat, cond in pivot.columns]
    pivot = pivot.reset_index()
    if 'llna_mean_atom_auc' in pivot and 'shuffled_mean_atom_auc' in pivot:
        pivot['delta'] = pivot['llna_mean_atom_auc'] - pivot['shuffled_mean_atom_auc']
    return pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llna-dir', default='explanations_llna_7method',
                        help='LLNA explanations directory under results/')
    parser.add_argument('--shuffled-dir', default='explanations_shuffled_7method',
                        help='Shuffled explanations directory under results/')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref = AOPReferenceSensitization()

    print(f'Gathering LLNA condition from {args.llna_dir} ...')
    df_llna = gather_per_mechanism('llna', PRIMARY_SEEDS, ref, args.llna_dir)
    print(f'  {len(df_llna)} rows')

    print(f'Gathering shuffled condition from {args.shuffled_dir} ...')
    df_shuf = gather_per_mechanism('shuffled', PRIMARY_SEEDS, ref, args.shuffled_dir)
    print(f'  {len(df_shuf)} rows')

    combined = pd.concat([df_llna, df_shuf], ignore_index=True)
    print(f'Total rows: {len(combined)}')

    print('Summarising ...')
    summary = summarize(combined)
    summary.to_json(args.output_dir / 'summary.json', orient='records', indent=2)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)

    print('Comparison (attention, LLNA vs shuffled):')
    comp = comparison_table(summary, 'attention')
    comp.to_csv(args.output_dir / 'comparison_attention.csv', index=False)
    print(comp.to_string(index=False))

    print('\nComparison (ensemble, LLNA vs shuffled):')
    comp_ens = comparison_table(summary, 'ensemble')
    comp_ens.to_csv(args.output_dir / 'comparison_ensemble.csv', index=False)
    print(comp_ens.to_string(index=False))

    combined.to_csv(args.output_dir / 'per_molecule_rows.csv', index=False)
    print(f'\nAll outputs in {args.output_dir}')


if __name__ == '__main__':
    main()
