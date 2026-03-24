"""
Conformal prediction calibration and evaluation.

Calibrates conformal predictor on validation set, evaluates coverage
on test set, and identifies "confident explanations" for alignment analysis.

Usage:
    python scripts/explain/run_conformal.py [--seeds 42 123 ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROJECT_ROOT, RESULTS_DIR
from src.explain.utils import (
    ALL_SEEDS, load_model, load_dataset, featurize_molecules, setup_device,
)
from src.modeling.causal_aop_gnn import MoleculeFeaturizer
from src.explain.conformal import ConformalPredictor
from src.explain.aop_reference import AOPReference
from src.explain.alignment_metrics import compute_batch_alignment


ALPHAS = [0.05, 0.10, 0.20]


def run_conformal_for_seed(
    seed: int,
    device: torch.device,
    output_dir: Path,
    explanations_dir: Path,
    aop_ref: AOPReference,
):
    """Run conformal prediction for one seed."""
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    # Load model
    model = load_model(seed, device)

    # Load data and split
    df, train_idx, val_idx, test_idx = load_dataset(seed)

    # Featurize val and test molecules
    def featurize_split(indices):
        graphs = []
        labels = []
        smiles_out = []
        valid_idx = []
        for i in indices:
            smi = df.loc[i, 'smiles']
            graph = MoleculeFeaturizer.smiles_to_graph(smi)
            if graph is not None:
                graphs.append(graph)
                labels.append(float(df.loc[i, 'sensitization_human']))
                smiles_out.append(smi)
                valid_idx.append(i)
        return graphs, np.array(labels), smiles_out, valid_idx

    val_graphs, val_labels, val_smiles, _ = featurize_split(val_idx)
    test_graphs, test_labels, test_smiles, _ = featurize_split(test_idx)

    print(f"  Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Calibrate conformal predictor
    cp = ConformalPredictor(model, target_key='sensitization')
    cp.calibrate(val_graphs, val_labels, device=device)
    print(f"  Calibrated with {len(val_graphs)} molecules")

    # Evaluate coverage at multiple alpha levels
    coverage_results = cp.evaluate_coverage(test_graphs, test_labels, alphas=ALPHAS, device=device)

    print("\n  Conformal Prediction Results:")
    print(f"  {'Alpha':>6} {'Target':>8} {'Empirical':>10} {'Avg Size':>9} {'Singleton':>10}")
    print(f"  {'-'*50}")
    for key, metrics in coverage_results.items():
        print(f"  {metrics['alpha']:>6.2f} "
              f"{metrics['target_coverage']:>8.2f} "
              f"{metrics['empirical_coverage']:>10.3f} "
              f"{metrics['avg_set_size']:>9.3f} "
              f"{metrics['singleton_rate']:>10.3f}")

    # "Confident explanations" analysis
    # Load saved explanations if available
    confident_alignment = {}
    expl_path = explanations_dir / f'seed_{seed}' / 'explanations.json'

    if expl_path.exists():
        with open(expl_path) as f:
            expl_data = json.load(f)

        # Get confident mask at alpha=0.10
        confident_mask = cp.get_confident_mask(test_graphs, alpha=0.10, device=device)
        uncertain_mask = ~confident_mask

        print(f"\n  Confident explanations (alpha=0.10): {confident_mask.sum()} / {len(test_graphs)}")

        # Get AOP masks for test set
        aop_data = aop_ref.annotate_dataset(test_smiles, test_labels)
        aop_masks = aop_data['masks']

        # Compare alignment for confident vs uncertain
        methods = [m for m in expl_data['explanations'].keys() if m != 'ensemble']
        expl_smiles = expl_data['smiles']

        # Match test_smiles to explanation indices
        expl_smi_to_idx = {s: i for i, s in enumerate(expl_smiles)}

        for method_name in methods:
            explanations = expl_data['explanations'][method_name]

            for subset_name, subset_mask in [('confident', confident_mask), ('uncertain', uncertain_mask)]:
                importances = []
                references = []

                for i in range(len(test_smiles)):
                    if not subset_mask[i]:
                        continue
                    if test_labels[i] != 1:
                        continue
                    if aop_masks[i].sum() == 0:
                        continue

                    smi = test_smiles[i]
                    if smi in expl_smi_to_idx:
                        expl_idx = expl_smi_to_idx[smi]
                        imp = torch.tensor(explanations[expl_idx], dtype=torch.float32)
                        ref = aop_masks[i]
                        if imp.numel() == ref.numel():
                            importances.append(imp)
                            references.append(ref)

                if importances:
                    metrics = compute_batch_alignment(importances, references)
                    confident_alignment[f'{method_name}_{subset_name}'] = metrics
                    auc = metrics.get('mean_atom_auc', float('nan'))
                    print(f"    {method_name} ({subset_name}, n={len(importances)}): atom-AUC={auc:.4f}")

    # Save results
    seed_dir = output_dir / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    results = {
        'seed': seed,
        'coverage': to_serializable(coverage_results),
        'confident_alignment': to_serializable(confident_alignment),
    }

    with open(seed_dir / 'conformal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {seed_dir / 'conformal_results.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run conformal prediction')
    parser.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'conformal'))
    parser.add_argument('--explanations-dir', type=str,
                        default=str(RESULTS_DIR / 'explanations'))
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    explanations_dir = Path(args.explanations_dir)
    aop_ref = AOPReference(use_extended=True)

    all_results = []
    for seed in args.seeds:
        try:
            result = run_conformal_for_seed(
                seed, device, output_dir, explanations_dir, aop_ref
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"Summary ({len(all_results)} seeds)")
        print(f"{'='*60}")

        for alpha_key in [f'alpha_{a}' for a in ALPHAS]:
            coverages = [r['coverage'][alpha_key]['empirical_coverage']
                         for r in all_results if alpha_key in r['coverage']]
            if coverages:
                print(f"  {alpha_key}: coverage={np.mean(coverages):.3f} +/- {np.std(coverages):.3f}")


if __name__ == '__main__':
    main()
