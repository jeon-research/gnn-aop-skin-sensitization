"""
Compare GNN explanations to AOP reference labels.

Computes alignment metrics (atom-AUC, HitRate@K, Precision@K, IoU)
for each explanation method, stratified by mechanism type and other factors.

Uses reactive center masks (strict) rather than full substructure masks.
Reports TP-only alignment as primary metric (sensitizers correctly predicted).

Usage:
    python scripts/explain/compute_alignment.py [--seeds 42 123 ...] [--explanations-dir results/explanations]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROJECT_ROOT, RESULTS_DIR
from src.explain.aop_reference import AOPReference
from src.explain.alignment_metrics import (
    compute_alignment_metrics,
    compute_batch_alignment,
    compute_stratified_alignment,
)
from src.explain.utils import ALL_SEEDS


def load_explanations(seed: int, explanations_dir: Path) -> dict:
    """Load saved explanations for a seed."""
    path = explanations_dir / f'seed_{seed}' / 'explanations.json'
    if not path.exists():
        raise FileNotFoundError(f"Explanations not found: {path}")
    with open(path) as f:
        return json.load(f)


def compute_alignment_for_seed(
    seed: int,
    explanations_dir: Path,
    aop_ref: AOPReference,
    output_dir: Path,
):
    """Compute alignment metrics for one seed."""
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    data = load_explanations(seed, explanations_dir)
    smiles_list = data['smiles']
    labels = np.array(data['labels'])
    predictions = np.array(data['predictions'])
    methods = {k: v for k, v in data['explanations'].items()}

    print(f"  {len(smiles_list)} molecules, {len(methods)} methods")

    # Get AOP reference masks — use REACTIVE CENTER (strict) masks
    reactive_masks = []
    full_masks = []
    mechanisms = []
    for smi in smiles_list:
        rc_mask, rc_info = aop_ref.get_reactive_center_mask(smi)
        full_mask, full_info = aop_ref.get_atom_mask(smi)
        mech = aop_ref.classify_mechanism(smi)
        reactive_masks.append(rc_mask)
        full_masks.append(full_mask)
        mechanisms.append(mech)

    # Coverage stats
    n_with_rc = sum(1 for m in reactive_masks if m.numel() > 0 and m.sum() > 0)
    n_with_full = sum(1 for m in full_masks if m.numel() > 0 and m.sum() > 0)
    sens_mask_np = labels == 1
    n_sens = sens_mask_np.sum()
    n_sens_with_rc = sum(1 for m, s in zip(reactive_masks, sens_mask_np) if s and m.numel() > 0 and m.sum() > 0)

    print(f"  Reactive center coverage: {n_with_rc}/{len(smiles_list)} ({n_with_rc/len(smiles_list):.1%})")
    print(f"  Full substructure coverage: {n_with_full}/{len(smiles_list)} ({n_with_full/len(smiles_list):.1%})")
    print(f"  Sensitizers with reactive centers: {n_sens_with_rc}/{n_sens}")

    # Primary filter: TP sensitizers with reactive center atoms
    pred_classes = (predictions > 0.5).astype(int)
    tp_mask = (labels == 1) & (pred_classes == 1) & np.array([m.sum() > 0 for m in reactive_masks])
    tp_indices = np.where(tp_mask)[0]

    # Also compute for all sensitizers with reactive centers (including FN)
    all_sens_mask = (labels == 1) & np.array([m.sum() > 0 for m in reactive_masks])
    all_sens_indices = np.where(all_sens_mask)[0]

    print(f"  TP with reactive centers (PRIMARY): {len(tp_indices)}")
    print(f"  All sensitizers with reactive centers: {len(all_sens_indices)}")

    all_results = {}

    for method_name, explanations in methods.items():
        # --- PRIMARY: TP-only with reactive center mask ---
        if len(tp_indices) > 0:
            tp_imps = [torch.tensor(explanations[i], dtype=torch.float32) for i in tp_indices]
            tp_refs = [reactive_masks[i] for i in tp_indices]
            tp_metrics = compute_batch_alignment(tp_imps, tp_refs)
            all_results[method_name] = tp_metrics  # This IS the primary metric now

            print(f"\n  {method_name} (TP, reactive center, n={len(tp_indices)}):")
            for key in ['mean_atom_auc', 'mean_atom_ap', 'mean_hit_rate_at_k', 'mean_precision_at_3']:
                if key in tp_metrics:
                    val = tp_metrics[key]
                    std_key = key.replace('mean_', 'std_')
                    std = tp_metrics.get(std_key, 0)
                    print(f"    {key}: {val:.4f} +/- {std:.4f}")

        # --- SECONDARY: All sensitizers with reactive center mask ---
        if len(all_sens_indices) > 0:
            all_imps = [torch.tensor(explanations[i], dtype=torch.float32) for i in all_sens_indices]
            all_refs = [reactive_masks[i] for i in all_sens_indices]
            all_metrics = compute_batch_alignment(all_imps, all_refs)
            all_results[f'{method_name}_all_sensitizers'] = all_metrics

        # --- TERTIARY: TP-only with full substructure mask (for comparison) ---
        tp_full_mask = (labels == 1) & (pred_classes == 1) & np.array([m.sum() > 0 for m in full_masks])
        tp_full_indices = np.where(tp_full_mask)[0]
        if len(tp_full_indices) > 0:
            full_imps = [torch.tensor(explanations[i], dtype=torch.float32) for i in tp_full_indices]
            full_refs = [full_masks[i] for i in tp_full_indices]
            full_metrics = compute_batch_alignment(full_imps, full_refs)
            all_results[f'{method_name}_full_substructure'] = full_metrics

        # Stratified by mechanism (TP only, reactive center)
        if len(tp_indices) > 0:
            strata = [mechanisms[i]['primary_mechanism'] for i in tp_indices]
            tp_imps = [torch.tensor(explanations[i], dtype=torch.float32) for i in tp_indices]
            tp_refs = [reactive_masks[i] for i in tp_indices]
            stratified = compute_stratified_alignment(tp_imps, tp_refs, strata)
            all_results[f'{method_name}_by_mechanism'] = {k: v for k, v in stratified.items()}

        # Stratified by direct vs pro-hapten
        if len(tp_indices) > 0:
            direct_idx = []
            prohapten_idx = []
            for i in tp_indices:
                pm = mechanisms[i]['primary_mechanism']
                if pm == 'pro_hapten':
                    prohapten_idx.append(i)
                else:
                    direct_idx.append(i)

            for name, idx_list in [('direct', direct_idx), ('pro_hapten', prohapten_idx)]:
                if idx_list:
                    imps = [torch.tensor(explanations[i], dtype=torch.float32) for i in idx_list]
                    refs = [reactive_masks[i] for i in idx_list]
                    metrics = compute_batch_alignment(imps, refs)
                    all_results[f'{method_name}_{name}'] = metrics

    # Stratify by prediction confidence (TP only)
    for method_name, explanations in methods.items():
        high_conf = tp_mask & (predictions > 0.8)
        low_conf = tp_mask & (predictions >= 0.2) & (predictions <= 0.8)

        for name, conf_mask in [('high_confidence', high_conf), ('low_confidence', low_conf)]:
            if conf_mask.sum() > 0:
                idx = np.where(conf_mask)[0]
                imps = [torch.tensor(explanations[i], dtype=torch.float32) for i in idx]
                refs = [reactive_masks[i] for i in idx]
                metrics = compute_batch_alignment(imps, refs)
                all_results[f'{method_name}_{name}'] = metrics

    # Stratify by molecule size (TP only)
    for method_name, explanations in methods.items():
        if len(tp_indices) > 0:
            sizes = np.array([len(explanations[i]) for i in tp_indices])
            small = sizes < 15
            medium = (sizes >= 15) & (sizes <= 30)
            large = sizes > 30

            for name, size_mask in [('small', small), ('medium', medium), ('large', large)]:
                sub_idx = tp_indices[size_mask]
                if len(sub_idx) > 0:
                    imps = [torch.tensor(explanations[i], dtype=torch.float32) for i in sub_idx]
                    refs = [reactive_masks[i] for i in sub_idx]
                    metrics = compute_batch_alignment(imps, refs)
                    all_results[f'{method_name}_size_{name}'] = metrics

    # Save results
    seed_dir = output_dir / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj

    output_path = seed_dir / 'alignment_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

    # Save AOP summary
    aop_summary = {
        'n_molecules': len(smiles_list),
        'n_sensitizers': int(n_sens),
        'n_with_reactive_center': n_with_rc,
        'n_with_full_substructure': n_with_full,
        'n_tp_with_reactive_center': len(tp_indices),
        'n_all_sens_with_reactive_center': len(all_sens_indices),
    }
    aop_path = seed_dir / 'aop_summary.json'
    with open(aop_path, 'w') as f:
        json.dump(to_serializable(aop_summary), f, indent=2)

    print(f"\n  Saved to {output_path}")
    return all_results


def aggregate_across_seeds(seeds: list, output_dir: Path):
    """Aggregate alignment metrics across seeds."""
    print(f"\n{'='*60}")
    print("Aggregating across seeds")
    print(f"{'='*60}")

    all_seed_results = {}
    for seed in seeds:
        path = output_dir / f'seed_{seed}' / 'alignment_metrics.json'
        if path.exists():
            with open(path) as f:
                all_seed_results[seed] = json.load(f)

    if not all_seed_results:
        print("  No results to aggregate")
        return

    # Aggregate per-method metrics
    method_names = ['ig', 'gradcam', 'attention', 'gnnexplainer', 'ensemble']
    metric_keys = ['mean_atom_auc', 'mean_atom_ap', 'mean_hit_rate_at_k',
                   'mean_precision_at_3', 'mean_precision_at_5', 'mean_iou_at_k']

    summary = {}
    for method in method_names:
        method_values = {}
        for metric in metric_keys:
            values = []
            for seed, results in all_seed_results.items():
                if method in results and metric in results[method]:
                    val = results[method][metric]
                    if not np.isnan(val):
                        values.append(val)
            if values:
                method_values[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'n_seeds': len(values),
                }
        if method_values:
            summary[method] = method_values

    # Also aggregate direct vs pro_hapten splits
    for suffix in ['_direct', '_pro_hapten', '_full_substructure', '_all_sensitizers']:
        for method in method_names:
            key = f'{method}{suffix}'
            method_values = {}
            for metric in metric_keys:
                values = []
                for seed, results in all_seed_results.items():
                    if key in results and metric in results[key]:
                        val = results[key][metric]
                        if not np.isnan(val):
                            values.append(val)
                if values:
                    method_values[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'n_seeds': len(values),
                    }
            if method_values:
                summary[key] = method_values

    # Print summary table
    print(f"\n  PRIMARY: TP + Reactive Center Mask")
    print(f"  {'Method':<15} {'atom-AUC':>12} {'AP':>12} {'HitRate@K':>12} {'P@3':>12} {'IoU@K':>12}")
    print(f"  {'-'*65}")
    for method in method_names:
        if method in summary:
            s = summary[method]
            auc = s.get('mean_atom_auc', {})
            ap = s.get('mean_atom_ap', {})
            hr = s.get('mean_hit_rate_at_k', {})
            p3 = s.get('mean_precision_at_3', {})
            iou = s.get('mean_iou_at_k', {})
            print(f"  {method:<15} "
                  f"{auc.get('mean', float('nan')):>5.3f}+/-{auc.get('std', 0):>4.3f} "
                  f"{ap.get('mean', float('nan')):>5.3f}+/-{ap.get('std', 0):>4.3f} "
                  f"{hr.get('mean', float('nan')):>5.3f}+/-{hr.get('std', 0):>4.3f} "
                  f"{p3.get('mean', float('nan')):>5.3f}+/-{p3.get('std', 0):>4.3f} "
                  f"{iou.get('mean', float('nan')):>5.3f}+/-{iou.get('std', 0):>4.3f}")

    # Print direct vs pro-hapten comparison
    print(f"\n  Direct vs Pro-hapten (atom-AUC):")
    for method in method_names:
        direct_key = f'{method}_direct'
        pro_key = f'{method}_pro_hapten'
        d_val = summary.get(direct_key, {}).get('mean_atom_auc', {}).get('mean', float('nan'))
        p_val = summary.get(pro_key, {}).get('mean_atom_auc', {}).get('mean', float('nan'))
        print(f"    {method:<15} direct={d_val:.3f}  pro_hapten={p_val:.3f}")

    # Save aggregated summary
    with open(output_dir / 'aggregated_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved aggregated metrics to {output_dir / 'aggregated_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description='Compute GNN-AOP alignment')
    parser.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    parser.add_argument('--explanations-dir', type=str,
                        default=str(RESULTS_DIR / 'explanations'))
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'alignment'))
    parser.add_argument('--use-extended', action='store_true', default=True,
                        help='Use extended SMARTS patterns')
    args = parser.parse_args()

    explanations_dir = Path(args.explanations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aop_ref = AOPReference(use_extended=args.use_extended)

    for seed in args.seeds:
        try:
            compute_alignment_for_seed(seed, explanations_dir, aop_ref, output_dir)
        except Exception as e:
            print(f"ERROR processing seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    aggregate_across_seeds(args.seeds, output_dir)


if __name__ == '__main__':
    main()
