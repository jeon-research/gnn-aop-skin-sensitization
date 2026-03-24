"""
End-to-end MechGNN pipeline: train → extract explanations → compute alignment → stats.

Runs all steps for a grid of seeds × lambda values, then compares to
plain GNN baseline.

Usage:
    python scripts/explain/run_mechgnn_pipeline.py
    python scripts/explain/run_mechgnn_pipeline.py --seeds 42 --lambdas 0.0 0.5 1.0
    python scripts/explain/run_mechgnn_pipeline.py --skip-training  # reuse existing checkpoints
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RESULTS_DIR
from src.explain.utils import ALL_SEEDS


DEFAULT_LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0]


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n  >> {description}")
    print(f"     {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"     [{status}] {elapsed:.0f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='MechGNN end-to-end pipeline')
    parser.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    parser.add_argument('--lambdas', type=float, nargs='+', default=DEFAULT_LAMBDAS)
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, use existing checkpoints')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train from scratch (ablation)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    total_start = time.time()
    n_models = len(args.seeds) * len(args.lambdas)
    print(f"MechGNN Pipeline")
    print(f"  Seeds: {args.seeds}")
    print(f"  Lambdas: {args.lambdas}")
    print(f"  Total models: {n_models}")
    print(f"  From scratch: {args.from_scratch}")

    # Phase 1: Train MechGNN models
    if not args.skip_training:
        print(f"\n{'='*70}")
        print(f"Phase 1: Training MechGNN ({n_models} models)")
        print(f"{'='*70}")

        train_results = {}
        for lam in args.lambdas:
            for seed in args.seeds:
                output_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / f'seed_{seed}'
                if (output_dir / 'best_model.pt').exists():
                    print(f"\n  Skipping λ={lam}, seed={seed} (already trained)")
                    train_results[(lam, seed)] = True
                    continue

                cmd = [
                    sys.executable, 'scripts/explain/train_mechgnn.py',
                    '--seed', str(seed),
                    '--lambda-mie', str(lam),
                    '--epochs', str(args.epochs),
                    '--batch-size', str(args.batch_size),
                ]
                if args.from_scratch:
                    cmd.append('--from-scratch')

                success = run_command(cmd, f"Training λ={lam}, seed={seed}")
                train_results[(lam, seed)] = success

                if not success:
                    print(f"  WARNING: Training failed for λ={lam}, seed={seed}")

        n_ok = sum(1 for v in train_results.values() if v)
        print(f"\n  Training complete: {n_ok}/{n_models} succeeded")
    else:
        print("\n  Skipping training phase")

    # Phase 2: Extract explanations for each trained MechGNN
    print(f"\n{'='*70}")
    print(f"Phase 2: Extract explanations")
    print(f"{'='*70}")

    for lam in args.lambdas:
        # Check if explanations already exist for all seeds of this lambda
        expl_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / 'explanations'
        seeds_to_extract = []
        for seed in args.seeds:
            checkpoint = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / f'seed_{seed}' / 'best_model.pt'
            expl_file = expl_dir / f'seed_{seed}' / 'explanations.json'
            if not checkpoint.exists():
                print(f"  Skipping λ={lam}, seed={seed} (no checkpoint)")
                continue
            if expl_file.exists():
                print(f"  Skipping λ={lam}, seed={seed} (explanations exist)")
                continue
            seeds_to_extract.append(seed)

        if not seeds_to_extract:
            continue

        cmd = [
            sys.executable, 'scripts/explain/extract_explanations.py',
            '--seeds', *[str(s) for s in seeds_to_extract],
            '--model-type', 'mechgnn',
            '--mechgnn-lambda', str(lam),
        ]
        run_command(cmd, f"Extracting explanations for λ={lam}, seeds={seeds_to_extract}")

    # Phase 3: Compute alignment metrics for each lambda
    print(f"\n{'='*70}")
    print(f"Phase 3: Compute alignment")
    print(f"{'='*70}")

    for lam in args.lambdas:
        expl_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / 'explanations'
        align_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / 'alignment'

        if not expl_dir.exists():
            print(f"  Skipping λ={lam} (no explanations)")
            continue

        cmd = [
            sys.executable, 'scripts/explain/compute_alignment.py',
            '--seeds', *[str(s) for s in args.seeds],
            '--explanations-dir', str(expl_dir),
            '--output-dir', str(align_dir),
        ]
        run_command(cmd, f"Computing alignment for λ={lam}")

    # Phase 4: Statistical analysis — compare MechGNN lambdas to plain GNN
    print(f"\n{'='*70}")
    print(f"Phase 4: Statistical analysis")
    print(f"{'='*70}")

    # Run standard stats for each lambda
    for lam in args.lambdas:
        align_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / 'alignment'
        stats_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / 'statistical_analysis'

        if not align_dir.exists():
            continue

        cmd = [
            sys.executable, 'scripts/explain/statistical_analysis.py',
            '--seeds', *[str(s) for s in args.seeds],
            '--alignment-dir', str(align_dir),
            '--output-dir', str(stats_dir),
        ]
        run_command(cmd, f"Statistical analysis for λ={lam}")

    # Phase 5: Cross-lambda comparison summary
    print(f"\n{'='*70}")
    print(f"Phase 5: Cross-lambda comparison")
    print(f"{'='*70}")
    print_lambda_comparison(args.lambdas, args.seeds)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Pipeline complete in {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")


def print_lambda_comparison(lambdas, seeds):
    """Print a summary comparing results across lambda values."""
    import json
    import numpy as np

    print(f"\n  {'Lambda':>8} | {'Test AUC':>10} | {'Test AP':>10} | {'MIE AUC':>10} | "
          f"{'atom-AUC (IG)':>14} | {'atom-AUC (ens)':>15}")
    print(f"  {'-'*75}")

    for lam in lambdas:
        # Collect training results
        test_aucs = []
        test_aps = []
        mie_aucs = []
        for seed in seeds:
            results_path = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / f'seed_{seed}' / 'results.json'
            if results_path.exists():
                with open(results_path) as f:
                    r = json.load(f)
                    test_aucs.append(r['test_sens_auc'])
                    test_aps.append(r['test_sens_ap'])
                    mie_aucs.append(r['test_mie_auc'])

        # Collect alignment results
        ig_aucs = []
        ens_aucs = []
        align_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{lam}' / 'alignment'
        if align_dir.exists():
            agg_path = align_dir / 'aggregated_metrics.json'
            if agg_path.exists():
                with open(agg_path) as f:
                    agg = json.load(f)
                    if 'ig' in agg and 'mean_atom_auc' in agg['ig']:
                        ig_aucs.append(agg['ig']['mean_atom_auc']['mean'])
                    if 'ensemble' in agg and 'mean_atom_auc' in agg['ensemble']:
                        ens_aucs.append(agg['ensemble']['mean_atom_auc']['mean'])

        def fmt(vals):
            if vals:
                return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
            return "N/A"

        ig_str = f"{ig_aucs[0]:.3f}" if ig_aucs else "N/A"
        ens_str = f"{ens_aucs[0]:.3f}" if ens_aucs else "N/A"

        print(f"  {lam:>8.1f} | {fmt(test_aucs):>10} | {fmt(test_aps):>10} | "
              f"{fmt(mie_aucs):>10} | {ig_str:>14} | {ens_str:>15}")

    # Also show plain GNN baseline
    plain_align_dir = RESULTS_DIR / 'alignment'
    if plain_align_dir.exists():
        agg_path = plain_align_dir / 'aggregated_metrics.json'
        if agg_path.exists():
            with open(agg_path) as f:
                agg = json.load(f)
            ig_str = "N/A"
            ens_str = "N/A"
            if 'ig' in agg and 'mean_atom_auc' in agg['ig']:
                ig_str = f"{agg['ig']['mean_atom_auc']['mean']:.3f}"
            if 'ensemble' in agg and 'mean_atom_auc' in agg['ensemble']:
                ens_str = f"{agg['ensemble']['mean_atom_auc']['mean']:.3f}"
            print(f"  {'plain':>8} | {'~0.857':>10} | {'N/A':>10} | "
                  f"{'N/A':>10} | {ig_str:>14} | {ens_str:>15}")


if __name__ == '__main__':
    main()
