"""
Statistical analysis of GNN-AOP alignment results.

Bootstrap confidence intervals, hypothesis tests, and summary tables.

Usage:
    python scripts/explain/statistical_analysis.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RESULTS_DIR
from src.explain.utils import ALL_SEEDS
METHODS = ['ig', 'gradcam', 'attention', 'gnnexplainer', 'pgexplainer', 'graphmask', 'ensemble']


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha)
    upper = np.percentile(boot_means, 100 * (1 - alpha))

    return float(np.mean(values)), float(lower), float(upper)


def load_alignment_results(alignment_dir: Path, seeds: list) -> dict:
    """Load alignment results across seeds."""
    results = {}
    for seed in seeds:
        path = alignment_dir / f'seed_{seed}' / 'alignment_metrics.json'
        if path.exists():
            with open(path) as f:
                results[seed] = json.load(f)
    return results


def load_fingerprint_results(fp_dir: Path, seeds: list) -> dict:
    """Load fingerprint baseline results across seeds."""
    results = {}
    for seed in seeds:
        path = fp_dir / f'seed_{seed}' / 'fingerprint_baseline.json'
        if path.exists():
            with open(path) as f:
                results[seed] = json.load(f)
    return results


def load_conformal_results(conformal_dir: Path, seeds: list) -> dict:
    """Load conformal prediction results across seeds."""
    results = {}
    for seed in seeds:
        path = conformal_dir / f'seed_{seed}' / 'conformal_results.json'
        if path.exists():
            with open(path) as f:
                results[seed] = json.load(f)
    return results


def main_comparison_table(alignment_results: dict, output_dir: Path):
    """Table 1: Main method comparison — TP + reactive center mask."""
    print("\n" + "="*70)
    print("TABLE 1: Explanation Method Alignment (TP, Reactive Center)")
    print("="*70)

    metrics = ['mean_atom_auc', 'mean_atom_ap', 'mean_hit_rate_at_k', 'mean_iou_at_k']
    metric_labels = ['Atom-AUC', 'Atom-AP', 'HitRate@K', 'IoU@K']

    rows = []
    for method in METHODS:
        row = {'Method': method}
        for metric, label in zip(metrics, metric_labels):
            values = []
            for seed, results in alignment_results.items():
                if method in results and metric in results[method]:
                    v = results[method][metric]
                    if not np.isnan(v):
                        values.append(v)

            if values:
                mean, ci_lo, ci_hi = bootstrap_ci(values)
                row[label] = f"{mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
                row[f'{label}_mean'] = mean
            else:
                row[label] = "N/A"
                row[f'{label}_mean'] = float('nan')
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df[['Method'] + metric_labels].to_string(index=False))

    df.to_csv(output_dir / 'table1_method_comparison.csv', index=False)
    return df


def gnn_vs_fingerprint_test(alignment_results: dict, fp_results: dict, output_dir: Path):
    """Statistical test: GNN vs Fingerprint baseline."""
    print("\n" + "="*70)
    print("TABLE 2: GNN vs Fingerprint Baseline (Wilcoxon Signed-Rank)")
    print("="*70)

    results = []
    for method in METHODS:
        gnn_aucs = []
        fp_aucs = []

        for seed in alignment_results:
            if method in alignment_results[seed]:
                gnn_auc = alignment_results[seed][method].get('mean_atom_auc', float('nan'))
                if not np.isnan(gnn_auc) and seed in fp_results:
                    fp_auc = fp_results[seed]['alignment'].get('mean_atom_auc', float('nan'))
                    if not np.isnan(fp_auc):
                        gnn_aucs.append(gnn_auc)
                        fp_aucs.append(fp_auc)

        if len(gnn_aucs) >= 3:
            stat, p_value = stats.wilcoxon(gnn_aucs, fp_aucs, alternative='two-sided')
            diff = np.mean(gnn_aucs) - np.mean(fp_aucs)

            # Bonferroni correction (5 methods)
            p_corrected = min(p_value * len(METHODS), 1.0)

            results.append({
                'Method': method,
                'GNN atom-AUC': f"{np.mean(gnn_aucs):.3f}",
                'FP atom-AUC': f"{np.mean(fp_aucs):.3f}",
                'Diff': f"{diff:+.3f}",
                'p-value': f"{p_value:.4f}",
                'p-corrected': f"{p_corrected:.4f}",
                'Significant': '*' if p_corrected < 0.05 else '',
            })

    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv(output_dir / 'table2_gnn_vs_fingerprint.csv', index=False)


def mechanism_analysis(alignment_results: dict, output_dir: Path):
    """Table 3: Alignment by reaction mechanism."""
    print("\n" + "="*70)
    print("TABLE 3: Alignment by Reaction Mechanism (best method per mechanism)")
    print("="*70)

    mechanism_data = {}

    for method in METHODS:
        mech_key = f'{method}_by_mechanism'
        for seed, results in alignment_results.items():
            if mech_key in results:
                for mechanism, metrics in results[mech_key].items():
                    if mechanism not in mechanism_data:
                        mechanism_data[mechanism] = {}
                    if method not in mechanism_data[mechanism]:
                        mechanism_data[mechanism][method] = []

                    auc = metrics.get('mean_atom_auc', float('nan'))
                    if not np.isnan(auc):
                        mechanism_data[mechanism][method].append(auc)

    rows = []
    for mechanism in sorted(mechanism_data.keys()):
        row = {'Mechanism': mechanism}
        best_method = None
        best_auc = -1

        for method in METHODS:
            if method in mechanism_data[mechanism]:
                values = mechanism_data[mechanism][method]
                if values:
                    mean_auc = np.mean(values)
                    row[method] = f"{mean_auc:.3f}"
                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_method = method
                else:
                    row[method] = "N/A"
            else:
                row[method] = "N/A"

        row['Best'] = best_method if best_method else "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv(output_dir / 'table3_mechanism_analysis.csv', index=False)


def direct_vs_prohapten(alignment_results: dict, output_dir: Path):
    """Table 3b: Direct electrophile vs pro-hapten alignment."""
    print("\n" + "="*70)
    print("TABLE 3b: Direct Electrophile vs Pro-hapten Alignment")
    print("="*70)

    rows = []
    for method in METHODS:
        direct_vals = []
        pro_vals = []

        for seed, results in alignment_results.items():
            dk = f'{method}_direct'
            pk = f'{method}_pro_hapten'
            if dk in results and 'mean_atom_auc' in results[dk]:
                v = results[dk]['mean_atom_auc']
                if not np.isnan(v):
                    direct_vals.append(v)
            if pk in results and 'mean_atom_auc' in results[pk]:
                v = results[pk]['mean_atom_auc']
                if not np.isnan(v):
                    pro_vals.append(v)

        row = {'Method': method}
        if direct_vals:
            mean, lo, hi = bootstrap_ci(direct_vals)
            row['Direct AUC'] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
            row['Direct_mean'] = mean
        else:
            row['Direct AUC'] = "N/A"
            row['Direct_mean'] = float('nan')

        if pro_vals:
            mean, lo, hi = bootstrap_ci(pro_vals)
            row['Pro-hapten AUC'] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
            row['ProHapten_mean'] = mean
        else:
            row['Pro-hapten AUC'] = "N/A"
            row['ProHapten_mean'] = float('nan')

        if direct_vals and pro_vals and len(direct_vals) >= 3 and len(pro_vals) >= 3:
            # Paired test if same number of seeds
            if len(direct_vals) == len(pro_vals):
                stat, p_value = stats.wilcoxon(direct_vals, pro_vals, alternative='two-sided')
                row['p-value'] = f"{p_value:.4f}"
            else:
                stat, p_value = stats.mannwhitneyu(direct_vals, pro_vals, alternative='two-sided')
                row['p-value'] = f"{p_value:.4f}"
        else:
            row['p-value'] = "N/A"

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        print(df[['Method', 'Direct AUC', 'Pro-hapten AUC', 'p-value']].to_string(index=False))
        df.to_csv(output_dir / 'table3b_direct_vs_prohapten.csv', index=False)


def conformal_analysis(conformal_results: dict, output_dir: Path):
    """Table 4: Conformal prediction coverage and confident explanations."""
    print("\n" + "="*70)
    print("TABLE 4: Conformal Prediction Coverage")
    print("="*70)

    rows = []
    for alpha_key in ['alpha_0.05', 'alpha_0.1', 'alpha_0.2']:
        coverages = []
        set_sizes = []
        singleton_rates = []

        for seed, results in conformal_results.items():
            if alpha_key in results['coverage']:
                metrics = results['coverage'][alpha_key]
                coverages.append(metrics['empirical_coverage'])
                set_sizes.append(metrics['avg_set_size'])
                singleton_rates.append(metrics['singleton_rate'])

        if coverages:
            alpha = float(alpha_key.split('_')[1])
            rows.append({
                'Alpha': alpha,
                'Target': f"{1-alpha:.2f}",
                'Coverage': f"{np.mean(coverages):.3f} +/- {np.std(coverages):.3f}",
                'Avg Set Size': f"{np.mean(set_sizes):.3f} +/- {np.std(set_sizes):.3f}",
                'Singleton Rate': f"{np.mean(singleton_rates):.3f} +/- {np.std(singleton_rates):.3f}",
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv(output_dir / 'table4_conformal.csv', index=False)

    # Confident vs Uncertain alignment comparison
    print("\n  Confident vs Uncertain Explanations:")
    for method in ['ig', 'gradcam', 'attention', 'gnnexplainer']:
        conf_values = []
        unc_values = []

        for seed, results in conformal_results.items():
            ca = results.get('confident_alignment', {})
            if f'{method}_confident' in ca:
                v = ca[f'{method}_confident'].get('mean_atom_auc', float('nan'))
                if not np.isnan(v):
                    conf_values.append(v)
            if f'{method}_uncertain' in ca:
                v = ca[f'{method}_uncertain'].get('mean_atom_auc', float('nan'))
                if not np.isnan(v):
                    unc_values.append(v)

        if conf_values and unc_values:
            print(f"    {method}: confident={np.mean(conf_values):.3f}, "
                  f"uncertain={np.mean(unc_values):.3f}, "
                  f"diff={np.mean(conf_values)-np.mean(unc_values):+.3f}")


def chance_level_test(alignment_results: dict, output_dir: Path):
    """Test if atom-AUC is significantly above chance (0.5)."""
    print("\n" + "="*70)
    print("Hypothesis Test: Atom-AUC > 0.5 (chance level)")
    print("="*70)

    rows = []
    for method in METHODS:
        values = []
        for seed, results in alignment_results.items():
            if method in results:
                v = results[method].get('mean_atom_auc', float('nan'))
                if not np.isnan(v):
                    values.append(v)

        if len(values) >= 3:
            t_stat, p_value = stats.ttest_1samp(values, 0.5)
            # One-sided test (greater than 0.5)
            p_onesided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
            # Bonferroni correction
            p_corrected = min(p_onesided * len(METHODS), 1.0)

            sig = '***' if p_corrected < 0.001 else '**' if p_corrected < 0.01 else '*' if p_corrected < 0.05 else 'ns'
            print(f"  {method:<15}: {np.mean(values):.3f} +/- {np.std(values):.3f}, "
                  f"t={t_stat:.2f}, p={p_onesided:.4f}, p_corr={p_corrected:.4f} {sig}")

            rows.append({
                'Method': method,
                'Mean': f"{np.mean(values):.3f}",
                'Std': f"{np.std(values):.3f}",
                't-stat': f"{t_stat:.2f}",
                'p (one-sided)': f"{p_onesided:.4f}",
                'p (Bonferroni)': f"{p_corrected:.4f}",
                'Significant': sig,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_dir / 'table5_chance_level_test.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis')
    parser.add_argument('--alignment-dir', type=str,
                        default=str(RESULTS_DIR / 'alignment'))
    parser.add_argument('--fp-dir', type=str,
                        default=str(RESULTS_DIR / 'fingerprint_baseline'))
    parser.add_argument('--conformal-dir', type=str,
                        default=str(RESULTS_DIR / 'conformal'))
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'statistical_analysis'))
    parser.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_results = load_alignment_results(Path(args.alignment_dir), args.seeds)
    fp_results = load_fingerprint_results(Path(args.fp_dir), args.seeds)
    conformal_results = load_conformal_results(Path(args.conformal_dir), args.seeds)

    print(f"Loaded results: {len(alignment_results)} alignment, "
          f"{len(fp_results)} fingerprint, {len(conformal_results)} conformal")

    # Run all analyses
    if alignment_results:
        main_comparison_table(alignment_results, output_dir)
        mechanism_analysis(alignment_results, output_dir)
        direct_vs_prohapten(alignment_results, output_dir)
        chance_level_test(alignment_results, output_dir)

    if alignment_results and fp_results:
        gnn_vs_fingerprint_test(alignment_results, fp_results, output_dir)

    if conformal_results:
        conformal_analysis(conformal_results, output_dir)

    print(f"\nAll tables saved to {output_dir}")


if __name__ == '__main__':
    main()
