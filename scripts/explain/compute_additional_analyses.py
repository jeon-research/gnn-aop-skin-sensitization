"""
Additional analyses:
1. Residual analysis: Does GNN attention capture chemistry beyond heteroatom identity?
2. Size-normalized pro-hapten vs direct comparison using Precision@K
3. Attention faithfulness: Do high-attention atoms causally influence predictions?

Usage:
    python scripts/explain/compute_additional_analyses.py [--seeds 42 123 ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROJECT_ROOT, RESULTS_DIR
from rdkit import Chem
from src.explain.aop_reference import AOPReference
from src.explain.utils import ALL_SEEDS, load_model, load_dataset, featurize_molecules
from src.modeling.causal_aop_gnn import MoleculeFeaturizer


def get_valid_molecules(seed, explanations_dir, aop_ref):
    """Get TP sensitizers with SMARTS matches for a seed."""
    path = explanations_dir / f'seed_{seed}' / 'explanations.json'
    with open(path) as f:
        data = json.load(f)

    smiles_list = data['smiles']
    labels = np.array(data['labels'])
    predictions = np.array(data['predictions'])
    explanations = data['explanations']

    # Get AOP masks and mechanisms
    reactive_masks = []
    mechanisms = []
    for smi in smiles_list:
        rc_mask, _ = aop_ref.get_reactive_center_mask(smi)
        mech = aop_ref.classify_mechanism(smi)
        reactive_masks.append(rc_mask)
        mechanisms.append(mech)

    # TP filter
    pred_classes = (predictions > 0.5).astype(int)
    tp_mask = (labels == 1) & (pred_classes == 1) & np.array(
        [m.sum() > 0 if m.numel() > 0 else False for m in reactive_masks]
    )
    tp_indices = np.where(tp_mask)[0]

    return data, smiles_list, labels, predictions, explanations, reactive_masks, mechanisms, tp_indices


# ========================================================================
# Analysis 1: Residual analysis (partialing out heteroatom signal)
# ========================================================================
ALL_METHODS = ['ig', 'gradcam', 'attention', 'gnnexplainer', 'pgexplainer', 'graphmask', 'ensemble']


def compute_residual_for_method(method_scores, smiles_list, reactive_masks, tp_indices):
    """Compute raw, residual, and heteroatom-only AUC for a single method's scores."""
    raw_aucs, residual_aucs, heteroatom_only_aucs = [], [], []

    for idx in tp_indices:
        smi = smiles_list[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        scores = np.array(method_scores[idx])
        mie_mask = reactive_masks[idx].numpy()
        n_atoms = len(scores)

        if len(mie_mask) != n_atoms or mie_mask.sum() == 0:
            continue

        # Build heteroatom mask (1 = non-carbon, 0 = carbon)
        hetero = np.array([0 if mol.GetAtomWithIdx(i).GetAtomicNum() == 6 else 1
                           for i in range(n_atoms)])

        # Group means
        hetero_mean = scores[hetero == 1].mean() if hetero.sum() > 0 else 0
        carbon_mean = scores[hetero == 0].mean() if (1 - hetero).sum() > 0 else 0

        # Residual: subtract group mean
        residual = scores.copy()
        residual[hetero == 1] -= hetero_mean
        residual[hetero == 0] -= carbon_mean

        if len(np.unique(mie_mask)) == 2:
            raw_aucs.append(roc_auc_score(mie_mask, scores))
            residual_aucs.append(roc_auc_score(mie_mask, residual))
            heteroatom_only_aucs.append(roc_auc_score(mie_mask, hetero.astype(float)))

    return {
        'n_valid': len(raw_aucs),
        'raw_auc': float(np.mean(raw_aucs)) if raw_aucs else None,
        'residual_auc': float(np.mean(residual_aucs)) if residual_aucs else None,
        'heteroatom_only_auc': float(np.mean(heteroatom_only_aucs)) if heteroatom_only_aucs else None,
        'raw_aucs': [float(x) for x in raw_aucs],
        'residual_aucs': [float(x) for x in residual_aucs],
    }


def compute_per_element_residual_for_method(method_scores, smiles_list, reactive_masks, tp_indices):
    """Compute residual AUC after subtracting per-element means (C, N, O, S, etc.)."""
    raw_aucs, residual_aucs = [], []

    for idx in tp_indices:
        smi = smiles_list[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        scores = np.array(method_scores[idx])
        mie_mask = reactive_masks[idx].numpy()
        n_atoms = len(scores)

        if len(mie_mask) != n_atoms or mie_mask.sum() == 0:
            continue

        # Get element for each atom
        elements = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)]
        elements = np.array(elements)

        # Subtract per-element group mean
        residual = scores.copy()
        for elem in np.unique(elements):
            mask = elements == elem
            if mask.sum() > 0:
                residual[mask] -= scores[mask].mean()

        if len(np.unique(mie_mask)) == 2:
            raw_aucs.append(roc_auc_score(mie_mask, scores))
            residual_aucs.append(roc_auc_score(mie_mask, residual))

    return {
        'n_valid': len(raw_aucs),
        'raw_auc': float(np.mean(raw_aucs)) if raw_aucs else None,
        'residual_auc': float(np.mean(residual_aucs)) if residual_aucs else None,
        'raw_aucs': [float(x) for x in raw_aucs],
        'residual_aucs': [float(x) for x in residual_aucs],
    }


def compute_residual_analysis(seed, explanations_dir, aop_ref, methods=None):
    """Compute residual analysis for all methods (or a subset)."""
    if methods is None:
        methods = ALL_METHODS

    data, smiles_list, labels, predictions, explanations, reactive_masks, mechanisms, tp_indices = \
        get_valid_molecules(seed, explanations_dir, aop_ref)

    results = {}
    for method in methods:
        if method not in explanations:
            continue
        results[method] = compute_residual_for_method(
            explanations[method], smiles_list, reactive_masks, tp_indices
        )
    return results


def compute_per_element_residual_analysis(seed, explanations_dir, aop_ref, methods=None):
    """Compute per-element residual analysis for all methods."""
    if methods is None:
        methods = ALL_METHODS

    data, smiles_list, labels, predictions, explanations, reactive_masks, mechanisms, tp_indices = \
        get_valid_molecules(seed, explanations_dir, aop_ref)

    results = {}
    for method in methods:
        if method not in explanations:
            continue
        results[method] = compute_per_element_residual_for_method(
            explanations[method], smiles_list, reactive_masks, tp_indices
        )
    return results


# ========================================================================
# Analysis 2: Size-normalized pro-hapten vs direct (Precision@3, Precision@5)
# ========================================================================
def compute_size_normalized_comparison(seed, alignment_dir):
    """Extract fixed-K metrics (P@3, P@5) for pro-hapten vs direct from existing data."""
    path = alignment_dir / f'seed_{seed}' / 'alignment_metrics.json'
    with open(path) as f:
        m = json.load(f)

    results = {}
    for method in ['ig', 'gradcam', 'attention', 'gnnexplainer', 'ensemble']:
        direct_key = f'{method}_direct'
        prohapten_key = f'{method}_pro_hapten'
        if direct_key in m and prohapten_key in m:
            results[method] = {
                'direct_p3': m[direct_key]['mean_precision_at_3'],
                'prohapten_p3': m[prohapten_key]['mean_precision_at_3'],
                'direct_p5': m[direct_key]['mean_precision_at_5'],
                'prohapten_p5': m[prohapten_key]['mean_precision_at_5'],
                'direct_atom_auc': m[direct_key]['mean_atom_auc'],
                'prohapten_atom_auc': m[prohapten_key]['mean_atom_auc'],
            }
    return results


# ========================================================================
# Analysis 3: Attention faithfulness (masking test)
# ========================================================================
def compute_faithfulness(seed, explanations_dir, aop_ref, device, n_random=50, mask_fraction=0.2):
    """Test if masking high-attention atoms changes predictions more than random masking."""
    data, smiles_list, labels, predictions, explanations, reactive_masks, mechanisms, tp_indices = \
        get_valid_molecules(seed, explanations_dir, aop_ref)

    # Load model
    model = load_model(seed, device)

    attention_scores = explanations['attention']
    results_per_mol = []

    for idx in tp_indices:
        smi = smiles_list[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        attn = np.array(attention_scores[idx])
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 3:
            continue

        # Featurize
        graph = MoleculeFeaturizer.smiles_to_graph(smi)
        if graph is None:
            continue
        graph = graph.to(device)
        # Add batch tensor for single-graph inference
        if graph.batch is None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

        # Baseline prediction
        with torch.no_grad():
            baseline_out = model(graph)
            if isinstance(baseline_out, dict):
                baseline_logit = baseline_out['sensitization'].item()
            else:
                baseline_logit = baseline_out.item()
        baseline_prob = torch.sigmoid(torch.tensor(baseline_logit)).item()

        # K = number of atoms to mask (top mask_fraction or at least 1)
        K = max(1, int(mask_fraction * n_atoms))

        # Mask top-K attended atoms (zero their features)
        top_k_indices = np.argsort(attn)[-K:]

        def predict_with_mask(mask_indices):
            g = graph.clone()
            x = g.x.clone()
            x[list(mask_indices)] = 0.0
            g.x = x
            if g.batch is None:
                g.batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(g)
                if isinstance(out, dict):
                    logit = out['sensitization'].item()
                else:
                    logit = out.item()
            return torch.sigmoid(torch.tensor(logit)).item()

        masked_top_prob = predict_with_mask(top_k_indices)
        top_drop = abs(baseline_prob - masked_top_prob)

        # Random masking (n_random repeats)
        rng = np.random.RandomState(seed + idx)
        random_drops = []
        for _ in range(n_random):
            rand_indices = rng.choice(n_atoms, size=K, replace=False)
            masked_rand_prob = predict_with_mask(rand_indices)
            random_drops.append(abs(baseline_prob - masked_rand_prob))

        mean_random_drop = float(np.mean(random_drops))

        results_per_mol.append({
            'smiles': smi,
            'n_atoms': n_atoms,
            'K': K,
            'baseline_prob': baseline_prob,
            'top_attention_drop': float(top_drop),
            'mean_random_drop': mean_random_drop,
            'faithfulness_ratio': float(top_drop / mean_random_drop) if mean_random_drop > 1e-6 else float('inf'),
            'top_beats_random': float(top_drop > mean_random_drop),
        })

    return results_per_mol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=ALL_SEEDS)
    parser.add_argument('--explanations-dir', type=Path, default=RESULTS_DIR / 'explanations')
    parser.add_argument('--alignment-dir', type=Path, default=RESULTS_DIR / 'alignment')
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR / 'additional_analyses')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    aop_ref = AOPReference()

    # ============================================================
    # Analysis 1: Residual analysis (all methods)
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Residual Analysis — All Methods (Heteroatom Partialed Out)")
    print("=" * 60)

    # Collect per-seed, per-method results
    all_residual = {method: {'raw': [], 'residual': [], 'heteroatom': []} for method in ALL_METHODS}
    per_seed_results = {}

    for seed in args.seeds:
        r = compute_residual_analysis(seed, args.explanations_dir, aop_ref)
        per_seed_results[seed] = r
        for method, mr in r.items():
            if mr['raw_auc'] is not None:
                all_residual[method]['raw'].append(mr['raw_auc'])
                all_residual[method]['residual'].append(mr['residual_auc'])
                all_residual[method]['heteroatom'].append(mr['heteroatom_only_auc'])

    # Print table and compute stats
    residual_summary = {'per_seed': per_seed_results, 'per_method': {}}
    print(f"\n  {'Method':<15} {'Raw AUC':>12} {'Residual AUC':>14} {'Heteroatom':>12} {'Drop':>8} {'Res>0.5 p':>10}")
    print(f"  {'-'*73}")

    for method in ALL_METHODS:
        raw = all_residual[method]['raw']
        res = all_residual[method]['residual']
        het = all_residual[method]['heteroatom']

        if not raw:
            continue

        drop = np.mean(raw) - np.mean(res)
        sd_raw = np.std(raw, ddof=1) if len(raw) > 1 else 0
        sd_res = np.std(res, ddof=1) if len(res) > 1 else 0

        # Statistical tests require >= 3 seeds
        w_res, p_res, w_drop, p_drop = 0, 1.0, 0, 1.0
        if len(raw) >= 3:
            try:
                w_res, p_res = stats.wilcoxon([v - 0.5 for v in res], alternative='greater')
            except Exception:
                pass
            try:
                w_drop, p_drop = stats.wilcoxon([r - s for r, s in zip(raw, res)], alternative='greater')
            except Exception:
                pass

        sig = '***' if p_res < 0.001 else '**' if p_res < 0.01 else '*' if p_res < 0.05 else ''

        print(f"  {method:<15} {np.mean(raw):>8.3f}±{sd_raw:.3f}"
              f"  {np.mean(res):>8.3f}±{sd_res:.3f}"
              f"  {np.mean(het):>8.3f}  {drop:>+.3f}  {p_res:>8.4f} {sig}")

        residual_summary['per_method'][method] = {
            'raw_mean': float(np.mean(raw)),
            'raw_sd': float(sd_raw),
            'residual_mean': float(np.mean(res)),
            'residual_sd': float(sd_res),
            'heteroatom_mean': float(np.mean(het)),
            'drop': float(drop),
            'residual_vs_chance_W': float(w_res),
            'residual_vs_chance_p': float(p_res),
            'raw_vs_residual_W': float(w_drop),
            'raw_vs_residual_p': float(p_drop),
            'n_seeds': len(raw),
        }

    with open(args.output_dir / 'residual_analysis.json', 'w') as f:
        json.dump(residual_summary, f, indent=2)

    # ============================================================
    # Analysis 1b: Per-element residual analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 1b: Per-Element Residual Analysis")
    print("=" * 60)

    all_per_elem = {method: {'raw': [], 'residual': []} for method in ALL_METHODS}

    for seed in args.seeds:
        r = compute_per_element_residual_analysis(seed, args.explanations_dir, aop_ref)
        for method, mr in r.items():
            if mr['raw_auc'] is not None:
                all_per_elem[method]['raw'].append(mr['raw_auc'])
                all_per_elem[method]['residual'].append(mr['residual_auc'])

    per_elem_summary = {}
    print(f"\n  {'Method':<15} {'Raw AUC':>12} {'Per-Elem Res':>14} {'Drop':>8} {'Res>0.5 p':>10}")
    print(f"  {'-'*61}")

    for method in ALL_METHODS:
        raw = all_per_elem[method]['raw']
        res = all_per_elem[method]['residual']

        if not raw:
            continue

        drop = np.mean(raw) - np.mean(res)
        sd_res = np.std(res, ddof=1) if len(res) > 1 else 0

        w_res, p_res = 0, 1.0
        if len(res) >= 3:
            try:
                w_res, p_res = stats.wilcoxon([v - 0.5 for v in res], alternative='greater')
            except Exception:
                pass

        sig = '***' if p_res < 0.001 else '**' if p_res < 0.01 else '*' if p_res < 0.05 else ''

        print(f"  {method:<15} {np.mean(raw):>8.3f}±{np.std(raw, ddof=1):.3f}"
              f"  {np.mean(res):>8.3f}±{sd_res:.3f}"
              f"  {drop:>+.3f}  {p_res:>8.4f} {sig}")

        per_elem_summary[method] = {
            'raw_mean': float(np.mean(raw)),
            'per_element_residual_mean': float(np.mean(res)),
            'per_element_residual_sd': float(sd_res),
            'drop': float(drop),
            'residual_vs_chance_p': float(p_res),
            'n_seeds': len(raw),
        }

    with open(args.output_dir / 'per_element_residual_analysis.json', 'w') as f:
        json.dump(per_elem_summary, f, indent=2)

    # ============================================================
    # Analysis 2: Size-normalized pro-hapten vs direct
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Size-Normalized Pro-hapten vs Direct")
    print("=" * 60)

    size_norm = {}
    for seed in args.seeds:
        size_norm[seed] = compute_size_normalized_comparison(seed, args.alignment_dir)

    # Aggregate and test
    methods = ['ig', 'gradcam', 'attention', 'gnnexplainer', 'ensemble']
    size_norm_summary = {}
    for method in methods:
        d_p3 = [size_norm[s][method]['direct_p3'] for s in args.seeds if method in size_norm[s]]
        ph_p3 = [size_norm[s][method]['prohapten_p3'] for s in args.seeds if method in size_norm[s]]
        d_p5 = [size_norm[s][method]['direct_p5'] for s in args.seeds if method in size_norm[s]]
        ph_p5 = [size_norm[s][method]['prohapten_p5'] for s in args.seeds if method in size_norm[s]]
        d_auc = [size_norm[s][method]['direct_atom_auc'] for s in args.seeds if method in size_norm[s]]
        ph_auc = [size_norm[s][method]['prohapten_atom_auc'] for s in args.seeds if method in size_norm[s]]

        # Wilcoxon on P@3 and P@5
        try:
            _, p3_p = stats.wilcoxon([p - d for p, d in zip(ph_p3, d_p3)], alternative='greater')
        except Exception:
            p3_p = 1.0
        try:
            _, p5_p = stats.wilcoxon([p - d for p, d in zip(ph_p5, d_p5)], alternative='greater')
        except Exception:
            p5_p = 1.0

        size_norm_summary[method] = {
            'direct_p3': float(np.mean(d_p3)),
            'prohapten_p3': float(np.mean(ph_p3)),
            'p3_diff': float(np.mean(ph_p3) - np.mean(d_p3)),
            'p3_p_value': float(p3_p),
            'direct_p5': float(np.mean(d_p5)),
            'prohapten_p5': float(np.mean(ph_p5)),
            'p5_diff': float(np.mean(ph_p5) - np.mean(d_p5)),
            'p5_p_value': float(p5_p),
            'direct_atom_auc': float(np.mean(d_auc)),
            'prohapten_atom_auc': float(np.mean(ph_auc)),
            'atom_auc_diff': float(np.mean(ph_auc) - np.mean(d_auc)),
        }

        print(f"\n  {method}:")
        print(f"    P@3: direct={np.mean(d_p3):.3f}, prohapten={np.mean(ph_p3):.3f}, "
              f"diff={np.mean(ph_p3) - np.mean(d_p3):+.3f}, p={p3_p:.4f}")
        print(f"    P@5: direct={np.mean(d_p5):.3f}, prohapten={np.mean(ph_p5):.3f}, "
              f"diff={np.mean(ph_p5) - np.mean(d_p5):+.3f}, p={p5_p:.4f}")
        print(f"    AUC: direct={np.mean(d_auc):.3f}, prohapten={np.mean(ph_auc):.3f}, "
              f"diff={np.mean(ph_auc) - np.mean(d_auc):+.3f}")

    with open(args.output_dir / 'size_normalized_comparison.json', 'w') as f:
        json.dump(size_norm_summary, f, indent=2)

    # ============================================================
    # Analysis 3: Attention faithfulness
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Attention Faithfulness (Masking Test)")
    print("=" * 60)

    faithfulness_results = {}
    seed_faithfulness_ratios = []
    seed_top_beats_random_fracs = []

    for seed in args.seeds:
        print(f"\n  Seed {seed}...")
        mol_results = compute_faithfulness(seed, args.explanations_dir, aop_ref, device)

        ratios = [r['faithfulness_ratio'] for r in mol_results if r['faithfulness_ratio'] != float('inf')]
        top_beats = [r['top_beats_random'] for r in mol_results]
        top_drops = [r['top_attention_drop'] for r in mol_results]
        rand_drops = [r['mean_random_drop'] for r in mol_results]

        mean_ratio = float(np.mean(ratios)) if ratios else 0
        frac_top_beats = float(np.mean(top_beats)) if top_beats else 0

        faithfulness_results[seed] = {
            'n_molecules': len(mol_results),
            'mean_faithfulness_ratio': mean_ratio,
            'frac_top_beats_random': frac_top_beats,
            'mean_top_drop': float(np.mean(top_drops)),
            'mean_random_drop': float(np.mean(rand_drops)),
            'per_molecule': mol_results,
        }
        seed_faithfulness_ratios.append(mean_ratio)
        seed_top_beats_random_fracs.append(frac_top_beats)

        print(f"    n_mol={len(mol_results)}, ratio={mean_ratio:.2f}, "
              f"top_beats_random={frac_top_beats:.1%}, "
              f"mean_top_drop={np.mean(top_drops):.4f}, "
              f"mean_random_drop={np.mean(rand_drops):.4f}")

    # Test: is faithfulness ratio > 1 across seeds?
    diffs_ratio = [r - 1.0 for r in seed_faithfulness_ratios]
    try:
        w_faith, p_faith = stats.wilcoxon(diffs_ratio, alternative='greater')
    except Exception:
        w_faith, p_faith = 0, 1.0

    # Test: paired comparison of drops
    all_top_drops = [faithfulness_results[s]['mean_top_drop'] for s in args.seeds]
    all_rand_drops = [faithfulness_results[s]['mean_random_drop'] for s in args.seeds]
    try:
        w_drops, p_drops = stats.wilcoxon(
            [t - r for t, r in zip(all_top_drops, all_rand_drops)],
            alternative='greater'
        )
    except Exception:
        w_drops, p_drops = 0, 1.0

    faithfulness_summary = {
        'mean_faithfulness_ratio': float(np.mean(seed_faithfulness_ratios)),
        'sd_faithfulness_ratio': float(np.std(seed_faithfulness_ratios, ddof=1)),
        'mean_frac_top_beats_random': float(np.mean(seed_top_beats_random_fracs)),
        'sd_frac_top_beats_random': float(np.std(seed_top_beats_random_fracs, ddof=1)),
        'mean_top_drop': float(np.mean(all_top_drops)),
        'mean_random_drop': float(np.mean(all_rand_drops)),
        'ratio_vs_1_W': float(w_faith),
        'ratio_vs_1_p': float(p_faith),
        'drops_paired_W': float(w_drops),
        'drops_paired_p': float(p_drops),
        'per_seed': {str(s): {k: v for k, v in faithfulness_results[s].items() if k != 'per_molecule'}
                     for s in args.seeds},
    }

    print(f"\n  Summary:")
    print(f"    Mean faithfulness ratio: {np.mean(seed_faithfulness_ratios):.2f} ± "
          f"{np.std(seed_faithfulness_ratios, ddof=1):.2f}")
    print(f"    Frac top > random: {np.mean(seed_top_beats_random_fracs):.1%} ± "
          f"{np.std(seed_top_beats_random_fracs, ddof=1):.1%}")
    print(f"    Mean top drop: {np.mean(all_top_drops):.4f}")
    print(f"    Mean random drop: {np.mean(all_rand_drops):.4f}")
    print(f"    Ratio > 1 test: W={w_faith}, p={p_faith:.4f}")
    print(f"    Paired drops: W={w_drops}, p={p_drops:.4f}")

    with open(args.output_dir / 'faithfulness_analysis.json', 'w') as f:
        json.dump(faithfulness_summary, f, indent=2)

    # Save per-molecule faithfulness details
    with open(args.output_dir / 'faithfulness_per_molecule.json', 'w') as f:
        json.dump(faithfulness_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("All analyses complete. Results saved to:", args.output_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
