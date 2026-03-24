"""
Control baselines for GNN explanation alignment evaluation.

Three baselines to disentangle learned chemistry from structural artifacts:
1. Random attribution — chance-level alignment control
2. Untrained GNN attention — architecture bias vs learned chemistry
3. Heteroatom heuristic — simple structural feature baseline

Usage:
    python scripts/explain/compute_control_baselines.py [--seeds 42 123 ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RESULTS_DIR
from src.explain.utils import (
    ALL_SEEDS, load_dataset, featurize_molecules, setup_device,
)
from src.explain.aop_reference import AOPReference
from src.explain.alignment_metrics import compute_batch_alignment
from src.explain.attention_extractor import AttentionExtractor
from src.modeling.ablation_model import AblationGNN


def random_baseline(n_atoms: int, rng: np.random.RandomState) -> torch.Tensor:
    """Uniform random importance scores."""
    return torch.tensor(rng.rand(n_atoms), dtype=torch.float32)


def heteroatom_baseline(smiles: str) -> torch.Tensor:
    """Importance = 1 for non-carbon atoms, 0 for carbon."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.tensor([])
    scores = []
    for atom in mol.GetAtoms():
        scores.append(0.0 if atom.GetAtomicNum() == 6 else 1.0)
    return torch.tensor(scores, dtype=torch.float32)


def compute_baselines_for_seed(
    seed: int,
    device: torch.device,
    aop_ref: AOPReference,
    output_dir: Path,
    n_random_repeats: int = 100,
):
    """Compute all control baselines for one seed."""
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    # Load data (same split as GNN)
    df, train_idx, val_idx, test_idx = load_dataset(seed)
    test_df = df.loc[test_idx].reset_index(drop=True)
    smiles_list = test_df['smiles'].tolist()
    labels = test_df['sensitization_human'].values.astype(int)

    # Featurize
    graphs, valid_indices = featurize_molecules(smiles_list)
    valid_smiles = [smiles_list[i] for i in valid_indices]
    valid_labels = labels[valid_indices]
    print(f"  {len(graphs)} valid molecules")

    # Get AOP reference masks
    reactive_masks = []
    for smi in valid_smiles:
        rc_mask, _ = aop_ref.get_reactive_center_mask(smi)
        reactive_masks.append(rc_mask)

    # TP filter (matches GNN evaluation)
    # Load GNN predictions to use same TP set
    expl_path = RESULTS_DIR / 'explanations' / f'seed_{seed}' / 'explanations.json'
    if expl_path.exists():
        with open(expl_path) as f:
            expl_data = json.load(f)
        gnn_predictions = np.array(expl_data['predictions'])
        pred_classes = (gnn_predictions > 0.5).astype(int)
    else:
        # Fallback: use all sensitizers
        print("  WARNING: No GNN predictions found, using all sensitizers")
        pred_classes = valid_labels

    tp_mask = (valid_labels == 1) & (pred_classes == 1) & \
              np.array([m.sum() > 0 if m.numel() > 0 else False for m in reactive_masks])
    tp_indices = np.where(tp_mask)[0]
    print(f"  TP with reactive centers: {len(tp_indices)}")

    if len(tp_indices) == 0:
        print("  No valid molecules for alignment — skipping")
        return

    tp_refs = [reactive_masks[i] for i in tp_indices]

    results = {}

    # === 1. Random baseline (averaged over repeats) ===
    random_aucs = []
    for rep in range(n_random_repeats):
        rng = np.random.RandomState(seed * 1000 + rep)
        rand_imps = [random_baseline(graphs[i].x.size(0), rng) for i in tp_indices]
        metrics = compute_batch_alignment(rand_imps, tp_refs)
        if 'mean_atom_auc' in metrics and not np.isnan(metrics['mean_atom_auc']):
            random_aucs.append(metrics['mean_atom_auc'])

    results['random'] = {
        'mean_atom_auc': float(np.mean(random_aucs)),
        'std_atom_auc': float(np.std(random_aucs)),
        'n_repeats': n_random_repeats,
        'n_valid': len(tp_indices),
    }
    print(f"  Random: atom-AUC = {results['random']['mean_atom_auc']:.4f} "
          f"+/- {results['random']['std_atom_auc']:.4f} ({n_random_repeats} repeats)")

    # === 2. Untrained GNN attention ===
    untrained_model = AblationGNN(condition='plain')
    untrained_model = untrained_model.to(device)
    untrained_model.eval()

    attention_extractor = AttentionExtractor(untrained_model, target_key='sensitization')
    untrained_imps = []
    for i in tp_indices:
        graph = graphs[i].clone().to(device)
        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
        try:
            weights = attention_extractor.attribute(graph, device=device)
            untrained_imps.append(weights)
        except Exception as e:
            print(f"    Warning: untrained attention failed for molecule {i}: {e}")
            untrained_imps.append(torch.ones(graph.x.size(0)) / graph.x.size(0))

    untrained_metrics = compute_batch_alignment(untrained_imps, tp_refs)
    results['untrained'] = untrained_metrics
    print(f"  Untrained GNN: atom-AUC = {untrained_metrics.get('mean_atom_auc', float('nan')):.4f}")

    # === 3. Heteroatom heuristic ===
    hetero_imps = [heteroatom_baseline(valid_smiles[i]) for i in tp_indices]
    hetero_metrics = compute_batch_alignment(hetero_imps, tp_refs)
    results['heteroatom'] = hetero_metrics
    print(f"  Heteroatom: atom-AUC = {hetero_metrics.get('mean_atom_auc', float('nan')):.4f}")

    # === 4. Atom degree baseline (bonus: tests if connectivity alone explains alignment) ===
    degree_imps = []
    for i in tp_indices:
        mol = Chem.MolFromSmiles(valid_smiles[i])
        if mol is not None:
            degrees = torch.tensor(
                [atom.GetDegree() for atom in mol.GetAtoms()], dtype=torch.float32
            )
            degree_imps.append(degrees)
        else:
            degree_imps.append(torch.ones(graphs[i].x.size(0)))

    degree_metrics = compute_batch_alignment(degree_imps, tp_refs)
    results['atom_degree'] = degree_metrics
    print(f"  Atom degree: atom-AUC = {degree_metrics.get('mean_atom_auc', float('nan')):.4f}")

    # Save
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

    with open(seed_dir / 'control_baselines.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"  Saved to {seed_dir / 'control_baselines.json'}")

    return results


def aggregate(seeds: list, output_dir: Path):
    """Aggregate control baseline results across seeds."""
    print(f"\n{'='*60}")
    print("Aggregating control baselines")
    print(f"{'='*60}")

    all_results = {}
    for seed in seeds:
        path = output_dir / f'seed_{seed}' / 'control_baselines.json'
        if path.exists():
            with open(path) as f:
                all_results[seed] = json.load(f)

    if not all_results:
        print("  No results to aggregate")
        return

    baselines = ['random', 'untrained', 'heteroatom', 'atom_degree']
    summary = {}

    for baseline in baselines:
        aucs = []
        for seed, results in all_results.items():
            if baseline in results:
                auc = results[baseline].get('mean_atom_auc', float('nan'))
                if not np.isnan(auc):
                    aucs.append(auc)

        if aucs:
            summary[baseline] = {
                'mean_atom_auc': float(np.mean(aucs)),
                'std_atom_auc': float(np.std(aucs)),
                'n_seeds': len(aucs),
            }
            print(f"  {baseline:<15}: atom-AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} "
                  f"(n={len(aucs)} seeds)")

    with open(output_dir / 'control_baselines_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {output_dir / 'control_baselines_summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Compute control baselines')
    parser.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'control_baselines'))
    parser.add_argument('--n-random-repeats', type=int, default=100)
    args = parser.parse_args()

    device = setup_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aop_ref = AOPReference(use_extended=True)

    for seed in args.seeds:
        try:
            compute_baselines_for_seed(
                seed=seed,
                device=device,
                aop_ref=aop_ref,
                output_dir=output_dir,
                n_random_repeats=args.n_random_repeats,
            )
        except Exception as e:
            print(f"ERROR processing seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    aggregate(args.seeds, output_dir)


if __name__ == '__main__':
    main()
