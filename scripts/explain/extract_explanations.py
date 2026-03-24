"""
Extract GNN explanations for all test set molecules across 20 seeds.

Runs Integrated Gradients, GradCAM, Attention extraction, and GNNExplainer
on each trained AblationGNN or MechGNN checkpoint, saving per-molecule
atom-level importance scores.

Usage:
    python scripts/explain/extract_explanations.py [--seeds 42 123 ...] [--methods ig gradcam attention gnnexplainer]
    python scripts/explain/extract_explanations.py --model-type mechgnn --mechgnn-lambda 0.5 --seeds 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RESULTS_DIR
from src.explain.utils import (
    ALL_SEEDS, find_checkpoint, load_model, load_dataset,
    featurize_molecules, setup_device, set_seed,
    load_mechgnn_model,
)
from src.explain.integrated_gradients import IntegratedGradients
from src.explain.gradcam import GradCAM
from src.explain.attention_extractor import AttentionExtractor
from src.explain.gnn_explainer import GNNExplainerWrapper
from src.explain.pg_explainer import PGExplainerWrapper
from src.explain.graphmask_explainer import GraphMaskExplainerWrapper
from src.explain.ensemble_explanation import EnsembleExplanation


METHOD_NAMES = ['ig', 'gradcam', 'attention', 'gnnexplainer', 'pgexplainer', 'graphmask']

# Methods that require AttentiveFP internals and cannot be used with GCN/GIN
ATTENTIVEFP_ONLY_METHODS = {'gradcam', 'attention', 'pgexplainer'}


def extract_for_seed(
    seed: int,
    methods: list,
    device: torch.device,
    output_dir: Path,
    max_molecules: int = None,
    model_type: str = 'plain',
    mechgnn_lambda: float = None,
    architecture: str = 'attentivefp',
):
    """Extract explanations for one seed."""
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"Seed {seed} (model_type={model_type}, arch={architecture})")
    print(f"{'='*60}")

    # Filter out methods incompatible with non-AttentiveFP architectures
    if architecture != 'attentivefp':
        skipped = [m for m in methods if m in ATTENTIVEFP_ONLY_METHODS]
        methods = [m for m in methods if m not in ATTENTIVEFP_ONLY_METHODS]
        if skipped:
            print(f"  Skipping {skipped} (not compatible with {architecture})")

    # Load model
    if model_type == 'mechgnn':
        model = load_mechgnn_model(seed, mechgnn_lambda, device)
        print(f"  Loaded MechGNN (λ={mechgnn_lambda}) for seed {seed}")
    else:
        model = load_model(seed, device, architecture=architecture)
        print(f"  Loaded AblationGNN ({architecture}) from {find_checkpoint(seed, architecture=architecture)}")

    # Load and split data
    df, train_idx, val_idx, test_idx = load_dataset(seed)
    test_df = df.loc[test_idx].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} molecules")

    if max_molecules:
        test_df = test_df.head(max_molecules)
        print(f"  Limiting to {max_molecules} molecules")

    # Featurize test molecules
    smiles_list = test_df['smiles'].tolist()
    graphs, valid_indices = featurize_molecules(smiles_list)
    print(f"  Featurized: {len(graphs)} valid molecules")

    # Initialize explainers
    explainers = {}
    if 'ig' in methods:
        explainers['ig'] = IntegratedGradients(model, target_key='sensitization', n_steps=50)
    if 'gradcam' in methods:
        explainers['gradcam'] = GradCAM(model, target_key='sensitization')
    if 'attention' in methods:
        explainers['attention'] = AttentionExtractor(model, target_key='sensitization')
    if 'gnnexplainer' in methods:
        explainers['gnnexplainer'] = GNNExplainerWrapper(
            model, target_key='sensitization', n_steps=200, lr=0.01
        )
    if 'pgexplainer' in methods:
        pg = PGExplainerWrapper(model, target_key='sensitization', epochs=30, lr=0.003)
        # Train PGExplainer on a subsample of test graphs
        pg.train_on_loader(graphs[:min(50, len(graphs))], device)
        explainers['pgexplainer'] = pg
        print(f"  PGExplainer trained on {min(50, len(graphs))} graphs")
    if 'graphmask' in methods:
        explainers['graphmask'] = GraphMaskExplainerWrapper(
            model, target_key='sensitization', n_steps=100, lr=0.01
        )

    # Extract explanations
    results = {method: [] for method in methods}
    predictions = []

    for i, (graph, orig_idx) in enumerate(zip(graphs, valid_indices)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing molecule {i+1}/{len(graphs)}...")

        # Get prediction
        data = graph.clone().to(device)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            model.eval()
            outputs = model(data)
            logit = outputs['sensitization']
            prob = torch.sigmoid(logit).item()
        predictions.append(prob)

        # Extract explanations
        for method_name, explainer in explainers.items():
            try:
                importance = explainer.attribute(graph, device=device)
                results[method_name].append(importance.numpy().tolist())
            except Exception as e:
                print(f"    Warning: {method_name} failed for molecule {i}: {e}")
                n_atoms = graph.x.size(0)
                results[method_name].append([0.0] * n_atoms)

    # Compute ensemble
    if len(explainers) >= 2:
        ensemble = EnsembleExplanation(min_methods_for_consensus=3)
        ensemble_results = []
        agreement_scores = []

        for i in range(len(graphs)):
            method_scores = {}
            for method_name in methods:
                scores = results[method_name][i]
                method_scores[method_name] = torch.tensor(scores, dtype=torch.float32)

            combined = ensemble.combine(method_scores)
            ensemble_results.append(combined['rank_importance'].numpy().tolist())
            agreement_scores.append(combined['method_agreement'].item())

        results['ensemble'] = ensemble_results

    # Save results
    seed_dir = output_dir / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'seed': seed,
        'n_molecules': len(graphs),
        'smiles': [smiles_list[i] for i in valid_indices],
        'labels': [float(test_df.iloc[i]['sensitization_human']) for i in valid_indices],
        'predictions': predictions,
        'explanations': results,
        'valid_indices': valid_indices,
    }

    output_path = seed_dir / 'explanations.json'
    with open(output_path, 'w') as f:
        json.dump(output, f)
    print(f"  Saved to {output_path}")

    # Summary
    print(f"  Prediction stats: mean={np.mean(predictions):.3f}, "
          f"positive rate={np.mean([p > 0.5 for p in predictions]):.3f}")

    return output


def main():
    parser = argparse.ArgumentParser(description='Extract GNN explanations')
    parser.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    parser.add_argument('--methods', type=str, nargs='+', default=METHOD_NAMES)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (auto-set based on model type if not given)')
    parser.add_argument('--max-molecules', type=int, default=None,
                        help='Limit number of test molecules (for debugging)')
    parser.add_argument('--model-type', type=str, default='plain',
                        choices=['plain', 'mechgnn'],
                        help='Model type to extract explanations from')
    parser.add_argument('--mechgnn-lambda', type=float, default=None,
                        help='Lambda value for MechGNN (required if model-type=mechgnn)')
    parser.add_argument('--architecture', type=str, default='attentivefp',
                        choices=['attentivefp', 'gcn', 'gin'],
                        help='GNN architecture (attentivefp, gcn, gin)')
    args = parser.parse_args()

    if args.model_type == 'mechgnn' and args.mechgnn_lambda is None:
        parser.error("--mechgnn-lambda is required when --model-type=mechgnn")

    device = setup_device()
    print(f"Device: {device}")

    # Set output directory based on model type and architecture
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.model_type == 'mechgnn':
        output_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{args.mechgnn_lambda}' / 'explanations'
    elif args.architecture != 'attentivefp':
        output_dir = RESULTS_DIR / f'explanations_{args.architecture}'
    else:
        output_dir = RESULTS_DIR / 'explanations'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seeds: {args.seeds}")
    print(f"Methods: {args.methods}")
    print(f"Model type: {args.model_type}")
    print(f"Architecture: {args.architecture}")
    if args.model_type == 'mechgnn':
        print(f"MechGNN lambda: {args.mechgnn_lambda}")
    print(f"Output: {output_dir}")

    start_time = time.time()

    for seed in args.seeds:
        try:
            extract_for_seed(
                seed=seed,
                methods=list(args.methods),
                device=device,
                output_dir=output_dir,
                max_molecules=args.max_molecules,
                model_type=args.model_type,
                mechgnn_lambda=args.mechgnn_lambda,
                architecture=args.architecture,
            )
        except Exception as e:
            print(f"ERROR processing seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
