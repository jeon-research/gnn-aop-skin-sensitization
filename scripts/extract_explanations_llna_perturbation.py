"""Extract GNNExplainer / PGExplainer / GraphMask atom-level explanations
from LLNA-primary AttentiveFP checkpoints. Output is written to a separate
directory and merged with the gradient/attention/ensemble outputs by
merge_llna_explanations.py.

Usage:
    python3 scripts/extract_explanations_llna_perturbation.py \\
        --training-root results/llna_training \\
        --label-source llna \\
        --explanations-root results/explanations_llna_perturbation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
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

from src.explain.utils import setup_device, featurize_molecules, set_seed
from src.explain.gnn_explainer import GNNExplainerWrapper
from src.explain.pg_explainer import PGExplainerWrapper
from src.explain.graphmask_explainer import GraphMaskExplainerWrapper
from src.modeling.ablation_model import AblationGNN


from senslib.seeds import ALL as DEFAULT_SEEDS  # noqa: E402


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> AblationGNN:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = AblationGNN(
        condition='plain',
        hidden_dim=cfg.get('hidden_dim', 256),
        node_dim=cfg.get('node_dim', 64),
        num_gnn_layers=cfg.get('num_gnn_layers', 3),
        dropout=cfg.get('dropout', 0.3),
        use_continuous_features=False,
        architecture=cfg.get('architecture', 'attentivefp'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _load_split(label_source: str, seed: int) -> Dict:
    path = REVISION_ROOT / 'results' / 'splits' / f'{label_source}_seed_{seed}.json'
    if not path.exists():
        raise FileNotFoundError(f"Missing persisted split: {path}")
    return json.loads(path.read_text())


def extract_for_seed(seed: int, training_root: Path, label_source: str,
                     explanations_root: Path, device: torch.device,
                     methods: List[str]) -> Dict:
    set_seed(seed)

    ckpt_path = training_root / f'seed_{seed}' / 'best_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = _load_checkpoint(ckpt_path, device)

    split = _load_split(label_source, seed)
    test_smiles: List[str] = split['test_smiles']

    label_col = 'llna_result'
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'causal_aop_comprehensive_v6.csv')
    df = df.dropna(subset=['smiles', label_col])
    if label_source == 'shuffled_llna':
        from senslib.llna_loader import load_dataset_shuffled
        df_, _, _, test_idx = load_dataset_shuffled(seed, label_col=label_col)
        test_labels = df_.loc[test_idx, label_col].tolist()
    else:
        from senslib.llna_loader import load_dataset_llna
        df_, _, _, test_idx = load_dataset_llna(seed)
        test_labels = df_.loc[test_idx, label_col].tolist()

    actual = df_.loc[test_idx, 'smiles'].tolist()
    if actual != test_smiles:
        raise RuntimeError(f"seed={seed}: persisted split drift detected")

    graphs, valid = featurize_molecules(test_smiles)
    valid_smiles = [test_smiles[i] for i in valid]
    valid_labels = [float(test_labels[i]) for i in valid]

    explainers: Dict[str, object] = {}
    if 'gnnexplainer' in methods:
        explainers['gnnexplainer'] = GNNExplainerWrapper(
            model, target_key='sensitization', n_steps=200, lr=0.01
        )
    if 'pgexplainer' in methods:
        pg = PGExplainerWrapper(model, target_key='sensitization',
                                epochs=30, lr=0.003)
        pg.train_on_loader(graphs[:min(50, len(graphs))], device)
        explainers['pgexplainer'] = pg
        print(f"  [seed={seed}] PGExplainer trained on "
              f"{min(50, len(graphs))} graphs")
    if 'graphmask' in methods:
        explainers['graphmask'] = GraphMaskExplainerWrapper(
            model, target_key='sensitization', n_steps=100, lr=0.01
        )

    results: Dict[str, List[List[float]]] = {k: [] for k in explainers}
    predictions: List[float] = []

    t_seed = time.time()
    for i, graph in enumerate(graphs):
        data = graph.clone().to(device)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long,
                                     device=device)
        with torch.no_grad():
            out = model(data)
            prob = torch.sigmoid(out['sensitization']).item()
        predictions.append(prob)

        for name, expl in explainers.items():
            try:
                imp = expl.attribute(graph, device=device)
                results[name].append(imp.cpu().numpy().tolist())
            except Exception as e:
                results[name].append([0.0] * graph.x.size(0))

    out = {
        'seed': seed,
        'label_source': label_source,
        'n_molecules': len(graphs),
        'smiles': valid_smiles,
        'labels': valid_labels,
        'predictions': predictions,
        'explanations': results,
    }

    seed_dir = explanations_root / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / 'explanations.json').write_text(json.dumps(out))
    elapsed = time.time() - t_seed
    print(f"  [seed={seed}] saved {len(graphs)} mols, "
          f"methods={list(explainers.keys())}, "
          f"elapsed={elapsed:.1f}s")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--training-root', type=Path,
                   default=Path('results/llna_training'))
    p.add_argument('--label-source', type=str, default='llna',
                   choices=['llna', 'shuffled_llna'])
    p.add_argument('--explanations-root', type=Path,
                   default=Path('results/explanations_llna_perturbation'))
    p.add_argument('--methods', type=str, nargs='+',
                   default=['gnnexplainer', 'pgexplainer', 'graphmask'],
                   choices=['gnnexplainer', 'pgexplainer', 'graphmask'])
    args = p.parse_args()

    training_root = (REVISION_ROOT / args.training_root).resolve() \
        if not args.training_root.is_absolute() else args.training_root
    explanations_root = (REVISION_ROOT / args.explanations_root).resolve() \
        if not args.explanations_root.is_absolute() else args.explanations_root
    explanations_root.mkdir(parents=True, exist_ok=True)

    device = setup_device()
    print(f"Device: {device}")
    print(f"Training root:    {training_root}")
    print(f"Explanations root: {explanations_root}")
    print(f"Methods: {args.methods}")
    print(f"Seeds ({len(args.seeds)}): {args.seeds}")

    t0 = time.time()
    for seed in args.seeds:
        try:
            extract_for_seed(seed, training_root, args.label_source,
                             explanations_root, device, args.methods)
        except Exception as exc:
            import traceback
            print(f"  [seed={seed}] FAILED: {exc}")
            traceback.print_exc()

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
