"""Train AblationGNN on the 1,118-molecule human-label subset
(broader cross-check, §3.5 onward). Mirrors scripts/train_llna_labels.py
with sensitization_human as the label column.

Outputs:
  results/human_training/seed_<N>/                (primary)
  results/shuffle_control_human/seed_<N>/         (--shuffle)
  results/human_training_pw{4,8,12}/seed_<N>/     (--pos-weight, --output-tag)

Usage:
  python3 scripts/train_human_labels.py --shuffle
  python3 scripts/train_human_labels.py --pos-weight 4 --output-tag pw4
  python3 scripts/train_human_labels.py --pos-weight 8 --output-tag pw8
  python3 scripts/train_human_labels.py --pos-weight 12 --output-tag pw12
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
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, average_precision_score

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

from src.explain.utils import set_seed, setup_device, featurize_molecules  # type: ignore
from src.modeling.ablation_model import AblationGNN                          # type: ignore
from senslib.human_loader import (
    HUMAN_LABEL_COL,
    load_dataset_human,
    load_dataset_human_shuffled,
)
from senslib.llna_loader import persist_split


from senslib.seeds import PRIMARY as DEFAULT_SEEDS  # noqa: E402
OUTPUT_ROOT_HUMAN   = REVISION_ROOT / 'results' / 'human_training'
OUTPUT_ROOT_SHUFFLE = REVISION_ROOT / 'results' / 'shuffle_control_human'


from senslib._training import build_split, make_loader, train_epoch, evaluate  # noqa: E402


def train_one_seed(seed: int, n_epochs: int, batch_size: int, patience: int,
                   lr: float, shuffle_labels: bool,
                   pos_weight_override: float = None,
                   output_tag: str = None):
    set_seed(seed)
    device = setup_device()

    label_source = 'shuffled_human' if shuffle_labels else 'human'
    out_root = OUTPUT_ROOT_SHUFFLE if shuffle_labels else OUTPUT_ROOT_HUMAN
    if output_tag:
        out_root = out_root.parent / f'{out_root.name}_{output_tag}'
    out_dir = out_root / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if shuffle_labels:
        df, train_idx, val_idx, test_idx = load_dataset_human_shuffled(seed)
    else:
        df, train_idx, val_idx, test_idx = load_dataset_human(seed)
    persist_split(seed, label_source, df, train_idx, val_idx, test_idx)

    print(f"\n[seed={seed}] {label_source} training")
    print(f"  dataset: {len(df)} mols, label={HUMAN_LABEL_COL}")
    print(f"  split:   train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    smiles = df['smiles'].tolist()
    graphs, valid = featurize_molecules(smiles)
    orig_to_graph = {o: i for i, o in enumerate(valid)}

    train_g, train_y = build_split(df, train_idx, HUMAN_LABEL_COL, graphs, orig_to_graph)
    val_g,   val_y   = build_split(df, val_idx,   HUMAN_LABEL_COL, graphs, orig_to_graph)
    test_g,  test_y  = build_split(df, test_idx,  HUMAN_LABEL_COL, graphs, orig_to_graph)
    print(f"  after featurization: train={len(train_g)} val={len(val_g)} test={len(test_g)}")

    train_loader = make_loader(train_g, batch_size, shuffle=True, seed=seed)
    val_loader   = make_loader(val_g,   batch_size, shuffle=False, seed=seed)
    test_loader  = make_loader(test_g,  batch_size, shuffle=False, seed=seed)

    if train_y.size == 0:
        raise RuntimeError(f"seed={seed}: empty training set")
    n_pos = float((train_y == 1).sum())
    n_neg = float((train_y == 0).sum())
    data_pos_weight = max(1.0, n_neg / max(n_pos, 1.0))
    pos_weight = (pos_weight_override if pos_weight_override is not None
                  else data_pos_weight)
    print(f"  class balance: pos={int(n_pos)} neg={int(n_neg)}  "
          f"data_pos_weight={data_pos_weight:.2f}  using={pos_weight:.2f}")

    model = AblationGNN(
        condition='plain', architecture='attentivefp',
        hidden_dim=256, node_dim=64, num_gnn_layers=3,
        dropout=0.3, use_continuous_features=False,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = -1.0
    best_epoch = 0
    history: List[Dict] = []
    since_best = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        rec = {'epoch': epoch, 'train_loss': train_loss,
               'val_auc': val_metrics['auc'], 'val_ap': val_metrics['ap']}
        history.append(rec)

        if not np.isnan(val_metrics['auc']) and val_metrics['auc'] > best_val:
            best_val = val_metrics['auc']
            best_epoch = epoch
            since_best = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'seed': seed,
                'label_source': label_source,
                'val_auc': val_metrics['auc'],
                'val_ap':  val_metrics['ap'],
                'config': {
                    'hidden_dim': 256, 'node_dim': 64,
                    'num_gnn_layers': 3, 'dropout': 0.3,
                    'architecture': 'attentivefp',
                },
            }, out_dir / 'best_model.pt')
        else:
            since_best += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_auc={val_metrics['auc']:.4f}  best={best_val:.4f}@{best_epoch}")

        if since_best >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device)

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s — test AUC={test_metrics['auc']:.4f} "
          f"AP={test_metrics['ap']:.4f}  (n={test_metrics['n']}, pos={test_metrics['n_pos']})")

    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    (out_dir / 'metrics.json').write_text(json.dumps({
        'seed': seed, 'label_source': label_source,
        'best_val_auc': best_val, 'best_epoch': best_epoch,
        'test_auc':   test_metrics['auc'],
        'test_ap':    test_metrics['ap'],
        'test_n':     test_metrics['n'],
        'test_n_pos': test_metrics['n_pos'],
        'test_probs': test_metrics.get('probs'),
        'test_labels': test_metrics.get('labels'),
        'pos_weight': pos_weight,
        'elapsed_sec': elapsed,
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--shuffle', action='store_true',
                   help='Shuffle labels deterministically per seed (negative control).')
    p.add_argument('--pos-weight', type=float, default=None,
                   help='Override pos_weight (default: data-derived).')
    p.add_argument('--output-tag', type=str, default=None,
                   help='Suffix appended to output root (for pos_weight sweeps).')
    args = p.parse_args()

    base = OUTPUT_ROOT_SHUFFLE if args.shuffle else OUTPUT_ROOT_HUMAN
    root = base.parent / f'{base.name}_{args.output_tag}' if args.output_tag else base
    root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        train_one_seed(seed, args.epochs, args.batch_size, args.patience,
                       args.lr,
                       shuffle_labels=args.shuffle,
                       pos_weight_override=args.pos_weight,
                       output_tag=args.output_tag)


if __name__ == '__main__':
    main()
