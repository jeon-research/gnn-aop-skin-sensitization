"""
Train MechGNN: GNN with atom-level MIE supervision.

Fine-tunes from pretrained AblationGNN (plain) checkpoints, adding a per-atom
MIE prediction head alongside the existing sensitization head.

Loss = L_sens + λ * L_atom_mie

Usage:
    python scripts/explain/train_mechgnn.py --seed 42 --lambda-mie 0.5
    python scripts/explain/train_mechgnn.py --seed 42 --lambda-mie 0.5 --from-scratch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RESULTS_DIR
from src.explain.utils import (
    ALL_SEEDS, load_dataset, featurize_molecules, setup_device, find_checkpoint,
    set_seed,
)
from src.explain.aop_reference import AOPReference
from src.modeling.mech_gnn import MechGNN


def generate_atom_mie_labels(smiles_list: list, graphs: list, valid_indices: list):
    """Generate atom-level MIE masks for all molecules.

    Uses AOPReference.get_atom_mask() which matches the full MIE substructure
    (broader than reactive-center-only). Labels ALL molecules regardless of
    sensitization status — teaches pattern recognition independent of outcome.

    Returns:
        atom_mie_masks: list of [n_atoms] tensors aligned with graphs.
        stats: dict with labeling statistics.
    """
    aop_ref = AOPReference(use_extended=True)
    atom_mie_masks = []
    n_with_mie = 0
    total_atoms = 0
    total_mie_atoms = 0

    for idx in valid_indices:
        smi = smiles_list[idx]
        mask, info = aop_ref.get_atom_mask(smi)
        atom_mie_masks.append(mask)
        if mask.numel() > 0 and mask.sum() > 0:
            n_with_mie += 1
        total_atoms += mask.numel()
        total_mie_atoms += int(mask.sum())

    mie_fraction = total_mie_atoms / total_atoms if total_atoms > 0 else 0.0
    stats = {
        'n_molecules': len(valid_indices),
        'n_with_mie': n_with_mie,
        'total_atoms': total_atoms,
        'total_mie_atoms': total_mie_atoms,
        'mie_atom_fraction': mie_fraction,
        'suggested_pos_weight': (1.0 - mie_fraction) / mie_fraction if mie_fraction > 0 else 10.0,
    }
    return atom_mie_masks, stats


def collate_with_mie(batch_items):
    """Custom collation that includes atom-level MIE masks in the batch.

    Args:
        batch_items: list of (graph, sens_label, atom_mie_mask) tuples.

    Returns:
        batched_data: Batch with .y for sensitization labels.
        mie_mask: [total_atoms] concatenated MIE mask.
    """
    graphs, labels, mie_masks = zip(*batch_items)

    # Set labels on graphs for batching
    graph_list = []
    for g, label in zip(graphs, labels):
        g_clone = g.clone()
        g_clone.y = torch.tensor([label], dtype=torch.float32)
        graph_list.append(g_clone)

    batched = Batch.from_data_list(graph_list)
    mie_mask = torch.cat(mie_masks, dim=0)
    return batched, mie_mask


class MechGNNDataset:
    """Simple dataset for MechGNN training."""

    def __init__(self, graphs, labels, mie_masks):
        self.graphs = graphs
        self.labels = labels
        self.mie_masks = mie_masks

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.mie_masks[idx]


def create_dataloader(dataset, batch_size=32, shuffle=True, seed=42):
    """Create DataLoader with custom collation."""
    from torch.utils.data import DataLoader
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_with_mie,
        drop_last=False,
        generator=generator,
    )


def train_epoch(model, loader, optimizer, sens_criterion, mie_criterion,
                lambda_mie, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_sens_loss = 0.0
    total_mie_loss = 0.0
    n_batches = 0

    for batch_data, mie_mask in loader:
        batch_data = batch_data.to(device)
        mie_mask = mie_mask.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)

        # Sensitization loss
        sens_logits = outputs['sensitization']
        sens_labels = batch_data.y
        loss_sens = sens_criterion(sens_logits, sens_labels)

        # Atom MIE loss
        atom_mie_logits = outputs['atom_mie_logits']
        loss_mie = mie_criterion(atom_mie_logits, mie_mask)

        # Combined loss
        loss = loss_sens + lambda_mie * loss_mie

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_sens_loss += loss_sens.item()
        total_mie_loss += loss_mie.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'sens_loss': total_sens_loss / n_batches,
        'mie_loss': total_mie_loss / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, sens_criterion, mie_criterion, lambda_mie, device):
    """Evaluate model on a dataset split."""
    model.eval()
    all_sens_logits = []
    all_sens_labels = []
    all_mie_logits = []
    all_mie_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch_data, mie_mask in loader:
        batch_data = batch_data.to(device)
        mie_mask = mie_mask.to(device)

        outputs = model(batch_data)

        sens_logits = outputs['sensitization']
        sens_labels = batch_data.y
        loss_sens = sens_criterion(sens_logits, sens_labels)

        atom_mie_logits = outputs['atom_mie_logits']
        loss_mie = mie_criterion(atom_mie_logits, mie_mask)
        loss = loss_sens + lambda_mie * loss_mie

        all_sens_logits.append(sens_logits.cpu())
        all_sens_labels.append(sens_labels.cpu())
        all_mie_logits.append(atom_mie_logits.cpu())
        all_mie_labels.append(mie_mask.cpu())
        total_loss += loss.item()
        n_batches += 1

    all_sens_logits = torch.cat(all_sens_logits)
    all_sens_labels = torch.cat(all_sens_labels)
    all_mie_logits = torch.cat(all_mie_logits)
    all_mie_labels = torch.cat(all_mie_labels)

    sens_probs = torch.sigmoid(all_sens_logits).numpy()
    sens_labels_np = all_sens_labels.numpy()
    mie_probs = torch.sigmoid(all_mie_logits).numpy()
    mie_labels_np = all_mie_labels.numpy()

    metrics = {
        'loss': total_loss / n_batches,
    }

    # Sensitization AUC
    try:
        metrics['sens_auc'] = roc_auc_score(sens_labels_np, sens_probs)
        metrics['sens_ap'] = average_precision_score(sens_labels_np, sens_probs)
    except ValueError:
        metrics['sens_auc'] = 0.5
        metrics['sens_ap'] = 0.5

    # Atom MIE AUC (only meaningful if there are positive atoms)
    if mie_labels_np.sum() > 0 and mie_labels_np.sum() < len(mie_labels_np):
        try:
            metrics['mie_auc'] = roc_auc_score(mie_labels_np, mie_probs)
        except ValueError:
            metrics['mie_auc'] = 0.5
    else:
        metrics['mie_auc'] = 0.5

    return metrics


def train_mechgnn(
    seed: int,
    lambda_mie: float,
    from_scratch: bool = False,
    n_epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    output_dir: Path = None,
):
    """Train a MechGNN model for one seed and lambda value."""
    set_seed(seed)
    device = setup_device()

    if output_dir is None:
        output_dir = RESULTS_DIR / 'mechgnn' / f'lambda_{lambda_mie}' / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MechGNN Training: seed={seed}, λ_mie={lambda_mie}, from_scratch={from_scratch}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load and split data
    df, train_idx, val_idx, test_idx = load_dataset(seed)
    smiles_list = df['smiles'].tolist()
    labels = df['sensitization_human'].values

    print(f"  Dataset: {len(df)} molecules")
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Featurize all molecules
    all_graphs, all_valid = featurize_molecules(smiles_list)
    valid_set = set(all_valid)

    # Build index map: original_idx -> position in all_graphs
    orig_to_graph = {orig_idx: pos for pos, orig_idx in enumerate(all_valid)}

    # Generate atom-level MIE labels for all valid molecules
    print("  Generating atom-level MIE labels...")
    atom_mie_masks, mie_stats = generate_atom_mie_labels(smiles_list, all_graphs, all_valid)
    print(f"  MIE stats: {mie_stats['n_with_mie']}/{mie_stats['n_molecules']} molecules "
          f"have MIE atoms ({mie_stats['mie_atom_fraction']:.3f} atom fraction)")

    # Build split datasets
    def build_split(split_indices):
        split_graphs = []
        split_labels = []
        split_mie = []
        for idx in split_indices:
            if idx in orig_to_graph:
                pos = orig_to_graph[idx]
                split_graphs.append(all_graphs[pos])
                split_labels.append(float(labels[idx]))
                split_mie.append(atom_mie_masks[pos])
        return MechGNNDataset(split_graphs, split_labels, split_mie)

    train_dataset = build_split(train_idx)
    val_dataset = build_split(val_idx)
    test_dataset = build_split(test_idx)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)

    # Create model
    model = MechGNN(hidden_dim=256, node_dim=64, num_gnn_layers=3, dropout=0.3)

    if not from_scratch:
        checkpoint_path = find_checkpoint(seed)
        model.load_pretrained(str(checkpoint_path), device=device)
        lr = 1e-4  # Fine-tune learning rate
    else:
        lr = 1e-3  # From-scratch learning rate
        print("  Training from scratch")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Loss functions
    # Sensitization: class-imbalanced (pos_weight=8.0, same as original training)
    sens_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([8.0], device=device)
    )
    # Atom MIE: heavily imbalanced (compute from data)
    mie_pos_weight = torch.tensor([mie_stats['suggested_pos_weight']], device=device)
    mie_criterion = nn.BCEWithLogitsLoss(pos_weight=mie_pos_weight)
    print(f"  MIE pos_weight: {mie_pos_weight.item():.1f}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training loop
    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, sens_criterion, mie_criterion,
            lambda_mie, device,
        )
        val_metrics = evaluate(
            model, val_loader, sens_criterion, mie_criterion, lambda_mie, device,
        )

        epoch_record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_sens_loss': train_metrics['sens_loss'],
            'train_mie_loss': train_metrics['mie_loss'],
            'val_loss': val_metrics['loss'],
            'val_sens_auc': val_metrics['sens_auc'],
            'val_sens_ap': val_metrics['sens_ap'],
            'val_mie_auc': val_metrics['mie_auc'],
        }
        history.append(epoch_record)

        # Early stopping on val sensitization AUC
        if val_metrics['sens_auc'] > best_val_auc:
            best_val_auc = val_metrics['sens_auc']
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_sens_auc': best_val_auc,
                'lambda_mie': lambda_mie,
                'seed': seed,
                'from_scratch': from_scratch,
            }, output_dir / 'best_model.pt')
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d} | "
                  f"Train loss: {train_metrics['loss']:.4f} "
                  f"(sens: {train_metrics['sens_loss']:.4f}, mie: {train_metrics['mie_loss']:.4f}) | "
                  f"Val AUC: {val_metrics['sens_auc']:.4f} | "
                  f"Val MIE AUC: {val_metrics['mie_auc']:.4f} | "
                  f"{elapsed:.0f}s")

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Evaluate on test set with best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(
        model, test_loader, sens_criterion, mie_criterion, lambda_mie, device,
    )

    elapsed = time.time() - t_start
    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Val AUC: {best_val_auc:.4f}")
    print(f"  Test AUC: {test_metrics['sens_auc']:.4f} | "
          f"Test AP: {test_metrics['sens_ap']:.4f} | "
          f"Test MIE AUC: {test_metrics['mie_auc']:.4f}")
    print(f"  Time: {elapsed:.0f}s")

    # Save results
    results = {
        'seed': seed,
        'lambda_mie': lambda_mie,
        'from_scratch': from_scratch,
        'best_epoch': best_epoch,
        'val_sens_auc': best_val_auc,
        'test_sens_auc': test_metrics['sens_auc'],
        'test_sens_ap': test_metrics['sens_ap'],
        'test_mie_auc': test_metrics['mie_auc'],
        'mie_stats': mie_stats,
        'n_params': n_params,
        'lr': lr,
        'elapsed_seconds': elapsed,
        'history': history,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Train MechGNN')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--lambda-mie', type=float, default=0.5,
                        help='Weight for atom MIE loss (default: 0.5)')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train from scratch instead of fine-tuning')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    train_mechgnn(
        seed=args.seed,
        lambda_mie=args.lambda_mie,
        from_scratch=args.from_scratch,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    main()
