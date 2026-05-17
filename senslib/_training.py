"""Training helpers shared by scripts/train_llna_labels.py and scripts/train_human_labels.py."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Batch


def build_split(df, indices, label_col, graphs, orig_to_graph) -> Tuple[list, np.ndarray]:
    xs, ys = [], []
    for idx in indices:
        if idx not in orig_to_graph:
            continue
        val = df.iloc[idx][label_col]
        if np.isnan(val):
            continue
        g = graphs[orig_to_graph[idx]].clone()
        g.y = torch.tensor([float(val)], dtype=torch.float32)
        xs.append(g)
        ys.append(float(val))
    return xs, np.asarray(ys)


def make_loader(graphs, batch_size, shuffle, seed):
    from torch.utils.data import DataLoader
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        graphs, batch_size=batch_size, shuffle=shuffle,
        collate_fn=Batch.from_data_list, drop_last=False, generator=gen,
    )


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out['sensitization'], batch.y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        p = torch.sigmoid(out['sensitization']).detach().cpu().numpy().ravel()
        probs.extend(p.tolist())
        labels.extend(np.atleast_1d(batch.y.reshape(-1).cpu().numpy()).tolist())
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {'auc': float('nan'), 'ap': float('nan'),
                'n': int(len(labels)), 'n_pos': int(labels.sum())}
    return {
        'auc': float(roc_auc_score(labels, probs)),
        'ap':  float(average_precision_score(labels, probs)),
        'n':   int(len(labels)),
        'n_pos': int(labels.sum()),
        'probs': probs.tolist(),
        'labels': labels.tolist(),
    }
