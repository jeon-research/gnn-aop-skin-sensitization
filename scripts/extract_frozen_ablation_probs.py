"""Extract raw sensitization probabilities from the architecture-ablation
checkpoints (plain AttentiveFP / GCN / GIN) so classification_confusion.py
can compute confusion matrices at threshold=0.5.

Loads each checkpoint, runs inference on its test split, and writes:
  results/frozen_ablation_probs/{condition}_{arch}/seed_<N>/probs.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
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

from src.explain.utils import (
    set_seed, setup_device, load_dataset, featurize_molecules, load_model,
)  # type: ignore


from senslib.seeds import ALL as ALL_SEEDS  # noqa: E402

OUT_ROOT = REVISION_ROOT / 'results' / 'frozen_ablation_probs'


def _run_arch(architecture: str, condition_label: str, seeds: List[int],
              device: torch.device) -> int:
    """Iterate seeds, write probs.json per seed. Returns how many succeeded."""
    ok = 0
    for seed in seeds:
        out_dir = OUT_ROOT / f'{condition_label}_{architecture}' / f'seed_{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            set_seed(seed)
            model = load_model(seed, device, architecture=architecture)
        except Exception as e:
            print(f"  [seed={seed} arch={architecture}] load failed: {e}")
            continue

        df, _, _, test_idx = load_dataset(seed)
        test_df = df.loc[test_idx].reset_index(drop=True)
        smiles = test_df['smiles'].tolist()
        labels = test_df['sensitization_human'].astype(float).tolist()

        graphs, valid = featurize_molecules(smiles)
        valid_labels = [labels[i] for i in valid]

        probs: List[float] = []
        for g in graphs:
            g = g.clone().to(device)
            if not hasattr(g, 'batch') or g.batch is None:
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(g)
                probs.append(float(torch.sigmoid(out['sensitization']).item()))

        payload = {
            'seed':         seed,
            'architecture': architecture,
            'condition':    condition_label,
            'n':            len(valid_labels),
            'n_pos':        int(sum(1 for l in valid_labels if l == 1)),
            'probs':        probs,
            'labels':       valid_labels,
        }
        (out_dir / 'probs.json').write_text(json.dumps(payload))
        print(f"  [seed={seed} arch={architecture}] wrote {len(probs)} preds "
              f"(pos_rate={sum(1 for l in valid_labels if l==1) / max(len(valid_labels), 1):.2f})")
        ok += 1
    return ok


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = setup_device()
    print(f'Device: {device}')

    # We only need the 20 primary seeds; these match the paper's reported set.
    for arch, cond in (('attentivefp', 'plain'),
                       ('gcn',         'plain_gcn'),
                       ('gin',         'plain_gin')):
        print(f'\n=== {cond} ({arch}) ===')
        n = _run_arch(arch, cond, ALL_SEEDS, device)
        print(f'  done: {n} seeds')


if __name__ == '__main__':
    main()
