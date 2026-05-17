"""Merge 4-method LLNA explanations with 3-method perturbation explanations
into a combined 7-method directory. Per-seed JSON has the same schema; we
just take the union of the explanations dictionary.

Source dirs:
  results/explanations_llna/seed_*/explanations.json (ig, gradcam, attention, ensemble)
  results/explanations_llna_perturbation/seed_*/explanations.json (gnnexplainer, pgexplainer, graphmask)

Output:
  results/explanations_llna_7method/seed_*/explanations.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REVISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REVISION_ROOT))


def merge_seed(seed: int, src_a: Path, src_b: Path, dst: Path) -> None:
    a_path = src_a / f'seed_{seed}' / 'explanations.json'
    b_path = src_b / f'seed_{seed}' / 'explanations.json'
    if not a_path.exists() or not b_path.exists():
        raise FileNotFoundError(f"seed {seed}: missing {a_path} or {b_path}")

    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())

    # Sanity checks: SMILES list and labels must match
    if a['smiles'] != b['smiles']:
        raise RuntimeError(f"seed {seed}: SMILES mismatch between sources")
    if a['labels'] != b['labels']:
        raise RuntimeError(f"seed {seed}: labels mismatch between sources")
    if a['n_molecules'] != b['n_molecules']:
        raise RuntimeError(f"seed {seed}: n_molecules mismatch")

    merged_explanations = dict(a['explanations'])
    for k, v in b['explanations'].items():
        if k in merged_explanations:
            raise RuntimeError(f"seed {seed}: method {k} present in both sources")
        merged_explanations[k] = v

    out = {
        'seed': a['seed'],
        'label_source': a['label_source'],
        'n_molecules': a['n_molecules'],
        'smiles': a['smiles'],
        'labels': a['labels'],
        'predictions': a['predictions'],  # predictions from the 4-method run
        'explanations': merged_explanations,
    }

    seed_dir = dst / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / 'explanations.json').write_text(json.dumps(out))
    print(f"  [seed={seed}] merged "
          f"{len(merged_explanations)} methods: {sorted(merged_explanations.keys())}")


def main():
    from senslib.seeds import ALL
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=list(ALL))
    p.add_argument('--source-a', type=Path,
                   default=REVISION_ROOT / 'results' / 'explanations_llna')
    p.add_argument('--source-b', type=Path,
                   default=REVISION_ROOT / 'results' / 'explanations_llna_perturbation')
    p.add_argument('--output', type=Path,
                   default=REVISION_ROOT / 'results' / 'explanations_llna_7method')
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Merging:\n  A: {args.source_a}\n  B: {args.source_b}\n  -> {args.output}")
    for seed in args.seeds:
        try:
            merge_seed(seed, args.source_a, args.source_b, args.output)
        except Exception as e:
            print(f"  [seed={seed}] FAILED: {e}")


if __name__ == '__main__':
    main()
