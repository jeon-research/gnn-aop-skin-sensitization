"""Six data-free per-atom attribution baselines (Section S5):
random, atom degree, aromatic indicator, heteroatom indicator, atomic mass,
heavy-atom count. Reuses SMILES and test-set boundaries from a reference
explanations.json so downstream alignment can compare directly to GNN
methods. Output layout mirrors a normal extraction
(results/<output-root>/seed_<N>/explanations.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdkit import Chem

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

from senslib.seeds import PRIMARY as DEFAULT_SEEDS  # noqa: E402


def _random_importance(n_atoms: int, rng: np.random.RandomState) -> List[float]:
    return rng.rand(n_atoms).tolist()


def _degree_importance(mol: Chem.Mol) -> List[float]:
    return [float(a.GetDegree()) for a in mol.GetAtoms()]


def _aromatic_importance(mol: Chem.Mol) -> List[float]:
    return [1.0 if a.GetIsAromatic() else 0.0 for a in mol.GetAtoms()]


def _heteroatom_importance(mol: Chem.Mol) -> List[float]:
    return [0.0 if a.GetAtomicNum() == 6 else 1.0 for a in mol.GetAtoms()]


def _atomic_mass_importance(mol: Chem.Mol) -> List[float]:
    return [float(a.GetMass()) for a in mol.GetAtoms()]


def _heavy_atom_importance(mol: Chem.Mol) -> List[float]:
    n_heavy = float(mol.GetNumHeavyAtoms())
    return [n_heavy] * mol.GetNumAtoms()


def build_for_seed(seed: int, reference: Dict, output_dir: Path,
                   rng_seed: int) -> Dict:
    smiles = reference['smiles']
    labels = reference.get('labels', [0.0] * len(smiles))

    rng = np.random.RandomState(rng_seed)

    explanations: Dict[str, List[List[float]]] = {
        'random':       [],
        'degree':       [],
        'aromatic':     [],
        'heteroatom':   [],
        'atomic_mass':  [],
        'heavy_atom':   [],
    }

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for k in explanations:
                explanations[k].append([0.0])
            continue
        n = mol.GetNumAtoms()
        explanations['random'].append(_random_importance(n, rng))
        explanations['degree'].append(_degree_importance(mol))
        explanations['aromatic'].append(_aromatic_importance(mol))
        explanations['heteroatom'].append(_heteroatom_importance(mol))
        explanations['atomic_mass'].append(_atomic_mass_importance(mol))
        explanations['heavy_atom'].append(_heavy_atom_importance(mol))

    payload = {
        'seed': seed,
        'n_molecules': len(smiles),
        'smiles': smiles,
        'labels': labels,
        'predictions': [0.5] * len(smiles),
        'explanations': explanations,
        'source': 'neutral-baselines (random/degree/aromatic/heteroatom)',
    }

    seed_dir = output_dir / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / 'explanations.json').write_text(json.dumps(payload))
    print(f"  [seed={seed}] wrote {len(smiles)} mols, {len(explanations)} baselines")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--reference-root', type=Path,
                   default=REVISION_ROOT / 'results' / 'explanations_llna',
                   help='Use SMILES and labels from this explanations tree.')
    p.add_argument('--output-root', type=Path,
                   default=REVISION_ROOT / 'results' / 'explanations_baselines')
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Reference root: {args.reference_root}")
    print(f"Output root:    {args.output_root}")

    for seed in args.seeds:
        ref_path = args.reference_root / f'seed_{seed}' / 'explanations.json'
        if not ref_path.exists():
            print(f"  [seed={seed}] missing reference, skipping")
            continue
        reference = json.loads(ref_path.read_text())
        # Use seed XOR pattern so random is reproducible but decoupled from
        # training seed.
        build_for_seed(seed, reference, args.output_root, rng_seed=seed ^ 0xBAADC0DE)


if __name__ == '__main__':
    main()
