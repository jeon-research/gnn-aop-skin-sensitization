"""
Train Random Forest + Morgan Fingerprint baseline for comparison.

Uses the same scaffold split as GNN models. Extracts feature importances
and maps them back to molecular substructures for AOP alignment comparison.

Usage:
    python scripts/explain/train_fingerprint_baseline.py [--seeds 42 123 ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROJECT_ROOT, RESULTS_DIR
from src.explain.utils import ALL_SEEDS, scaffold_split
from src.explain.aop_reference import AOPReference
from src.explain.alignment_metrics import compute_alignment_metrics, compute_batch_alignment
FP_RADIUS = 2
FP_NBITS = 2048


def smiles_to_fingerprint(smiles: str) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    return np.array(fp)


def get_bit_atom_mapping(smiles: str) -> dict:
    """Map Morgan FP bits to atom indices.

    Returns dict mapping bit_index -> set of atom indices that contribute to that bit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS, bitInfo=bit_info)

    # bit_info: {bit_idx: [(atom_center, radius), ...]}
    bit_to_atoms = {}
    for bit_idx, atom_radius_list in bit_info.items():
        atoms = set()
        for atom_center, radius in atom_radius_list:
            # Include the center atom and its neighborhood at the given radius
            if radius == 0:
                atoms.add(atom_center)
            else:
                # Get atoms within radius
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_center)
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())
        bit_to_atoms[bit_idx] = atoms

    return bit_to_atoms


def fp_importance_to_atom_importance(
    smiles: str,
    feature_importances: np.ndarray,
) -> torch.Tensor:
    """Map RF feature importances to atom-level importance scores.

    For each atom, sum the importances of all FP bits that involve that atom.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.tensor([])

    n_atoms = mol.GetNumAtoms()
    atom_importance = np.zeros(n_atoms)

    bit_to_atoms = get_bit_atom_mapping(smiles)

    for bit_idx, atoms in bit_to_atoms.items():
        for atom_idx in atoms:
            if atom_idx < n_atoms:
                atom_importance[atom_idx] += feature_importances[bit_idx]

    return torch.tensor(atom_importance, dtype=torch.float32)


def train_and_evaluate_seed(
    seed: int,
    output_dir: Path,
    aop_ref: AOPReference,
):
    """Train RF baseline and compute alignment for one seed."""
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    # Load data
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'causal_aop_comprehensive_v6.csv'
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['smiles', 'sensitization_human']).reset_index(drop=True)

    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(df, smiles_col='smiles', seed=seed)

    # Compute fingerprints
    fps = {}
    for idx in df.index:
        smi = df.loc[idx, 'smiles']
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps[idx] = fp

    # Filter to molecules with valid fingerprints
    valid_train = [i for i in train_idx if i in fps]
    valid_val = [i for i in val_idx if i in fps]
    valid_test = [i for i in test_idx if i in fps]

    X_train = np.array([fps[i] for i in valid_train])
    y_train = df.loc[valid_train, 'sensitization_human'].values.astype(int)
    X_val = np.array([fps[i] for i in valid_val])
    y_val = df.loc[valid_val, 'sensitization_human'].values.astype(int)
    X_test = np.array([fps[i] for i in valid_test])
    y_test = df.loc[valid_test, 'sensitization_human'].values.astype(int)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    test_probs = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    test_ap = average_precision_score(y_test, test_probs)
    print(f"  RF Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}")

    # Feature importances (MDI)
    feature_importances = rf.feature_importances_

    # Map to atom-level importance for alignment
    test_smiles = df.loc[valid_test, 'smiles'].tolist()

    # Get AOP reference
    aop_data = aop_ref.annotate_dataset(test_smiles, y_test)
    masks = aop_data['masks']
    mechanisms = aop_data['mechanisms']

    # Compute atom-level importances and alignment
    atom_importances = []
    valid_for_alignment = []

    for i, smi in enumerate(test_smiles):
        atom_imp = fp_importance_to_atom_importance(smi, feature_importances)
        atom_importances.append(atom_imp)

        if y_test[i] == 1 and masks[i].numel() > 0 and masks[i].sum() > 0:
            valid_for_alignment.append(i)

    # Compute alignment metrics
    valid_imps = [atom_importances[i] for i in valid_for_alignment]
    valid_refs = [masks[i] for i in valid_for_alignment]
    alignment = compute_batch_alignment(valid_imps, valid_refs)

    print(f"\n  Fingerprint Alignment (n={alignment.get('n_valid', 0)}):")
    for key in ['mean_atom_auc', 'mean_atom_ap', 'mean_hit_rate_at_k', 'mean_iou_at_k']:
        if key in alignment:
            std_key = key.replace('mean_', 'std_')
            print(f"    {key}: {alignment[key]:.4f} +/- {alignment.get(std_key, 0):.4f}")

    # Save results
    seed_dir = output_dir / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'seed': seed,
        'rf_test_auc': float(test_auc),
        'rf_test_ap': float(test_ap),
        'n_test': len(X_test),
        'alignment': {k: float(v) if isinstance(v, (float, np.floating)) else v
                      for k, v in alignment.items()},
    }

    with open(seed_dir / 'fingerprint_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {seed_dir / 'fingerprint_baseline.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train fingerprint baseline')
    parser.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'fingerprint_baseline'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aop_ref = AOPReference(use_extended=True)

    all_results = []
    for seed in args.seeds:
        try:
            result = train_and_evaluate_seed(seed, output_dir, aop_ref)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        aucs = [r['rf_test_auc'] for r in all_results]
        print(f"\n{'='*60}")
        print(f"RF Baseline Summary ({len(all_results)} seeds)")
        print(f"  Test AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

        atom_aucs = [r['alignment'].get('mean_atom_auc', float('nan')) for r in all_results]
        atom_aucs = [a for a in atom_aucs if not np.isnan(a)]
        if atom_aucs:
            print(f"  Atom-AUC alignment: {np.mean(atom_aucs):.4f} +/- {np.std(atom_aucs):.4f}")

        # Save summary
        summary = {
            'n_seeds': len(all_results),
            'rf_test_auc': {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))},
        }
        if atom_aucs:
            summary['atom_auc_alignment'] = {
                'mean': float(np.mean(atom_aucs)),
                'std': float(np.std(atom_aucs)),
            }
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
