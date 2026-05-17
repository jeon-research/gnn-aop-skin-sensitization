"""Sanity tests for the sensitization-specific alert set.

Intentionally small: known sensitizers must hit a relevant mechanism, and
known non-sensitizers must not produce spurious hits. Run with:

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repository root importable so senslib resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senslib.aop_reference_sensitization import (
    AOPReferenceSensitization,
    pattern_count,
    pattern_inventory,
)


@pytest.fixture(scope='module')
def ref() -> AOPReferenceSensitization:
    return AOPReferenceSensitization()


# --- known skin sensitizers -------------------------------------------------

SENSITIZERS = {
    # SMILES, expected primary mechanism
    'cinnamaldehyde':      ('O=C/C=C/c1ccccc1', 'michael_addition'),
    'DNCB':                ('O=[N+]([O-])c1ccc(Cl)c([N+](=O)[O-])c1', 'snar'),
    'PPD':                 ('Nc1ccc(N)cc1', 'pro_hapten'),
    'formaldehyde':        ('C=O', 'schiff_base'),
    'glutaraldehyde':      ('O=CCCCC=O', 'schiff_base'),
    'eugenol':             ('C=CCc1ccc(O)c(OC)c1', 'pro_hapten'),
    'methyl_acrylate':     ('C=CC(=O)OC', 'michael_addition'),
    'glycidol':            ('OCC1CO1', 'sn2'),                    # epoxide
    'benzyl_chloride':     ('ClCc1ccccc1', 'sn2'),
    'phthalic_anhydride':  ('O=C1OC(=O)c2ccccc12', 'acyl_transfer'),
}


@pytest.mark.parametrize("name, smi_mech", SENSITIZERS.items())
def test_known_sensitizer_matches(ref, name, smi_mech):
    smi, expected_mech = smi_mech
    mask, info = ref.get_atom_mask(smi)
    assert mask.numel() > 0, f"{name}: could not parse SMILES"
    assert mask.sum() > 0, f"{name}: no atoms matched, expected {expected_mech}"

    cls = ref.classify_mechanism(smi)
    assert expected_mech in cls['mechanisms'], (
        f"{name}: expected {expected_mech}, got {cls['mechanisms']}"
    )


# --- known non-sensitizers --------------------------------------------------

NON_SENSITIZERS = {
    'n_hexane':       'CCCCCC',
    'benzene':        'c1ccccc1',
    'sucrose':        'OCC1OC(OC2(COC3(CO)OC(CO)C(O)C3O)OC(CO)C2O)C(O)C(O)C1O',
    'glucose':        'OCC1OC(O)C(O)C(O)C1O',
    'ethanol':        'CCO',
    'glycine':        'NCC(=O)O',
}


@pytest.mark.parametrize("name, smi", NON_SENSITIZERS.items())
def test_known_non_sensitizer_clean(ref, name, smi):
    mask, info = ref.get_atom_mask(smi)
    # A non-sensitizer may hit nothing, or it may hit incidentally through a
    # pro-hapten alert (e.g. primary amines in glycine aren't aromatic so
    # should not fire). We require zero matched patterns.
    assert info.get('n_mie_atoms', 0) == 0, (
        f"{name}: unexpectedly matched {info.get('matched_patterns')}"
    )


# --- interface checks -------------------------------------------------------

def test_reactive_center_mask_strictly_subset_of_union(ref):
    smi = 'O=C/C=C/c1ccccc1'  # cinnamaldehyde
    full, _ = ref.get_atom_mask(smi, mode='union')
    center, _ = ref.get_reactive_center_mask(smi)
    assert center.sum() <= full.sum()
    assert center.sum() >= 1


def test_intersection_mode_is_subset_of_union(ref):
    # PPD has two amino groups; each is independently matched only by the
    # prohapten_para_phenylenediamine pattern, so intersection (≥2 distinct
    # patterns per atom) should be empty or smaller than union.
    smi = 'Nc1ccc(N)cc1'
    union, _ = ref.get_atom_mask(smi, mode='union')
    inter, _ = ref.get_atom_mask(smi, mode='intersection')
    assert inter.sum() <= union.sum()


def test_pattern_inventory_covers_five_domains_plus_prohapten(ref):
    inv = pattern_inventory()
    assert set(inv) == {
        'michael_addition', 'schiff_base', 'acyl_transfer',
        'sn2', 'snar', 'pro_hapten',
    }
    # Each domain should have at least one pattern.
    assert all(len(v) >= 1 for v in inv.values())


def test_pattern_count_is_reasonable():
    n = pattern_count()
    # Intentionally narrow vs the old set (~80+ patterns). Keep a sanity
    # envelope — 30 to 70 feels right for a sensitization-only library.
    assert 30 <= n <= 70, f"pattern_count={n} outside sanity envelope"
