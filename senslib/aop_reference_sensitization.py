"""
Sensitization-specific SMARTS alert set.

A 50-pattern alert set restricted to chemistry that is well-established as
relevant to the skin-sensitization molecular initiating event. The patterns
are drawn from:

  - Enoch, S. J.; Madden, J. C.; Cronin, M. T. D.
    "Identification of mechanisms of toxic action for skin sensitisation using
     a SMARTS pattern based approach."
    SAR and QSAR in Environmental Research 19 (5-6), 555-578 (2008).

  - Patlewicz, G. et al.
    "TIMES-SS — A Promising Tool for the Assessment of Skin Sensitization
     Hazard."
    Regulatory Toxicology and Pharmacology 48, 225-239 (2007).

  - Roberts, D. W.; Aptula, A. O.
    "Determinants of Skin Sensitisation Potential."
    Journal of Applied Toxicology 28, 377-387 (2008).

Mechanistic scope (Enoch 2008, five reaction domains):
  1. Michael addition — α,β-unsaturated carbonyls, quinones, vinyl sulfones.
  2. Schiff base formation — aldehydes, α-ketoaldehydes.
  3. Acyl transfer — anhydrides, isocyanates, acyl halides, activated esters.
  4. SN2 — alkyl halides, epoxides, aziridines, sulfonate esters.
  5. SNAr — activated aryl halides (nitro-activated, perfluoro-activated).

Plus a narrow pro-hapten set for chemistry whose activation to skin-sensitizing
electrophiles is well-documented (aromatic amines → diazonium / quinone imine;
phenols / hydroquinones / catechols → quinones; allylphenols → quinone
methides). Metal chelators, hydroperoxides, nitroso compounds, and generic
thiols are excluded as they are not mechanistically tied to the
skin-sensitization MIE.

The module mirrors the interface of src/explain/aop_reference.AOPReference so
it can be used as a drop-in replacement in the alignment pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem


# Pattern library — sensitization-specific, grouped by Enoch 2008 mechanism.

# 1. Michael addition
MICHAEL_PATTERNS: Dict[str, str] = {
    'michael_enone':            '[#6]=[#6]-C(=O)-[#6]',          # α,β-unsat ketone (Enoch 2008)
    'michael_enal':             '[#6]=[#6]-[CH]=O',              # α,β-unsat aldehyde (cinnamaldehyde)
    'michael_acrylate':         '[#6]=[#6]-C(=O)O',              # acrylate ester
    'michael_acrylamide':       '[#6]=[#6]-C(=O)N',              # acrylamide (Enoch 2008)
    'michael_acrylonitrile':    '[#6]=[#6]-C#N',                 # acrylonitrile
    'michael_vinyl_sulfone':    '[#6]=[#6]-S(=O)(=O)',           # vinyl sulfone (Patlewicz 2007)
    'michael_maleimide':        'O=C1[#6]=[#6]C(=O)N1',          # maleimide
    'michael_benzoquinone':     'O=C1C=CC(=O)C=C1',              # para-benzoquinone
    'michael_naphthoquinone':   'O=C1C=CC(=O)c2ccccc12',         # naphthoquinone
    'michael_butenolide':       'O=C1OCC=C1',                    # α,β-unsat γ-lactone
    'michael_cyanoacrylate':    '[#6]=[#6](C#N)C(=O)O',          # cyanoacrylate
}

# 2. Schiff base formation
SCHIFF_PATTERNS: Dict[str, str] = {
    'schiff_aliphatic_aldehyde':  '[CX4][CH]=O',                 # saturated aliphatic aldehyde
    'schiff_aromatic_aldehyde':   'c[CH]=O',                     # benzaldehyde-type
    'schiff_formaldehyde':        '[CH2]=O',                     # formaldehyde / releasers
    'schiff_glutaraldehyde':      'O=CCCCC=O',                   # dialdehyde bridging (Roberts 2008)
    'schiff_glyoxal':             'O=CC=O',                      # 1,2-dicarbonyl
    'schiff_alpha_ketoaldehyde':  'O=CC(=O)[#6]',                # pyruvaldehyde-type
}

# 3. Acyl transfer
ACYL_PATTERNS: Dict[str, str] = {
    'acyl_anhydride':             'C(=O)OC(=O)',                 # mixed / symmetric anhydride
    'acyl_cyclic_anhydride':      'O=C1OC(=O)[#6][#6]1',         # succinic / maleic anhydride
    'acyl_isocyanate':            'N=C=O',                       # isocyanate
    'acyl_isothiocyanate':        'N=C=S',                       # isothiocyanate (Enoch 2008)
    'acyl_chloride':              'C(=O)Cl',                     # acyl chloride
    'acyl_bromide':               'C(=O)Br',
    'acyl_fluoride':              'C(=O)F',
    'acyl_sulfonyl_halide':       'S(=O)(=O)[Cl,Br]',            # sulfonyl halide
    'acyl_nhs_ester':             'C(=O)ON1C(=O)CC1=O',          # N-hydroxysuccinimide ester
    'acyl_beta_lactone':          'O=C1CCO1',                    # β-propiolactone (strained)
    'acyl_beta_lactam':           'O=C1CCN1',                    # β-lactam
}

# 4. SN2
SN2_PATTERNS: Dict[str, str] = {
    'sn2_alkyl_halide_primary':   '[CH2;X4][Cl,Br,I]',           # primary alkyl halide
    'sn2_benzyl_halide':          'c[CH2][Cl,Br,I]',             # benzyl halide
    'sn2_allyl_halide':           '[#6]=[#6]C[Cl,Br,I]',         # allyl halide
    'sn2_alpha_haloketone':       'C(=O)C[Cl,Br,I]',             # α-halo ketone
    'sn2_alpha_halonitrile':      '[Cl,Br,I]CC#N',
    'sn2_epoxide':                '[#6]1O[#6]1',                 # epoxide (ring opening)
    'sn2_aziridine':              '[#6]1N[#6]1',                 # aziridine
    'sn2_episulfide':             '[#6]1S[#6]1',                 # thiirane
    'sn2_mesylate':               'CS(=O)(=O)O[CX4]',
    'sn2_tosylate':               'Cc1ccc(S(=O)(=O)O[CX4])cc1',
    'sn2_sultone':                'O=S1(=O)OC[#6]1',             # cyclic sulfonate (Patlewicz 2007)
}

# 5. SNAr
SNAR_PATTERNS: Dict[str, str] = {
    'snar_nitro_halide':          '[cH0]([N+](=O)[O-])[cH0][F,Cl,Br,I]',   # DNCB-type
    'snar_dinitro':               'c1c([F,Cl,Br])c([N+](=O)[O-])ccc1[N+](=O)[O-]',
    'snar_pentafluoro':           'c1c(F)c(F)c(F)c(F)c1F',        # perfluoroaryl
    'snar_triazine_halide':       'c1nc([Cl,Br])nc([Cl,Br])n1',   # cyanuric chloride
    'snar_pyrimidine_halide':     'c1cnc([Cl,F])nc1',
}

# Pro-haptens — narrow set, each with documented activation to a skin-sensitizing
# electrophile.
PROHAPTEN_PATTERNS: Dict[str, str] = {
    'prohapten_para_phenylenediamine': 'Nc1ccc(N)cc1',           # PPD → Bandrowski's base
    'prohapten_aminophenol':           'Nc1ccc(O)cc1',           # p-aminophenol → benzoquinone imine
    'prohapten_hydroquinone':          'Oc1ccc(O)cc1',           # → benzoquinone
    'prohapten_catechol':              'Oc1ccccc1O',             # → ortho-quinone
    'prohapten_allylphenol':           '[#6]=[#6]Cc1ccc(O)cc1',  # eugenol-type → quinone methide
    'prohapten_aminoquinoline':        'Nc1ccnc2ccccc12',        # known skin pro-hapten
}

# Mechanism map

MECHANISM_MAP: Dict[str, str] = {}
for name in MICHAEL_PATTERNS:       MECHANISM_MAP[name] = 'michael_addition'
for name in SCHIFF_PATTERNS:        MECHANISM_MAP[name] = 'schiff_base'
for name in ACYL_PATTERNS:          MECHANISM_MAP[name] = 'acyl_transfer'
for name in SN2_PATTERNS:           MECHANISM_MAP[name] = 'sn2'
for name in SNAR_PATTERNS:          MECHANISM_MAP[name] = 'snar'
for name in PROHAPTEN_PATTERNS:     MECHANISM_MAP[name] = 'pro_hapten'


_ALL_PATTERNS: Dict[str, str] = {}
for group in (MICHAEL_PATTERNS, SCHIFF_PATTERNS, ACYL_PATTERNS,
              SN2_PATTERNS, SNAR_PATTERNS, PROHAPTEN_PATTERNS):
    _ALL_PATTERNS.update(group)

_COMPILED: Dict[str, Chem.Mol] = {}


def _compiled_patterns() -> Dict[str, Chem.Mol]:
    if not _COMPILED:
        for name, smarts in _ALL_PATTERNS.items():
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                raise ValueError(f"Invalid SMARTS for {name}: {smarts}")
            _COMPILED[name] = mol
    return _COMPILED


class AOPReferenceSensitization:
    """Drop-in replacement for AOPReference restricted to skin-sensitization
    chemistry. Mirrors the old interface (get_atom_mask, get_reactive_center_mask,
    classify_mechanism) so existing downstream code can consume it unchanged."""

    def __init__(self):
        self.patterns = _compiled_patterns()

    # -- atom masks -----------------------------------------------------------

    def get_atom_mask(
        self,
        smiles: str,
        mode: str = 'union',
    ) -> Tuple[torch.Tensor, Dict]:
        """Binary [n_atoms] mask. mode='union' (any pattern hit) or
        'intersection' (atoms hit by ≥2 distinct patterns).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.tensor([]), {}

        n_atoms = mol.GetNumAtoms()
        hits_per_atom: List[set] = [set() for _ in range(n_atoms)]
        matches: Dict[str, List[Tuple[int, ...]]] = {}

        for name, pattern in self.patterns.items():
            pm = mol.GetSubstructMatches(pattern)
            if pm:
                matches[name] = list(pm)
                for match in pm:
                    for idx in match:
                        if idx < n_atoms:
                            hits_per_atom[idx].add(name)

        mask = torch.zeros(n_atoms, dtype=torch.float32)
        if mode == 'union':
            for i, names in enumerate(hits_per_atom):
                if names:
                    mask[i] = 1.0
        elif mode == 'intersection':
            for i, names in enumerate(hits_per_atom):
                if len(names) >= 2:
                    mask[i] = 1.0
        else:
            raise ValueError(f"mode must be 'union' or 'intersection', got {mode!r}")

        info = {
            'matched_patterns': list(matches.keys()),
            'n_mie_atoms': int(mask.sum()),
            'n_total_atoms': n_atoms,
            'matches': matches,
            'mode': mode,
        }
        return mask, info

    def get_reactive_center_mask(self, smiles: str) -> Tuple[torch.Tensor, Dict]:
        """Strict mask — only the electrophilic atom within each match, chosen
        per mechanism. Kept identical in spirit to the old implementation so
        numbers are directly comparable."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.tensor([]), {}

        n_atoms = mol.GetNumAtoms()
        mask = torch.zeros(n_atoms, dtype=torch.float32)
        reactive: set = set()

        for name, pattern in self.patterns.items():
            pm = mol.GetSubstructMatches(pattern)
            if not pm:
                continue
            mechanism = MECHANISM_MAP[name]

            for match in pm:
                if mechanism == 'michael_addition':
                    # β-carbon — first atom in C=C-...
                    if match:
                        reactive.add(match[0])
                elif mechanism == 'schiff_base':
                    # carbonyl carbon
                    for idx in match:
                        a = mol.GetAtomWithIdx(idx)
                        if a.GetAtomicNum() == 6 and any(
                                n.GetAtomicNum() == 8 and
                                mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 2.0
                                for n in a.GetNeighbors()):
                            reactive.add(idx)
                            break
                elif mechanism == 'acyl_transfer':
                    # carbonyl / electrophilic carbon
                    for idx in match:
                        a = mol.GetAtomWithIdx(idx)
                        if a.GetAtomicNum() == 6 and any(
                                n.GetAtomicNum() == 8 for n in a.GetNeighbors()):
                            reactive.add(idx)
                            break
                    else:
                        if match:
                            reactive.add(match[0])
                elif mechanism == 'sn2':
                    # carbon bearing the leaving group / strained-ring carbon
                    for idx in match:
                        a = mol.GetAtomWithIdx(idx)
                        if a.GetAtomicNum() == 6:
                            reactive.add(idx)
                            break
                elif mechanism == 'snar':
                    # aromatic carbon bearing the leaving halide
                    for idx in match:
                        a = mol.GetAtomWithIdx(idx)
                        if a.GetAtomicNum() == 6 and a.GetIsAromatic() and any(
                                n.GetAtomicNum() in (9, 17, 35, 53) for n in a.GetNeighbors()):
                            reactive.add(idx)
                            break
                elif mechanism == 'pro_hapten':
                    # the N / O that is oxidised during activation
                    for idx in match:
                        a = mol.GetAtomWithIdx(idx)
                        if a.GetAtomicNum() in (7, 8):
                            reactive.add(idx)

        for idx in reactive:
            if idx < n_atoms:
                mask[idx] = 1.0

        info = {
            'n_reactive_centers': int(mask.sum()),
            'n_total_atoms': n_atoms,
        }
        return mask, info

    # -- mechanism classification ---------------------------------------------

    def classify_mechanism(self, smiles: str) -> Dict[str, object]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'mechanisms': [], 'primary_mechanism': 'unknown'}

        mechanism_hits: Dict[str, List[str]] = {}
        for name, pattern in self.patterns.items():
            if mol.GetSubstructMatches(pattern):
                mechanism = MECHANISM_MAP[name]
                mechanism_hits.setdefault(mechanism, []).append(name)

        if not mechanism_hits:
            return {'mechanisms': [], 'primary_mechanism': 'none',
                    'pattern_counts': {}}

        primary = max(mechanism_hits, key=lambda m: len(mechanism_hits[m]))
        return {
            'mechanisms': list(mechanism_hits.keys()),
            'primary_mechanism': primary,
            'pattern_counts': {m: len(p) for m, p in mechanism_hits.items()},
        }


def pattern_count() -> int:
    """How many SMARTS patterns this alert set defines."""
    return len(_ALL_PATTERNS)


def pattern_inventory() -> Dict[str, List[str]]:
    """For citation in the response letter: mapping of mechanism -> pattern names."""
    inv: Dict[str, List[str]] = {}
    for name, mech in MECHANISM_MAP.items():
        inv.setdefault(mech, []).append(name)
    return inv
