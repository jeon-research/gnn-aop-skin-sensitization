"""
AOP-40 mechanistic reference labels for skin sensitization.

Provides atom-level "ground truth" from AOP-40 MIE chemistry:
- 23 core SMARTS patterns for electrophilic reactive centers
- Extended structural alerts (Aptula & Roberts 2006, Enoch et al. 2011)
- Mechanism classification for stratified analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem

# Core MIE SMARTS patterns for electrophilic reactive centers in skin sensitization.
# Reference: OECD AOP 40 — Skin Sensitisation (https://aopwiki.org/aops/40)
MIE_SMARTS_PATTERNS: Dict[str, str] = {
    'michael_acceptor': '[C]=[C]-[C]=O',
    'michael_acceptor_ester': '[C]=[C]-C(=O)O',
    'acyl_fluoride': 'C(=O)F',
    'acyl_chloride': 'C(=O)Cl',
    'acyl_bromide': 'C(=O)Br',
    'epoxide': '[C]1[O][C]1',
    'aldehyde': '[CX3H1]=O',
    'formaldehyde': '[CH2]=O',
    'snar_nitro_halide': '[cH0:1]([N+](=O)[O-])[cH0:2][F,Cl,Br,I]',
    'snar_dinitro': '[cH0]([N+](=O)[O-])[cH0][N+](=O)[O-]',
    'aryl_halide_activated': 'c1cc([F,Cl,Br])cc([N+](=O)[O-])c1',
    'alkyl_halide': '[CX4][F,Cl,Br,I]',
    'benzyl_halide': '[cH0]C[F,Cl,Br,I]',
    'sulfonyl_halide': 'S(=O)(=O)[Cl,Br]',
    'isocyanate': 'N=C=O',
    'isothiocyanate': 'N=C=S',
    'anhydride': 'C(=O)OC(=O)',
    'nhs_ester': 'C(=O)ON1C(=O)CC1=O',
    'quinone': 'O=C1C=CC(=O)C=C1',
    'diketone_1_2': 'C(=O)C(=O)',
    'acrylate': 'C=CC(=O)O',
    'methacrylate': 'C(=C)C(=O)O',
    'alpha_halo_nitrile': '[F,Cl,Br]CC#N',
}

_COMPILED_CORE_PATTERNS: Dict[str, Chem.Mol] = {}


def _get_compiled_patterns() -> Dict[str, Chem.Mol]:
    """Lazily compile core MIE SMARTS patterns."""
    global _COMPILED_CORE_PATTERNS
    if not _COMPILED_CORE_PATTERNS:
        for name, smarts in MIE_SMARTS_PATTERNS.items():
            mol = Chem.MolFromSmarts(smarts)
            if mol is not None:
                _COMPILED_CORE_PATTERNS[name] = mol
    return _COMPILED_CORE_PATTERNS


def get_mie_atom_indices(smiles: str) -> Tuple[List[int], Dict[str, List[Tuple[int, ...]]]]:
    """Find atoms involved in MIE-relevant functional groups."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], {}

    patterns = _get_compiled_patterns()
    all_atoms = set()
    matches = {}

    for name, pattern in patterns.items():
        pattern_matches = mol.GetSubstructMatches(pattern)
        if pattern_matches:
            matches[name] = list(pattern_matches)
            for match in pattern_matches:
                all_atoms.update(match)

    return sorted(all_atoms), matches

# Extended SMARTS patterns from literature
# Sources: Aptula & Roberts (2006), Enoch et al. (2011, 2012), OECD QSAR Toolbox,
#          Toxtree skin sensitization module, Derek Nexus public alerts,
#          Natsch & Gfeller (2008), Roberts & Aptula (2014)
EXTENDED_SMARTS_PATTERNS: Dict[str, str] = {
    # ----------------------------------------------------------------
    # Michael acceptors (additional)
    # ----------------------------------------------------------------
    'michael_acceptor_amide': '[C]=[C]-C(=O)N',
    'michael_acceptor_nitrile': '[C]=[C]-C#N',
    'chalcone': 'O=C(/C=C/c1ccccc1)c1ccccc1',
    'maleimide': 'O=C1C=CC(=O)N1',
    'vinyl_ketone': 'C=CC(=O)[C,c]',           # Enoch et al. 2012
    'vinyl_pyridine': 'C=Cc1ccncc1',            # Activated vinyl
    'propiolate': 'C#CC(=O)O',                  # Triple bond Michael
    'cyanoacrylate': 'C=C(C#N)C(=O)O',          # Super glue monomer
    'acrylamide': 'C=CC(=O)N',                   # Enoch et al. 2012
    'itaconate': 'C(=C)C(=O)OCC(=O)O',          # Bifunctional Michael
    'coumarin_activated': 'O=c1ccc2ccccc2o1',    # Activated coumarin
    'nitroalkene': '[C]=[C][N+](=O)[O-]',        # Enoch et al. 2012

    # ----------------------------------------------------------------
    # Aldehydes (extended)
    # ----------------------------------------------------------------
    'aromatic_aldehyde': 'c[CH]=O',
    'alpha_beta_unsat_aldehyde': 'C=CC=O',
    'aliphatic_aldehyde': '[CX4][CH]=O',          # Natsch & Gfeller 2008
    'dialdehyde': 'O=CCC=O',                      # Glutaraldehyde type

    # ----------------------------------------------------------------
    # Diketones / 1,2-dicarbonyls (react with arginine)
    # ----------------------------------------------------------------
    'glyoxal': 'O=CC=O',
    'alpha_ketoaldehyde': 'O=CC(=O)[C,c]',        # Pyruvaldehyde type

    # ----------------------------------------------------------------
    # Lactones (strained)
    # ----------------------------------------------------------------
    'beta_lactone': 'C1CC(=O)O1',
    'beta_propiolactone': 'O=C1CCO1',
    'gamma_butyrolactone_unsat': 'O=C1OCC=C1',    # Unsaturated gamma-lactone

    # ----------------------------------------------------------------
    # Ring-opening electrophiles (expanded)
    # ----------------------------------------------------------------
    'aziridine': 'C1NC1',
    'aziridine_n_subst': 'C1N([C,c])C1',          # N-substituted aziridine
    'episulfonium': 'C1SC1',                        # Thiirane (episulfide)
    'oxetane': 'C1COC1',                            # Strained 4-membered ring
    'cyclopropane_activated': 'C1(C(=O))CC1',       # Donor-acceptor cyclopropane

    # ----------------------------------------------------------------
    # Activated double bonds
    # ----------------------------------------------------------------
    'vinyl_sulfone': 'C=CS(=O)(=O)',
    'butenolide': 'O=C1C=CCO1',
    'vinyl_phosphonate': 'C=CP(=O)',                # Phosphonate Michael
    'vinyl_sulfonate': 'C=CS(=O)(=O)O',             # Sulfonate vinyl
    'vinyl_sulfonamide': 'C=CS(=O)(=O)N',           # Sulfonamide vinyl

    # ----------------------------------------------------------------
    # Thiols and disulfides
    # ----------------------------------------------------------------
    'thiol': '[SH]',
    'disulfide': 'SS',
    'thiuram': 'N(C=S)SC(=S)N',
    'thioester': 'C(=O)S',                          # Roberts & Aptula 2014
    'dithiocarbamate': 'N(C=S)S',                    # Metal ion dependent
    'thiocyanate': 'SC#N',                            # Electrophilic sulfur
    'sulfenamide': 'SN',                              # S-N bond

    # ----------------------------------------------------------------
    # Peroxides & radical generators (expanded)
    # ----------------------------------------------------------------
    'peroxide': 'OO',
    'hydroperoxide': '[OX2H]O',                       # ROOH
    'peracid': 'C(=O)OO',                             # Peroxyacid
    'azo_compound': 'N=N',                             # Azo radical source
    'nitroso': '[N]=O',                                # Nitroso radical
    'quinone_imine': 'N=C1C=CC(=O)C=C1',              # Quinone imine oxidant
    'nitro_aromatic': 'c[N+](=O)[O-]',                # Aromatic nitro (radical)
    'hydroperoxide_cumene': 'c1ccccc1C(C)(C)OO',      # Cumene hydroperoxide

    # ----------------------------------------------------------------
    # Diazonium and nitrene precursors
    # ----------------------------------------------------------------
    'diazonium': '[N+]#N',
    'azide': '[N-]=[N+]=[N-]',                         # Organic azide → nitrene

    # ----------------------------------------------------------------
    # Aromatic electrophiles (SNAr and related)
    # ----------------------------------------------------------------
    'nitrobenzene_ortho_halide': 'c1c([F,Cl,Br])cc([N+](=O)[O-])cc1',
    'pyridine_halide': 'c1ccnc([F,Cl,Br])c1',
    'triazine_halide': 'c1nc(Cl)nc(Cl)n1',
    'pyrimidine_halide': 'c1cnc([F,Cl])nc1',            # Pyrimidine SNAr
    'quinoline_halide': 'c1ccc2c([F,Cl,Br])ccnc2c1',    # Quinoline SNAr
    'dichloropyridine': 'c1c(Cl)cnc(Cl)c1',             # Di-halo pyridine

    # ----------------------------------------------------------------
    # SN2 electrophiles (additional leaving groups)
    # ----------------------------------------------------------------
    'mesylate': 'CS(=O)(=O)O',
    'tosylate': 'Cc1ccc(S(=O)(=O)O)cc1',
    'triflate': 'C(F)(F)(F)S(=O)(=O)O',                 # Triflate leaving group
    'epichlorohydrin': 'ClCC1CO1',                       # Epi + SN2
    'mustard_thioether': 'Cl[CH2][CH2]S',               # Mustard gas type
    'alpha_haloketone': 'C(=O)C[F,Cl,Br,I]',            # Alpha-haloketone
    'allyl_halide': 'C=CC[Cl,Br,I]',                     # Allylic SN2/SN2'
    'propargyl_halide': 'C#CC[Cl,Br,I]',                 # Propargylic SN2

    # ----------------------------------------------------------------
    # Metal chelators (expanded)
    # ----------------------------------------------------------------
    'hydroxamic_acid': 'C(=O)NO',
    'catechol': 'Oc1ccccc1O',
    'thiourea': 'NC(=S)N',                               # Cu/Ni chelation
    'dithiocarbamate_chelator': 'S=C(N)S',               # Strong metal chelator
    'oxime': 'C=NO',                                      # Aldoxime/ketoxime
    'hydroxypyridone': 'Oc1cc[nH]c(=O)c1',              # 3-hydroxypyridinone
    'bipyridyl': 'c1ccnc(-c2ccccn2)c1',                  # Bipyridine chelator
    'salicylate': 'OC(=O)c1ccccc1O',                     # o-Hydroxybenzoate
    'picolinic_acid': 'OC(=O)c1ccccn1',                  # Pyridine-2-carboxylic

    # ----------------------------------------------------------------
    # Isothiazolinones (common preservative sensitizers)
    # ----------------------------------------------------------------
    'isothiazolinone': 'c1cc(=O)n(C)s1',                     # MIT (N-methyl, aromatic)
    'isothiazolinone_nh': 'c1cc(=O)[nH]s1',               # MIT (N-H form)
    'isothiazolinone_oxidized': 'O=C1NS(=O)C=C1',         # Oxidized form
    'benzisothiazolinone': 'O=c1[nH]s2ccccc12',
    'chloroisothiazolinone': 'c1c(Cl)c(=O)n(C)s1',        # MCI (Kathon, aromatic)
    'thiazolinone': 'c1csn(C)c1=O',                       # Thiazolinone aromatic

    # ----------------------------------------------------------------
    # Formaldehyde releasers & Schiff base precursors (expanded)
    # ----------------------------------------------------------------
    'formaldehyde_releaser': 'O[CH2]N',
    'imidazolidinone': 'O=C1NCCN1',                      # DMDM hydantoin type
    'oxazolidine': 'C1OCNC1',                             # Oxazolidine FR

    # ----------------------------------------------------------------
    # Beta-lactams and acylating agents (expanded)
    # ----------------------------------------------------------------
    'beta_lactam': 'O=C1CCN1',
    'sultone': 'O=S1(=O)OCC1',                            # Cyclic sulfonate
    'cyclic_anhydride': 'O=C1OC(=O)CC1',                  # Succinic anhydride
    'chloroformate': 'ClC(=O)O',                           # Chloroformate

    # ----------------------------------------------------------------
    # Pro-haptens (metabolically activated) — expanded
    # ----------------------------------------------------------------
    'amine_aromatic': '[NH2]c1ccccc1',
    'aminophenol': 'Oc1ccc([NH2])cc1',
    'hydroquinone': 'Oc1ccc(O)cc1',
    'phenol_para_amino': '[NH2]c1ccc(O)cc1',
    'diaminobenzene': '[NH2]c1ccc([NH2])cc1',             # PPD type
    'eugenol_type': 'C=CCc1ccc(O)c(OC)c1',               # Allylphenol pro-hapten
    'resorcinol': 'Oc1cccc(O)c1',                         # 1,3-dihydroxybenzene
    'phenylenediamine_n_subst': 'Nc1ccc(NC)cc1',          # N-substituted PPD
    'cinnamic_acid': 'OC(=O)/C=C/c1ccccc1',              # Cinnamic pro-hapten
    'methylenedioxybenzene': 'c1cc2c(cc1)OCO2',           # Safrole type
    'polyphenol': 'Oc1cc(O)cc(O)c1',                      # Trihydroxybenzene
    'phenothiazine': 'c1ccc2c(c1)Sc1ccccc1N2',            # Phenothiazine pro-hapten
    'tertiary_aromatic_amine': 'c1ccccc1N(C)C',           # Tert aromatic amine
    'secondary_aromatic_amine': 'c1ccccc1NC',             # Sec aromatic amine
    'aromatic_amine_general': '[NH2,NH,N]c1ccccc1',       # Any aromatic amine
    'diphenylamine': 'c1ccc(Nc2ccccc2)cc1',               # Diphenylamine type
    'aminoquinoline': 'Nc1ccnc2ccccc12',                  # Aminoquinoline pro-hapten
    'piperidine_phenyl': 'c1ccc(C2CCNCC2)cc1',            # Arylpiperidine

    # ----------------------------------------------------------------
    # Additional Schiff base / imine patterns
    # ----------------------------------------------------------------
    'imine_aromatic': 'c/C=N/c',                           # Pre-formed Schiff base
    'imine_aliphatic': 'C/C=N/C',                          # Aliphatic imine

    # ----------------------------------------------------------------
    # Vinyl halides and allyl systems
    # ----------------------------------------------------------------
    'vinyl_halide': 'C=C[Cl,Br,I]',                       # Vinyl halide
    'chloroallyl': 'ClC=CC',                               # Chloroallyl

    # ----------------------------------------------------------------
    # Unsaturated lactones (broader)
    # ----------------------------------------------------------------
    'unsaturated_lactone': 'C=C1C(=O)OC1',                  # General unsat lactone
    'phthalide_unsat': 'C=C1OC(=O)c2ccccc12',             # Phthalide with exo C=C

    # ----------------------------------------------------------------
    # Aminoglycosides / polyamine
    # ----------------------------------------------------------------
    'aminosugar': '[NH2]C1C[C,O]OC1',                       # Aminosugar ring
    'aminocyclitol': '[NH2]C1CCCCC1O',                     # Aminocyclitol (neomycin)
    'polyamine': '[NH2]CC[NH2,NH]',                        # Polyamine chain

    # ----------------------------------------------------------------
    # Hexaminium / quaternary nitrogen
    # ----------------------------------------------------------------
    'quaternary_nitrogen': '[N+](C)(C)(C)C',               # Quaternary ammonium
}

# Mechanism classification mapping
# Each pattern -> primary reaction mechanism
MECHANISM_MAP: Dict[str, str] = {
    # Michael acceptors
    'michael_acceptor': 'michael_addition',
    'michael_acceptor_ester': 'michael_addition',
    'michael_acceptor_amide': 'michael_addition',
    'michael_acceptor_nitrile': 'michael_addition',
    'chalcone': 'michael_addition',
    'maleimide': 'michael_addition',
    'quinone': 'michael_addition',
    'acrylate': 'michael_addition',
    'methacrylate': 'michael_addition',
    'vinyl_sulfone': 'michael_addition',
    'alpha_beta_unsat_aldehyde': 'michael_addition',
    'butenolide': 'michael_addition',
    'vinyl_ketone': 'michael_addition',
    'vinyl_pyridine': 'michael_addition',
    'propiolate': 'michael_addition',
    'cyanoacrylate': 'michael_addition',
    'acrylamide': 'michael_addition',
    'itaconate': 'michael_addition',
    'coumarin_activated': 'michael_addition',
    'nitroalkene': 'michael_addition',
    'vinyl_phosphonate': 'michael_addition',
    'vinyl_sulfonate': 'michael_addition',
    'vinyl_sulfonamide': 'michael_addition',

    # Isothiazolinones
    'isothiazolinone': 'michael_addition',
    'isothiazolinone_nh': 'michael_addition',
    'isothiazolinone_oxidized': 'michael_addition',
    'benzisothiazolinone': 'michael_addition',
    'chloroisothiazolinone': 'michael_addition',
    'thiazolinone': 'michael_addition',

    # Acyl transfer
    'acyl_fluoride': 'acyl_transfer',
    'acyl_chloride': 'acyl_transfer',
    'acyl_bromide': 'acyl_transfer',
    'anhydride': 'acyl_transfer',
    'nhs_ester': 'acyl_transfer',
    'isocyanate': 'acyl_transfer',
    'isothiocyanate': 'acyl_transfer',
    'beta_lactone': 'acyl_transfer',
    'beta_propiolactone': 'acyl_transfer',
    'beta_lactam': 'acyl_transfer',
    'gamma_butyrolactone_unsat': 'acyl_transfer',
    'sultone': 'acyl_transfer',
    'cyclic_anhydride': 'acyl_transfer',
    'chloroformate': 'acyl_transfer',

    # SN2 alkylation
    'alkyl_halide': 'sn2',
    'benzyl_halide': 'sn2',
    'sulfonyl_halide': 'sn2',
    'alpha_halo_nitrile': 'sn2',
    'mesylate': 'sn2',
    'tosylate': 'sn2',
    'triflate': 'sn2',
    'epichlorohydrin': 'sn2',
    'mustard_thioether': 'sn2',
    'alpha_haloketone': 'sn2',
    'allyl_halide': 'sn2',
    'propargyl_halide': 'sn2',

    # SNAr
    'snar_nitro_halide': 'snar',
    'snar_dinitro': 'snar',
    'aryl_halide_activated': 'snar',
    'nitrobenzene_ortho_halide': 'snar',
    'pyridine_halide': 'snar',
    'triazine_halide': 'snar',
    'pyrimidine_halide': 'snar',
    'quinoline_halide': 'snar',
    'dichloropyridine': 'snar',

    # Schiff base formation
    'aldehyde': 'schiff_base',
    'formaldehyde': 'schiff_base',
    'aromatic_aldehyde': 'schiff_base',
    'aliphatic_aldehyde': 'schiff_base',
    'dialdehyde': 'schiff_base',
    'diketone_1_2': 'schiff_base',
    'glyoxal': 'schiff_base',
    'alpha_ketoaldehyde': 'schiff_base',
    'formaldehyde_releaser': 'schiff_base',
    'imidazolidinone': 'schiff_base',
    'oxazolidine': 'schiff_base',

    # Ring opening
    'epoxide': 'ring_opening',
    'aziridine': 'ring_opening',
    'aziridine_n_subst': 'ring_opening',
    'episulfonium': 'ring_opening',
    'oxetane': 'ring_opening',
    'cyclopropane_activated': 'ring_opening',

    # Pre-/pro-haptens (require metabolic activation)
    'amine_aromatic': 'pro_hapten',
    'aminophenol': 'pro_hapten',
    'hydroquinone': 'pro_hapten',
    'phenol_para_amino': 'pro_hapten',
    'catechol': 'pro_hapten',
    'diaminobenzene': 'pro_hapten',
    'eugenol_type': 'pro_hapten',
    'resorcinol': 'pro_hapten',
    'phenylenediamine_n_subst': 'pro_hapten',
    'cinnamic_acid': 'pro_hapten',
    'methylenedioxybenzene': 'pro_hapten',
    'polyphenol': 'pro_hapten',

    # Thiol reactivity
    'thiol': 'thiol_exchange',
    'disulfide': 'thiol_exchange',
    'thiuram': 'thiol_exchange',
    'thioester': 'thiol_exchange',
    'dithiocarbamate': 'thiol_exchange',
    'thiocyanate': 'thiol_exchange',
    'sulfenamide': 'thiol_exchange',

    # Radical generators
    'peroxide': 'radical',
    'hydroperoxide': 'radical',
    'peracid': 'radical',
    'azo_compound': 'radical',
    'nitroso': 'radical',
    'quinone_imine': 'radical',
    'nitro_aromatic': 'radical',
    'hydroperoxide_cumene': 'radical',
    'diazonium': 'radical',
    'azide': 'radical',

    # Chelation
    'hydroxamic_acid': 'chelation',
    'thiourea': 'chelation',
    'dithiocarbamate_chelator': 'chelation',
    'oxime': 'chelation',
    'hydroxypyridone': 'chelation',
    'bipyridyl': 'chelation',
    'salicylate': 'chelation',
    'picolinic_acid': 'chelation',

    # Additional pro-haptens
    'phenothiazine': 'pro_hapten',
    'tertiary_aromatic_amine': 'pro_hapten',
    'secondary_aromatic_amine': 'pro_hapten',
    'aromatic_amine_general': 'pro_hapten',
    'diphenylamine': 'pro_hapten',
    'aminoquinoline': 'pro_hapten',
    'piperidine_phenyl': 'pro_hapten',
    'aminosugar': 'pro_hapten',
    'aminocyclitol': 'pro_hapten',
    'polyamine': 'pro_hapten',

    # Schiff bases (pre-formed)
    'imine_aromatic': 'schiff_base',
    'imine_aliphatic': 'schiff_base',

    # Vinyl halides and allyl (SN2')
    'vinyl_halide': 'sn2',
    'chloroallyl': 'sn2',

    # Additional lactones
    'unsaturated_lactone': 'michael_addition',
    'phthalide_unsat': 'michael_addition',

    # Quaternary nitrogen (general electrophilic N)
    'quaternary_nitrogen': 'sn2',
}

ALL_MECHANISMS = [
    'michael_addition', 'acyl_transfer', 'sn2', 'snar',
    'schiff_base', 'ring_opening', 'pro_hapten',
    'thiol_exchange', 'radical', 'chelation',
]

# Compile all patterns (core + extended)
_ALL_COMPILED_PATTERNS: Dict[str, Chem.Mol] = {}


def _get_all_compiled_patterns() -> Dict[str, Chem.Mol]:
    """Lazily compile all SMARTS patterns (core + extended)."""
    global _ALL_COMPILED_PATTERNS
    if not _ALL_COMPILED_PATTERNS:
        all_patterns = {**MIE_SMARTS_PATTERNS, **EXTENDED_SMARTS_PATTERNS}
        for name, smarts in all_patterns.items():
            mol = Chem.MolFromSmarts(smarts)
            if mol is not None:
                _ALL_COMPILED_PATTERNS[name] = mol
    return _ALL_COMPILED_PATTERNS


class AOPReference:
    """AOP-40 mechanistic reference labels for atom-level ground truth."""

    def __init__(self, use_extended: bool = True):
        """
        Args:
            use_extended: If True, use both core (23) and extended (~50) patterns.
                If False, use only the core 23 MIE patterns.
        """
        self.use_extended = use_extended

    def get_atom_mask(self, smiles: str) -> Tuple[torch.Tensor, Dict]:
        """Get binary atom-level MIE mask for a molecule.

        Args:
            smiles: SMILES string.

        Returns:
            (mask, info): Binary [n_atoms] mask and dict with match details.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.tensor([]), {}

        n_atoms = mol.GetNumAtoms()

        if self.use_extended:
            patterns = _get_all_compiled_patterns()
        else:
            patterns = _get_compiled_patterns()

        all_atoms = set()
        matches = {}

        for name, pattern in patterns.items():
            pattern_matches = mol.GetSubstructMatches(pattern)
            if pattern_matches:
                matches[name] = list(pattern_matches)
                for match in pattern_matches:
                    all_atoms.update(match)

        mask = torch.zeros(n_atoms, dtype=torch.float32)
        for idx in all_atoms:
            if idx < n_atoms:
                mask[idx] = 1.0

        info = {
            'matched_patterns': list(matches.keys()),
            'n_mie_atoms': int(mask.sum()),
            'n_total_atoms': n_atoms,
            'matches': matches,
        }

        return mask, info

    def classify_mechanism(self, smiles: str) -> Dict[str, any]:
        """Classify molecule by primary reaction mechanism.

        Args:
            smiles: SMILES string.

        Returns:
            Dict with mechanism classification details.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'mechanisms': [], 'primary_mechanism': 'unknown'}

        patterns = _get_all_compiled_patterns()
        mechanism_hits = {}

        for name, pattern in patterns.items():
            pattern_matches = mol.GetSubstructMatches(pattern)
            if pattern_matches:
                mechanism = MECHANISM_MAP.get(name, 'unknown')
                if mechanism not in mechanism_hits:
                    mechanism_hits[mechanism] = []
                mechanism_hits[mechanism].append(name)

        if not mechanism_hits:
            return {
                'mechanisms': [],
                'primary_mechanism': 'none',
                'pattern_counts': {},
            }

        # Primary mechanism: the one with most pattern matches
        primary = max(mechanism_hits, key=lambda m: len(mechanism_hits[m]))

        return {
            'mechanisms': list(mechanism_hits.keys()),
            'primary_mechanism': primary,
            'pattern_counts': {m: len(p) for m, p in mechanism_hits.items()},
        }

    def get_reactive_center_mask(self, smiles: str) -> Tuple[torch.Tensor, Dict]:
        """Get strict reactive center mask (only the electrophilic atom itself).

        For electrophilic mechanisms, the reactive center is typically:
        - Michael acceptors: the beta-carbon
        - Aldehydes: the carbonyl carbon
        - SN2: the carbon bearing the leaving group
        - Epoxides: the ring carbons

        This is stricter than get_atom_mask which includes the full substructure.

        Returns:
            (mask, info): Binary [n_atoms] strict mask.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.tensor([]), {}

        n_atoms = mol.GetNumAtoms()
        mask = torch.zeros(n_atoms, dtype=torch.float32)
        reactive_atoms = set()

        patterns = _get_all_compiled_patterns()
        for name, pattern in patterns.items():
            pattern_matches = mol.GetSubstructMatches(pattern)
            if not pattern_matches:
                continue

            mechanism = MECHANISM_MAP.get(name, 'unknown')
            for match in pattern_matches:
                # Identify the key reactive atom(s) based on mechanism
                if mechanism == 'michael_addition':
                    # Beta carbon (first atom in C=C-C=O pattern)
                    if len(match) >= 1:
                        reactive_atoms.add(match[0])
                elif mechanism in ('schiff_base',):
                    # Carbonyl carbon
                    if len(match) >= 1:
                        reactive_atoms.add(match[0])
                elif mechanism == 'sn2':
                    # Carbon bearing leaving group (first atom in C-X pattern)
                    if len(match) >= 1:
                        reactive_atoms.add(match[0])
                elif mechanism == 'ring_opening':
                    # Ring carbons (first and last in epoxide C-O-C)
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetAtomicNum() == 6:  # Carbon
                            reactive_atoms.add(idx)
                elif mechanism == 'acyl_transfer':
                    # Carbonyl carbon
                    if len(match) >= 1:
                        reactive_atoms.add(match[0])
                elif mechanism == 'snar':
                    # Carbon bearing leaving group
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
                            neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
                            if any(z in (9, 17, 35, 53) for z in neighbors):
                                reactive_atoms.add(idx)
                                break
                elif mechanism == 'pro_hapten':
                    # For pro-haptens: the nitrogen or oxidizable atom
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetAtomicNum() == 7:  # Nitrogen
                            reactive_atoms.add(idx)
                        elif atom.GetAtomicNum() == 8:  # Oxygen (hydroquinone)
                            reactive_atoms.add(idx)
                elif mechanism == 'thiol_exchange':
                    # Sulfur atoms
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetAtomicNum() == 16:  # Sulfur
                            reactive_atoms.add(idx)
                else:
                    # Default for radical, chelation, unknown: first atom only
                    if match:
                        reactive_atoms.add(match[0])

        for idx in reactive_atoms:
            if idx < n_atoms:
                mask[idx] = 1.0

        info = {
            'n_reactive_centers': int(mask.sum()),
            'n_total_atoms': n_atoms,
        }

        return mask, info

    def annotate_dataset(
        self,
        smiles_list: List[str],
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """Annotate a full dataset with AOP reference information.

        Args:
            smiles_list: List of SMILES strings.
            labels: Optional binary labels (1=sensitizer, 0=non-sensitizer).

        Returns:
            Dict with per-molecule masks, mechanisms, and summary statistics.
        """
        masks = []
        mechanisms = []
        infos = []

        for smi in smiles_list:
            mask, info = self.get_atom_mask(smi)
            mech = self.classify_mechanism(smi)
            masks.append(mask)
            mechanisms.append(mech)
            infos.append(info)

        # Summary statistics
        n_with_mie = sum(1 for m in masks if m.numel() > 0 and m.sum() > 0)
        coverage = n_with_mie / len(smiles_list) if smiles_list else 0

        mechanism_counts = {}
        for mech in mechanisms:
            pm = mech['primary_mechanism']
            mechanism_counts[pm] = mechanism_counts.get(pm, 0) + 1

        summary = {
            'n_molecules': len(smiles_list),
            'n_with_mie_atoms': n_with_mie,
            'mie_coverage': coverage,
            'mechanism_distribution': mechanism_counts,
        }

        # If labels provided, check MIE coverage by class
        if labels is not None:
            labels = np.asarray(labels)
            sensitizers = labels == 1
            non_sensitizers = labels == 0

            sens_masks = [m for m, s in zip(masks, sensitizers) if s]
            nonsens_masks = [m for m, ns in zip(masks, non_sensitizers) if ns]

            n_sens_with_mie = sum(
                1 for m in sens_masks if m.numel() > 0 and m.sum() > 0
            )
            n_nonsens_with_mie = sum(
                1 for m in nonsens_masks if m.numel() > 0 and m.sum() > 0
            )

            summary['sensitizer_mie_coverage'] = (
                n_sens_with_mie / len(sens_masks) if sens_masks else 0
            )
            summary['nonsensitizer_mie_coverage'] = (
                n_nonsens_with_mie / len(nonsens_masks) if nonsens_masks else 0
            )

        return {
            'masks': masks,
            'mechanisms': mechanisms,
            'infos': infos,
            'summary': summary,
        }
