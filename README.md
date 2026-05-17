# GNN Attribution Alignment with AOP-40 Reactive Centers

Code and data for the manuscript "Systematic Validation of Graph Neural
Network Explanations Against Adverse Outcome Pathway Reactive Centers for
Skin Sensitization" (Journal of Cheminformatics).

## Quick start

```bash
conda env create -f environment.yml
conda activate gnn-aop-skin-sensitization
python scripts/train_llna_labels.py --seeds 42                       # LLNA-primary
python scripts/train_llna_labels.py --seeds 42 --shuffle             # shuffle null
python scripts/extract_explanations_llna.py --seeds 42               # 4 methods
python scripts/extract_explanations_llna_perturbation.py --seeds 42  # 3 methods
python scripts/alignment_sensitization.py --seeds 42
```

The 20-seed manifest is in `seeds.json`.

## Layout

```
senslib/        50-pattern sensitization-specific alert set + LLNA/human loaders
src/            Attribution methods (IG, GradCAM, attention, GNNExplainer,
                PGExplainer, GraphMask, ensemble) and model classes
scripts/        Entry points — training, extraction, alignment, classification
data/processed/ Compiled dataset
results/splits/ Per-seed train/val/test splits (20 seeds × 2 label sets)
tests/          Sanity checks for the alert set
```

---

## Reproduction guide

Map of which scripts produce which numbers in the manuscript and
supplementary.

## Environment

Python 3.10, PyTorch with CUDA. The public repository layout:

```
<repo>/
├── data/processed/causal_aop_comprehensive_v6.csv   # input dataset
├── environment.yml                                   # pinned conda env
├── seeds.json                                        # 20-seed manifest
├── src/explain/                                      # IG, GradCAM, attention,
│   │                                                   GNNExplainer, PGExplainer,
│   │                                                   GraphMask, ensemble,
│   │                                                   AOP reference, alignment
│   │                                                   metrics, utilities
│   └── modeling/ablation_model.py                    # AblationGNN backbone
├── senslib/                                          # 50-pattern alert set,
│                                                       LLNA loader, human loader,
│                                                       seed manifest, training helpers
├── scripts/                                          # this directory (entry points)
├── results/splits/                                   # 40 per-seed split manifests
│                                                       (20 LLNA + 20 human)
└── tests/                                            # sanity test for the alert set
```

Scripts auto-detect the repo root by walking up to find `src/explain/utils.py`,
so `senslib.*` and `src.*` imports resolve without manual `sys.path` setup.
Install dependencies with `conda env create -f environment.yml`.

## Compute requirements

| Stage                                       | Hardware    | Wall time per seed |
| ------------------------------------------- | ----------- | ------------------ |
| Train AttentiveFP (LLNA-primary, 1 seed)    | 1 GPU       | ~10 min            |
| Train shuffled-label control (1 seed)       | 1 GPU       | ~10 min            |
| Train pos_weight variant (1 seed)           | 1 GPU       | ~10 min            |
| Extract IG/GradCAM/Attention/Ensemble       | 1 GPU       | ~3 min             |
| Extract GNNExplainer/PGExplainer/GraphMask  | 1 GPU       | ~15 min            |
| Alignment / stats / baselines / report      | CPU         | seconds            |

Stages 3–6 (alignment / classification / report / build) need only the
per-seed checkpoints and explanations from stages 1–2; they run in
seconds on CPU.

## Seeds

Twenty seeds throughout: `42, 123, 456, 789, 1024, 2048, 3141, 4096,
5555, 7777, 1111, 2222, 3333, 4444, 6666, 8888, 9999, 1234, 5678, 9876`.
Deterministic scaffold split — identical test set across seeds.

## Pipeline

### 1. Train

| Output                                    | Script                                                          |
| ----------------------------------------- | --------------------------------------------------------------- |
| `results/llna_training/seed_<N>/`         | `train_llna_labels.py`                                          |
| `results/shuffle_control/seed_<N>/`       | `train_llna_labels.py --shuffle`                                |
| `results/llna_training_pw{4,8,12}/`       | `train_llna_labels.py --pos-weight <W> --output-tag pw<W>`      |
| `results/human_training/seed_<N>/`        | `train_human_labels.py` (broader cross-check, §3.5 onward)      |
| `results/shuffle_control_human/seed_<N>/` | `train_human_labels.py --shuffle`                               |
| `results/human_training_pw{4,8,12}/`      | `train_human_labels.py --pos-weight <W> --output-tag pw<W>`     |

### 2. Extract attributions

| Methods                                            | Script                                       |
| -------------------------------------------------- | -------------------------------------------- |
| IG / GradCAM / Attention / Ensemble (LLNA)         | `extract_explanations_llna.py`               |
| GNNExplainer / PGExplainer / GraphMask (LLNA)      | `extract_explanations_llna_perturbation.py`  |
| Merge into 7-method tree                           | `merge_llna_explanations.py`                 |
| Same 4 methods (human-label cross-check)           | `extract_explanations_human.py`              |
| Probabilities from architecture-ablation ckpts     | `extract_frozen_ablation_probs.py`           |

### 3. Compute alignment

| Output (Manuscript table)                              | Script                                  |
| ------------------------------------------------------ | --------------------------------------- |
| Per-seed atom-AUC under 50-pattern alerts (Tables 2, 3) | `alignment_sensitization.py`            |
| Bootstrap CIs across seeds                             | `alignment_stats.py`                    |
| Per-mechanism stratification (Table 8)                 | `mechanism_stratified_analysis.py`      |
| Six neutral baselines (Table S5)                       | `baselines_neutral.py`                  |

### 4. Classification metrics (Table 1, Table S11)

| Output                                          | Script                          |
| ----------------------------------------------- | ------------------------------- |
| Confusion matrices for every model              | `classification_confusion.py`   |
| pos_weight sweep summary                        | `posweight_summary.py`          |
| Human-label cross-check summary (§3.5 onward)   | `summarize_human_label_results.py` |

### 5. Decision gates (Limitations §4.7)

| Output                                          | Script                          |
| ----------------------------------------------- | ------------------------------- |
| Gate A / Gate B verdict (attention, primary)    | `check_decision_gates.py`       |
| Console headline summary                        | `headline_report.py`            |

### 6. Deliverables

Additional file 1 (supplementary PDF) and Additional file 2 (per-molecule
.xlsx workbook) ship as built artifacts alongside this repository; the
scripts that assemble them (`build_additional_file_1.py`,
`regen_additional_file_2_sheets.py`, `update_additional_file_2_readme.py`)
are not included because their outputs are static.
