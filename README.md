# GNN-AOP Skin Sensitization

Code and data for: *"Systematic Validation of Graph Neural Network Explanations Against Adverse Outcome Pathway Reactive Centers for Skin Sensitization"*

## Installation

```bash
conda create -n gnn-aop python=3.10
conda activate gnn-aop

# PyTorch (adjust CUDA version as needed)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# PyTorch Geometric
pip install torch-geometric==2.7.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.9.1+cu128.html

# Remaining dependencies
pip install -r requirements.txt
```

## Reproducing Results

### Step 1: Train Models (20 seeds)

```bash
# AttentiveFP (main architecture)
for seed in 42 123 456 789 1024 2048 3141 4096 5555 7777 1111 2222 3333 4444 6666 8888 9999 1234 5678 9876; do
    python scripts/train_ablation.py --condition plain --seed $seed
done

# GCN and GIN
for arch in gcn gin; do
    for seed in 42 123 456 789 1024 2048 3141 4096 5555 7777 1111 2222 3333 4444 6666 8888 9999 1234 5678 9876; do
        python scripts/train_ablation.py --condition plain --architecture $arch --seed $seed
    done
done
```

### Step 2: Extract Explanations

```bash
python scripts/explain/extract_explanations.py \
    --methods ig gradcam attention gnnexplainer pgexplainer graphmask \
    --seeds 42 123 456 789 1024 2048 3141 4096 5555 7777 1111 2222 3333 4444 6666 8888 9999 1234 5678 9876
```

### Step 3: Compute Alignment

```bash
python scripts/explain/compute_alignment.py \
    --seeds 42 123 456 789 1024 2048 3141 4096 5555 7777 1111 2222 3333 4444 6666 8888 9999 1234 5678 9876
```

### Step 4: Baselines

```bash
# Fingerprint baseline (Morgan FP + Random Forest)
python scripts/explain/train_fingerprint_baseline.py

# Control baselines (random, untrained, heteroatom, degree)
python scripts/explain/compute_control_baselines.py

# Conformal prediction
python scripts/explain/run_conformal.py
```

### Step 5: Statistical Analysis

```bash
python scripts/explain/statistical_analysis.py
python scripts/explain/compute_additional_analyses.py
```

### Step 6: MechGNN Ablation

```bash
python scripts/explain/run_mechgnn_pipeline.py
```

## Software Versions

| Package | Version |
|---------|---------|
| Python | 3.10.12 |
| PyTorch | 2.9.1+cu128 |
| PyTorch Geometric | 2.7.0 |
| RDKit | 2023.09.6 |
| scikit-learn | 1.6.1 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Optuna | 4.7.0 |

## License

MIT License. See [LICENSE](LICENSE).
