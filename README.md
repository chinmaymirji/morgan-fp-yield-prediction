# Reaction Yield Prediction with Morgan Fingerprints and ANN

This repository contains training and ablation pipelines for reaction yield prediction on the **Suzuki–Miyaura** and **Buchwald–Hartwig** benchmark datasets using **Morgan fingerprint-based reaction features** and **artificial neural network (ANN)** models.

## Repository Structure

```text
.
├── utils/
│   ├── __init__.py
│   └── common.py
├── train_morgan_ann_sm.py
├── train_morgan_ann_bh.py
├── fp_ablation_sm_all_splits.py
├── fp_ablation_bh_all_splits.py
├── environment.yml
└── README.md
```

## What This Project Includes

- **Training pipelines**
  - `train_morgan_ann_sm.py` for Suzuki–Miyaura
  - `train_morgan_ann_bh.py` for Buchwald–Hartwig

- **Fingerprint ablation studies**
  - `fp_ablation_sm_all_splits.py`
  - `fp_ablation_bh_all_splits.py`

- **Shared utilities**
  - `utils/common.py` for reaction parsing, fingerprint generation, feature construction, metrics, plotting, and reusable training helpers

## Feature Representation

The current training and ablation scripts use a **6-block reaction feature layout** based on Morgan count fingerprints:

1. Reactant fingerprint sum  
2. Middle/reagent fingerprint sum  
3. Product fingerprint sum  
4. Product minus reactant  
5. Product minus (reactant + middle)  
6. Absolute value of block 5  

These fingerprint blocks are followed by scalar reaction descriptors.

## Environment Setup

Create the Conda environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate chem_ml
```

## Dataset Setup

Update the dataset paths in the config block at the top of each script before running.

Expected locations currently used by the scripts:

- **Suzuki–Miyaura**
  - `../data/Suzuki-Miyaura/random_splits`

- **Buchwald–Hartwig**
  - `../data/data/split`

Make sure the required TSV or NPZ split files are present in those directories.

## Running the Training Pipelines

### Suzuki–Miyaura

```bash
python train_morgan_ann_sm.py
```

### Buchwald–Hartwig

```bash
python train_morgan_ann_bh.py
```

These scripts run across all configured splits and typically produce:

- model checkpoints
- `metrics.json`
- summary CSV files
- measured-vs-predicted scatter plots

## Running the Ablation Pipelines

### Suzuki–Miyaura

```bash
python fp_ablation_sm_all_splits.py
```

### Buchwald–Hartwig

```bash
python fp_ablation_bh_all_splits.py
```

These scripts evaluate multiple fingerprint configurations and report:

- R²
- RMSE
- MAE
- feature generation time
- training time

## Notes

- All main scripts use a **config block at the top** instead of command-line arguments.
- Shared reusable code lives in `utils/common.py`.
- Buchwald–Hartwig scripts support NPZ-based split files.
- The project is structured to keep feature generation and training logic consistent across datasets.
- If certain libraries are missing or won't run due to dependencies please run the following command to ensure it isn't the library path mismatch.
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Goal

The goal of this repository is to provide a clean, reproducible Morgan fingerprint baseline for cross-coupling reaction yield prediction and systematic fingerprint ablation experiments.
