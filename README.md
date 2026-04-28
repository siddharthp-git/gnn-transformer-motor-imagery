# EEG Brain Connectivity — GNN Classification Pipeline

A complete pipeline for classifying **Motor Imagery EEG signals** (BCI Competition IV 2a dataset) using a Graph Neural Network, followed by statistical analysis and connectivity visualization of learned graph features.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [File Reference — What Each File Does](#3-file-reference--what-each-file-does)
4. [Full Pipeline Walkthrough](#4-full-pipeline-walkthrough)
5. [Model Architecture](#5-model-architecture)
6. [Data Shapes Reference](#6-data-shapes-reference)
7. [Statistical Analysis](#7-statistical-analysis)
8. [Visualization Guide](#8-visualization-guide)
9. [Dependencies](#9-dependencies)
10. [Quick-Start Commands](#10-quick-start-commands)

---

## 1. Project Overview

- **Goal**: Train a GNN to binary-classify pairs of motor imagery tasks (e.g., Left Hand vs Right Hand), then identify *which* EEG channels (nodes) and *which* inter-channel connections (edges) most significantly drive classification.
- **Dataset**: BCI Competition IV 2a — 9 subjects, 4 motor imagery classes (Left Hand=0, Right Hand=1, Both Feet=2, Tongue=3), 22 EEG channels, 250 Hz sampling rate.
- **Strategy**: Leave-One-Subject-Out (LOSO) cross-validation over all 6 pairwise class combinations.

---

## 2. Repository Structure

```
Brainconnectivity/
│
├── Final_Pipeline/                  # Preprocessing scripts (run first)
│   ├── 1.data_filter.py
│   └── 2.ICA.m
│
├── BCI_Kaggle/                      # Raw data in .mat format (A01T.mat … A09T.mat, A01E.mat … A09E.mat)
│
├── experiment_with_logging.py       # ★ Main training script
├── extract_all_features.py          # ★ Post-training feature extraction
├── extract_cheb_features.py         # Legacy: ChebConv-only extraction for pair 0v1
├── perform_stats_all_pairs.py       # ★ Statistical tests (all pairs, all layers)
├── perform_statistical_tests.py     # Legacy: Stats for a single pair (hardcoded)
├── analyze_wilcoxon.py              # Interactive Wilcoxon analysis CLI
├── report_significant_features.py   # Summary report of significant features
│
├── plot_feature_topomaps.py         # ★ Average connectivity topoplot
├── plot_significant_topomaps.py     # ★ Significant + Difference connectivity topoplot
├── visualize_from_pickle.py         # Topoplot from GNNExplainer pickle file
├── save_explanations.py             # Run GNNExplainer and save results to pickle
├── calculate_kappa_from_explanations.py  # Cohen's Kappa from explanation pickle files
│
├── inspect_npy.py                   # Utility: inspect any .npy file's shape/stats
├── open_pickle.py                   # Utility: print contents of a pickle file
│
├── binary_label_{c1}_vs_{c2}/       # Per-pair trained models (one folder per pair)
│   └── subject_{N}/
│       └── model.pth
│
├── binary_{c1}_vs_{c2}_feature/     # Per-pair extracted features & stats
│   ├── class{c}_avg_edges.npy       # Averaged edge weights (spectral coherence)
│   └── {graph|cheb|sage}/
│       ├── class{c}_correct.npy     # Node features (22×16×N_trials)
│       ├── p_values.npy             # T-test p-values (22×16)
│       ├── t_stats.npy              # T-statistics (22×16)
│       ├── mean_diff.npy            # Mean difference C1−C2 (22×16)
│       ├── wilcoxon_p_values.npy    # Wilcoxon p-values (22×16)
│       ├── wilcoxon_stats.npy       # Wilcoxon z-statistics (22×16)
│       ├── global_p_values.npy      # Node-averaged T-test p-values (16,)
│       ├── global_t_stats.npy       # Node-averaged T-statistics (16,)
│       ├── global_wilcoxon_p_values.npy
│       ├── global_wilcoxon_stats.npy
│       └── *.png                    # Generated topoplot images
│
└── explanation_data*/               # GNNExplainer pickles (optional)
```

---

## 3. File Reference — What Each File Does

### Preprocessing

#### `Final_Pipeline/1.data_filter.py`
- **Input**: Raw `.gdf` files from `BCICIV_2a_gdf/` (subjects A01T–A09T).
- **What it does**:
  1. Loads raw GDF with MNE, drops EOG channels.
  2. Renames EEG channels to standard 10-20 names (e.g., `EEG-Fz` → `Fz`).
  3. Sets average EEG reference and standard 10-20 montage.
  4. Extracts motor imagery epochs (`769`=Left, `770`=Right), window `−1.5 s` to `6 s`.
  5. Applies baseline correction (`−1.5 s` to `0 s`).
  6. Crops to `3 s – 6 s` (active motor imagery window).
  7. Applies **50 Hz notch filter** (spectrum_fit method).
  8. Applies **0.5–48 Hz 5th-order Butterworth bandpass filter**.
- **Output**: `Filtered/filtered_data_subA0{i}T.mat` — filtered epoched data per subject.

#### `Final_Pipeline/2.ICA.m` *(MATLAB / EEGLAB)*
- **Input**: Filtered `.mat` files from step 1.
- **What it does**: Runs **Independent Component Analysis (ICA)** using EEGLAB to identify and remove ocular/muscular artifact components.
- **Output**: Cleaned data in `filtered_Artifacts_removed/` — this is the data loaded by the GNN training pipeline.

> **Note**: Steps 1 and 2 must be completed before any Python training scripts are run.

---

### Training & Inference

#### `experiment_with_logging.py` ⭐
The **core module** of the entire pipeline. Contains:

| Class / Function | Description |
|---|---|
| `bandpass_filter(data, ...)` | 4th-order Butterworth bandpass (8–30 Hz) — applied inside `load_BCI2a_data` |
| `compute_spectral_coherence(data)` | Computes mean-squared coherence between all channel pairs using Welch's method. Returns shape `(N_trials, 231)` |
| `load_BCI2a_data(path, subject, training)` | Loads `.mat` files from `BCI_Kaggle/`, filters to 22 channels, applies bandpass, windows `2 s – 6 s`. Returns `(data, labels)` |
| `EEGGraphDataset` | PyTorch Geometric `Dataset`. Converts `(N_trials, 22, 16)` CSP features + coherence into `Data(x, edge_index, edge_attr, y)` objects |
| `GraphConvBlock` | 3-layer GraphConv with LayerNorm + ReLU + Global Mean Pool |
| `ChebConvBlock` | 3-layer ChebConv (K=3) with LayerNorm + ReLU + Global Mean Pool |
| `SAGEConvBlock` | 3-layer SAGEConv with LayerNorm + ReLU + Global Mean Pool |
| `TransformerBlock` | Standard `nn.TransformerEncoder` for cross-branch attention |
| `GraphTransformerNet` | Full model: 3 parallel GNN branches → Concat → Transformer → MLP |
| `train_model(...)` | One epoch of training with BCEWithLogitsLoss + Adam |
| `evaluate_model(...)` | Evaluation loop returning accuracy, kappa, features |
| `extract_trial_explanation(...)` | Runs GNNExplainer on a single trial |

**Usage (training)**:
```bash
python3 experiment_with_logging.py --c1 0 --c2 1
```
Saves trained models to `binary_label_{c1}_vs_{c2}/subject_{N}/model.pth`.

---

### Feature Extraction

#### `extract_all_features.py` ⭐
- **Depends on**: Trained models in `binary_label_{c1}_vs_{c2}/`.
- **What it does**:
  1. Iterates over all 6 class pairs and all 9 subjects (LOSO).
  2. Replicates the exact training-time CSP fit on training subjects, transforms the test subject.
  3. Runs each test trial through the trained model with `return_node_feats=True`.
  4. For **correctly classified** trials only, extracts node-level features from each GNN branch:
     - `graph`: indices `0:16` of the 48-dim node feature vector
     - `cheb`: indices `16:32`
     - `sage`: indices `32:48`
  5. Averages and saves spectral coherence edge weights per class.
- **Output**:
  - `binary_{c1}_vs_{c2}_feature/{layer}/class{c}_correct.npy` — shape `(22, 16, N_trials)`
  - `binary_{c1}_vs_{c2}_feature/class{c}_avg_edges.npy` — shape `(231,)`

```bash
python3 extract_all_features.py
```

#### `extract_cheb_features.py`
- **Legacy script**: Extracts ChebConv features for pair `0 vs 1` only (hardcoded).
- Superseded by `extract_all_features.py`. Kept for reference.
- Saves to `binary_label_0_vs_1/cheb_features_class{0|1}_correct.npy`.

---

### Statistical Analysis

#### `perform_stats_all_pairs.py` ⭐
- **Depends on**: `binary_{c1}_vs_{c2}_feature/{layer}/class{c}_correct.npy`
- **What it does**: For every (pair × layer) combination:
  1. **Node-wise analysis**: Welch's T-test + Wilcoxon Rank-Sum test on each of the `22 nodes × 16 features` → `(22, 16)` p-value matrices.
  2. **Global analysis**: Same tests on node-averaged features → `(16,)` p-value vectors.
- **Output** (saved into each `layer_dir`):

  | File | Shape | Contents |
  |---|---|---|
  | `p_values.npy` | (22, 16) | T-test p-values |
  | `t_stats.npy` | (22, 16) | T-statistics |
  | `mean_diff.npy` | (22, 16) | Class1 − Class2 mean difference |
  | `wilcoxon_p_values.npy` | (22, 16) | Wilcoxon p-values |
  | `wilcoxon_stats.npy` | (22, 16) | Wilcoxon z-statistics |
  | `global_p_values.npy` | (16,) | Node-averaged T-test p-values |
  | `global_t_stats.npy` | (16,) | Node-averaged T-statistics |
  | `global_wilcoxon_p_values.npy` | (16,) | Node-averaged Wilcoxon p-values |
  | `global_wilcoxon_stats.npy` | (16,) | Node-averaged Wilcoxon z-stats |

```bash
python3 perform_stats_all_pairs.py
```

#### `perform_statistical_tests.py`
- **Legacy script**: Hardcoded for pair `0 vs 1`, loads old-style ChebConv features from `binary_label_0_vs_1/`.
- Superseded by `perform_stats_all_pairs.py`. Kept for reference/single-pair debugging.

#### `analyze_wilcoxon.py`
- **Interactive CLI** for inspecting Wilcoxon results for a specific pair and layer.
- Supports multiple output modes:

  | Flag | Output |
  |---|---|
  | *(default)* | Lists all significant (node, feature) pairs with p-value and direction |
  | `--show_all` | Full 22×16 p-value matrix |
  | `--simple` | Per-node minimum p-value |
  | `--features` | Per-feature maximum p-value |

```bash
python3 analyze_wilcoxon.py --c1 0 --c2 1 --layer cheb --alpha 0.05
python3 analyze_wilcoxon.py --c1 0 --c2 1 --layer cheb --simple
python3 analyze_wilcoxon.py --c1 0 --c2 1 --layer cheb --features
```

#### `report_significant_features.py`
- Scans **all** pairs and layers and prints a summary table of which (node, feature) combinations are significant (p < 0.05).
- Useful for a high-level overview of where the model's discrimination power lies.

```bash
python3 report_significant_features.py
```

---

### Visualization

#### `plot_feature_topomaps.py` ⭐
- **What it does**: Plots the **average connectivity** topomap for each class. All 231 edges are shown (no significance filter). Top 10 nodes by feature value are labelled.
- **Inputs**: `class{c}_correct.npy` (node features), `class{c}_avg_edges.npy` (edge weights).
- **Output**: `binary_{c1}_vs_{c2}_feature/{layer}/connectivity_{pair}_{layer}_feat{F}_class{C}.png`

```bash
python3 plot_feature_topomaps.py --c1 0 --c2 1 --layer cheb --feature_idx 11
```

#### `plot_significant_topomaps.py` ⭐
- **What it does**: Generates **two statistically filtered plots** per call:
  1. **Significant Connectivity**: Shows only the Top 10 nodes by Wilcoxon p-value. Edges drawn only if *both* endpoints are in the Top 10 set.
  2. **Difference Map**: Shows `Class1 − Class2` for nodes and edges, using a diverging `RdBu_r` colormap. Symmetric color scale centered at zero.
- **Inputs**: `class{c}_correct.npy`, `class{c}_avg_edges.npy`, `p_values.npy`.
- **Output**:
  - `sig_connectivity_{pair}_{layer}_feat{F}_class{C}.png`
  - `sig_diff_{pair}_{layer}_feat{F}.png`

```bash
python3 plot_significant_topomaps.py --c1 0 --c2 1 --layer cheb --feature_idx 11
```

#### `visualize_from_pickle.py`
- Reads a GNNExplainer output `.pkl` file and plots the node importance and edge mask as a connectivity topomap.
- Accepts the pickle path as a command-line argument.

```bash
python3 visualize_from_pickle.py binary_label_0_vs_1/subject_0/explanation_trial_0.pkl
```

#### `save_explanations.py`
- Runs `GNNExplainer` (PyG) on individual test trials and saves the resulting node/edge masks to `.pkl` files.
- Arguments: `--trial_idx N` for a single trial, or `--all_trials` for the full test set.

```bash
python3 save_explanations.py --trial_idx 0 --output_dir explanation_data
python3 save_explanations.py --all_trials --output_dir explanation_data
```

---

### Utilities

#### `inspect_npy.py`
- Quick utility to print the **shape, dtype, min, max, mean** of any `.npy` file.
- Accepts path as command-line argument or edit the hardcoded path at the top.

```bash
python3 inspect_npy.py binary_0_vs_1_feature/cheb/p_values.npy
```

#### `open_pickle.py`
- Prints the raw contents of a `.pkl` file. Useful for inspecting GNNExplainer outputs.

#### `calculate_kappa_from_explanations.py`
- Reads all `explanation_trial_*.pkl` files in a subject directory and computes **Cohen's Kappa** from the saved `predicted_label` vs `true_label`.
- Iterates through all subjects for a given pair and prints a summary table.

```bash
python3 calculate_kappa_from_explanations.py
```

---

## 4. Full Pipeline Walkthrough

```
Step 1 — Preprocessing (MATLAB + Python)
  Final_Pipeline/1.data_filter.py    →  Filtered/filtered_data_subA0{i}T.mat
  Final_Pipeline/2.ICA.m             →  filtered_Artifacts_removed/  (ICA-cleaned)

Step 2 — Training
  experiment_with_logging.py         →  binary_label_{c1}_vs_{c2}/subject_{N}/model.pth
  (LOSO cross-validation, all 6 class pairs)

Step 3 — Feature Extraction
  extract_all_features.py            →  binary_{c1}_vs_{c2}_feature/{layer}/class{c}_correct.npy
                                        binary_{c1}_vs_{c2}_feature/class{c}_avg_edges.npy

Step 4 — Statistical Analysis
  perform_stats_all_pairs.py         →  p_values.npy, wilcoxon_p_values.npy, mean_diff.npy, etc.

Step 5 — Visualization
  plot_feature_topomaps.py           →  Average connectivity PNG per class
  plot_significant_topomaps.py       →  Significance-filtered PNG + Difference PNG
```

---

## 5. Model Architecture

The `GraphTransformerNet` (defined in `experiment_with_logging.py`) processes each EEG trial as a graph.

### Input Graph (per trial)
| Tensor | Shape | Description |
|---|---|---|
| `x` | `(22, 16)` | CSP features — same 16-D vector broadcast to all 22 electrode nodes |
| `edge_index` | `(2, 231)` | Fully-connected graph — all unique pairs of 22 nodes |
| `edge_attr` | `(231,)` | Spectral coherence per channel pair |

### Architecture
```
Input (22 nodes × 16 features)
        │
   ┌────┴────────────┬──────────────┐
GraphConvBlock   ChebConvBlock   SAGEConvBlock
  (3 GraphConv)  (3 ChebConv,K=3) (3 SAGEConv)
  16→32→32→16   16→32→32→16     16→32→32→16
  LayerNorm+ReLU at each layer
        │               │              │
  GlobalMeanPool   GlobalMeanPool  GlobalMeanPool
   (batch, 16)      (batch, 16)    (batch, 16)
        │               │              │
        └───────────────┴──────────────┘
                   Concatenate
                  (batch, 48)
                       │
             Unsqueeze seq dim
             (batch, 1, 48)
                       │
          TransformerEncoder (8 heads, 3 layers)
                       │
                 (batch, 48)
                       │
              MLP Classifier
           48→32 → ReLU → LN → Dropout(0.3)
           32→16 → ReLU → LN → Dropout(0.3)
           16→1
                       │
                  Binary Logit
```

**Training hyperparameters**: Adam (lr=1e-5, weight_decay=5e-4), BCEWithLogitsLoss, batch=16, max 200 epochs, early stopping patience=30.

---

## 6. Data Shapes Reference

| Array | Shape | Description |
|---|---|---|
| Raw EEG | `(N_trials, 22, 1000)` | 22 channels, 4 s at 250 Hz |
| CSP features | `(N_trials, 16)` | After `CSP.transform()` |
| Node features (input) | `(22, 16)` | CSP vector repeated per node |
| Spectral coherence | `(N_trials, 231)` | One scalar per channel pair |
| Node features (extracted) | `(22, 16, N_trials)` | From `class{c}_correct.npy` |
| Edge weights (avg) | `(231,)` | From `class{c}_avg_edges.npy` |
| P-value matrix | `(22, 16)` | Node × Feature statistics |
| Global p-values | `(16,)` | Feature-level statistics (node-averaged) |

---

## 7. Statistical Analysis

For each class pair and GNN layer, `perform_stats_all_pairs.py` compares the feature distribution of correctly classified **Class 1** vs **Class 2** trials:

- **Welch's T-test** (`scipy.stats.ttest_ind`, `equal_var=False`): Tests if the means differ, accounting for unequal variance.
- **Wilcoxon Rank-Sum** (`scipy.stats.ranksums`): Non-parametric test, robust to non-normality. The sign of the z-statistic indicates direction (`C0 > C1` or `C1 > C0`).

Both tests are run at two granularities:
- **Node-wise** `(22 × 16)`: Which specific electrode × CSP component pairs are discriminative?
- **Global** `(16,)`: Averaged across all electrodes — which CSP components are globally significant?

---

## 8. Visualization Guide

### Plot Types

| Script | Type | Node Value | Edge Value | Colormap |
|---|---|---|---|---|
| `plot_feature_topomaps.py` | Average Activation | Mean feature across trials | Mean spectral coherence | `Reds` / `plasma` |
| `plot_significant_topomaps.py` | Significant Connectivity | Top-10 significant nodes only | Edges between Top-10 pairs only | `Reds` / `plasma` |
| `plot_significant_topomaps.py` | Difference Map | C1 − C2 (masked to Top-10) | Coherence C1 − C2 (masked) | `RdBu_r` (diverging) |

### Thresholding Logic

1. **Node filtering**: Sort all 22 nodes by their Wilcoxon p-value for the selected feature. Keep only the **Top 10** (lowest p-values).
2. **Edge filtering**: An edge is plotted only if **both** of its endpoint nodes are in the Top-10 significant set.
3. **Visual encoding**: Node size ∝ feature magnitude; edge width and color ∝ weight magnitude.
4. **Global color scaling**: The color axis is computed globally across both classes to allow direct visual comparison.

---

## 9. Dependencies

```
Python >= 3.9
torch >= 2.0
torch-geometric
numpy
scipy
scikit-learn
mne >= 1.3
matplotlib
tqdm
```

Install:
```bash
pip install torch torch-geometric numpy scipy scikit-learn mne matplotlib tqdm
```

MATLAB with EEGLAB is required only for `Final_Pipeline/2.ICA.m`.

---

## 10. Quick-Start Commands

```bash
# 1. Preprocess (run once — requires GDF files and EEGLAB)
python3 Final_Pipeline/1.data_filter.py
# Then run 2.ICA.m in MATLAB

# 2. Train all pairs (LOSO, ~hours depending on hardware)
python3 experiment_with_logging.py --c1 0 --c2 1
python3 experiment_with_logging.py --c1 0 --c2 2
# ... repeat for all 6 pairs

# 3. Extract features for all pairs at once
python3 extract_all_features.py

# 4. Run statistical analysis
python3 perform_stats_all_pairs.py

# 5. View significance summary
python3 report_significant_features.py

# 6. Inspect a specific pair/layer interactively
python3 analyze_wilcoxon.py --c1 0 --c2 1 --layer cheb --simple

# 7. Generate topoplots (example: pair 0v1, ChebConv, feature 11)
python3 plot_feature_topomaps.py --c1 0 --c2 1 --layer cheb --feature_idx 11
python3 plot_significant_topomaps.py --c1 0 --c2 1 --layer cheb --feature_idx 11

# 8. Inspect any saved .npy file
python3 inspect_npy.py binary_0_vs_1_feature/cheb/p_values.npy
```

---

## Class Label Reference

| Integer Label | Motor Imagery Task |
|---|---|
| 0 | Left Hand |
| 1 | Right Hand |
| 2 | Both Feet |
| 3 | Tongue |

**Pair naming**: `binary_{c1}_vs_{c2}_feature/` e.g., `binary_0_vs_1_feature/` = Left Hand vs Right Hand.

## Channel Order (22 electrodes)

```
Index: 0    1    2    3    4    5    6   7   8   9   10  11  12
Name:  Fz  FC3  FC1  FCz  FC2  FC4  C5  C3  C1  Cz  C2  C4  C6

Index: 13   14   15   16   17  18  19  20  21
Name:  CP3  CP1  CPz  CP2  CP4  P1  Pz  P2  POz
```
