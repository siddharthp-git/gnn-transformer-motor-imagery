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
- **Data path**: The GNN loads directly from `BCI_Kaggle/` (`.mat` files) and applies an internal **8–30 Hz Butterworth bandpass filter** — no external preprocessing step is required.

---

## 2. Repository Structure

```
gnn-transformer-motor-imagery/
│
├── BCI_Kaggle/                      # Raw data (A01T.mat … A09T.mat, A01E.mat … A09E.mat)
│                                    # ← NOT committed (too large). Download separately.
│
├── Final_Pipeline/
│   └── 1.data_filter.py             # Reference script: original GDF → .mat conversion
│
├── experiment_with_logging.py       # ★ Main: model definition + LOSO training
├── extract_all_features.py          # ★ Post-training node/edge feature extraction
├── perform_stats_all_pairs.py       # ★ Statistical tests (T-test + Wilcoxon, all pairs)
├── analyze_wilcoxon.py              # Interactive Wilcoxon CLI inspector
├── report_significant_features.py   # Summary report across all pairs & layers
│
├── plot_feature_topomaps.py         # ★ Average connectivity topoplot
├── plot_significant_topomaps.py     # ★ Significance-filtered + difference topoplot
├── save_explanations.py             # Run GNNExplainer → save masks to .pkl
├── visualize_from_pickle.py         # Topoplot from GNNExplainer .pkl file
├── calculate_kappa_from_explanations.py  # Cohen's Kappa from explanation pickles
│
├── inspect_npy.py                   # Utility: inspect any .npy file's shape/stats
├── open_pickle.py                   # Utility: print contents of a .pkl file
│
├── binary_label_{c1}_vs_{c2}/       # Per-pair trained models (LOSO, one per subject)
│   └── subject_{N}/
│       ├── model.pth
│       └── explanation_trial_*.pkl  # GNNExplainer outputs per trial
│
├── binary_{c1}_vs_{c2}_feature/     # Per-pair extracted features & statistics
│   ├── class{c}_avg_edges.npy       # Averaged spectral coherence edge weights (231,)
│   └── {graph|cheb|sage}/
│       ├── class{c}_correct.npy     # Node features (22×16×N_trials)
│       ├── p_values.npy             # T-test p-values (22×16)
│       ├── t_stats.npy              # T-statistics (22×16)
│       ├── mean_diff.npy            # Mean difference C1−C2 (22×16)
│       ├── wilcoxon_p_values.npy    # Wilcoxon p-values (22×16)
│       ├── wilcoxon_stats.npy       # Wilcoxon z-statistics (22×16)
│       ├── global_p_values.npy      # Node-averaged T-test p-values (16,)
│       ├── global_wilcoxon_p_values.npy
│       └── *.png                    # Generated topoplot images
│
├── results/
│   ├── log.txt                      # Training log
│   └── best_models.txt              # Best model summary per pair
│
└── loso_*.png                       # LOSO learning curves + overall performance
```

---

## 3. File Reference — What Each File Does

### Data Reference

#### `Final_Pipeline/1.data_filter.py`
Reference script showing how the original BCI Competition IV `.gdf` files were converted to `.mat` format. Applies notch filtering (50 Hz) and Butterworth bandpass (0.5–48 Hz), then saves per-subject `.mat` files.

> **Note**: The GNN training pipeline does **not** use these files. It loads `BCI_Kaggle/*.mat` directly and applies its own internal 8–30 Hz bandpass.

---

### Training & Inference

#### `experiment_with_logging.py` ⭐
The **core module** of the entire pipeline. Contains all model and data code.

| Class / Function | Description |
|---|---|
| `bandpass_filter(data, ...)` | 4th-order Butterworth 8–30 Hz filter (applied inside `load_BCI2a_data`) |
| `compute_spectral_coherence(data)` | Welch's method coherence between all channel pairs → shape `(N_trials, 231)` |
| `load_BCI2a_data(path, subject, training)` | Loads `BCI_Kaggle/*.mat`, selects 22 EEG channels, applies bandpass, windows `2–6 s` |
| `EEGGraphDataset` | PyTorch Geometric `Dataset`. Converts CSP features + coherence into graph `Data` objects |
| `GraphConvBlock` | 3-layer GraphConv with LayerNorm + ReLU + Global Mean Pool |
| `ChebConvBlock` | 3-layer ChebConv (K=3) with LayerNorm + ReLU + Global Mean Pool |
| `SAGEConvBlock` | 3-layer SAGEConv with LayerNorm + ReLU + Global Mean Pool |
| `GraphTransformerNet` | Full model: 3 parallel GNN branches → concat → TransformerEncoder → MLP |
| `train_model(...)` | One training epoch — BCEWithLogitsLoss + Adam |
| `evaluate_model(...)` | Evaluation loop returning accuracy, kappa, and features |

**Run training** (one pair at a time):
```bash
python3 experiment_with_logging.py --c1 0 --c2 1
```
Saves models to `binary_label_{c1}_vs_{c2}/subject_{N}/model.pth`.

---

### Feature Extraction

#### `extract_all_features.py` ⭐
- **Depends on**: trained models in `binary_label_{c1}_vs_{c2}/`
- **What it does**:
  1. Iterates all 6 class pairs × 9 subjects (LOSO).
  2. Replicates training-time CSP fit on training subjects, transforms test subject.
  3. Runs each test trial through the trained model.
  4. For **correctly classified trials only**, extracts node features from each GNN branch:
     - `graph`: feature vector indices `0:16`
     - `cheb`: indices `16:32`
     - `sage`: indices `32:48`
  5. Saves class-averaged spectral coherence edge weights.
- **Output**:
  - `binary_{c1}_vs_{c2}_feature/{layer}/class{c}_correct.npy` — shape `(22, 16, N_trials)`
  - `binary_{c1}_vs_{c2}_feature/class{c}_avg_edges.npy` — shape `(231,)`

```bash
python3 extract_all_features.py
```

---

### Statistical Analysis

#### `perform_stats_all_pairs.py` ⭐
For every (pair × layer) combination:
1. **Node-wise**: Welch's T-test + Wilcoxon Rank-Sum on each `22 nodes × 16 features` → `(22, 16)` matrices.
2. **Global**: Same tests on node-averaged features → `(16,)` vectors.

| Output file | Shape | Contents |
|---|---|---|
| `p_values.npy` | (22, 16) | T-test p-values |
| `t_stats.npy` | (22, 16) | T-statistics |
| `mean_diff.npy` | (22, 16) | Class1 − Class2 mean difference |
| `wilcoxon_p_values.npy` | (22, 16) | Wilcoxon p-values |
| `wilcoxon_stats.npy` | (22, 16) | Wilcoxon z-statistics (sign = direction) |
| `global_p_values.npy` | (16,) | Node-averaged T-test p-values |
| `global_wilcoxon_p_values.npy` | (16,) | Node-averaged Wilcoxon p-values |

```bash
python3 perform_stats_all_pairs.py
```

#### `analyze_wilcoxon.py`
Interactive CLI to inspect Wilcoxon results for a specific pair and layer.

| Flag | Output |
|---|---|
| *(default)* | All significant (node, feature) pairs with p-value and direction |
| `--show_all` | Full 22×16 p-value matrix |
| `--simple` | Per-node minimum p-value |
| `--features` | Per-feature maximum p-value across nodes |

```bash
python3 analyze_wilcoxon.py --c1 0 --c2 1 --layer cheb --alpha 0.05
python3 analyze_wilcoxon.py --c1 0 --c2 1 --layer cheb --simple
```

#### `report_significant_features.py`
Scans all pairs and layers and prints a summary of which (node, feature) combinations are significant (p < 0.05). Useful for a quick high-level overview.

```bash
python3 report_significant_features.py
```

---

### Visualization

#### `plot_feature_topomaps.py` ⭐
Plots the **average connectivity** topomap per class. All 231 edges are shown (no significance filter). Top 10 nodes by feature value are labelled.

```bash
python3 plot_feature_topomaps.py --c1 0 --c2 1 --layer cheb --feature_idx 11
```

#### `plot_significant_topomaps.py` ⭐
Generates two plots per call:
1. **Significant Connectivity**: Only Top-10 nodes by Wilcoxon p-value; edges only between Top-10 pairs.
2. **Difference Map**: Class1 − Class2 on nodes and edges using a diverging `RdBu_r` colormap.

```bash
python3 plot_significant_topomaps.py --c1 0 --c2 1 --layer cheb --feature_idx 11
```

#### `save_explanations.py`
Runs `GNNExplainer` (PyG) on individual test trials and saves node/edge masks to `.pkl` files.

```bash
python3 save_explanations.py --trial_idx 0 --output_dir explanation_data
python3 save_explanations.py --all_trials --output_dir explanation_data
```

#### `visualize_from_pickle.py`
Reads a GNNExplainer `.pkl` file and plots the node importance and edge mask as a connectivity topomap.

```bash
python3 visualize_from_pickle.py binary_label_0_vs_1/subject_0/explanation_trial_0.pkl
```

---

### Utilities

#### `inspect_npy.py`
Prints shape, dtype, min, max, mean of any `.npy` file.
```bash
python3 inspect_npy.py binary_0_vs_1_feature/cheb/p_values.npy
```

#### `open_pickle.py`
Prints raw contents of a `.pkl` file.

#### `calculate_kappa_from_explanations.py`
Reads all `explanation_trial_*.pkl` files in a subject directory and computes **Cohen's Kappa** from saved `predicted_label` vs `true_label`. Prints a summary table per subject.
```bash
python3 calculate_kappa_from_explanations.py
```

---

## 4. Full Pipeline Walkthrough

```
Step 1 — (Optional) GDF → MAT conversion reference
  Final_Pipeline/1.data_filter.py    →  (reference only, output not used by GNN)

Step 2 — Training  [loads from BCI_Kaggle/ with internal 8-30Hz bandpass]
  experiment_with_logging.py         →  binary_label_{c1}_vs_{c2}/subject_{N}/model.pth
  (runs LOSO for all 6 class pairs)

Step 3 — Feature Extraction
  extract_all_features.py            →  binary_{c1}_vs_{c2}_feature/{layer}/class{c}_correct.npy
                                        binary_{c1}_vs_{c2}_feature/class{c}_avg_edges.npy

Step 4 — Statistical Analysis
  perform_stats_all_pairs.py         →  p_values.npy, wilcoxon_p_values.npy, mean_diff.npy, …

Step 5 — Visualization
  plot_feature_topomaps.py           →  Average connectivity PNG per class
  plot_significant_topomaps.py       →  Significance-filtered PNG + Difference PNG
  save_explanations.py + visualize_from_pickle.py  →  GNNExplainer topoplots
```

### Data loading inside the GNN

```
BCI_Kaggle/A0{i}T.mat  (BCI Competition IV 2a, standard .mat format)
      ↓
load_BCI2a_data()       (inside experiment_with_logging.py)
  - selects 22 EEG channels (drops EOG)
  - windows to 2–6 s post-cue
      ↓
bandpass_filter(8–30 Hz)  (4th-order Butterworth, applied internally)
      ↓
CSP (16 components)  +  Spectral Coherence (231 channel pairs)
      ↓
EEGGraphDataset  →  GraphTransformerNet
```

---

## 5. Model Architecture

### Input Graph (per trial)
| Tensor | Shape | Description |
|---|---|---|
| `x` | `(22, 16)` | 16 CSP features broadcast to all 22 electrode nodes |
| `edge_index` | `(2, 231)` | Fully-connected graph — all unique pairs of 22 nodes |
| `edge_attr` | `(231,)` | Spectral coherence weight per channel pair |

### Architecture
```
Input (22 nodes × 16 features)
        │
   ┌────┴────────────┬──────────────┐
GraphConvBlock   ChebConvBlock   SAGEConvBlock
  (3 GraphConv)  (3 ChebConv,K=3) (3 SAGEConv)
  16→32→32→16   16→32→32→16     16→32→32→16
  LayerNorm + ReLU at each layer
        │               │              │
  GlobalMeanPool   GlobalMeanPool  GlobalMeanPool
   (batch, 16)      (batch, 16)    (batch, 16)
        │               │              │
        └───────────────┴──────────────┘
                   Concatenate
                  (batch, 48)
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

**Training**: Adam (lr=1e-5, weight_decay=5e-4), BCEWithLogitsLoss, batch=16, max 200 epochs, early stopping patience=30.

---

## 6. Data Shapes Reference

| Array | Shape | Description |
|---|---|---|
| Raw EEG | `(N_trials, 22, 1000)` | 22 channels, 4 s @ 250 Hz |
| CSP features | `(N_trials, 16)` | After `CSP.transform()` |
| Node features (input) | `(22, 16)` | CSP vector repeated per node |
| Spectral coherence | `(N_trials, 231)` | One scalar per channel pair |
| Node features (saved) | `(22, 16, N_trials)` | From `class{c}_correct.npy` |
| Edge weights (saved) | `(231,)` | From `class{c}_avg_edges.npy` |
| P-value matrix | `(22, 16)` | Node × Feature statistics |
| Global p-values | `(16,)` | Feature-level stats (node-averaged) |

---

## 7. Statistical Analysis

Both tests are run at two levels:

- **Welch's T-test** (`equal_var=False`): Tests if means of Class 1 vs Class 2 differ, robust to unequal variance.
- **Wilcoxon Rank-Sum**: Non-parametric alternative. The sign of the z-statistic indicates direction (`z > 0` → Class 0 higher).

| Level | Shape | Purpose |
|---|---|---|
| Node-wise | (22, 16) | Which electrode × CSP component pairs are discriminative? |
| Global | (16,) | Which CSP components are significant across all electrodes? |

---

## 8. Visualization Guide

| Script | Plot type | Node value | Edge value | Colormap |
|---|---|---|---|---|
| `plot_feature_topomaps.py` | Average activation | Mean feature across trials | Mean spectral coherence | `Reds` / `plasma` |
| `plot_significant_topomaps.py` | Significant connectivity | Top-10 nodes by p-value | Edges within Top-10 set only | `Reds` / `plasma` |
| `plot_significant_topomaps.py` | Difference map | C1 − C2 (Top-10 masked) | Coherence C1 − C2 (masked) | `RdBu_r` (diverging) |

### Thresholding logic
1. Sort all 22 nodes by Wilcoxon p-value for the chosen feature. Keep **Top 10** (lowest p).
2. An edge is drawn only if **both** endpoint nodes are in the Top-10 set.
3. Node size ∝ feature magnitude; edge width and color ∝ weight magnitude.
4. Color axis is computed globally across both classes for direct comparison.

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

```bash
pip install torch torch-geometric numpy scipy scikit-learn mne matplotlib tqdm
```

---

## 10. Quick-Start Commands

```bash
# 1. Train all pairs (LOSO — repeat for all 6 pairs)
python3 experiment_with_logging.py --c1 0 --c2 1
python3 experiment_with_logging.py --c1 0 --c2 2
# ... (0v3, 1v2, 1v3, 2v3)

# 2. Extract features for all pairs at once
python3 extract_all_features.py

# 3. Run statistical analysis
python3 perform_stats_all_pairs.py

# 4. View significance summary
python3 report_significant_features.py

# 5. Inspect a specific pair/layer interactively
python3 analyze_wilcoxon.py --c1 0 --c2 1 --layer cheb --simple

# 6. Generate topoplots (example: pair 0v1, ChebConv, feature 11)
python3 plot_feature_topomaps.py --c1 0 --c2 1 --layer cheb --feature_idx 11
python3 plot_significant_topomaps.py --c1 0 --c2 1 --layer cheb --feature_idx 11

# 7. Run GNNExplainer and visualize
python3 save_explanations.py --all_trials --output_dir explanation_data
python3 visualize_from_pickle.py explanation_data/explanation_trial_0.pkl

# 8. Inspect any .npy file
python3 inspect_npy.py binary_0_vs_1_feature/cheb/p_values.npy
```

---

## Class Label Reference

| Label | Motor Imagery Task |
|---|---|
| 0 | Left Hand |
| 1 | Right Hand |
| 2 | Both Feet |
| 3 | Tongue |

**Pair naming**: `binary_{c1}_vs_{c2}_feature/` — e.g., `binary_0_vs_1_feature/` = Left Hand vs Right Hand.

## Channel Order (22 electrodes)

```
Index:  0    1    2    3    4    5    6   7   8    9   10  11  12
Name:  Fz  FC3  FC1  FCz  FC2  FC4  C5  C3  C1   Cz  C2  C4  C6

Index: 13   14   15   16   17  18  19  20  21
Name: CP3  CP1  CPz  CP2  CP4  P1  Pz  P2  POz
```
