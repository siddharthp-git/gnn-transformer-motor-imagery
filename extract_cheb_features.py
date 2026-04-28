import torch
import numpy as np
import os
import pickle
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from experiment_with_logging import (
    GraphTransformerNet, 
    load_BCI2a_data, 
    compute_spectral_coherence, 
    EEGGraphDataset
)
from mne.decoding import CSP
import sys

# Configuration
c1, c2 = 0, 1
pair_dir = f"binary_label_{c1}_vs_{c2}"
subjects = range(9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Containers for collected features
# Shape will be list of (22, 16) arrays, to be stacked later
cheb_class0 = []
cheb_class1 = []

print(f"--- Extracting ChebConv Features for Pair {c1} vs {c2} ---\n")

for test_sub in subjects:
    subject_dir = os.path.join(pair_dir, f"subject_{test_sub}")
    model_path = os.path.join(subject_dir, "model.pth")
    
    if not os.path.exists(model_path):
        print(f"Skipping Subject {test_sub}: Model not found at {model_path}")
        continue
        
    print(f"Processing Subject {test_sub}...")

    # 1. Load Data (Exact same logic as experiment_with_logging.py)
    # We need to recreate the exact test set transformation
    # NOTE: To get the correct CSP transformation, we strictly need to fit CSP on the training set 
    # (subjects != test_sub) and apply to test_sub. 
    # However, for simply extracting features from a *trained* model, we assume the model handles inputs.
    # BUT! The model expects CSP-transformed inputs. 
    # So we MUST redo the CSP fit on train and transform on test to feed correct inputs to the model.
    # This is slightly expensive but necessary for correctness.
    
    # --- Load Training Data for CSP Fit ---
    train_data_all = []
    train_labels_all = []
    
    for subi in range(9):
        if subi != test_sub:
            d1, l1 = load_BCI2a_data('BCI_Kaggle/', subject=subi, training=True, all_trials=True)
            d2, l2 = load_BCI2a_data('BCI_Kaggle/', subject=subi, training=False, all_trials=True)
            train_data_all.extend([d1, d2])
            train_labels_all.extend([l1, l2])
            
    train_data_all = np.concatenate(train_data_all, axis=0)
    train_labels_all = np.concatenate(train_labels_all, axis=0)
    
    # --- Load Test Data ---
    td1, tl1 = load_BCI2a_data('BCI_Kaggle/', subject=test_sub, training=True, all_trials=True)
    td2, tl2 = load_BCI2a_data('BCI_Kaggle/', subject=test_sub, training=False, all_trials=True)
    test_data = np.concatenate([td1, td2], axis=0)
    test_labels = np.concatenate([tl1, tl2], axis=0)

    # --- Filter for Pair ---
    train_mask = np.isin(train_labels_all, [c1, c2])
    test_mask = np.isin(test_labels, [c1, c2])

    train_data_binary = train_data_all[train_mask]
    train_labels_raw = train_labels_all[train_mask]
    test_data_binary = test_data[test_mask]
    test_labels_raw = test_labels[test_mask]

    # Remap
    train_labels_binary = np.where(train_labels_raw == c1, 0, 1)
    test_labels_binary = np.where(test_labels_raw == c1, 0, 1)

    # --- CSP ---
    csp = CSP(n_components=16, reg=None, log=True, norm_trace=False, transform_into='average_power')
    csp.fit(train_data_binary, train_labels_binary)
    test_csp = csp.transform(test_data_binary)
    
    test_features = np.repeat(test_csp[:, np.newaxis, :], 22, axis=1)

    # --- Spectral Coherence ---
    test_spec_coh = compute_spectral_coherence(test_data_binary)

    # --- Dataset & Loader ---
    test_dataset = EEGGraphDataset(test_features, test_labels_binary, np.arange(len(test_labels_binary)), 250, test_spec_coh)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size 1 for easy handling

    # 2. Load Model
    model = GraphTransformerNet(in_channels=16, hidden_channels=32, out_channels=1, K=3, transformer_heads=8, dropouts=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Extract Features
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            out, features, node_feats = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_node_feats=True)
            
            # Prediction
            probs = torch.sigmoid(out)
            pred = (probs > 0.5).long().item()
            true_label = batch.y.item()
            
            # Filter: Correct Classifications Only
            if pred == true_label:
                # node_feats shape: (1, 22, 48)
                # Slice ChebConv: Indices 16 to 32
                cheb_feats = node_feats[0, :, 16:32].cpu().numpy() # Shape (22, 16)
                
                # Check original label (0 or 1 mapped)
                # Note: true_label is 0 or 1 because we remapped
                if true_label == 0:
                    cheb_class0.append(cheb_feats)
                else:
                    cheb_class1.append(cheb_feats)

# 4. Stack and Save
print(f"\nFeature Extraction Complete.")
print(f"Class 0 (Correctly Classified) Count: {len(cheb_class0)}")
print(f"Class 1 (Correctly Classified) Count: {len(cheb_class1)}")

if len(cheb_class0) > 0:
    # Stack along new axis (Trials) -> (22, 16, N)
    arr_c0 = np.stack(cheb_class0, axis=2)
    save_path_c0 = os.path.join(pair_dir, "cheb_features_class0_correct.npy")
    np.save(save_path_c0, arr_c0)
    print(f"Saved Class 0 features to: {save_path_c0} | Shape: {arr_c0.shape}")

if len(cheb_class1) > 0:
    arr_c1 = np.stack(cheb_class1, axis=2)
    save_path_c1 = os.path.join(pair_dir, "cheb_features_class1_correct.npy")
    np.save(save_path_c1, arr_c1)
    print(f"Saved Class 1 features to: {save_path_c1} | Shape: {arr_c1.shape}")
