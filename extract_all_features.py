import torch
import numpy as np
import os
import pickle
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from itertools import combinations
from experiment_with_logging import (
    GraphTransformerNet, 
    load_BCI2a_data, 
    compute_spectral_coherence, 
    EEGGraphDataset
)
from mne.decoding import CSP
import sys

def main():
    subjects = range(9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Iterate over all pairs
    class_pairs = list(combinations([0, 1, 2, 3], 2))

    for (c1, c2) in class_pairs:
        print(f"\n==========================================")
        print(f" PROCESSING PAIR: Class {c1} vs Class {c2} ")
        print(f"==========================================\n")
        
        pair_dir_in = f"binary_label_{c1}_vs_{c2}"
        
        # Prepare storage for this pair
        # Structure: layer -> class_label -> list of arrays
        results = {
            'graph': {c1: [], c2: []},
            'cheb':  {c1: [], c2: []},
            'sage':  {c1: [], c2: []}
        }
        
        # Structure: class_label -> list of (N_edges,) arrays
        edge_results = {c1: [], c2: []}
        
        # Check if pair directory exists at all before diving in
        if not os.path.exists(pair_dir_in):
            print(f"Warning: Directory {pair_dir_in} does not exist. Skipping pair.")
            continue

        for test_sub in subjects:
            subject_dir = os.path.join(pair_dir_in, f"subject_{test_sub}")
            model_path = os.path.join(subject_dir, "model.pth")
            
            if not os.path.exists(model_path):
                # print(f"Skipping Subject {test_sub}: Model not found.")
                continue
                
            print(f"  Subject {test_sub}...", end="\r")

            # --- Load Data & Transform (Replicating Pipeline) ---
            # 1. Load Train (for CSP fit) and Test
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
            
            td1, tl1 = load_BCI2a_data('BCI_Kaggle/', subject=test_sub, training=True, all_trials=True)
            td2, tl2 = load_BCI2a_data('BCI_Kaggle/', subject=test_sub, training=False, all_trials=True)
            test_data = np.concatenate([td1, td2], axis=0)
            test_labels = np.concatenate([tl1, tl2], axis=0)

            # 2. Filter for Pair
            train_mask = np.isin(train_labels_all, [c1, c2])
            test_mask = np.isin(test_labels, [c1, c2])

            train_data_binary = train_data_all[train_mask]
            train_labels_raw = train_labels_all[train_mask]
            test_data_binary = test_data[test_mask]
            test_labels_raw = test_labels[test_mask]

            # 3. Remap Labels (c1->0, c2->1) strictly for model compatibility
            train_labels_binary = np.where(train_labels_raw == c1, 0, 1)
            test_labels_binary = np.where(test_labels_raw == c1, 0, 1)

            # 4. CSP
            csp = CSP(n_components=16, reg=None, log=True, norm_trace=False, transform_into='average_power')
            csp.fit(train_data_binary, train_labels_binary)
            test_csp = csp.transform(test_data_binary)
            test_features = np.repeat(test_csp[:, np.newaxis, :], 22, axis=1)

            # 5. Spec Coherence
            test_spec_coh = compute_spectral_coherence(test_data_binary)

            # 6. DataLoader
            test_dataset = EEGGraphDataset(test_features, test_labels_binary, np.arange(len(test_labels_binary)), 250, test_spec_coh)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            # 7. Model Load
            model = GraphTransformerNet(in_channels=16, hidden_channels=32, out_channels=1, K=3, transformer_heads=8, dropouts=0.3).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # 8. Inference & Extraction
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    # Forward
                    out, _, node_feats = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_node_feats=True)
                    
                    probs = torch.sigmoid(out)
                    pred_bin = (probs > 0.5).long().item()
                    true_bin = batch.y.item()
                    
                    # Original class label for storage
                    # if true_bin is 0 (remapped), it's c1. If 1, it's c2.
                    original_true_label = c1 if true_bin == 0 else c2
                    
                    # Correctness Check
                    if pred_bin == true_bin:
                        # node_feats: (1, 22, 48)
                        nf = node_feats[0].cpu().numpy() # (22, 48)
                        
                        # Slice Layers
                        # Graph: 0-16
                        # Cheb:  16-32
                        # SAGE:  32-48
                        g_feat = nf[:, 0:16]
                        c_feat = nf[:, 16:32]
                        s_feat = nf[:, 32:48]
                        
                        results['graph'][original_true_label].append(g_feat)
                        results['cheb'][original_true_label].append(c_feat)
                        results['sage'][original_true_label].append(s_feat)
                        
                        # Save Edge Weights (Layer Independent)
                        # batch.edge_attr: (N_edges, 1) or (N_edges)
                        # We want it as numpy array
                        weights = batch.edge_attr.cpu().numpy()
                        edge_results[original_true_label].append(weights)
        
        print("") # Newline after subject loop
        
        # --- Save Results for this pair ---
        # Output struct: binary_{c1}_vs_{c2}_feature/{layer}/class{L}_correct.npy
        base_out_dir = f"binary_{c1}_vs_{c2}_feature"
        
        layers = ['graph', 'cheb', 'sage']
        for layer in layers:
            layer_dir = os.path.join(base_out_dir, layer)
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save for both classes in pair
            for cls in [c1, c2]:
                data_list = results[layer][cls]
                if len(data_list) > 0:
                    # Stack: (22, 16, N)
                    arr = np.stack(data_list, axis=2)
                    fname = f"class{cls}_correct.npy"
                    save_path = os.path.join(layer_dir, fname)
                    np.save(save_path, arr)
                    print(f"  Saved {layer}/{fname} shape: {arr.shape}")
                else:
                    print(f"  Warning: No correct trials found for {layer} Class {cls}")

        # --- Save Averaged Edge Weights for this Pair ---
        print(f"  Saving Averaged Edge Weights...")
        for cls in [c1, c2]:
            edges_list = edge_results[cls]
            if len(edges_list) > 0:
                # Stack: (N_edges, N_trials) -> Mean -> (N_edges,)
                # edges_list items could be (N_edges,) or (N_edges, 1)
                # Ensure they are flat
                flat_edges = [e.flatten() for e in edges_list]
                stacked_edges = np.stack(flat_edges, axis=1) # (N_edges, N_trials)
                avg_edges = np.mean(stacked_edges, axis=1)   # (N_edges,)
                
                fname = f"class{cls}_avg_edges.npy"
                save_path = os.path.join(base_out_dir, fname)
                np.save(save_path, avg_edges)
                print(f"    Saved {fname} shape: {avg_edges.shape}")
            else:
                print(f"    Warning: No edge data for Class {cls}")

    print("\n--- All Extraction Complete ---")

if __name__ == "__main__":
    main()
