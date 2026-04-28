import torch
import numpy as np
import os
import argparse
from experiment_with_logging import (
    GraphTransformerNet, 
    load_BCI2a_data, 
    compute_spectral_coherence, 
    EEGGraphDataset, 
    extract_trial_explanation
)
from mne.decoding import CSP
import pickle

def main():
    parser = argparse.ArgumentParser(description="Extract and save GNN explanations for trials.")
    parser.add_argument('--trial_idx', type=int, default=0, help="Index of the trial to explain (default: 0)")
    parser.add_argument('--all_trials', action='store_true', help="If set, run for ALL trials in the test set (Warning: Slow)")
    parser.add_argument('--output_dir', type=str, default="explanation_data", help="Directory to save output files")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Data (Subject 0 Test)
    print("Loading data for Subject 0...")
    target_subject = 0
    d1, l1 = load_BCI2a_data('BCI_Kaggle/', subject=target_subject, training=True, all_trials=True)
    d2, l2 = load_BCI2a_data('BCI_Kaggle/', subject=target_subject, training=False, all_trials=True)
    
    data = np.concatenate([d1, d2], axis=0)
    labels = np.concatenate([l1, l2], axis=0)
    
    mask = np.isin(labels, [0, 1])
    data_binary = data[mask]
    labels_binary = labels[mask]
    
    print("Computing CSP...")
    csp = CSP(n_components=16, reg=None, log=True, norm_trace=False, transform_into='average_power')
    csp_features = csp.fit_transform(data_binary, labels_binary)
    features = np.repeat(csp_features[:, np.newaxis, :], 22, axis=1)
    
    print("Computing Spectral Coherence...")
    spec_coh = compute_spectral_coherence(data_binary)
    
    dataset = EEGGraphDataset(features, labels_binary, np.arange(len(labels_binary)), 250, spec_coh)
    
    # 2. Load Model
    print("Loading Model...")
    model = GraphTransformerNet(in_channels=16, hidden_channels=32, out_channels=1, K=3, transformer_heads=8, dropouts=0.3).to(device)
    
    model_path = "model_subject_0.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Warning: Model checkpoint not found. Using random weights.")

    # 3. Determine trials to process
    if args.all_trials:
        trial_indices = range(len(dataset))
    else:
        trial_indices = [args.trial_idx]
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing {len(trial_indices)} trials...")
    
    for idx in trial_indices:
        print(f"Explaining Trial {idx}...")
        
        # Run GNNExplainer
        # Note: 'epochs' controls optimization time. 200 is decent.
        edge_mask, node_mask, pred, true_label, edge_index = extract_trial_explanation(
            model, dataset, idx, device, epochs=200
        )
        
        # Save Dictionary
        save_data = {
            "trial_idx": idx,
            "node_mask": node_mask,      # (22,) Node importance
            "edge_mask": edge_mask,      # (NumEdges,) Edge importance
            "edge_index": edge_index,    # (2, NumEdges) Connectivity
            "predicted_label": pred,
            "true_label": true_label,
            "raw_features": dataset[idx].x.numpy() if hasattr(dataset[idx].x, 'numpy') else dataset[idx].x.cpu().numpy()
        }
        
        filename = os.path.join(args.output_dir, f"explanation_trial_{idx}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(save_data, f)
            
        print(f"Saved: {filename}")
        
    print("Done.")

if __name__ == "__main__":
    main()
