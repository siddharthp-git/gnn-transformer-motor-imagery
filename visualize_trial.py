import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mne
import os
from experiment_with_logging import (
    GraphTransformerNet, 
    load_BCI2a_data, 
    compute_spectral_coherence, 
    EEGGraphDataset, 
    extract_trial_explanation
)
from torch_geometric.data import DataLoader
from mne.decoding import CSP
import math

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Data for Subject 0
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

    # 3. Explain Feature for Trial 0
    trial_idx = 0
    print(f"Running GNNExplainer for trial {trial_idx}...")
    
    edge_mask, node_mask, pred, true_label, edge_index = extract_trial_explanation(model, dataset, trial_idx, device, epochs=200)
    
    print(f"Trial {trial_idx}:")
    print(f"  True Label: {true_label}")
    print(f"  Predicted:  {pred}")

    # 5. Feature Visualization Match
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    
    title = f"Topoplot with gradient edges (Class {pred})"
    plot_connectivity_graph(node_mask, edge_mask, edge_index, ch_names, title)

def plot_connectivity_graph(node_imp, edge_weights, edge_index, ch_names, title, 
                          edge_thresh=0.0, node_scaling=500, edge_scaling=4.0):
    """
    Plots a connectivity graph matching the user's style:
    - Discrete nodes (Scatter) colored by importance
    - Edges colored by weight
    - Two separate colorbars
    """
    
    # 1. Get 2D positions
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]
    pos = np.array([pos_dict[ch][:2] for ch in ch_names])
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14)

    # 2. Setup Colormaps
    # Edges: Purple to Yellow (Viridis or Plasma or similar)
    edge_cmap_name = "plasma" 
    edge_cmap = cm.get_cmap(edge_cmap_name)
    edge_norm = mcolors.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())

    # Nodes: White to Red
    node_cmap_name = "Reds"
    node_cmap = cm.get_cmap(node_cmap_name)
    node_norm = mcolors.Normalize(vmin=node_imp.min(), vmax=node_imp.max())

    # 3. Plot Edges
    # Sort edges by weight so stronger ones are on top
    sorted_indices = np.argsort(edge_weights)
    for i in sorted_indices:
        w = edge_weights[i]
        if w >= edge_thresh:
            u, v = edge_index[0, i], edge_index[1, i]
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            color = edge_cmap(edge_norm(w))
            lw = 1.0 + edge_scaling * w
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.9, zorder=1)

    # 4. Plot Nodes
    # Scale size by importance
    node_sizes = node_scaling * (node_imp - node_imp.min()) / (node_imp.max() - node_imp.min() + 1e-8) + 100
    
    sc = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=node_imp, cmap=node_cmap_name, norm=node_norm,
        s=node_sizes, edgecolor="k", zorder=3
    )

    # 5. Add Colorbars
    # Edge Colorbar
    sm_edge = cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    sm_edge.set_array([])
    cbar_edge = plt.colorbar(sm_edge, ax=ax, fraction=0.046, pad=0.04)
    cbar_edge.set_label("Edge Weight", fontsize=12)
    
    # Node Colorbar (We need a trick to place it next to the first one or separate)
    # Let's place it on the left or just create a new axis
    # Standard way in matplotlib is usually multiple axes, but let's try just adding another
    # We can shrink the main plot or use dividers.
    # Simple hack: create a new axes for the second colorbar
    
    # Actually, let's just use `plt.colorbar` again with specific axes
    # We can position them manually or use `pad`.
    
    sm_node = cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_node.set_array([])
    cbar_node = plt.colorbar(sm_node, ax=ax, fraction=0.046, pad=0.10) # Pad to move it further out
    cbar_node.set_label("Node Importance", fontsize=12)

    output_path = "topoplot_connectivity_trial_0.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved connectivity topoplot to {output_path}")

if __name__ == "__main__":
    main()
