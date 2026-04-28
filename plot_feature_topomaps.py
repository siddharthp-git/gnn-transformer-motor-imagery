import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mne
import os
import argparse
from itertools import combinations

def plot_connectivity_graph(node_imp, edge_weights, edge_index, ch_names, title, output_path,
                          edge_thresh=0.0, node_scaling=500, edge_scaling=4.0,
                          vmin=None, vmax=None, edge_vmin=None, edge_vmax=None,
                          node_cmap_name="Reds", edge_cmap_name="plasma",
                          label_indices=None):
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
    # Edges: Purple to Yellow
    edge_cmap = cm.get_cmap(edge_cmap_name)
    if edge_vmin is None: edge_vmin = edge_weights.min()
    if edge_vmax is None: edge_vmax = edge_weights.max()
    edge_norm = mcolors.Normalize(vmin=edge_vmin, vmax=edge_vmax)

    # Nodes: White to Red
    node_cmap = cm.get_cmap(node_cmap_name)
    if vmin is None: vmin = node_imp.min()
    if vmax is None: vmax = node_imp.max()
    node_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

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

    if label_indices is not None:
        for idx in label_indices:
            u, v = pos[idx]
            ch_name = ch_names[idx]
            ax.text(u + 0.005, v + 0.005, ch_name, fontsize=10, fontweight='bold', 
                    ha='left', va='bottom', zorder=5, 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # 5. Add Colorbars
    # Edge Colorbar
    sm_edge = cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    sm_edge.set_array([])
    cbar_edge = plt.colorbar(sm_edge, ax=ax, fraction=0.046, pad=0.04)
    cbar_edge.set_label("Edge Weight (Avg)", fontsize=12)
    
    # Node Colorbar
    sm_node = cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_node.set_array([])
    cbar_node = plt.colorbar(sm_node, ax=ax, fraction=0.046, pad=0.10)
    cbar_node.set_label("Node Importance (Avg Feature)", fontsize=12)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_topomap_for_class(data_path, edge_path, class_label, layer, feature_idx, pair_name, output_dir,
                           vmin=None, vmax=None, edge_vmin=None, edge_vmax=None, label_indices=None):
    """
    Loads data, selects feature, averages, loads edges, and plots connectivity.
    """
    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        return

    # --- Load Node Features ---
    # (22, 16, N_trials)
    try:
        data = np.load(data_path)
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return

    if data.ndim != 3:
        print(f"Error: Expected 3D array (Node, Feat, Trial), got {data.shape}")
        return

    # Extract specific feature across all trials: (22, N_trials)
    if feature_idx >= data.shape[1]:
        print(f"Error: Feature index {feature_idx} out of bounds for shape {data.shape}")
        return

    feat_data = data[:, feature_idx, :]
    
    # Average across trials: (22,)
    avg_node_values = np.mean(feat_data, axis=1)
    
    # --- Load Edge Weights ---
    # (N_edges,)
    if not os.path.exists(edge_path):
        print(f"Error: Edge file not found: {edge_path}")
        # Could proceed with zeros if needed, but better to stop if user expects edges
        return 
        
    avg_edge_values = np.load(edge_path)
    
    # --- Check Shapes ---
    # We expect 22 nodes
    n_nodes = 22
    # Combinations of 22 nodes = 231 edges
    n_edges_expected = 231
    
    if avg_edge_values.shape[0] != n_edges_expected:
        print(f"Warning: Edge weights shape {avg_edge_values.shape} != {n_edges_expected}. Edges might largely fail.")

    # Channel Names (Standard BCI2a 22 channels)
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    
    # Reconstruct Edge Index (Fully Connected)
    node_ids = list(range(len(ch_names)))
    edge_list = list(combinations(node_ids, 2))
    # Shape (2, N_edges)
    edge_index = np.array(edge_list).T 
    
    # --- Plot ---
    out_name = f"connectivity_{pair_name}_{layer}_feat{feature_idx}_class{class_label}.png"
    out_path = os.path.join(output_dir, out_name)
    
    title = f"Connectivity: Pair {pair_name} | Class {class_label} | {layer.upper()} | Feat {feature_idx}"
    
    plot_connectivity_graph(avg_node_values, avg_edge_values, edge_index, ch_names, title, out_path,
                           vmin=vmin, vmax=vmax, edge_vmin=edge_vmin, edge_vmax=edge_vmax,
                           label_indices=label_indices)

def main():
    parser = argparse.ArgumentParser(description="Plot Feature Connectivity Topomaps (Averaged over trials)")
    parser.add_argument("--c1", type=int, required=True, help="Class 1 Label (e.g. 0)")
    parser.add_argument("--c2", type=int, required=True, help="Class 2 Label (e.g. 1)")
    parser.add_argument("--layer", type=str, required=True, choices=['graph', 'cheb', 'sage'], help="GNN Layer")
    parser.add_argument("--feature_idx", type=int, required=True, help="Feature Index (0-15)")
    
    args = parser.parse_args()
    
    c1 = args.c1
    c2 = args.c2
    layer = args.layer.lower()
    feat_idx = args.feature_idx
    
    pair_name = f"{c1}v{c2}"
    base_dir = f"binary_{c1}_vs_{c2}_feature"
    layer_dir = os.path.join(base_dir, layer) # Features are here
    
    # Edges are in the base_dir (layer independent)
    # e.g., binary_0_vs_1_feature/class0_avg_edges.npy
    
    if not os.path.exists(layer_dir):
        print(f"Error: Directory {layer_dir} not found.")
        return
        
    # --- Global Scaling ---
    edge_vmin, edge_vmax = None, None
    f1_edges = os.path.join(base_dir, f"class{c1}_avg_edges.npy")
    f2_edges = os.path.join(base_dir, f"class{c2}_avg_edges.npy")
    if os.path.exists(f1_edges) and os.path.exists(f2_edges):
        all_e = np.concatenate([np.load(f1_edges), np.load(f2_edges)])
        edge_vmin, edge_vmax = all_e.min(), all_e.max()

    node_vmin, node_vmax = None, None
    f1_nodes = os.path.join(layer_dir, f"class{c1}_correct.npy")
    f2_nodes = os.path.join(layer_dir, f"class{c2}_correct.npy")
    if os.path.exists(f1_nodes) and os.path.exists(f2_nodes):
        n1 = np.load(f1_nodes)[:, feat_idx, :]
        n2 = np.load(f2_nodes)[:, feat_idx, :]
        all_n = np.concatenate([np.mean(n1, axis=1), np.mean(n2, axis=1)])
        node_vmin, node_vmax = all_n.min(), all_n.max()
        
    # Find labels (top 10 nodes for each class)
    l1 = np.argsort(np.mean(np.load(f1_nodes)[:, feat_idx, :], axis=1))[-10:]
    l2 = np.argsort(np.mean(np.load(f2_nodes)[:, feat_idx, :], axis=1))[-10:]

    print(f"--- Generating Connectivity Plots for Pair {c1}vs{c2}, Layer {layer.upper()}, Feature {feat_idx} ---")
    
    # Process Class 1
    plot_topomap_for_class(f1_nodes, f1_edges, c1, layer, feat_idx, pair_name, layer_dir,
                           vmin=node_vmin, vmax=node_vmax, edge_vmin=edge_vmin, edge_vmax=edge_vmax,
                           label_indices=l1)
    
    # Process Class 2
    plot_topomap_for_class(f2_nodes, f2_edges, c2, layer, feat_idx, pair_name, layer_dir,
                           vmin=node_vmin, vmax=node_vmax, edge_vmin=edge_vmin, edge_vmax=edge_vmax,
                           label_indices=l2)

if __name__ == "__main__":
    main()
