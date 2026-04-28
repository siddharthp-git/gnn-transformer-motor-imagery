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
                          absolute_scaling=False, label_indices=None):
    """
    Plots a connectivity graph.
    absolute_scaling: If True, uses np.abs() for list scaling and node sizing (useful for diffs).
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
    edge_cmap = cm.get_cmap(edge_cmap_name)
    if edge_vmin is None: edge_vmin = edge_weights.min()
    if edge_vmax is None: edge_vmax = edge_weights.max()
    edge_norm = mcolors.Normalize(vmin=edge_vmin, vmax=edge_vmax)

    node_cmap = cm.get_cmap(node_cmap_name)
    if vmin is None: vmin = node_imp.min()
    if vmax is None: vmax = node_imp.max()
    node_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 3. Plot Edges
    # Sort edges by weight magnitude if absolute, else value
    if absolute_scaling:
        sort_metrics = np.abs(edge_weights)
    else:
        sort_metrics = edge_weights
        
    sorted_indices = np.argsort(sort_metrics)
    
    for i in sorted_indices:
        w = edge_weights[i]
        
        # Threshold check (magnitude if absolute)
        val_for_thresh = np.abs(w) if absolute_scaling else w
        
        if val_for_thresh >= edge_thresh:
            u, v = edge_index[0, i], edge_index[1, i]
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            color = edge_cmap(edge_norm(w))
            
            # Width scaling
            val_for_width = np.abs(w) if absolute_scaling else w
            # Ensure non-negative width base
            if val_for_width < 0: val_for_width = 0
            
            lw = 1.0 + edge_scaling * val_for_width
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.9, zorder=1)

    # 4. Plot Nodes
    # Scale size by importance
    val_for_size = np.abs(node_imp) if absolute_scaling else node_imp
    
    # Normalize size calc to 0-1 range approx for scaling
    size_min = val_for_size.min()
    size_max = val_for_size.max()
    denom = size_max - size_min
    if denom < 1e-8: denom = 1e-8
    
    node_sizes = node_scaling * (val_for_size - size_min) / denom + 100
    
    sc = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=node_imp, cmap=node_cmap_name, norm=node_norm,
        s=node_sizes, edgecolor="k", zorder=3
    )

    # 4b. Add Text Labels for specific nodes (Top significant ones)
    if label_indices is not None:
        for idx in label_indices:
            u, v = pos[idx]
            ch_name = ch_names[idx]
            # Offset the text slightly so it doesn't overlap the node center
            ax.text(u + 0.005, v + 0.005, ch_name, fontsize=10, fontweight='bold', 
                    ha='left', va='bottom', zorder=5, 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # 5. Add Colorbars
    # Edge Colorbar
    sm_edge = cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    sm_edge.set_array([])
    cbar_edge = plt.colorbar(sm_edge, ax=ax, fraction=0.046, pad=0.04)
    cbar_edge.set_label("Edge Weight", fontsize=12)
    
    # Node Colorbar
    sm_node = cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_node.set_array([])
    cbar_node = plt.colorbar(sm_node, ax=ax, fraction=0.046, pad=0.10)
    cbar_node.set_label("Node Feature", fontsize=12)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_topomap_with_sig_filter(data_path, edge_path, p_val_path, class_label, layer, feature_idx, pair_name, output_dir, 
                                 edge_vmin=None, edge_vmax=None, vmin=None, vmax=None):
    """
    Loads data/edges/p-values, filters top 10 significant nodes, and plots.
    """
    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        return

    # --- Load P-Values ---
    # Shape (22, 16)
    if not os.path.exists(p_val_path):
        print(f"Error: P-values file not found: {p_val_path}")
        return
        
    p_values = np.load(p_val_path)
    # Extract column for this feature
    try:
        p_col = p_values[:, feature_idx] # (22,)
    except IndexError:
        print(f"Error: Feature index {feature_idx} out of bounds for p-values {p_values.shape}")
        return
        
    # Find Indices of Top 10 smallest p-values
    # helper for sorting: argsort gives indices that sort the array
    sorted_indices = np.argsort(p_col)
    top_10_indices = sorted_indices[:10]
    
    print(f"Top 10 Significant Nodes for Feat {feature_idx}: {top_10_indices}")
    print(f"  P-values: {p_col[top_10_indices]}")

    # --- Load Node Features ---
    # (22, 16, N_trials)
    try:
        data = np.load(data_path)
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return

    feat_data = data[:, feature_idx, :]
    # Average across trials: (22,)
    avg_node_values = np.mean(feat_data, axis=1)
    
    # --- Mask Non-Significant Nodes ---
    masked_node_values = np.zeros_like(avg_node_values)
    masked_node_values[top_10_indices] = avg_node_values[top_10_indices]
    
    # --- Load Edge Weights ---
    # (N_edges,)
    if not os.path.exists(edge_path):
        print(f"Error: Edge file not found: {edge_path}")
        return 
        
    avg_edge_values = np.load(edge_path)
    
    # --- Check Shapes ---
    n_nodes = 22
    n_edges_expected = 231
    if avg_edge_values.shape[0] != n_edges_expected:
        print(f"Warning: Edge weights shape {avg_edge_values.shape} != {n_edges_expected}.")

    # Channel Names
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    
    # Reconstruct Edge Index (Fully Connected)
    node_ids = list(range(len(ch_names)))
    edge_list = list(combinations(node_ids, 2))
    edge_index = np.array(edge_list).T 

    # --- Filter Edges: Only keep edges between significant nodes ---
    filtered_edge_values = avg_edge_values.copy()
    top_10_set = set(top_10_indices)
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        # Check if BOTH nodes are significant
        if u not in top_10_set or v not in top_10_set:
            # Set to -1.0 so it falls below threshold (0.0) and won't be plotted
            filtered_edge_values[i] = -1.0
 
    
    # --- Make plot ---
    out_name = f"sig_connectivity_{pair_name}_{layer}_feat{feature_idx}_class{class_label}.png"
    out_path = os.path.join(output_dir, out_name)
    
    title = f"Significant Connectivity (Top 10): Pair {pair_name} | Class {class_label} | {layer.upper()} | Feat {feature_idx}"
    
    # Calculate limits from the ORIGINAL full data to keep colors consistent if not provided
    if vmin is None: vmin = avg_node_values.min()
    if vmax is None: vmax = avg_node_values.max()
    
    # Label top 10 significant nodes
    label_indices = top_10_indices[:10]
    
    plot_connectivity_graph(masked_node_values, filtered_edge_values, edge_index, ch_names, title, out_path,
                           vmin=vmin, vmax=vmax, edge_vmin=edge_vmin, edge_vmax=edge_vmax,
                           label_indices=label_indices)

def main():
    parser = argparse.ArgumentParser(description="Plot Significant Feature Connectivity Topomaps (Top 10 Nodes)")
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
    layer_dir = os.path.join(base_dir, layer) # Features and P-values are here
    
    if not os.path.exists(layer_dir):
        print(f"Error: Directory {layer_dir} not found.")
        return
    
    p_val_file = os.path.join(layer_dir, "p_values.npy")
    
    print(f"--- Generating Significant Connectivity Plots for Pair {c1}vs{c2}, Layer {layer.upper()}, Feature {feat_idx} ---")
    
    # Paths
    f1 = os.path.join(layer_dir, f"class{c1}_correct.npy")
    e1_path = os.path.join(base_dir, f"class{c1}_avg_edges.npy")
    
    f2 = os.path.join(layer_dir, f"class{c2}_correct.npy")
    e2_path = os.path.join(base_dir, f"class{c2}_avg_edges.npy")

    # --- Pre-calculate Global Edge Limits for Consistent Coloring ---
    edge_vmin = None
    edge_vmax = None
    
    if os.path.exists(e1_path) and os.path.exists(e2_path):
        e1_data = np.load(e1_path)
        e2_data = np.load(e2_path)
        
        # Combine to find global min/max
        all_edges = np.concatenate([e1_data, e2_data])
        edge_vmin = all_edges.min()
        edge_vmax = all_edges.max()
        print(f"Global Edge Scaling: [{edge_vmin:.4f}, {edge_vmax:.4f}]")
    else:
        print("Warning: One or both edge files missing. Using per-plot scaling.")

    # --- Pre-calculate Global Node Limits for Consistent Coloring ---
    global_node_vmin = None
    global_node_vmax = None
    
    if os.path.exists(f1) and os.path.exists(f2):
        n1_data = np.load(f1)[:, feat_idx, :]
        n2_data = np.load(f2)[:, feat_idx, :]
        avg_n1 = np.mean(n1_data, axis=1)
        avg_n2 = np.mean(n2_data, axis=1)
        
        global_node_vmin = min(avg_n1.min(), avg_n2.min())
        global_node_vmax = max(avg_n1.max(), avg_n2.max())
        print(f"Global Node Scaling: [{global_node_vmin:.4f}, {global_node_vmax:.4f}]")

    # Process Class 1
    plot_topomap_with_sig_filter(f1, e1_path, p_val_file, c1, layer, feat_idx, pair_name, layer_dir, 
                                 edge_vmin=edge_vmin, edge_vmax=edge_vmax,
                                 vmin=global_node_vmin, vmax=global_node_vmax)
    
    # Process Class 2
    plot_topomap_with_sig_filter(f2, e2_path, p_val_file, c2, layer, feat_idx, pair_name, layer_dir,
                                 edge_vmin=edge_vmin, edge_vmax=edge_vmax,
                                 vmin=global_node_vmin, vmax=global_node_vmax)
                                 
    # --- Process Difference (C1 - C2) ---
    print(f"--- Generating Difference Plot ({c1} - {c2}) ---")
    
    # Load Data Manually for Diff
    # Nodes
    d1 = np.load(f1)[:, feat_idx, :]
    d2 = np.load(f2)[:, feat_idx, :]
    avg_n1 = np.mean(d1, axis=1)
    avg_n2 = np.mean(d2, axis=1)
    diff_nodes = avg_n1 - avg_n2
    
    # P-Values for filtering
    p_vals = np.load(p_val_file)[:, feat_idx]
    top_10 = np.argsort(p_vals)[:10]
    
    # Mask Nodes
    masked_diff_nodes = np.zeros_like(diff_nodes)
    masked_diff_nodes[top_10] = diff_nodes[top_10]
    
    # Edges
    if os.path.exists(e1_path) and os.path.exists(e2_path):
        e1_data = np.load(e1_path)
        e2_data = np.load(e2_path)
        diff_edges = e1_data - e2_data
        
        # Filter Edges (Keep only if BOTH nodes in Top 10)
        # Reconstruct Edge Index (Fully Connected) - Same as in function
        ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
        ]
        node_ids = list(range(len(ch_names)))
        edge_list = list(combinations(node_ids, 2))
        edge_index = np.array(edge_list).T 
        
        filtered_diff_edges = diff_edges.copy()
        top_10_set = set(top_10)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[:, i]
            if u not in top_10_set or v not in top_10_set:
                filtered_diff_edges[i] = 0.0 # Zero overlap for diff means white in RdBu
        
        # Plot Diff
        out_name = f"sig_diff_{pair_name}_{layer}_feat{feat_idx}.png"
        out_path = os.path.join(layer_dir, out_name)
        title = f"Difference ({c1} - {c2}): Pair {pair_name} | {layer.upper()} | Feat {feat_idx}"
        
        # Symmetric Limits for Diverging Colormap
        max_abs_node = np.max(np.abs(masked_diff_nodes))
        max_abs_edge = np.max(np.abs(filtered_diff_edges))
        
        plot_connectivity_graph(masked_diff_nodes, filtered_diff_edges, edge_index, ch_names, title, out_path,
                               edge_thresh=0.0001, # Show non-zero diffs
                               vmin=-max_abs_node, vmax=max_abs_node,
                               edge_vmin=-max_abs_edge, edge_vmax=max_abs_edge,
                               node_cmap_name="RdBu_r", edge_cmap_name="RdBu_r",
                               absolute_scaling=True, label_indices=top_10[:10])


if __name__ == "__main__":
    main()
