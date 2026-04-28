import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mne
import sys
import os

def main():
    if len(sys.argv) > 1:
        pickle_path = sys.argv[1]
    else:
        # Default fallback (or we could just error out)
        # Using the path specified in the user's open_pickle.py edit as default example
        pickle_path = '/Users/siddharth/Documents/Brainconnectivity/binary_label_0_vs_1/subject_0/explanation_trial_0.pkl'
    
    print(f"Loading pickle: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        print(f"Error: File not found: {pickle_path}")
        return

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract keys
    node_mask = data['node_mask']
    edge_mask = data['edge_mask']
    edge_index = data['edge_index']
    pred = data['predicted_label']
    true_label = data['true_label']
    
    print(f"Loaded explanation for Trial {data.get('trial_idx', 'Unknown')}")
    print(f"Predicted: {pred}, True: {true_label}")
    
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    
    output_filename = f"topoplot_from_pickle_trial_{data.get('trial_idx', 'custom')}.png"
    title = f"Connectivity: Pred {pred} (True {true_label})"
    
    plot_connectivity_graph(node_mask, edge_mask, edge_index, ch_names, title, output_path=output_filename)


def plot_connectivity_graph(node_imp, edge_weights, edge_index, ch_names, title, 
                          edge_thresh=0.0, node_scaling=500, edge_scaling=4.0, output_path="topoplot_connectivity.png"):
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

    # Node Colorbar
    sm_node = cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_node.set_array([])
    cbar_node = plt.colorbar(sm_node, ax=ax, fraction=0.046, pad=0.10) # Pad to move it further out
    cbar_node.set_label("Node Importance", fontsize=12)

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved connectivity topoplot to {output_path}")

if __name__ == "__main__":
    main()
