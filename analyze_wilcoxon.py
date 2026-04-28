import numpy as np
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Analyze Wilcoxon Stats for a specific Pair and Layer")
    parser.add_argument("--c1", type=int, default=0, help="Class 1 Label (e.g., 0)")
    parser.add_argument("--c2", type=int, default=1, help="Class 2 Label (e.g., 1)")
    parser.add_argument("--layer", type=str, default="cheb", choices=['graph', 'cheb', 'sage'], help="GNN Layer")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance Level")
    parser.add_argument("--show_all", action="store_true", help="Show all p-values, not just significant ones")
    parser.add_argument("--simple", action="store_true", help="Show simplified 'Node X - pvalue' (Min P) format")
    parser.add_argument("--features", action="store_true", help="Show simplified 'Feature X - pvalue' (Max P) format")
    
    args = parser.parse_args()
    
    c1, c2 = args.c1, args.c2
    layer = args.layer.lower()
    alpha = args.alpha
    
    base_dir = f"binary_{c1}_vs_{c2}_feature"
    layer_dir = os.path.join(base_dir, layer)
    
    print(f"--- Wilcoxon Analysis: Pair {c1}vs{c2} | Layer: {layer.upper()} ---")
    print(f"Directory: {layer_dir}")
    
    if not os.path.exists(layer_dir):
        print(f"Error: Directory not found. Have you run 'extract_all_features.py' or 'perform_stats_all_pairs.py'?")
        return

    # Files
    f_p_val = os.path.join(layer_dir, "wilcoxon_p_values.npy")
    f_stat = os.path.join(layer_dir, "wilcoxon_stats.npy")
    f_global_p = os.path.join(layer_dir, "global_wilcoxon_p_values.npy")
    f_mean_diff = os.path.join(layer_dir, "mean_diff.npy") # Often useful for direction even if from T-test context, or we can assume sign of W-stat? 
    # ranksums returns statistic: "The test statistic z-score." 
    # Positive z usually means x > y (Class 0 > Class 1).
    
    if not os.path.exists(f_p_val):
        print(f"Error: {f_p_val} not found.")
        return
        
    # Load
    p_values = np.load(f_p_val)          # (22, 16)
    w_stats = np.load(f_stat)            # (22, 16)
    
    # 1. Global Analysis
    print(f"\n[1] Global Features (Node-Averaged):")
    if os.path.exists(f_global_p):
        gp = np.load(f_global_p)
        sig_indices = np.where(gp < alpha)[0]
        
        if len(sig_indices) == 0:
            print("  No significant global features.")
        else:
            for idx in sig_indices:
                print(f"  Feature {idx}: p={gp[idx]:.5e}")
                
        # Find highest p-value
        max_p_idx = np.argmax(gp)
        max_p_val = gp[max_p_idx]
        print(f"\n  > Feature with Highest Global P-Value: Feature {max_p_idx} (p={max_p_val:.5e})")
        
    else:
        print("  (Global Wilcoxon stats file not found)")
        
    # 2. Node-wise Analysis
    print(f"\n[2] Significant Node-Features (p < {alpha}):")
    sig_rows, sig_cols = np.where(p_values < alpha)
    num_sig = len(sig_rows)
    print(f"  Total Significant Connections: {num_sig}")
    
    if args.simple:
        print(f"\n--- Node P-Values (Min P-value across 16 features) ---")
        for i in range(22):
            # Take the minimum p-value to represent the node's significance
            min_p = np.min(p_values[i, :])
            print(f"Node {i+1} - {min_p:.5e}")

    elif args.features:
        print(f"\n--- Feature P-Values (Max P-value across 22 nodes) ---")
        for j in range(16):
            # Take the maximum p-value across nodes as requested ("max p value for features")
            max_p = np.max(p_values[:, j])
            print(f"Feature {j} - {max_p:.5e}")
            
    elif args.show_all:
        print(f"\n--- ALL P-VALUES (Node x Feature) ---")
        for i in range(22):
            p_str = ", ".join([f"F{j}={p_values[i, j]:.4f}" for j in range(16)])
            print(f"  Node {i:02d}: {p_str}")
    
    if num_sig > 0 and not args.simple and not args.features:
        # Sort by p-value significance? Or group by node?
        # Grouping by node is usually more readable for topoplots
        unique_nodes = sorted(list(set(sig_rows)))
        
        for node_idx in unique_nodes:
            # Get features for this node
            mask = (sig_rows == node_idx)
            feats = sig_cols[mask]
            
            # Helper to format string
            details = []
            for f in feats:
                p = p_values[node_idx, f]
                stat = w_stats[node_idx, f]
                # Direction
                direction = "C0>C1" if stat > 0 else "C1>C0"
                details.append(f"F{f}(p={p:.4f}, {direction})")
                
            print(f"  Node {node_idx:02d}: {', '.join(details)}")
            
    print("\nDone.")

if __name__ == "__main__":
    main()
