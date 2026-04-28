import numpy as np
import os
from itertools import combinations
import sys

def main():
    p_value_threshold = 0.05
    class_pairs = list(combinations([0, 1, 2, 3], 2))
    layers = ['graph', 'cheb', 'sage']
    
    print(f"--- Significant Feature Report (p < {p_value_threshold}) ---")
    
    for (c1, c2) in class_pairs:
        pair_dir = f"binary_{c1}_vs_{c2}_feature"
        if not os.path.exists(pair_dir):
            continue
            
        print(f"\n==========================================")
        print(f" PAIR: {c1} vs {c2}")
        print(f"==========================================")
        
        for layer in layers:
            layer_dir = os.path.join(pair_dir, layer)
            p_val_path = os.path.join(layer_dir, "p_values.npy")
            t_stat_path = os.path.join(layer_dir, "t_stats.npy")
            global_p_path = os.path.join(layer_dir, "global_p_values.npy")
            
            if not os.path.exists(p_val_path):
                print(f"  [{layer.upper()}] Stats not found.")
                continue
                
            p_values = np.load(p_val_path) # (22, 16)
            t_stats = np.load(t_stat_path) # (22, 16)
            
            # 1. Global Features
            print(f"  [{layer.upper()}] Global Features (Averaged Nodes):")
            if os.path.exists(global_p_path):
                gp = np.load(global_p_path)
                sig_indices = np.where(gp < p_value_threshold)[0]
                if len(sig_indices) == 0:
                    print(f"    None.")
                else:
                    for idx in sig_indices:
                        print(f"    Feature {idx}: p={gp[idx]:.5f}")
            else:
                print(f"    (Global stats not found)")

            # 2. Node-wise Features
            # Find (Node, Feature) where p < 0.05
            sig_rows, sig_cols = np.where(p_values < p_value_threshold)
            num_sig = len(sig_rows)
            
            print(f"  [{layer.upper()}] Significant Node-Features: {num_sig} found")
            
            if num_sig > 0:
                # Group by Node for cleaner output
                for node_idx in sorted(list(set(sig_rows))):
                    feats = sig_cols[sig_rows == node_idx]
                    # Get p-values for these
                    p_vals_node = p_values[node_idx, feats]
                    # Format: "Node 0: Feats [1, 2(p=0.01)]"
                    feat_strs = [f"{f}(p={p:.4f})" for f, p in zip(feats, p_vals_node)]
                    print(f"    Node {node_idx}: {', '.join(feat_strs)}")

if __name__ == "__main__":
    main()
