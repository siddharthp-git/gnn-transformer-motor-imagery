import numpy as np
import os
from scipy import stats
from itertools import combinations
import sys

def run_statistical_analysis(data_c0, data_c1, output_dir):
    """
    Runs T-test and Wilcoxon Rank-Sum test on the two datasets.
    data_c0, data_c1: (22, 16, N_trials)
    output_dir: Directory to save results
    """
    n_nodes = data_c0.shape[0]
    n_feats = data_c0.shape[1]
    
    # --- Part 1: Node-wise Analysis ---
    t_stats = np.zeros((n_nodes, n_feats))
    p_values = np.zeros((n_nodes, n_feats))
    mean_diff = np.zeros((n_nodes, n_feats))
    wilcoxon_stats = np.zeros((n_nodes, n_feats))
    wilcoxon_p = np.zeros((n_nodes, n_feats))
    
    print(f"    Running Node-wise Stats ({n_nodes}x{n_feats})...")
    
    for i in range(n_nodes):
        for j in range(n_feats):
            v0 = data_c0[i, j, :]
            v1 = data_c1[i, j, :]
            
            # Welch's T-test
            t, p = stats.ttest_ind(v0, v1, equal_var=False)
            # Wilcoxon Rank-Sum
            w_stat, w_p = stats.ranksums(v0, v1)
            
            t_stats[i, j] = t
            p_values[i, j] = p
            mean_diff[i, j] = np.mean(v0) - np.mean(v1)
            wilcoxon_stats[i, j] = w_stat
            wilcoxon_p[i, j] = w_p
            
    np.save(os.path.join(output_dir, "t_stats.npy"), t_stats)
    np.save(os.path.join(output_dir, "p_values.npy"), p_values)
    np.save(os.path.join(output_dir, "mean_diff.npy"), mean_diff)
    np.save(os.path.join(output_dir, "wilcoxon_stats.npy"), wilcoxon_stats)
    np.save(os.path.join(output_dir, "wilcoxon_p_values.npy"), wilcoxon_p)
    
    # --- Part 2: Global (Node-Averaged) Analysis ---
    avg_c0 = np.mean(data_c0, axis=0) # (16, N0)
    avg_c1 = np.mean(data_c1, axis=0) # (16, N1)
    
    global_t = []
    global_p = []
    global_w_stat = []
    global_w_p = []
    
    for j in range(n_feats):
        v0 = avg_c0[j, :]
        v1 = avg_c1[j, :]
        
        t, p = stats.ttest_ind(v0, v1, equal_var=False)
        w_stat, w_p = stats.ranksums(v0, v1)
        
        global_t.append(t)
        global_p.append(p)
        global_w_stat.append(w_stat)
        global_w_p.append(w_p)
        
    np.save(os.path.join(output_dir, "global_t_stats.npy"), np.array(global_t))
    np.save(os.path.join(output_dir, "global_p_values.npy"), np.array(global_p))
    np.save(os.path.join(output_dir, "global_wilcoxon_stats.npy"), np.array(global_w_stat))
    np.save(os.path.join(output_dir, "global_wilcoxon_p_values.npy"), np.array(global_w_p))
    
    print(f"    -> Saved all statistics to {output_dir}")

def main():
    # Iterate over all pairs
    class_pairs = list(combinations([0, 1, 2, 3], 2))
    layers = ['graph', 'cheb', 'sage']
    
    print("Starting Post-Hoc Statistical Analysis for All Pairs...")

    for (c1, c2) in class_pairs:
        pair_dir = f"binary_{c1}_vs_{c2}_feature"
        
        if not os.path.exists(pair_dir):
            print(f"Skipping Pair {c1}vs{c2}: Directory {pair_dir} not found.")
            continue
            
        print(f"\n=== Processing Pair {c1} vs {c2} ===")
        
        for layer in layers:
            layer_dir = os.path.join(pair_dir, layer)
            if not os.path.exists(layer_dir):
                print(f"  Skipping Layer {layer}: Not found.")
                continue
                
            print(f"  Layer: {layer}")
            
            # Look for extracted feature files
            f1 = os.path.join(layer_dir, f"class{c1}_correct.npy")
            f2 = os.path.join(layer_dir, f"class{c2}_correct.npy")
            
            if os.path.exists(f1) and os.path.exists(f2):
                try:
                    d1 = np.load(f1)
                    d2 = np.load(f2)
                    
                    if d1.size > 0 and d2.size > 0:
                        run_statistical_analysis(d1, d2, layer_dir)
                    else:
                        print(f"    Error: Empty data files.")
                except Exception as e:
                    print(f"    Error processing files: {e}")
            else:
                print(f"    Missing feature files: class{c1}_correct.npy or class{c2}_correct.npy")

    print("\nDone.")

if __name__ == "__main__":
    main()
