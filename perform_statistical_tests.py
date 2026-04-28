import numpy as np
import os
from scipy import stats
import sys

def main():
    # Configuration
    c1, c2 = 0, 1
    pair_dir = f"binary_label_{c1}_vs_{c2}"
    
    # Input Files
    file_c0 = os.path.join(pair_dir, "cheb_features_class0_correct.npy")
    file_c1 = os.path.join(pair_dir, "cheb_features_class1_correct.npy")
    
    if not os.path.exists(file_c0) or not os.path.exists(file_c1):
        print(f"Error: Feature files not found in {pair_dir}")
        print(f"Expected: {file_c0} and {file_c1}")
        return

    # Load Data
    # Shapes: (22, 16, N_trials)
    data_c0 = np.load(file_c0)
    data_c1 = np.load(file_c1)
    
    print(f"Loaded Class 0: {data_c0.shape}")
    print(f"Loaded Class 1: {data_c1.shape}")
    
    n_nodes = data_c0.shape[0]   # 22
    n_feats = data_c0.shape[1]   # 16
    
    # Initialize Output Matrices
    t_stats = np.zeros((n_nodes, n_feats))
    p_values = np.zeros((n_nodes, n_feats))
    mean_diff = np.zeros((n_nodes, n_feats)) # Class 0 - Class 1
    
    # Wilcoxon matrices
    wilcoxon_stats = np.zeros((n_nodes, n_feats))
    wilcoxon_p = np.zeros((n_nodes, n_feats))
    
    print(f"\nRunning 2-sample T-test (Welch's) and Wilcoxon Rank-Sum for {n_nodes} nodes x {n_feats} features...")
    
    for i in range(n_nodes):
        for j in range(n_feats):
            # Extract distributions
            dist_c0 = data_c0[i, j, :]
            dist_c1 = data_c1[i, j, :]
            
            # Perform Welch's t-test (equal_var=False)
            # Null hypothesis: means are equal
            # Null hypothesis: means are equal
            t, p = stats.ttest_ind(dist_c0, dist_c1, equal_var=False)
            
            # Perform Wilcoxon Rank-Sum Test
            # Null hypothesis: distributions are equal
            w_stat, w_p = stats.ranksums(dist_c0, dist_c1)
            
            t_stats[i, j] = t
            p_values[i, j] = p
            mean_diff[i, j] = np.mean(dist_c0) - np.mean(dist_c1)
            wilcoxon_stats[i, j] = w_stat
            wilcoxon_p[i, j] = w_p
            
    # Save Outputs
    output_dir = pair_dir
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "t_stats.npy"), t_stats)
    np.save(os.path.join(output_dir, "p_values.npy"), p_values)
    np.save(os.path.join(output_dir, "t_stats.npy"), t_stats)
    np.save(os.path.join(output_dir, "p_values.npy"), p_values)
    np.save(os.path.join(output_dir, "mean_diff.npy"), mean_diff)
    np.save(os.path.join(output_dir, "wilcoxon_stats.npy"), wilcoxon_stats)
    np.save(os.path.join(output_dir, "wilcoxon_p_values.npy"), wilcoxon_p)
    
    print(f"\nAnalysis Complete.")
    print(f"Saved t_stats, p_values, mean_diff, wilcoxon_stats, wilcoxon_p to {output_dir}")
    
    # Statistical Summary
    # Count significant features (p < 0.05)
    sig_mask = p_values < 0.05
    n_sig = np.sum(sig_mask)
    print(f"\nSignificant Features (p < 0.05): {n_sig} / {n_nodes * n_feats}")
    
    # Directionality
    # Class 0 Higher (Mean Diff > 0)
    c0_higher = np.sum((sig_mask) & (mean_diff > 0))
    # Class 1 Higher (Mean Diff < 0)
    c1_higher = np.sum((sig_mask) & (mean_diff < 0))
    
    print(f"  Higher in Class 0: {c0_higher}")
    print(f"  Higher in Class 1: {c1_higher}")

    # --- PART 2: Global Feature Analysis (Averaged across Nodes) ---
    print(f"\n--- Part 2: Node-Averaged Feature Analysis ---")
    
    # Average across nodes (Axis 0)
    # Shapes become (16, N_trials)
    avg_data_c0 = np.mean(data_c0, axis=0) # (16, N0)
    avg_data_c1 = np.mean(data_c1, axis=0) # (16, N1)
    
    global_p_values = []
    global_p_values = []
    global_t_stats = []
    global_wilcoxon_p = []
    global_wilcoxon_stats = []
    
    print(f"Analyzing {n_feats} global features...")
    
    for j in range(n_feats):
        v0 = avg_data_c0[j, :]
        v1 = avg_data_c1[j, :]
        
        t, p = stats.ttest_ind(v0, v1, equal_var=False)
        w_stat, w_p = stats.ranksums(v0, v1)
        
        global_t_stats.append(t)
        global_p_values.append(p)
        global_wilcoxon_stats.append(w_stat)
        global_wilcoxon_p.append(w_p)
        
        print(f"  Feat {j}: T-test p={p:.5e}, Wilcoxon p={w_p:.5e}")

    global_p_values = np.array(global_p_values)
    global_wilcoxon_p = np.array(global_wilcoxon_p)
    
    # Find feature with HIGHEST P-value (Least Significant)
    # User asked for "feature number which has highest p value"
    max_p_idx = np.argmax(global_p_values)
    max_p_val = global_p_values[max_p_idx]
    
    # Also find LOWEST P-value (Most Significant)
    min_p_idx = np.argmin(global_p_values)
    min_p_val = global_p_values[min_p_idx]
    
    print(f"\nGlobal Analysis Results:")
    print(f"  Feature with HIGHEST P-value (Least diff): Feat {max_p_idx} (p={max_p_val:.5f})")
    print(f"  Feature with LOWEST P-value (Most diff):   Feat {min_p_idx} (p={min_p_val:.5e})")
    
    # Save these as well
    np.save(os.path.join(output_dir, "global_t_stats.npy"), np.array(global_t_stats))
    np.save(os.path.join(output_dir, "global_t_stats.npy"), np.array(global_t_stats))
    np.save(os.path.join(output_dir, "global_p_values.npy"), global_p_values)
    np.save(os.path.join(output_dir, "global_wilcoxon_stats.npy"), np.array(global_wilcoxon_stats))
    np.save(os.path.join(output_dir, "global_wilcoxon_p_values.npy"), global_wilcoxon_p)

if __name__ == "__main__":
    main()
