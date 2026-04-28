import os
import pickle
import numpy as np
from sklearn.metrics import cohen_kappa_score
import glob

def calculate_kappa_from_explanations(directory):
    # Find all explanation pickle files
    files = glob.glob(os.path.join(directory, "explanation_trial_*.pkl"))
    
    if not files:
        print(f"No explanation files found in {directory}")
        return None
    
    # Sort files by trial index to maintain order (though not strictly necessary for kappa)
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    y_true = []
    y_pred = []
    
    for f_path in files:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
            y_true.append(data['true_label'])
            y_pred.append(data['predicted_label'])
            
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa, len(y_true)

if __name__ == "__main__":
    pair_dir = "/Users/siddharth/Documents/Brainconnectivity/binary_label_0_vs_1"
    
    # Logic to find subject folders
    subj_dirs = sorted([d for d in os.listdir(pair_dir) if os.path.isdir(os.path.join(pair_dir, d)) and d.startswith("subject_")])
    
    results = []
    
    print(f"\nKappa Analysis for Pair 0v1 (LOSO Results)")
    print(f"{'Subject':<12} | {'Trials':<8} | {'Kappa':<8}")
    print("-" * 35)
    
    for s_dir in subj_dirs:
        full_path = os.path.join(pair_dir, s_dir)
        res = calculate_kappa_from_explanations(full_path)
        if res:
            kappa, count = res
            print(f"{s_dir:<12} | {count:<8} | {kappa:.4f}")
            results.append(kappa)
        else:
            print(f"{s_dir:<12} | No data")
            
    if results:
        print("-" * 35)
        print(f"{'AVERAGE':<12} | {'-':<8} | {np.mean(results):.4f}")
