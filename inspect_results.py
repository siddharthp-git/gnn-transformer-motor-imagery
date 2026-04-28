import pickle
import numpy as np
import sys

def inspect_pkl(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {filepath}")
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys (Subjects): {list(data.keys())}")
            for sub_id, sub_data in data.items():
                print(f"\n--- Subject {sub_id} ---")
                if isinstance(sub_data, dict):
                    for key, value in sub_data.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  {key}: {value}")
                            
                    # Detailed check on features
                    if 'features' in sub_data:
                        feats = sub_data['features']
                        print(f"  Feature example (first row, first 5 cols): {feats[0, :5]}")
                else:
                    print(f"  Value: {sub_data}")
                # Just show first subject for brevity if many
                # break 
        else:
            print(data)
            
    except Exception as e:
        print(f"Error loading pickle: {e}")

if __name__ == "__main__":
    filepath = "loso_gnn_features_results.pkl"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    inspect_pkl(filepath)
