import numpy as np
import os
import sys

# Replace this with the path you want to check
# Example: "binary_0_vs_1_feature/cheb/t_stats.npy"
file_path = "/Users/siddharth/Documents/Brainconnectivity/binary_0_vs_3_feature/graph/wilcoxon_p_values.npy"

if len(sys.argv) > 1:
    file_path = sys.argv[1]

print(f"Inspecting: {file_path}")

if not os.path.exists(file_path):
    print("File not found.")
    sys.exit(1)

data = np.load(file_path)

print(f"Shape: {data.shape}")
print(f"Type: {data.dtype}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")
print(f"Mean: {np.mean(data)}")

print("\nSample Data (First 5x5 if applicable):")
if data.ndim == 2:
    print(data[:5, :5])
elif data.ndim == 3:
    print(data[:5, :5, 0])
else:
    print(data.flat[:20])
