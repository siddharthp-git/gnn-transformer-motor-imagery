import numpy as np
import scipy.io

# Initialize an empty list to store features
features = []

# Iterate over subjects
for i in range(2, 10):
    file_path = f'/Users/siddharth/Documents/Brainconnectivity/filt_features/eeg_data_sub{i}_features.mat'
    
    # Load .mat file
    data = scipy.io.loadmat(file_path)
    
    # Extract relevant keys (e.g., from 'auc' to 'zork')
    relevant_keys = list(data.keys())[4:]
    
    # Stack extracted features for the current subject
    feature = np.stack([data[key] for key in relevant_keys], axis=-1)  # Shape: (num_trials, num_nodes, feature_dim)
    selected_features = feature[:, :, [8,9,10,11,12,13,14,15,16]]  # Shape: (num_trials, num_nodes, 5)
    # Append to the list
    features.append(selected_features)

# Convert the list of features into a NumPy array
features = np.array(features)  # Shape: (num_subjects, num_trials, num_nodes, feature_dim)

# Now, to get the desired shape of (trials, channels, features)
# Assume num_nodes corresponds to num_channels
# You can reshape the array to (num_trials, num_nodes, feature_dim) if needed

features = features.reshape(features.shape[0]*features.shape[1],22,9)  # Shape: (num_trials, num_channels, feature_dim)

# Print shape of the final features array
print("Extracted Features Shape:", features.shape)  # Expected: (num_trials, 22, feature_dim)