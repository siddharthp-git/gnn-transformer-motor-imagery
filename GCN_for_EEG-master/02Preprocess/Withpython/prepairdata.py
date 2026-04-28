import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import scipy.io as sio

# Path setup
DATA_DIR = '/Users/siddharth/Documents/Brainconnectivity/BCI_Kaggle/'
SAVE_DIR = '/Users/siddharth/Documents/Brainconnectivity/data_EEGMASTER/'
os.makedirs(SAVE_DIR, exist_ok=True)

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'[info] Using device: {device}')

all_data = []
all_labels = []

print('[info] Reading EEG files...')
for i in range(1, 10):
    n_channels = 22
    n_tests = 6 * 48
    window_Length = 7 * 250
    fs = 250
    t1 = int(2 * fs)
    t2 = int(6 * fs)

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))
    NO_valid_trial = 0

    a = sio.loadmat(DATA_DIR + f'A0{i}T.mat')
    a_data = a['data']

    for ii in range(a_data.size):
        a_data1 = a_data[0, ii][0, 0]
        a_X = a_data1[0]
        a_trial = a_data1[1]
        a_y = a_data1[2]

        for trial in range(a_trial.size):
            start_idx = int(a_trial[trial][0])
            trial_data = np.transpose(a_X[start_idx:(start_idx + window_Length), :22])
            data_return[NO_valid_trial, :, :] = trial_data
            class_return[NO_valid_trial] = int(a_y[trial][0])
            NO_valid_trial += 1

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)

    all_data.append(data_return)
    all_labels.append(class_return)

print('[info] Concatenating data...')
data_all = np.concatenate(all_data, axis=0)  # shape: (total_trials, channels, time)
labels_all = np.concatenate(all_labels, axis=0)  # shape: (total_trials,)

print('[info] Moving data to GPU...')
data_tensor = torch.tensor(data_all, dtype=torch.float32, device=device)
data_tensor = data_tensor.view(data_tensor.shape[0], data_tensor.shape[1], -1)
labels_tensor = torch.tensor(labels_all, dtype=torch.long, device=device)

print('[info] Normalizing data...')
data_tensor -= torch.min(data_tensor)
data_tensor /= torch.max(data_tensor)

print('[info] Computing connectivity matrices...')
data_flat = data_tensor.mean(dim=0)  # Average across trials: shape (channels, time)

mean = data_flat.mean(dim=1, keepdim=True)
centered = data_flat - mean
cov_matrix = centered @ centered.T / (data_flat.shape[1] - 1)

std_dev = data_flat.std(dim=1, keepdim=True)
zscore = (data_flat - mean) / std_dev
pearson_matrix = (zscore @ zscore.T) / zscore.shape[1]

abs_pearson = torch.abs(pearson_matrix)
adj_matrix = abs_pearson - torch.eye(abs_pearson.shape[0], device=device)
degree_vector = adj_matrix.sum(dim=1)
degree_matrix = torch.diag(degree_vector)
laplacian_matrix = degree_matrix - adj_matrix

print('[info] Saving matrices...')
np.savetxt(os.path.join(SAVE_DIR, 'Covariance_Matrix.csv'), cov_matrix.cpu().numpy())
np.savetxt(os.path.join(SAVE_DIR, 'Pearson_Matrix.csv'), pearson_matrix.cpu().numpy())
np.savetxt(os.path.join(SAVE_DIR, 'Absolute_Pearson_Matrix.csv'), abs_pearson.cpu().numpy())
np.save(os.path.join(SAVE_DIR, 'Adjacency_Matrix.npy'), adj_matrix.cpu().numpy())
np.savetxt(os.path.join(SAVE_DIR, 'Degree_Matrix.csv'), degree_matrix.cpu().numpy())
np.savetxt(os.path.join(SAVE_DIR, 'Laplacian_Matrix.csv'), laplacian_matrix.cpu().numpy())

print('[info] Preparing dataset splits...')
data_flat = data_tensor.view(data_tensor.shape[0], -1)
labels_all = labels_tensor.view(-1, 1)
full_data = torch.cat((data_flat, labels_all.float()), dim=1)

perm = torch.randperm(full_data.shape[0])
full_data = full_data[perm]

train_size = int(0.9 * full_data.shape[0])
train_data = full_data[:train_size, :-1].cpu().numpy()
train_labels = full_data[:train_size, -1].cpu().numpy()
test_data = full_data[train_size:, :-1].cpu().numpy()
test_labels = full_data[train_size:, -1].cpu().numpy()

np.save(os.path.join(SAVE_DIR, 'training_set.npy'), train_data)
np.save(os.path.join(SAVE_DIR, 'training_label.npy'), train_labels)
np.save(os.path.join(SAVE_DIR, 'test_set.npy'), test_data)
np.save(os.path.join(SAVE_DIR, 'test_label.npy'), test_labels)

print('[info] All processing complete and saved.')
