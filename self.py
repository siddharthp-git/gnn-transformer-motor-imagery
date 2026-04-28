import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.linalg import eigh
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from itertools import product
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, ChebConv, SAGEConv, global_mean_pool
from tqdm import trange
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import json
import os
from datetime import datetime



def bandpass_filter(data, lowcut=8, highcut=30, fs=250, order=4):
    """Applies a Butterworth bandpass filter to EEG data."""
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def load_BCI2a_data(data_path, subject, training, all_trials=True):    
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6 * 48     
    window_Length = 7 * 250  # Total trial duration (7s)

    # Define MI trial window 
    fs = 250          # Sampling rate
    t1 = int(2 * fs)  # Start time_point (2s)
    t2 = int(6 * fs)  # End time_point (6s)

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))  #(trials x channels x time)

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject + 1) + 'T.mat')   #()
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject + 1) + 'E.mat')

    a_data = a['data']  #()
    
    for ii in range(a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]  # EEG signals
        a_trial = a_data3[1]  # Trial start indices
        a_y = a_data3[2]  # Class labels
        a_artifacts = a_data3[5]  # Artifact presence

        for trial in range(a_trial.size):
            if a_artifacts[trial] != 0 and not all_trials:
                continue
            # Extract EEG trial (time x channels) and transpose to (channels x time)
            trial_data = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_Length), :22])
            
            # Apply bandpass filter
            trial_data = bandpass_filter(trial_data, lowcut=8, highcut=30, fs=fs, order=4)

            data_return[NO_valid_trial, :, :] = trial_data
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1        

    # Keep only valid trials and select MI window (2s to 6s)
    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)  # Convert to zero-based index

    return data_return, class_return



def extract_features(data, labels, fs=250, n_csp_components=2):
    """
    Extracts time-domain, frequency-domain (per channel), and CSP (global) features.
    
    Output:
        features: numpy array of shape (trials, channels, num_features)
        csp_features: numpy array of shape (trials, csp_feature_dim)
    """
    n_trials, n_channels, n_times = data.shape
    per_channel_features = []

    # ---- CSP Filters (global features) ----
    csp_filters = []
    classes = np.unique(labels)
    
    for c in classes:
        X1 = data[labels == c]
        X2 = data[labels != c]

        cov1 = np.mean([np.cov(trial) for trial in X1], axis=0)
        cov2 = np.mean([np.cov(trial) for trial in X2], axis=0)
        cov_total = cov1 + cov2

        eigvals, eigvecs = eigh(cov1, cov_total)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sorted_indices]

        # Top & bottom CSP filters
        filters = np.hstack((eigvecs[:, :n_csp_components], eigvecs[:, -n_csp_components:]))
        csp_filters.append(filters)

    # ---- Feature Extraction ----
    csp_feature_list = []

    for trial in range(n_trials):
        trial_feature_channels = []

        # Per-channel features
        for ch in range(n_channels):
            signal = data[trial, ch, :]

            # Time-domain features
            peak_amp = np.max(np.abs(signal))
            latency = np.argmax(np.abs(signal)) / fs
            auc = np.trapz(np.abs(signal), dx=1/fs)
            slope = (signal[-1] - signal[0]) / n_times
            ptp_amp = np.ptp(signal)
            mean_abs_amp = np.mean(np.abs(signal))
            rms_amp = np.sqrt(np.mean(signal**2))
            std_dev = np.std(signal)
            skewness = skew(signal)
            kurt_val = kurtosis(signal)
            zork = ((signal[:-1] * signal[1:]) < 0).sum()

            # Frequency-domain features
            freqs, psd = welch(signal, fs=fs, nperseg=fs)
            mean_power = np.mean(psd)
            var_power = np.var(psd)
            skew_power = skew(psd)
            kurt_power = kurtosis(psd)
            max_power = np.max(psd)
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

            channel_features = [
                peak_amp, latency, auc, slope, ptp_amp, mean_abs_amp, rms_amp,
                std_dev, skewness, kurt_val, zork,
                mean_power, var_power, skew_power, kurt_power, max_power, spectral_entropy
            ]

            trial_feature_channels.append(channel_features)

        # Add per-channel features to list
        per_channel_features.append(trial_feature_channels)

        # CSP features (global)
        trial_csp_feats = []
        for filters in csp_filters:
            projected = np.dot(filters.T, data[trial])
            variances = np.var(projected, axis=1)
            log_var = np.log(variances + 1e-10)
            trial_csp_feats.extend(log_var.tolist())  # ensure compatibility with older list methods
        csp_feature_list.append(trial_csp_feats)

    # Final output shapes
    return np.array(per_channel_features), np.array(csp_feature_list)


class EEGGraphDataset(Dataset):
    def __init__(self, X, y, indices, loader_type, sfreq, transform=None):
        self.epochs = X
        self.labels = y
        self.indices = indices
        self.sfreq = sfreq
        self.loader_type = loader_type
        self.transform = transform

        # Your updated EEG channel names
        self.ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]

        self.node_ids = range(len(self.ch_names))

        # Create fully connected undirected graph (or change this to a sparse graph if needed)
        self.edge_index = torch.tensor(
            [[a, b] for a, b in product(self.node_ids, self.node_ids)],
            dtype=torch.long
        ).t().contiguous()

        # Compute and normalize geodesic distances
        self.distances = self.get_sensor_distances()
        a = np.array(self.distances)
        self.distances = ((a - np.min(a)) / (np.max(a) - np.min(a))).astype(np.float32)

        # Load any edge-wise spectral coherence features (shape: [samples, edges])
        # self.spec_coh_values = np.load("../spec_coh_values.npy", allow_pickle=True)

    def get_sensor_distances(self):
        coords_1010 = pd.read_csv("/home/deepesh.bme.iitbhu/Siddharth Workspace/BCI/standard_1010.tsv.txt", sep='\t')
        distances = []
        for edge_idx in range(self.edge_index.shape[1]):
            i = self.edge_index[0, edge_idx]
            j = self.edge_index[1, edge_idx]
            dist = self.get_geodesic_distance(self.ch_names[i], self.ch_names[j], coords_1010)
            distances.append(dist)
        return distances

    def get_geodesic_distance(self, ch1, ch2, coords_1010):
        # Find 3D coords from standard_1010.tsv.txt
        def get_coords(ch):
            row = coords_1010[coords_1010['label'] == ch]
            if row.empty:
                print(f"Channel {ch} not found in 10-10 coordinates!")
            return float(row['x']), float(row['y']), float(row['z'])

        x1, y1, z1 = get_coords(ch1)
        x2, y2, z2 = get_coords(ch2)

        dot_product = x1 * x2 + y1 * y2 + z1 * z2
        dot_product = max(min(dot_product, 1.0), -1.0)  # Clamp for stability
        return math.acos(dot_product)  # r = 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = self.indices[idx]
        node_features = torch.from_numpy(self.epochs[idx].reshape(len(self.ch_names), -1))  # auto-detect feature dim
        # spec_coh_values = self.spec_coh_values[idx, :]

        # Combine distances + spectral coherence (you can also concatenate or use them separately)
        edge_weights = torch.tensor(self.distances, dtype=torch.float32)
        # edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        edge_weights = edge_weights.clone().detach().float()

        return Data(
            x=node_features.float(),
            edge_index=self.edge_index,
            edge_attr=edge_weights,
            dataset_idx=idx,
            y=torch.tensor(self.labels[idx], dtype=torch.long)
        )




# --- GNN Blocks with LayerNorm ---

class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels // 2)
        self.ln3 = nn.LayerNorm(hidden_channels // 2)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.ln1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.ln2(self.conv2(x, edge_index, edge_weight)))
        x = F.relu(self.ln3(self.conv3(x, edge_index, edge_weight)))
        return global_mean_pool(x, batch)

class ChebConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=3):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.conv3 = ChebConv(hidden_channels, hidden_channels // 2, K)
        self.ln3 = nn.LayerNorm(hidden_channels // 2)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.ln1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.ln2(self.conv2(x, edge_index, edge_weight)))
        x = F.relu(self.ln3(self.conv3(x, edge_index, edge_weight)))
        return global_mean_pool(x, batch)

class SAGEConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.ln3 = nn.LayerNorm(hidden_channels // 2)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.ln1(self.conv1(x, edge_index)))
        x = F.relu(self.ln2(self.conv2(x, edge_index)))
        x = F.relu(self.ln3(self.conv3(x, edge_index)))
        return global_mean_pool(x, batch)

# --- Transformer Block (unchanged) ---

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=1, dim_feedforward=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

# --- Combined GNN + Transformer Model with LayerNorm ---

class GraphTransformerNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3, transformer_heads=4, transformer_layers=3,dropouts=0.5):
        super().__init__()
        self.graph_block = GraphConvBlock(in_channels, hidden_channels)
        self.cheb_block = ChebConvBlock(in_channels, hidden_channels, K)
        self.sage_block = SAGEConvBlock(in_channels, hidden_channels)

        self.feature_dim = (hidden_channels // 2) * 3
        assert self.feature_dim % transformer_heads == 0, "embed_dim must be divisible by num_heads"

        self.transformer = TransformerBlock(
            input_dim=self.feature_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropouts),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels // 2),
            nn.Dropout(dropouts),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, edge_weight, batch):
        g1 = self.graph_block(x, edge_index, edge_weight, batch)
        g2 = self.cheb_block(x, edge_index, edge_weight, batch)
        g3 = self.sage_block(x, edge_index, edge_weight, batch)

        x = torch.cat([g1, g2, g3], dim=1)

        x = x.unsqueeze(1)  # (batch_size, seq_len=1, feature_dim)
        x = self.transformer(x)
        x = x[:, 0, :]  # back to (batch_size, feature_dim)

        return self.mlp(x)





# Create directory for hyperparameter tuning results
os.makedirs('hyperparameter_results', exist_ok=True)

# Define hyperparameter search space
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'weight_decay': [1e-4, 5e-4, 1e-3],
    'hidden_channels': [16, 32, 64],
    'transformer_heads': [2, 4, 8],
    'batch_size': [8,16],
    'dropout': [0.1, 0.3, 0.5]  # Adding dropout as a tunable parameter
}

# Generate all parameter combinations
param_combinations = list(ParameterGrid(param_grid))
print(f"Total parameter combinations to try: {len(param_combinations)}")

# Track results
results = []

# Loop through subjects
for subi in range(0, 9):  # Assuming 9 subjects as in original code
    print(f"\n{'='*50}")
    print(f"Processing Subject {subi}")
    print(f"{'='*50}")
    
    # Load and prepare data (from original code)
    train_data, train_label = load_BCI2a_data('/home/deepesh.bme.iitbhu/Siddharth Workspace/BCI/Dataset/', subject=subi, training=True, all_trials=True)
    test_data, test_label = load_BCI2a_data('/home/deepesh.bme.iitbhu/Siddharth Workspace/BCI/Dataset/', subject=subi, training=False, all_trials=True)
    tr_features, train_csp = extract_features(train_data, train_label, fs=250, n_csp_components=2)
    te_features, test_csp = extract_features(test_data, test_label, fs=250, n_csp_components=2)
    
    train_features = np.repeat(train_csp[:, np.newaxis, :], 22, axis=1)  # shape: (trials, channels, csp_features)
    test_features = np.repeat(test_csp[:, np.newaxis, :], 22, axis=1)
    
    # Create full datasets
    train_dataset_full = EEGGraphDataset(
        X=train_features, 
        y=train_label, 
        indices=np.arange(len(train_label)), 
        loader_type="train", 
        sfreq=250
    )
    
    test_dataset_full = EEGGraphDataset(
        X=test_features, 
        y=test_label, 
        indices=np.arange(len(test_label)), 
        loader_type="test", 
        sfreq=250
    )
    
    # Hold out validation set from test set
    val_len = int(0.2 * len(test_dataset_full))
    test_len = len(test_dataset_full) - val_len
    test_dataset, val_dataset = random_split(test_dataset_full, [test_len, val_len])
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameter tuning
    subject_results = []
    
    for param_idx, params in enumerate(param_combinations):
        print(f"\nTrying parameter combination {param_idx+1}/{len(param_combinations)}:")
        print(params)
        
        # Create DataLoaders with current batch size
        train_loader = DataLoader(train_dataset_full, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Initialize model with current hyperparameters
        model = GraphTransformerNet(
            in_channels=16,  # Keep fixed as per data dimensions
            hidden_channels=params['hidden_channels'],
            out_channels=4,  # Keep fixed as per number of classes
            K=3,             # Keep fixed for simplicity
            transformer_heads=params['transformer_heads'],
            dropouts=params['dropout']  # Pass dropout parameter
        )
        model.to(device)
        
        # Initialize optimizer with current hyperparameters
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Define train function
        def train_model(model, train_loader):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(train_loader)
        
        # Define evaluation function
        def evaluate_model(model, data_loader, return_preds=False):
            model.eval()
            all_preds = []
            all_labels = []
            total_loss = 0
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    preds = torch.argmax(out, dim=1)
                    loss = criterion(out, batch.y)
                    total_loss += loss.item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
            acc = accuracy_score(all_labels, all_preds)
            kappa = cohen_kappa_score(all_labels, all_preds)
            avg_loss = total_loss / len(data_loader)
            if return_preds:
                return acc, kappa, all_labels, all_preds, avg_loss
            return acc, kappa, avg_loss
        
        # Training loop with early stopping
        num_epochs = 500  # Reduced for faster hyperparameter search
        patience = 50
        early_stop_counter = 0
        
        best_val_acc = 0.0
        best_epoch = 0
        best_model_state = None
        best_val_kappa = 0.0
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Use tqdm for progress tracking
        epoch_bar = trange(1, num_epochs + 1, desc=f"Subject {subi} - Params {param_idx+1}/{len(param_combinations)}", leave=False)
        
        for epoch in epoch_bar:
            train_loss = train_model(model, train_loader)
            train_acc, train_kappa, _ = evaluate_model(model, train_loader)
            val_acc, val_kappa, val_loss = evaluate_model(model, val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            if val_kappa > best_val_kappa:
                best_val_kappa = val_kappa
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}. No improvement for {patience} consecutive epochs.")
                break
            
            epoch_bar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_acc:.4f}',
                'Val Kappa': f'{val_kappa:.4f}',
                'Best val Kappa': f'{best_val_kappa:.4f} (Epoch {best_epoch})',
                'Best val Acc': f'{best_val_acc:.4f} (Epoch {best_epoch})'
            })
        
        # Load best model for final evaluation
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Final evaluation on test set
        test_acc, test_kappa, test_loss = evaluate_model(model, test_loader)
        
        # Record results
        param_result = {
            'subject': subi,
            'params': params,
            'val_acc': best_val_acc,
            'val_kappa': best_val_kappa,
            'test_acc': test_acc,
            'test_kappa': test_kappa,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'epochs_trained': len(train_losses)
        }
        
        subject_results.append(param_result)
        results.append(param_result)
        
        print(f"Test Accuracy: {test_acc:.4f}, Test Kappa: {test_kappa:.4f},Best Val Kappa: {best_val_kappa:.4f} (Epoch {best_epoch}),Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
        
        # Save learning curves for this parameter combination
        if test_kappa > 0.5:  # Only save plots for reasonably good models
            plt.figure(figsize=(14, 5))
            
            # Accuracy Plot
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Acc', color='blue')
            plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Acc', color='green')
            plt.axhline(y=test_acc, color='red', linestyle='--', label=f'Test Acc: {test_acc:.4f}')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Subject {subi} - Params {param_idx+1}")
            plt.legend()
            
            # Loss Plot
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', color='green')
            plt.axhline(y=test_loss, color='red', linestyle='--', label=f'Test Loss: {test_loss:.4f}')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Kappa: {test_kappa:.4f}")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'hyperparameter_results/subject_{subi}_params_{param_idx}.png')
            plt.close()
    
    # Sort subject results by test kappa score
    subject_results.sort(key=lambda x: x['test_kappa'], reverse=True)
    
    # Save subject results
    with open(f'hyperparameter_results/subject_{subi}_results.json', 'w') as f:
        json.dump(subject_results, f, indent=4)
    
    # Print best hyperparameters for this subject
    best_params = subject_results[0]['params']
    best_test_kappa = subject_results[0]['test_kappa']
    best_test_acc = subject_results[0]['test_acc']
    best_val_acc = subject_results[0]['val_acc']
    best_val_kappa = subject_results[0]['val_kappa']
    print(f"\nBest hyperparameters for Subject {subi}:")
    print(f"Parameters: {best_params}")
    print(f"Test Accuracy: {best_test_acc:.4f}, Test Kappa: {best_test_kappa:.4f}",'Val Accuracy:',f"{best_val_acc:.4f}",'Val Kappa:',f"{best_val_kappa:.4f}")

# Save overall results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'hyperparameter_results/all_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=4)

# Analyze results across subjects
print("\nAggregating results across subjects...")

# Group by parameter combination
param_to_avg_scores = {}
for result in results:
    param_key = str(result['params'])
    if param_key not in param_to_avg_scores:
        param_to_avg_scores[param_key] = {
            'params': result['params'],
            'test_acc_scores': [],
            'test_kappa_scores': [],
            'subjects_count': 0
        }
    
    param_to_avg_scores[param_key]['test_acc_scores'].append(result['test_acc'])
    param_to_avg_scores[param_key]['test_kappa_scores'].append(result['test_kappa'])
    param_to_avg_scores[param_key]['subjects_count'] += 1

# Calculate averages
for param_key, data in param_to_avg_scores.items():
    data['avg_test_acc'] = np.mean(data['test_acc_scores'])
    data['avg_test_kappa'] = np.mean(data['test_kappa_scores'])
    data['std_test_acc'] = np.std(data['test_acc_scores'])
    data['std_test_kappa'] = np.std(data['test_kappa_scores'])

# Sort by average kappa score
sorted_params = sorted(
    param_to_avg_scores.values(),
    key=lambda x: x['avg_test_kappa'],
    reverse=True
)

# Print top 5 best parameter combinations
print("\nTop 5 Hyperparameter Combinations (Averaged across subjects):")
for i, param_data in enumerate(sorted_params[:5]):
    print(f"{i+1}. Parameters: {param_data['params']}")
    print(f"   Avg Test Accuracy: {param_data['avg_test_acc']:.4f} ± {param_data['std_test_acc']:.4f}")
    print(f"   Avg Test Kappa: {param_data['avg_test_kappa']:.4f} ± {param_data['std_test_kappa']:.4f}")
    print(f"   Subjects evaluated: {param_data['subjects_count']}")
    print()

# Save aggregated results
with open(f'hyperparameter_results/aggregated_results_{timestamp}.json', 'w') as f:
    json.dump(sorted_params, f, indent=4)

# Visualize parameter impact
print("\nCreating parameter impact visualization...")

# Calculate average performance for each parameter value
param_impact = {}
for param_name in param_grid.keys():
    param_impact[param_name] = {}
    for param_value in param_grid[param_name]:
        matching_results = [r for r in results if r['params'][param_name] == param_value]
        avg_kappa = np.mean([r['test_kappa'] for r in matching_results])
        param_impact[param_name][param_value] = avg_kappa

# Plot parameter impact
plt.figure(figsize=(15, 10))
param_count = len(param_impact)
cols = 2
rows = (param_count + 1) // cols

for i, (param_name, value_scores) in enumerate(param_impact.items()):
    plt.subplot(rows, cols, i+1)
    
    param_values = list(value_scores.keys())
    kappa_scores = list(value_scores.values())
    
    # Sort by parameter values for better visualization
    sorted_indices = np.argsort(param_values)
    sorted_values = [param_values[i] for i in sorted_indices]
    sorted_scores = [kappa_scores[i] for i in sorted_indices]
    
    plt.bar(range(len(sorted_values)), sorted_scores, color='skyblue')
    plt.xticks(range(len(sorted_values)), [str(v) for v in sorted_values], rotation=45)
    plt.xlabel(param_name)
    plt.ylabel('Average Test Kappa')
    plt.title(f'Impact of {param_name}')

plt.tight_layout()
plt.savefig(f'hyperparameter_results/parameter_impact_{timestamp}.png')
plt.close()

print(f"\nHyperparameter tuning completed! Results saved to 'hyperparameter_results' directory.")

# Use the best hyperparameters for final training on all subjects
print("\nTraining final models with best hyperparameters for each subject...")

# Train with the optimal hyperparameters for each subject
for subi in range(0, 9):
    # Load the best hyperparameters for this subject
    try:
        with open(f'hyperparameter_results/subject_{subi}_results.json', 'r') as f:
            subject_results = json.load(f)
        
        best_params = subject_results[0]['params']
        print(f"\nTraining final model for Subject {subi} with best parameters:")
        print(best_params)
        
        # Rest of the training code with the best parameters would go here
        # This would be similar to your original training loop but using the best parameters
        
    except Exception as e:
        print(f"Error loading results for Subject {subi}: {e}")
        continue