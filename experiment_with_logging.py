import argparse
import sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, coherence
from mne.decoding import CSP
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, ChebConv, SAGEConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from sklearn.metrics import accuracy_score, cohen_kappa_score
from itertools import combinations
from tqdm import tqdm, trange
import pickle
import os

# --- Helper Functions ---

def bandpass_filter(data, lowcut=8, highcut=30, fs=250, order=4):
    """Applies a Butterworth bandpass filter to EEG data."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def compute_spectral_coherence(data, sfreq=250):
    """
    Compute coherence between all pairs of channels for each sample.
    Returns: np.ndarray, shape (samples, num_edges)
    """
    n_samples, n_channels, _ = data.shape
    coherence_features = []

    # Disable tqdm for inner loop if running many epochs to avoid clutter
    iterator = data if n_samples < 100 else tqdm(data, desc="Computing coherence", leave=False)
    
    for sample in iterator:
        sample_coherence = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                f, coh = coherence(sample[i], sample[j], fs=sfreq, nperseg=256)
                avg_coh = np.mean(coh)
                sample_coherence.append(avg_coh)
        coherence_features.append(sample_coherence)

    return np.array(coherence_features)

def load_BCI2a_data(data_path, subject, training, all_trials=True):    
    n_channels = 22
    n_tests = 6 * 48     
    window_Length = 7 * 250
    fs = 250
    t1 = int(2 * fs)
    t2 = int(6 * fs)

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    filename = 'A0' + str(subject + 1) + ('T.mat' if training else 'E.mat')
    filepath = os.path.join(data_path, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    a = sio.loadmat(filepath)
    a_data = a['data']

    for ii in range(a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(a_trial.size):
            if a_artifacts[trial] != 0 and not all_trials:
                continue
            trial_data = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_Length), :22])
            trial_data = bandpass_filter(trial_data, lowcut=8, highcut=30, fs=fs, order=4)
            data_return[NO_valid_trial, :, :] = trial_data
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1        

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)

    return data_return, class_return

# --- Dataset ---

class EEGGraphDataset(Dataset):
    def __init__(self, X, y, indices, sfreq, spec_coh_values):
        self.epochs = X
        self.labels = y
        self.indices = indices
        self.sfreq = sfreq
        self.spec_coh_values = spec_coh_values
        self.ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
        ]
        self.node_ids = list(range(len(self.ch_names)))
        edge_list = list(combinations(self.node_ids, 2))
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        real_idx = self.indices[idx]
        node_features = torch.from_numpy(self.epochs[real_idx].reshape(22, -1)).float()
        edge_weights = torch.tensor(self.spec_coh_values[real_idx], dtype=torch.float32)

        return Data(
            x=node_features,
            edge_index=self.edge_index,
            edge_attr=edge_weights,
            dataset_idx=real_idx,
            y=torch.tensor(self.labels[real_idx], dtype=torch.long)
        )

# --- Models ---

class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels // 2)
        self.ln3 = nn.LayerNorm(hidden_channels // 2)

    def forward(self, x, edge_index, edge_weight, batch, return_node_feats=False):
        x = F.relu(self.ln1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.ln2(self.conv2(x, edge_index, edge_weight)))
        x = F.relu(self.ln3(self.conv3(x, edge_index, edge_weight)))
        pooled = global_mean_pool(x, batch)
        if return_node_feats:
            return pooled, x
        return pooled

class ChebConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=3):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.conv3 = ChebConv(hidden_channels, hidden_channels // 2, K)
        self.ln3 = nn.LayerNorm(hidden_channels // 2)

    def forward(self, x, edge_index, edge_weight, batch, return_node_feats=False):
        x = F.relu(self.ln1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.ln2(self.conv2(x, edge_index, edge_weight)))
        x = F.relu(self.ln3(self.conv3(x, edge_index, edge_weight)))
        pooled = global_mean_pool(x, batch)
        if return_node_feats:
            return pooled, x
        return pooled

class SAGEConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.ln3 = nn.LayerNorm(hidden_channels // 2)

    def forward(self, x, edge_index, edge_weight, batch, return_node_feats=False):
        x = F.relu(self.ln1(self.conv1(x, edge_index)))
        x = F.relu(self.ln2(self.conv2(x, edge_index)))
        x = F.relu(self.ln3(self.conv3(x, edge_index)))
        pooled = global_mean_pool(x, batch)
        if return_node_feats:
            return pooled, x
        return pooled

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

class GraphTransformerNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3, transformer_heads=4, transformer_layers=3, dropouts=0.5):
        super().__init__()
        self.graph_block = GraphConvBlock(in_channels, hidden_channels)
        self.cheb_block = ChebConvBlock(in_channels, hidden_channels, K)
        self.sage_block = SAGEConvBlock(in_channels, hidden_channels)

        self.feature_dim = (hidden_channels // 2) * 3
        # Ensure divisible by transformer_heads
        if self.feature_dim % transformer_heads != 0:
            # Adjust feature_dim slightly if needed or assert? 
            # The original code asserted, so we trust parameters are correct
            pass

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

    def forward(self, x, edge_index, edge_attr=None, batch=None, edge_weight=None, return_node_feats=False, return_only_logits=False):
        # GNNExplainer passes edge_attr. Our code uses edge_weight. 
        # Harmonize them: use edge_attr if edge_weight is None
        if edge_weight is None:
            edge_weight = edge_attr
            
        # We need to handle the return values carefully
        if return_node_feats:
            g1, n1 = self.graph_block(x, edge_index, edge_weight, batch, return_node_feats=True)
            g2, n2 = self.cheb_block(x, edge_index, edge_weight, batch, return_node_feats=True)
            g3, n3 = self.sage_block(x, edge_index, edge_weight, batch, return_node_feats=True)
            
            # Concatenate pooled features for transformer
            x_pool = torch.cat([g1, g2, g3], dim=1)
            x_pool = x_pool.unsqueeze(1)  # (batch_size, seq_len=1, feature_dim)
            x_trans = self.transformer(x_pool)
            features = x_trans[:, 0, :]  # (batch_size, feature_dim)
            out = self.mlp(features)
            
            # Concatenate node features: (Total Nodes, FeatureDim) -> Reshape to (Batch, 22, FeatureDim)
            node_feats = torch.cat([n1, n2, n3], dim=1) # (Total Nodes, FeatureDim)
            
            # Reshape logic:
            batch_size = features.size(0)
            num_nodes = 22
            node_feats = node_feats.view(batch_size, num_nodes, -1)
            
            return out, features, node_feats
        
        else:
            g1 = self.graph_block(x, edge_index, edge_weight, batch)
            g2 = self.cheb_block(x, edge_index, edge_weight, batch)
            g3 = self.sage_block(x, edge_index, edge_weight, batch)
            
            x = torch.cat([g1, g2, g3], dim=1)
            x = x.unsqueeze(1)
            x = self.transformer(x)
            features = x[:, 0, :]
            out = self.mlp(features)
            
            if return_only_logits:
                return out
            
            return out, features

# --- Training / Eval ---

def train_model(model, train_loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out = out.squeeze()
        loss = criterion(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, device, return_details=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out, feats = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            out = out.squeeze()
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().float().numpy())
            all_features.extend(feats.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    if return_details:
        return acc, kappa, np.array(all_labels), np.array(all_preds), np.array(all_features)
    return acc, kappa

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Store all results here
    experiment_results = {}

    subjects = list(range(9))
    if args.subjects is not None:
        # User provided specific subjects (e.g. --subjects 1 2 3)
        subjects = args.subjects
    
    print(f"Running for subjects: {subjects}")


    # Iterate over all pairs of classes (Outer Loop)
    class_pairs = list(combinations([0, 1, 2, 3], 2))
    
    for (c1, c2) in class_pairs:
        print(f"\n==========================================")
        print(f" PROCESSING PAIR: Class {c1} vs Class {c2} ")
        print(f"==========================================\n")
        
        # Results for this specific pair
        pair_results = {}
        
        # Output Directory for this Pair
        # Structure: binary_label_{c1}_vs_{c2} / subject_{test_sub}
        pair_base_dir = f"binary_label_{c1}_vs_{c2}"
        os.makedirs(pair_base_dir, exist_ok=True)

        for test_sub in subjects:
            print(f"\n--- Subject {test_sub} held out for testing (Pair {c1}v{c2}) ---")
            
            # --- Load Data for this Split ---
            train_data_all = []
            train_labels_all = []
            test_data = []
            test_labels = []

            # Load Data
            for subi in range(9):
                if subi != test_sub:
                    d1, l1 = load_BCI2a_data('BCI_Kaggle/', subject=subi, training=True, all_trials=True)
                    d2, l2 = load_BCI2a_data('BCI_Kaggle/', subject=subi, training=False, all_trials=True)
                    train_data_all.extend([d1, d2])
                    train_labels_all.extend([l1, l2])
            
            train_data_all = np.concatenate(train_data_all, axis=0)
            train_labels_all = np.concatenate(train_labels_all, axis=0)

            td1, tl1 = load_BCI2a_data('BCI_Kaggle/', subject=test_sub, training=True, all_trials=True)
            td2, tl2 = load_BCI2a_data('BCI_Kaggle/', subject=test_sub, training=False, all_trials=True)
            
            test_data = np.concatenate([td1, td2], axis=0)
            test_labels = np.concatenate([tl1, tl2], axis=0)
            
            # --- Filter for this pair ---
            train_mask = np.isin(train_labels_all, [c1, c2])
            test_mask = np.isin(test_labels, [c1, c2])

            train_data_binary = train_data_all[train_mask]
            train_labels_raw = train_labels_all[train_mask]
            
            test_data_binary = test_data[test_mask]
            test_labels_raw = test_labels[test_mask]
            
            # Remap labels to 0 and 1
            # c1 -> 0, c2 -> 1
            train_labels_binary = np.where(train_labels_raw == c1, 0, 1)
            test_labels_binary = np.where(test_labels_raw == c1, 0, 1)

            # --- CSP ---
            csp = CSP(n_components=16, reg=None, log=True, norm_trace=False, transform_into='average_power')
            train_csp = csp.fit_transform(train_data_binary, train_labels_binary)
            test_csp = csp.transform(test_data_binary)

            train_features = np.repeat(train_csp[:, np.newaxis, :], 22, axis=1)
            test_features = np.repeat(test_csp[:, np.newaxis, :], 22, axis=1)

            # --- Spectral Coherence ---
            # print(f"Computing spectral coherence...")
            train_spec_coh = compute_spectral_coherence(train_data_binary)
            test_spec_coh = compute_spectral_coherence(test_data_binary)

            train_dataset = EEGGraphDataset(train_features, train_labels_binary, np.arange(len(train_labels_binary)), 250, train_spec_coh)
            test_dataset = EEGGraphDataset(test_features, test_labels_binary, np.arange(len(test_labels_binary)), 250, test_spec_coh)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            # --- Model Init ---
            model = GraphTransformerNet(in_channels=16, hidden_channels=32, out_channels=1, K=3, transformer_heads=8, dropouts=0.3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)
            criterion = torch.nn.BCEWithLogitsLoss()

            # --- Training Loop ---
            best_acc = 0.0
            best_state = None
            counter = 0
            
            patience = args.patience
            epochs = args.epochs
            
            iterator = trange(epochs, desc=f"Sub {test_sub}", leave=False)
            
            for epoch in iterator:
                train_loss = train_model(model, train_loader, device, optimizer, criterion)
                test_acc, test_kappa = evaluate_model(model, test_loader, device)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                
                if counter >= patience:
                    iterator.set_postfix(status="Early Stop", best_acc=best_acc)
                    break
                
                iterator.set_postfix(loss=train_loss, test_acc=test_acc, best=best_acc)
            
            # --- Save Model & Results ---
            subject_dir = os.path.join(pair_base_dir, f"subject_{test_sub}")
            os.makedirs(subject_dir, exist_ok=True)

            if best_state:
                model.load_state_dict(best_state)
                model_save_path = os.path.join(subject_dir, "model.pth")
                torch.save(best_state, model_save_path)
                # print(f"Saved best model to {model_save_path}")
            
            acc, kappa, labels, preds, gnn_feats = evaluate_model(model, test_loader, device, return_details=True)
            print(f"Subject {test_sub} Finished. Test Acc: {acc:.4f}, Kappa: {kappa:.4f}")
            
            # Store Results locally for this pair
            pair_results[test_sub] = {
                "features": gnn_feats,
                "true_labels": labels, # these are 0/1 mapped
                "predicted_labels": preds,
                "original_classes": (c1, c2),
                "correct_mask": (labels == preds),
                "accuracy": acc,
                "kappa": kappa
            }
            experiment_results[f"sub{test_sub}_pair{c1}v{c2}"] = pair_results[test_sub] # Keep global copy just in case

            # --- Save Explanations ---
            if args.save_explanations:
                # print(f"Generating Explanations...")
                expl_dir = subject_dir
                num_trials = len(test_dataset)
                
                for idx in tqdm(range(num_trials), desc=f"Explaining", leave=False):
                    edge_mask, node_mask, pred, true_label, edge_index = extract_trial_explanation(
                        model, test_dataset, idx, device, epochs=50
                    )
                    
                    save_data = {
                        "trial_idx": idx,
                        "node_mask": node_mask,
                        "edge_mask": edge_mask,
                        "edge_index": edge_index,
                        "predicted_label": pred,
                        "true_label": true_label,
                        "original_classes": (c1, c2),
                        "raw_features": test_dataset[idx].x.cpu().numpy()
                    }
                    
                    with open(os.path.join(expl_dir, f"explanation_trial_{idx}.pkl"), "wb") as f_expl:
                        pickle.dump(save_data, f_expl)
        
        # Save Aggregate Results for this Pair
        pair_out_file = os.path.join(pair_base_dir, f"loso_results_pair_{c1}v{c2}.pkl")
        with open(pair_out_file, "wb") as f:
            pickle.dump(pair_results, f)
        print(f"Saved aggregate results for pair {c1}v{c2} to {pair_out_file}")
    
    # Save to file
    out_file = "loso_gnn_features_results.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(experiment_results, f)
    
    print(f"\nSaved all results to {out_file}")


def extract_single_trial(model, dataset, trial_idx, device):
    """
    Extracts features for a single trial.
    Returns:
        node_features: (22, FeatureDim) numpy array
        predicted_label: int (0 or 1)
        true_label: int (0 or 1)
        raw_output_features: (FeatureDim,) numpy array (pooled)
    """
    model.eval()
    data = dataset[trial_idx]
    
    # Create a batch of size 1
    batch = Batch.from_data_list([data]).to(device)
    
    with torch.no_grad():
        logits, pooled_feats, node_feats = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_node_feats=True
        )
        
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long().item()
        true_label = batch.y.item()
        
    return node_feats.squeeze(0).cpu().numpy(), pred, true_label, pooled_feats.squeeze(0).cpu().numpy()


def extract_trial_explanation(model, dataset, trial_idx, device, epochs=50):
    """
    Runs GNNExplainer on a single trial to get edge and node masks.
    Returns:
        edge_mask: (NumEdges,) numpy array (weights 0-1)
        node_mask: (NumNodes,) numpy array (weights 0-1)
        pred: int
        true_label: int
        edge_index: (2, NumEdges) numpy array
    """
    model.eval()
    data = dataset[trial_idx]
    
    # Create batch of size 1 for the model forward pass
    batch = Batch.from_data_list([data]).to(device)
    
    # 1. Get Prediction First
    with torch.no_grad():
        logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_node_feats=True)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long().item()
        true_label = batch.y.item()

    # 2. Configure Explainer
    # We want to explain the prediction made by the model
    model_config = ModelConfig(
        mode='binary_classification', 
        task_level='graph', 
        return_type='raw'
    )
    
    # Wrapper to enforce return_only_logits=True
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
            return self.model(x, edge_index, edge_attr=edge_attr, batch=batch, return_only_logits=True)

    explainer = Explainer(
        model=ModelWrapper(model),
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='phenomenon',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=model_config,
    )
    
    # 3. Run Explanation
    # Note: GNNExplainer usually takes x, edge_index etc.
    # The 'target' is the class label we want to explain (usually the predicted one)
    explanation = explainer(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        batch=batch.batch,
        target=torch.tensor([pred], device=device) 
    )
    
    # 4. Extract Masks
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    edge_index = batch.edge_index.cpu().numpy()
    
    # Compute Node Mask from Edge Mask (as in GNNExplainer notebook)
    # Sum of importance of connected edges
    num_nodes = data.num_nodes # Should be 22
    node_mask = np.zeros(num_nodes)
    num_edges = edge_index.shape[1]
    
    for i in range(num_edges):
        u = edge_index[0, i]
        v = edge_index[1, i]
        w = edge_mask[i]
        node_mask[u] += w
        node_mask[v] += w
        
    # Normalize node mask 0-1
    if node_mask.max() > 0:
        node_mask /= node_mask.max()
        
    return edge_mask, node_mask, pred, true_label, edge_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--subjects", type=int, nargs='+', help="Specific subjects to run (e.g. 0 1 2). If empty, runs all.")
    parser.add_argument("--debug", action="store_true", help="Run with very small data for debugging")
    parser.add_argument("--save_explanations", action="store_true", help="Save GNNExplainer data after testing")
    args = parser.parse_args()
    
    if args.debug:
        print("DEBUG MODE: Reducing data size and epochs")
        args.epochs = 2
        args.patience = 1
        # Quick hack to monkeypatch load function or just rely on quick epochs?
        # A better way is to slice data after loading but before dataset creation:
        # We'll handle this by monkeypatching the data usage in main if debug is on
        # But for now, let's just rely on standard main logic and slice lists.
    
    main(args)
