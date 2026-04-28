import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, ChebConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import ttest_ind
import mat73
import pandas as pd
import time
import bct

# Random Noise Injection for data augmentation
# Diffusion model for data augmentation
def forward_diffusion(x0, T, beta):
    """
    Forward diffusion process: Adds Gaussian noise to the data.
    Args:
        x0 (np.ndarray): Original data.
        T (int): Number of diffusion steps.
        beta (float): Noise scale parameter.
    Returns:
        xt (np.ndarray): Noisy data after T steps.
    """
    xt = x0
    for t in range(T):
        noise = np.random.normal(0, np.sqrt(beta), x0.shape)
        xt = np.sqrt(1 - beta) * xt + noise
    return xt

def backward_diffusion(xt, T, beta):
    """
    Backward diffusion process: Reconstructs data from noise.
    Args:
        xt (np.ndarray): Noisy data.
        T (int): Number of diffusion steps.
        beta (float): Noise scale parameter.
    Returns:
        x0 (np.ndarray): Reconstructed data.
    """
    x0 = xt
    for t in range(T-1, -1, -1):
        noise = np.random.normal(0, np.sqrt(beta), xt.shape)
        x0 = (x0 - noise) / np.sqrt(1 - beta)
    return x0

# Augment data using diffusion model
def augment_data_with_diffusion(data, T=100, noise_factor=0.01, num_augmentations=1000):
    avg_matrix = np.mean(data, axis=2)  # Compute average connectivity matrix

    augmented_data_list = []
    for _ in range(num_augmentations):
        noisy_data = forward_diffusion(avg_matrix, T, noise_factor)
        reconstructed_data = backward_diffusion(noisy_data, T, noise_factor)
        reconstructed_data_tril = np.tril(reconstructed_data) + np.tril(reconstructed_data, -1).T
        augmented_data_list.append(reconstructed_data_tril[..., np.newaxis])
    
    augmented_data = np.concatenate(augmented_data_list, axis=2)  # Shape: [122, 122, num_augmentations]
    return augmented_data

# Network measure functions
def compute_network_measures(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    if not np.allclose(adj_matrix, adj_matrix.T):
        raise ValueError("Adjacency matrix must be symmetric for undirected graphs.")
    node_strength = np.sum(adj_matrix, axis=1)
    clustering_coeff = bct.clustering_coef_wu(adj_matrix)
    eigenvector_centrality = bct.eigenvector_centrality_und(adj_matrix)
    return node_strength, clustering_coeff, eigenvector_centrality

# Generate edge index and weights
def generate_edge_index_and_weights(adj_matrix):
    edge_index = []
    edge_weights = []    
    rows, cols = np.triu_indices_from(adj_matrix, k=1)
    for i, j in zip(rows, cols):
        if adj_matrix[i, j] > 0:  # Consider only non-zero edges
            edge_index.append([i, j])
            edge_weights.append(adj_matrix[i, j])
    
    return np.array(edge_index).T, np.array(edge_weights)

# Define GCN+ChebConv model
class GraphConvBlock(nn.Module):
    """Block with two GraphConv layers in series and global mean pooling."""
    def __init__(self, in_channels, hidden_channels):
        super(GraphConvBlock, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # Batch Norm after first layer
        self.conv2 = GraphConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = global_mean_pool(x, batch)  # Graph-level feature
        return x

class ChebConvBlock(nn.Module):
    """Block with two ChebConv layers in series and global mean pooling."""
    def __init__(self, in_channels, hidden_channels, K=2):
        super(ChebConvBlock, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # Batch Norm after first layer
        self.conv2 = ChebConv(hidden_channels, hidden_channels // 2, K)  # Integer division

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = global_mean_pool(x, batch)  # Graph-level feature
        return x

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=20):
        super(GCN, self).__init__()
        self.graph_block = GraphConvBlock(in_channels, hidden_channels)
        self.cheb_block = ChebConvBlock(in_channels, hidden_channels, K)
        self.fc = nn.Linear(hidden_channels + hidden_channels, out_channels)  # Updated input size

    def forward(self, x, edge_index, edge_weight, batch):
        graph_features = self.graph_block(x, edge_index, edge_weight, batch)
        cheb_features = self.cheb_block(x, edge_index, edge_weight, batch)
        x = torch.cat([graph_features, cheb_features], dim=1)  # Concatenation
        x = self.fc(x)  # Classification layer
        return x

# Load data
def load_data(healthy_data_path, mci_data_path):
    healthy_data = mat73.loadmat(healthy_data_path)['h']
    mci_data = mat73.loadmat(mci_data_path)['m']
    return healthy_data, mci_data

# Save results to file
def save_results_to_file(results, confusion_matrices, output_path):
   
    # Add classification metrics columns
    df = pd.DataFrame(results, columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity'])
    
    # Add confusion matrix columns
    cm_columns = ['TN', 'FP', 'FN', 'TP']
    cm_flattened = []
    for cm in confusion_matrices:
        cm_flattened.extend(cm.flatten())  # Flatten the confusion matrix
        
    # Add confusion matrix to the results dataframe
    cm_df = pd.DataFrame([cm_flattened], columns=cm_columns)
    
    result_df = pd.concat([df, cm_df], axis=1)
    
    result_df.to_csv(output_path, index=False)

# Perform t-test between original and synthetic data
def perform_statistical_test(original_data, augmented_data):
    t_stat, p_value = ttest_ind(original_data.flatten(), augmented_data.flatten())
    return p_value

# Train and evaluate GCN model
def train_gcn_model(gcn_model, train_loader, optimizer, loss_fn, logfile):
    gcn_model.train()
    epoch_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        
        # Forward pass: Get model output
        output = gcn_model(data.x, data.edge_index, data.edge_weight, data.batch)
        print(f"Model output shape: {output.shape}")
                
        # Target labels should be [batch_size] for binary classification
        target = data.y.view(-1)  # Flatten to [batch_size], since each label corresponds to one subject
        print(f"Target shape: {target.shape}")
        
        # Calculate loss
        loss = loss_fn(output, target)
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

def evaluate_gcn_model(gcn_model, test_loader, logfile):
    gcn_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            # Pass 'batch' to the model
            output = gcn_model(data.x, data.edge_index, data.edge_weight, data.batch)
            _, predicted = torch.max(output, dim=1)
            all_preds.append(predicted)
            all_labels.append(data.y)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
    precision = precision_score(all_labels.numpy(), all_preds.numpy(), average='binary')
    recall = recall_score(all_labels.numpy(), all_preds.numpy(), average='binary')
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='binary')
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    
    logfile.write(f"{time.ctime()}: Evaluation Metrics:\n")
    logfile.write(f"Accuracy: {accuracy}\n")
    logfile.write(f"Precision: {precision}\n")
    logfile.write(f"Recall: {recall}\n")
    logfile.write(f"F1 Score: {f1}\n")
    logfile.write(f"Specificity: {specificity}\n")
    
    return accuracy, precision, recall, f1, specificity, cm

# Main Code
data_dir = "C:/Users/B00896414/OneDrive - Ulster University/CPM_AD/Kaniska/AvgTrial/Correlation_matrix/Alpha/SecondClick/"
healthy_data_path = os.path.join(data_dir, "healthy_data.mat")
mci_data_path = os.path.join(data_dir, "mci_data.mat")

logfile = open("training_log.txt", "w")
logfile.write(f"{time.ctime()}: Starting training and evaluation process.\n")

healthy_data, mci_data = load_data(healthy_data_path, mci_data_path)

results = []
confusion_matrices = []


for i in range(1):
    for j in range(1):
        train_data_temp = np.concatenate([healthy_data[:, :, :i], healthy_data[:, :, i+1:12]], axis=2)
        test_data_temp = np.concatenate([healthy_data[:, :, [i]], healthy_data[:, :, 12:], mci_data], axis=2)
        test_labels = np.array([1]*5 + [-1]*5)

        augmented_data = augment_data_with_diffusion(train_data_temp, noise_factor=0.01, num_augmentations=1000)
        p_value = perform_statistical_test(train_data_temp, augmented_data)
        while p_value <= 0.05:
            print("Synthetic data are not similar to original. Generating new data...")
            augmented_data = augment_data_with_diffusion(train_data_temp, noise_factor=0.01, num_augmentations=1000)
            p_value = perform_statistical_test(train_data_temp, augmented_data)

        combined_data = np.concatenate([train_data_temp, augmented_data], axis=2)
        print("Extracting features...")
        train_data_list = []
        for k in range(combined_data.shape[2]):  # Iterate over subjects
            subject_data = combined_data[:, :, k]
            node_strength, clustering_coeff, eigenvector_centrality = compute_network_measures(subject_data)
            subject_features = np.stack([node_strength, clustering_coeff, eigenvector_centrality], axis=1)
            edge_idx, edge_wts = generate_edge_index_and_weights(subject_data)
        
            # Create a Data object for each subject
            data = Data(
                x=torch.tensor(subject_features, dtype=torch.float),
                edge_index=torch.tensor(edge_idx, dtype=torch.long),
                edge_weight=torch.tensor(edge_wts, dtype=torch.float),
                y=torch.tensor([1], dtype=torch.long)  # Replace with actual labels if available
            )
            train_data_list.append(data)
        
        # Prepare DataLoader for training
        train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
        
        # Prepare testing data
        test_data_list = []
        for k in range(test_data_temp.shape[2]):  # Iterate over test subjects
            subject_data = test_data_temp[:, :, k]
            node_strength, clustering_coeff, eigenvector_centrality = compute_network_measures(subject_data)
            subject_features = np.stack([node_strength, clustering_coeff, eigenvector_centrality], axis=1)  # Shape: [122, 3]
            edge_idx, edge_wts = generate_edge_index_and_weights(subject_data)
            
            # Get the correct label for this subject (healthy or MCI)
            label = test_labels[k]
        
            # Create a Data object for each test subject
            data = Data(
                x=torch.tensor(subject_features, dtype=torch.float),
                edge_index=torch.tensor(edge_idx, dtype=torch.long),
                edge_weight=torch.tensor(edge_wts, dtype=torch.float),
                y=torch.tensor([label], dtype=torch.long)  # Replace with actual labels if available
            )
            test_data_list.append(data)
        
        # Prepare DataLoader for testing
        test_loader = DataLoader(test_data_list, batch_size=10, shuffle=False)
        
        print("Feature extraction completed.")

        gcn_model = GCN(in_channels=3, hidden_channels=122, out_channels=2)
        optimizer = optim.Adam(gcn_model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        print("Training GCN model")
        best_loss = float('inf')
        best_model_weights = None
        # Training loop
        for epoch in range(1):
            epoch_loss = train_gcn_model(
                gcn_model, 
                train_loader, 
                optimizer, 
                loss_fn, 
                logfile
            )
            logfile.write(f"{time.ctime()}: Epoch {epoch} Loss: {epoch_loss}\n")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = gcn_model.state_dict()
                logfile.write(f"\n\n{time.ctime()}: Epoch {epoch}, New best loss: {epoch_loss}\n\n")
        
        gcn_model.load_state_dict(best_model_weights)
        
        print("Testing GCN model")
        accuracy, precision, recall, f1, specificity, cm = evaluate_gcn_model(
            gcn_model, 
            test_loader, 
            logfile
        )
        results.append([accuracy, precision, recall, f1, specificity])
        confusion_matrices.append(cm)

save_results_to_file(results, confusion_matrices, "evaluation_results.csv")
logfile.close()
