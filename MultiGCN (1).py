import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, ChebConv, SAGEConv, global_mean_pool
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
# 4d --->3d
# merge accrods trail and frequency
# 102--> 64 1st layer
# 64 ---> 32 2nd layer
# 100X102X102X23 input
# trail, channel, channel, frequency
# concate in a way taht will be like 102X102X2300 to make it 3d
# (100,1) ---> label will be changed to (2300,1)
# First 23 will be same label for 1st trail 
# permute 102X102x2300 to 2300x102x102
# extract only hand and feet


# Custom Dataset class for loading brain connectivity networks
class BrainConnectivityDataset(Dataset):
    def __init__(self, adjacency_matrices, labels, transform=None, pre_transform=None):
        self.adjacency_matrices = adjacency_matrices
        self.labels = labels
        super(BrainConnectivityDataset, self).__init__(None, transform, pre_transform)

    def len(self):
        return len(self.labels)

    def get(self, idx):
        edge_index = self.adjacency_matrix_to_edge_index(self.adjacency_matrices[idx])
        x = torch.ones((self.adjacency_matrices.shape[1], 1), dtype=torch.float)
        y = torch.tensor([self.labels[idx]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    @staticmethod
    def adjacency_matrix_to_edge_index(matrix):
        edge_index = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j]:  # If there is an edge
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Load the .mat files
# adjacency_matrices = loadmat('path_to_adjacency_matrices.mat')['variable_name']
# labels = loadmat('path_to_labels.mat')['variable_name'].squeeze()
adjacency_matrices = np.random.randint(0, 2300, (2300, 306, 306))

# Generate random labels with shape (1, 2300)
labels = np.random.randint(0, 2, (1, 2300))
# Split the dataset into training and testing sets (Thos os no)
train_indices, test_indices = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
train_data = adjacency_matrices[train_indices]
test_data = adjacency_matrices[test_indices]
train_labels = labels[train_indices]
test_labels = labels[test_indices]

# Create the datasets
train_dataset = BrainConnectivityDataset(train_data, train_labels)
test_dataset = BrainConnectivityDataset(test_data, test_labels)

# GCN Model with additional layers in each branch
# number of nodes = munmber of channels

class BrainConnectivityNet(nn.Module):
    def __init__(self, num_nodes, num_classes):
        super(BrainConnectivityNet, self).__init__()
        # GCN branch
        self.gcn1 = GCNConv(num_nodes, 32)
        self.gcn2 = GCNConv(32, 64)
        # visulasation of the branch/feature (What are the information this branch is extracting)
        # GAT branch
        self.gat1 = GATConv(num_nodes, 32)
        self.gat2 = GATConv(32, 64)
        # visulasation of the branch/feature (What are the information this branch is extracting)
        # Chebyshev branch
        self.cheb1 = ChebConv(num_nodes, 32, K=2)
        self.cheb2 = ChebConv(32, 64, K=2)
        # visulasation of the branch/feature (What are the information this branch is extracting)
        # GraphSAGE branch
        self.sage1 = SAGEConv(num_nodes, 32)
        self.sage2 = SAGEConv(32, 64)
        # visulasation of the branch/feature (What are the information this branch is extracting)
        # Pooling layer
        self.pool = global_mean_pool

        # Classifier
        self.classifier = nn.Linear(64 * 4, num_classes)  # 4 branches with 64 features each

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node features are all ones
        x = torch.ones((data.num_nodes, 1), dtype=torch.float)

        # GCN branch
        x_gcn = F.relu(self.gcn1(x, edge_index))
        x_gcn = F.dropout(x_gcn, training=self.training)
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))

        # GAT branch
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.dropout(x_gat, training=self.training)
        x_gat = F.relu(self.gat2(x_gat, edge_index))

        # Chebyshev branch
        x_cheb = F.relu(self.cheb1(x, edge_index))
        x_cheb = F.dropout(x_cheb, training=self.training)
        x_cheb = F.relu(self.cheb2(x_cheb, edge_index))

        # GraphSAGE branch
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.dropout(x_sage, training=self.training)
        x_sage = F.relu(self.sage2(x_sage, edge_index))

        # Concatenate the outputs from all branches
        x = torch.cat((x_gcn, x_gat, x_cheb, x_sage), dim=1)

        # Apply the pooling layer to get graph-level representation
        x = self.pool(x, batch)

        # Apply the classification layer
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the model
model = BrainConnectivityNet(num_nodes=64, num_classes=2)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(200):
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        # img = cv2.imread('resize.png',0)
        # img = np.reshape(img, (1,800,64,1)) # (n_images, x_shape, y_shape, n_channels)
        # img.shape 
        # intermediate_layer_model = model(inputs=model.input,
        #                                   outputs=model.get_layer(gcn1).output)
        # intermediate_output = intermediate_layer_model.predict(img)
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()

print(f'Accuracy of the network on the test set: {100 * correct / total}%')

# Save the model
torch.save(model.state_dict(), 'brain_connectivity_model.pth')
