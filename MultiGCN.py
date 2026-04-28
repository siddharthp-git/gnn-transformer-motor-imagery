import torch
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from torch_geometric.nn import ChebConv, GCNConv, GraphConv, GATConv, SAGEConv, global_mean_pool
from torch.nn import functional as F
from torch_geometric.data import Data
from scipy.sparse import coo_matrix


# 102--> 64 1st layer
# 64 ---> 32 2nd layer
# 100X102X102X23 input
# trail, channel, channel, frequency
# concate in a way taht will be like 102X102X2300 to make it 3d.
# (100,1) ---> label will be changed to (2300,1)
# First 23 will be same label for 1st trail 
# permute 102X102x2300 to 2300x102x102.
# extract only hand and feet


# Load adjacency matrices and labels
adjacency_matrices = scipy.io.loadmat('path_to_adjacency_matrices.mat')['name_of_variable']
labels = scipy.io.loadmat('path_to_labels.mat')['name_of_variable']

adjacency_matrices = reshape(adjacency_matrices, [], size(adjacency_matrices, 2), size(adjacency_matrices, 3))
label = label.reshape(-1, 1)

# Split data into training and testing sets
adj_train, adj_test, labels_train, labels_test = train_test_split(adjacency_matrices, labels, test_size=0.2, random_state=42)

class MultiGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(MultiGCN, self).__init__()
        self.chebconv = ChebConv(num_features, 16, K=2)
        self.gcnconv = GCNConv(num_features, 16)
        self.graphconv = GraphConv(num_features, 16)
        self.gatconv = GATConv(num_features, 16)
        self.sageconv = SAGEConv(num_features, 16)
        self.fc = torch.nn.Linear(80, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # If nodes have no features, use a tensor of ones as node features
        if x is None:
            x = torch.ones((data.num_nodes, 1)).to(data.edge_index.device)
        # Multiple GCN layers in parallel
        x1 = F.relu(self.chebconv(x, edge_index))
        x2 = F.relu(self.gcnconv(x, edge_index))
        x3 = F.relu(self.graphconv(x, edge_index))
        x4 = F.relu(self.gatconv(x, edge_index))
        x5 = F.relu(self.sageconv(x, edge_index))

        x1 = F.dropout(x1, training=self.training)
        x2 = F.dropout(x2, training=self.training)
        x3 = F.dropout(x3, training=self.training)
        x4 = F.dropout(x4, training=self.training)
        x5 = F.dropout(x5, training=self.training)

        # Combine the outputs
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # Apply a pooling layer
        x = global_mean_pool(x, batch=data.batch)

        # Classification layer
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# Initialize the model
model = MultiGCN(num_features=1, num_classes=2)

# Create a PyG Data object for each adjacency matrix in the training set
train_data_list = []
for i in range(adj_train.shape[0]):
    coo = coo_matrix(adj_train[i])
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
    data = Data(edge_index=edge_index, y=labels_train[i])
    train_data_list.append(data)

# Similarly, create a PyG Data object for each adjacency matrix in the testing set
test_data_list = []
for i in range(adj_test.shape[0]):
    coo = coo_matrix(adj_test[i])
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
    data = Data(edge_index=edge_index, y=labels_test[i])
    test_data_list.append(data)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(100):
    for data in train_data_list:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct = 0
for data in test_data_list:
    out = model(data)
    pred = out.argmax(dim=1)
    correct += int((pred == data.y).sum())
accuracy = correct / len(test_data_list)

print('Accuracy: {:.4f}'.format(accuracy))
