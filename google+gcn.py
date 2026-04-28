# class Block1(nn.Module):
#      def __init__(self, inplace):
#           super().__init__()
#           self.conv1=nn.GCNConv(num_nodes, 64)
#           self.conv2=nn.GATConv(num_nodes, 64)
#           self.conv3=nn.ChebConv(num_nodes, 64)
#           self.re1=nn.ReLU()
#      def forward(self, x):
#       x1=self.conv1(x)
#       x2=self.conv2(x)
#       x3=self.conv3(x)
#       print(x1.shape,x2.shape,x3.shape)
#       x=torch.cat([x1,x3,x3],dim=1)
#       return x

# class Block2(nn.Module):
#      def __init__(self, inplace):
#           super().__init__()
#           self.conv1=nn.GCNConv(num_nodes, 32)
#           self.conv2=nn.GATConv(num_nodes, 32)
#           self.conv3=nn.ChebConv(num_nodes, 32)
#           self.re1=nn.ReLU()
#      def forward(self, x):
#       x1=self.conv1(x)
#       x2=self.conv2(x)
#       x3=self.conv3(x)
#       print(x1.shape,x2.shape,x3.shape)
#       x=torch.cat([x1,x3,x3],dim=1)
#       return x

# class ChronNet(nn.Module):
#   def __init__(self,channel):
#     super().__init__()
#     self.block1=Block1(102)
#     self.block2=Block2(64)
#     self.flatten=nn.Flatten()
#     self.fc1=nn.Linear(64,1)
#     self.relu=nn.ReLU()
#   def forward(self,x):
#     x=self.block1(x)
#     x=self.block2(x)
#     x=self.flatten(gru_out4)
#     x=self.fc1(x)
#     return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv

# Create synthetic data
num_samples = 100
num_trials = 100
num_channels = 102
num_datapoints = 23
input_data = torch.randn(num_samples, 1, num_trials, num_channels, num_datapoints)

# Create synthetic label dataset
labels = torch.randint(0, 2, (num_samples, 1))  # Binary labels, 0 or 1

# Model definition
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.block1 = nn.Sequential(
            GCNConv(23, 64),
            nn.ReLU(),
            GATConv(64, 64),
            nn.ReLU(),
            ChebConv(64, 64,K=2),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            GCNConv(64, 32),
            nn.ReLU(),
            GATConv(32, 32),
            nn.ReLU(),
            ChebConv(32, 32,K=2),
            nn.ReLU()
        )
        self.linear = nn.Linear(102*102*32, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, 102*102*32)  # Flatten the output
        x = self.linear(x)
        return x

# Instantiate the model
model = MyNet()

# Example usage
output = model(input_data)
print(output.shape)  # Output shape should be (100, 1) as expected
