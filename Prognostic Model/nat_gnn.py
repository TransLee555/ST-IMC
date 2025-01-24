import os
import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.model_selection import KFold


# Graph Attention Networks (GAT) Class
class GAT(nn.Module):
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, hidden3_channels, out_channels, num_heads=4, dropout=0.1):
        super(GAT, self).__init__()

        # GATConv layers with attention mechanism
        self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden1_channels, heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden1_channels * num_heads)

        self.conv2 = GATConv(in_channels=hidden1_channels * num_heads, out_channels=hidden2_channels, heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden2_channels * num_heads)

        self.conv3 = GATConv(in_channels=hidden2_channels * num_heads, out_channels=hidden3_channels, heads=6, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden3_channels * 6)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden3_channels * 6, 32)
        self.fc2 = nn.Linear(32, out_channels)

    def weighted_global_pool(self, x, batch, edge_index, attention_weights):
        attention_weights_mean = attention_weights[1].mean(dim=1)
        node_attention = torch.zeros(x.size(0), device=x.device)

        node_attention.scatter_add_(0, attention_weights[0][0], attention_weights_mean)
        node_attention.scatter_add_(0, attention_weights[0][1], attention_weights_mean)

        x_weighted = x * node_attention.view(-1, 1)
        x_pooled = global_mean_pool(x_weighted, batch)
        return x_pooled

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First GAT layer
        x1, attn_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x1 = self.norm1(x1)
        x1 = torch.relu(x1)

        # Second GAT layer
        x2, attn_weights2 = self.conv2(x1, edge_index, return_attention_weights=True)
        x2 = self.norm2(x2)
        x2 = torch.relu(x2)

        # Third GAT layer
        x3, attn_weights3 = self.conv3(x2, edge_index, return_attention_weights=True)
        x3 = torch.relu(x3)

        x_res = x + x3
        # x_pooled = self.weighted_global_pool(x_res, data.batch, edge_index, attn_weights3)

        # Average pooling
        x_pooled = global_mean_pool(x_res, data.batch)

        # Fully connected layers
        x_f1 = torch.relu(self.fc1(x_pooled))
        x_f2 = self.fc2(x_f1)

        return x_f2, attn_weights1, attn_weights2, attn_weights3


# GATv2 (Graph Attention Networks v2) Class
class GATV2(nn.Module):
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, hidden3_channels, out_channels, num_heads=4, dropout=0.1):
        super(GATV2, self).__init__()

        # GATv2Conv layers with attention mechanism
        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=hidden1_channels, heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden1_channels * num_heads)

        self.conv2 = GATv2Conv(in_channels=hidden1_channels * num_heads, out_channels=hidden2_channels, heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden2_channels * num_heads)

        self.conv3 = GATv2Conv(in_channels=hidden2_channels * num_heads, out_channels=hidden3_channels, heads=6, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden3_channels * 6)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden3_channels * 6, 32)
        self.fc2 = nn.Linear(32, out_channels)

    def weighted_global_pool(self, x, batch, edge_index, attention_weights):
        attention_weights_mean = attention_weights[1].mean(dim=1)
        node_attention = torch.zeros(x.size(0), device=x.device)

        node_attention.scatter_add_(0, attention_weights[0][0], attention_weights_mean)
        node_attention.scatter_add_(0, attention_weights[0][1], attention_weights_mean)

        x_weighted = x * node_attention.view(-1, 1)
        x_pooled = global_mean_pool(x_weighted, batch)
        return x_pooled

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First GATv2 layer
        x1, attn_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x1 = self.norm1(x1)
        x1 = torch.relu(x1)

        # Second GATv2 layer
        x2, attn_weights2 = self.conv2(x1, edge_index, return_attention_weights=True)
        x2 = self.norm2(x2)
        x2 = torch.relu(x2)

        # Third GATv2 layer
        x3, attn_weights3 = self.conv3(x2, edge_index, return_attention_weights=True)
        x3 = torch.relu(x3)

        x_res = x + x3
        # x_pooled = self.weighted_global_pool(x_res, data.batch, edge_index, attn_weights3)

        # Average pooling
        x_pooled = global_mean_pool(x_res, data.batch)

        # Fully connected layers
        x_f1 = torch.relu(self.fc1(x_pooled))
        x_f2 = self.fc2(x_f1)

        return x_f2, attn_weights1, attn_weights2, attn_weights3


# Graph Transformer Class
class GT(nn.Module):
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, hidden3_channels, out_channels, num_heads=4, dropout=0.1):
        super(GT, self).__init__()

        # TransformerConv layers
        self.conv1 = TransformerConv(in_channels=in_channels, out_channels=hidden1_channels, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm1 = nn.LayerNorm(hidden1_channels * num_heads)

        self.conv2 = TransformerConv(in_channels=hidden1_channels * num_heads, out_channels=hidden2_channels, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm2 = nn.LayerNorm(hidden2_channels * num_heads)

        self.conv3 = TransformerConv(in_channels=hidden2_channels * num_heads, out_channels=hidden3_channels, heads=6, dropout=dropout, edge_dim=1)
        self.norm3 = nn.LayerNorm(hidden3_channels * 6)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden3_channels * 6, 32)
        self.fc2 = nn.Linear(32, out_channels)

    def weighted_global_pool(self, x, batch, edge_index, attention_weights):
        attention_weights_mean = attention_weights.mean(dim=1)
        node_attention = torch.zeros(x.size(1), device=x.device)

        node_attention.scatter_add_(0, edge_index[0], attention_weights_mean)
        node_attention.scatter_add_(0, edge_index[1], attention_weights_mean)

        node_attention = torch.softmax(node_attention, dim=0)

        x_weighted = x * node_attention.view(-1, 1)
        x_pooled = global_mean_pool(x_weighted, batch)

        return x_pooled

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index += 1

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            edge_attr = edge_attr.unsqueeze(-1)

        # First TransformerConv layer
        x1, (edge_index1, attn_weights1) = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x1 = self.norm1(x1)
        x1 = torch.relu(x1)

        # Second TransformerConv layer
        x2, (edge_index2, attn_weights2) = self.conv2(x1, edge_index1, edge_attr=edge_attr, return_attention_weights=True)
        x2 = self.norm2(x2)
        x2 = torch.relu(x2)

        # Third TransformerConv layer
        x3, (edge_index3, attn_weights3) = self.conv3(x2, edge_index2, edge_attr=edge_attr, return_attention_weights=True)
        x3 = torch.relu(x3)

        x_res = x + x3

        # Average pooling
        x_pooled = global_mean_pool(x_res, data.batch)

        # Fully connected layers
        x_f1 = torch.relu(self.fc1(x_pooled))
        x_f2 = self.fc2(x_f1)

        return x_f2, attn_weights1, attn_weights2, attn_weights3


# Model Selection and Cross Validation Setup
def get_model(model_type, in_channels, hidden1_channels, hidden2_channels, hidden3_channels, out_channels):
    if model_type == 'GAT':
        return GAT(in_channels, hidden1_channels, hidden2_channels, hidden3_channels, out_channels)
    elif model_type == 'GATV2':
        return GATV2(in_channels, hidden1_channels, hidden2_channels, hidden3_channels, out_channels)
    elif model_type == 'GT':
        return GT(in_channels, hidden1_channels, hidden2_channels, hidden3_channels, out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Set random seed for reproducibility
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Cross-validation training function
def cross_validate(model_type, data, k=5, seed=94):
    set_random_seed(seed)
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"Training fold {fold + 1}/{k}")
        train_data = data[train_idx]
        val_data = data[val_idx]
        model = get_model(model_type, in_channels=138, hidden1_channels=64, hidden2_channels=128, hidden3_channels=256, out_channels=1)

        # Training loop setup
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        # Train the model on the current fold
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out, _ = model(train_data)
            out = torch.sigmoid(out)
            loss = loss_fn(out, train_data.y)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        # Validation
        model.eval()
        with torch.no_grad():
            out, _ = model(val_data)
            val_loss = loss_fn(out, val_data.y)
            print(f"Validation Loss: {val_loss.item()}")


# Example usage:
# Load your data here and use cross-validation for model evaluation
# data = YourDataClassHere
# cross_validate('GAT', data)





