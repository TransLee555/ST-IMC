import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool,global_add_pool
from torch_geometric.data import Data, Batch,DataListLoader,DataLoader
import networkx as nx
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


class GT(nn.Module):
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, hidden3_channels, out_channels, num_heads=4, dropout=0.1):
        super(GT, self).__init__()

        self.conv1 = TransformerConv(in_channels=in_channels, out_channels=hidden1_channels, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm1 = nn.LayerNorm(hidden1_channels * num_heads)

        self.conv2 = TransformerConv(in_channels=hidden1_channels * num_heads, out_channels=hidden2_channels, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm2 = nn.LayerNorm(hidden2_channels * num_heads)

        self.conv3 = TransformerConv(in_channels=hidden2_channels * num_heads, out_channels=hidden3_channels, heads=6, dropout=dropout, edge_dim=1)
        self.norm3 = nn.LayerNorm(hidden3_channels * 6)

        self.fc1 = nn.Linear(hidden3_channels*6, 32)
        self.fc2 = nn.Linear(32, out_channels)

    def weighted_global_pool(self, x, batch, edge_index, attention_weights):
        attention_weights_mean = attention_weights.mean(dim=1)

        node_attention = torch.zeros(x.size(1), device=x.device)

        node_attention.scatter_add_(0, edge_index[0], attention_weights_mean)
        node_attention.scatter_add_(0, edge_index[1], attention_weights_mean)

        x_weighted = x * node_attention.view(-1, 1)

        x_pooled = global_mean_pool(x_weighted, batch)

        return x_pooled, attention_weights_mean, node_attention

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index += 1

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            edge_attr = edge_attr.unsqueeze(-1)

        x1, (edge_index1, attn_weights1) = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x1 = self.norm1(x1)
        x1 = torch.relu(x1)

        x2, (edge_index2, attn_weights2) = self.conv2(x1, edge_index1, edge_attr=edge_attr, return_attention_weights=True)
        x2 = self.norm2(x2)
        x2 = torch.relu(x2)

        x3, (edge_index3, attn_weights3) = self.conv3(x2, edge_index2, edge_attr=edge_attr, return_attention_weights=True)
        x3 = torch.relu(x3)

        x_res = x + x3
        x_pooled, attention_weights_mean, node_attention = self.weighted_global_pool(x_res, data.batch, edge_index3, attn_weights3)

        x_f1 = torch.relu(self.fc1(x_pooled))
        x_f2 = self.fc2(x_f1)

        return x_f2, attn_weights1, attn_weights2, attn_weights3, attention_weights_mean, node_attention


def graph_to_data(graph):
    with open(graph) as f:
        graph = json.load(f)
        graph = nx.node_link_graph(graph)

        node_features = torch.tensor([graph.nodes[node]['data'] for node in graph.nodes()], dtype=torch.float)
        edge_index = []
        edge_attr = []
        for edge in graph.edges(data=True):
            edge_index.append([edge[0], edge[1]])
            edge_attr.append(edge[2]['weight'])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()-1
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, label_dict):
        self.root = root
        self.input_list = os.listdir(root)
        # random.shuffle(self.input_list)
        self.label_dict = label_dict

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        graph_path = os.path.join(self.root, self.input_list[idx])
        data = graph_to_data(graph_path)

        file_name = os.path.basename(graph_path)
        label = self.label_dict.get(int(file_name.split("_")[0]), -1)

        if label == -1:
            raise ValueError(f"Label for file {file_name} not found.")

        return data, torch.tensor(label, dtype=torch.float),file_name

def save_edge_attention_heatmap(data, edge_attention, node_attention, file_name):
    G = nx.Graph()

    edge_index = data.edge_index.cpu().numpy()
    edge_attention = edge_attention.cpu().numpy()
    node_attention = node_attention.cpu().numpy()

    data_attr = data.edge_attr.cpu().numpy()

    for i, (u, v) in enumerate(edge_index.T):
        G.add_edge(u, v, weight=edge_attention[i], cosine_similarity=data_attr[i])

    pos = nx.spring_layout(G, weight='cosine_similarity')
    edges = G.edges(data=True)
    weights = [edata['weight'] for _, _, edata in edges]

    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, node_color=[node_attention[n] for n in G.nodes()], cmap=plt.cm.coolwarm,
                           node_size=300, vmin=min(node_attention), vmax=max(node_attention))
    nx.draw_networkx_labels(G, pos)

    nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.coolwarm, edge_vmin=min(weights),
                           edge_vmax=max(weights), width=2)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=min(node_attention), vmax=max(node_attention)))
    sm.set_array([])
    plt.colorbar(sm)

    plt.savefig(r"../Community detection/graph_leiden_group/" + file_name + ".png", format='png', dpi=300)
    plt.close()


if __name__ == '__main__':
    root = r"../Community detection/graph_with_group"

    ## gene name
    gene = pd.read_csv(r'../Gene prediction/gene_val_correlation.csv')

    # Load the metadata CSV
    metadata = pd.read_excel(r'../Community detection/graph_with_group/Purdue_NC.xlsx',sheet_name=1)
    # Create a dictionary to map file names to labels
    label_dict = {row['patients']: row['pCR'] for _, row in metadata.iterrows()}

    # # Load the metadata CSV YALE
    # metadata = pd.read_csv(r'../Community detection/graph_with_group/Yale_trastuzumab_response_cohort_metadata_clean.csv')
    # # Create a dictionary to map file names to labels
    # label_dict = {row['Patient']: 1 if row['Responder'] == 'responder' else 0 for _, row in metadata.iterrows()}

    # Initialize the dataset
    dataset = CustomDataset(root,label_dict)

    in_channels = 138
    hidden1_channels = 64
    hidden2_channels = 64
    hidden3_channels = 23
    out_channels = 1

    model = Gt(in_channels, hidden1_channels, hidden2_channels,hidden3_channels, out_channels)

    checkpoint = torch.load(r"../Prognostic Model/best_gnn_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, label, file_name in val_loader:
            out, conv_attention_weights1, conv_attention_weights2, conv_attention_weights3, attention_weights_mean, node_attention =model(data)

            if not os.path.exists("../Community detection/graph_leiden_group/" + file_name[0].split(".")[0]):
                os.makedirs("../Community detection/graph_leiden_group/" + file_name[0].split(".")[0],exist_ok=True)

            for i in range(len(node_attention)):
                leiden_data = data.x[i]
                leiden_data = pd.DataFrame(leiden_data)
                leiden_data.index = gene
                leiden_data.columns = ["score"]
                leiden_data.to_csv(r"../Community detection/graph_leiden_group/" + file_name[0].split(".")[0]+f"/leiden{i}_{node_attention[i]}.csv")

