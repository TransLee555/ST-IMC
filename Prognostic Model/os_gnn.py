from lifelines import CoxPHFitter
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch,DataListLoader,DataLoader
import networkx as nx
import json
import os
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
import joblib
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from scipy.stats import chi2
from lifelines.plotting import add_at_risk_counts
from numpy import interp

class GAT(nn.Module):
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, hidden3_channels, num_heads=4, dropout=0.1):
        super(NodeAttentionModel, self).__init__()

        self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden1_channels, heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden1_channels * num_heads)

        self.conv2 = GATConv(in_channels=hidden1_channels * num_heads, out_channels=hidden2_channels, heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden2_channels * num_heads)

        self.conv3 = GATConv(in_channels=hidden2_channels * num_heads, out_channels=hidden3_channels, heads=6, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden3_channels * 6)

        self.dropout = nn.Dropout(dropout)  # Dropout layer after pooling

    def weighted_global_pool(self, x, batch, edge_index, attention_weights):
        attention_weights_mean = attention_weights[1].mean(dim=1)
        node_attention = torch.zeros(x.size(0), device=x.device)

        node_attention.scatter_add_(0, attention_weights[0][0], attention_weights_mean)
        node_attention.scatter_add_(0, attention_weights[0][1], attention_weights_mean)

        x_weighted = x * node_attention.view(-1, 1)
        x_pooled = global_mean_pool(x_weighted, batch)
        return x_pooled

    def compute_elastic_net_regularization(self, x_pooled, lambda_l1=0.8, lambda_l2=0.8):
        batch_size = x_pooled.size(0)
        l1_loss = lambda_l1 * torch.norm(x_pooled, p=1)
        l2_loss = lambda_l2 * torch.norm(x_pooled, p=2) ** 2
        return (l1_loss + l2_loss) / batch_size

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1, attn_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x1 = self.norm1(x1)
        x1 = torch.relu(x1)

        x2, attn_weights2 = self.conv2(x1, edge_index, return_attention_weights=True)
        x2 = self.norm2(x2)
        x2 = torch.relu(x2)

        x3, attn_weights3 = self.conv3(x2, edge_index, return_attention_weights=True)
        x3 = torch.relu(x3)

        x_res = x + x3
        # x_pooled = self.weighted_global_pool(x_res, data.batch, edge_index, attn_weights3)
        x_pooled = global_mean_pool(x_res, data.batch)

        x_pooled = self.dropout(x_pooled)

        return x_pooled

class GATv2(nn.Module):
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, hidden3_channels, num_heads=4, dropout=0.1):
        super(NodeAttentionModel, self).__init__()

        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=hidden1_channels, heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden1_channels * num_heads)

        self.conv2 = GATv2Conv(in_channels=hidden1_channels * num_heads, out_channels=hidden2_channels, heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden2_channels * num_heads)
        self.conv3 = GATv2Conv(in_channels=hidden2_channels * num_heads, out_channels=hidden3_channels, heads=6, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden3_channels * 6)

        self.dropout = nn.Dropout(dropout)  # Dropout layer after pooling

    def weighted_global_pool(self, x, batch, edge_index, attention_weights):
        attention_weights_mean = attention_weights[1].mean(dim=1)
        node_attention = torch.zeros(x.size(0), device=x.device)

        node_attention.scatter_add_(0, attention_weights[0][0], attention_weights_mean)
        node_attention.scatter_add_(0, attention_weights[0][1], attention_weights_mean)

        x_weighted = x * node_attention.view(-1, 1)
        x_pooled = global_mean_pool(x_weighted, batch)
        return x_pooled

    def compute_elastic_net_regularization(self, x_pooled, lambda_l1=0.8, lambda_l2=0.8):
        batch_size = x_pooled.size(0)
        l1_loss = lambda_l1 * torch.norm(x_pooled, p=1)
        l2_loss = lambda_l2 * torch.norm(x_pooled, p=2) ** 2
        return (l1_loss + l2_loss) / batch_size

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1, attn_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x1 = self.norm1(x1)
        x1 = torch.relu(x1)

        x2, attn_weights2 = self.conv2(x1, edge_index, return_attention_weights=True)
        x2 = self.norm2(x2)
        x2 = torch.relu(x2)

        x3, attn_weights3 = self.conv3(x2, edge_index, return_attention_weights=True)
        x3 = torch.relu(x3)

        x_res = x + x3
        x_pooled = self.weighted_global_pool(x_res, data.batch, edge_index, attn_weights3)
        # x_pooled = global_mean_pool(x_res, data.batch)

        x_pooled = self.dropout(x_pooled)

        return x_pooled

class GT(nn.Module):
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, hidden3_channels, num_heads=4, dropout=0.1):
        super(GT, self).__init__()

        self.conv1 = TransformerConv(in_channels=in_channels, out_channels=hidden1_channels, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm1 = nn.LayerNorm(hidden1_channels * num_heads)

        self.conv2 = TransformerConv(in_channels=hidden1_channels * num_heads, out_channels=hidden2_channels, heads=num_heads, dropout=dropout, edge_dim=1)
        self.norm2 = nn.LayerNorm(hidden2_channels * num_heads)

        self.conv3 = TransformerConv(in_channels=hidden2_channels * num_heads, out_channels=hidden3_channels, heads=6, dropout=dropout, edge_dim=1)
        self.norm3 = nn.LayerNorm(hidden3_channels * 6)

        # self.fc1 = nn.Linear(hidden3_channels*6, 15)

        self.dropout = nn.Dropout(dropout)  # Dropout layer after pooling

    def weighted_global_pool(self, x, batch, edge_index, attention_weights):
        attention_weights_mean = attention_weights.mean(dim=1)
        node_attention = torch.zeros(x.size(1), device=x.device)
        node_attention.scatter_add_(0, edge_index[0], attention_weights_mean)
        node_attention.scatter_add_(0, edge_index[1], attention_weights_mean)
        x_weighted = x * node_attention.view(-1, 1)
        x_pooled = global_mean_pool(x_weighted, batch)
        return x_pooled, attention_weights_mean, node_attention

    def compute_elastic_net_regularization(self, x_pooled, lambda_l1=0.8, lambda_l2=0.8):
        batch_size = x_pooled.size(0)  # 获取 batch 大小
        l1_loss = lambda_l1 * torch.norm(x_pooled, p=1)
        l2_loss = lambda_l2 * torch.norm(x_pooled, p=2) ** 2
        return (l1_loss + l2_loss) / batch_size

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

        x_pooled = self.dropout(x_pooled)
        # x_f1 = torch.relu(self.fc1(x_pooled))

        return x_pooled, attn_weights1, attn_weights2, attn_weights3, attention_weights_mean, node_attention


def train(model, data, os_, time, optimizer, cox_fitter):
    all_risk_scores = []
    all_follow_up_times = []
    all_events = []

    x_pooled, attn_weights1, attn_weights2, attn_weights3, attention_weights_mean, node_attention = model(data)

    df = pd.DataFrame(x_pooled.squeeze(0).detach().cpu().numpy())
    df['follow_up_time'] = time.cpu().numpy()
    df['event_occurred'] = os_.cpu().numpy()

    cox_fitter.fit(df, duration_col='follow_up_time', event_col='event_occurred')
    cox_loss = -cox_fitter.log_likelihood_

    elastic_net_loss = model.compute_elastic_net_regularization(x_pooled)

    total_loss = cox_loss + elastic_net_loss * 0.8

    total_loss.backward()
    optimizer.step()

    risk_scores = cox_fitter.predict_partial_hazard(df)
    all_risk_scores.extend(risk_scores)
    all_follow_up_times.extend(df['follow_up_time'])
    all_events.extend(df['event_occurred'])

    c_index = concordance_index(all_follow_up_times, -np.array(all_risk_scores), all_events)

    return total_loss.item(), c_index


def val(model, data, os_, time, cox_fitter):
    model.eval()
    all_risk_scores = []
    all_follow_up_times = []
    all_events = []

    x_pooled, attn_weights1, attn_weights2, attn_weights3, attention_weights_mean, node_attention = model(data)

    df = pd.DataFrame(x_pooled.squeeze(0).detach().cpu().numpy())
    df['follow_up_time'] = time.cpu().numpy()
    df['event_occurred'] = os_.cpu().numpy()

    risk_scores = cox_fitter.predict_partial_hazard(df)
    all_risk_scores.extend(risk_scores)
    all_follow_up_times.extend(df['follow_up_time'])
    all_events.extend(df['event_occurred'])

    c_index = concordance_index(all_follow_up_times, -np.array(all_risk_scores), all_events)

    auc = roc_auc_score(os_.cpu().numpy(), risk_scores)
    fpr, tpr, _ = roc_curve(os_.cpu().numpy(), risk_scores)

    return c_index, auc, fpr, tpr, risk_scores

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
    def __init__(self, root, os_dict, time_dict):
        self.root = root
        self.input_list = os.listdir(root)
        # random.shuffle(self.input_list)
        self.os_dict = os_dict
        self.time_dict = time_dict

        # Store labels and time for stratified sampling
        self._label_list = []
        self._time_list = []
        for file_name in self.input_list:
            key = file_name[:12]
            os_ = os_dict.get(key, -1)
            time = time_dict.get(key, -1)
            if os_ != -1 and time != -1:
                self._label_list.append(os_)
                self._time_list.append(time / 365)  # Normalized time
            else:
                raise ValueError(f"Label for file {file_name} not found.")

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        graph_path = os.path.join(self.root, self.input_list[idx])
        data = graph_to_data(graph_path)

        # Extract file name from path to get the corresponding label
        file_name = os.path.basename(graph_path)
        os_ = self.os_dict.get(file_name[:12], -1)
        time = self.time_dict.get(file_name[:12], -1)

        if os_ == -1 or time == -1:
            raise ValueError(f"Label for file {file_name} not found.")

        return data, torch.tensor(self._label_list[idx], dtype=torch.float), torch.tensor(self._time_list[idx],
                                                                                          dtype=torch.float)

    @property
    def labels(self):
        return self._label_list


def aggregate_roc_results(fprs, tprs, aucs,c_indexs):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
    mean_auc = np.mean(aucs)
    mean_c_index = np.mean(c_indexs)
    std_tpr = np.std([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
    std_auc = np.std(aucs)
    std_c_index = np.std(c_indexs)
    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, mean_c_index, std_c_index


def plot_roc_with_folds(mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, mean_c_index, std_c_index):
    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f},'
                                                     f'C_index = {mean_c_index:.2f} ± {std_c_index:.2f})'
             )
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve - 10-Fold Cross-Validation', fontsize=16)
    plt.legend(loc='lower right')
    plt.savefig(r"../Prognostic Model/roc.png", dpi=300)
    plt.grid(True)
    plt.show()


def aggregate_km_results(times, events, risk_scores):
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    all_times_high = []
    all_events_high = []
    all_times_low = []
    all_events_low = []

    for fold_times, fold_events, fold_risk_scores in zip(times, events, risk_scores):
        median_risk = np.median(fold_risk_scores)

        high_risk_idx = fold_risk_scores > median_risk
        low_risk_idx = ~high_risk_idx

        all_times_high.extend(fold_times[high_risk_idx])
        all_events_high.extend(fold_events[high_risk_idx])

        all_times_low.extend(fold_times[low_risk_idx])
        all_events_low.extend(fold_events[low_risk_idx])

    kmf_high.fit(all_times_high, all_events_high, label='High Risk')

    kmf_low.fit(all_times_low, all_events_low, label='Low Risk')

    ax = kmf_high.plot()
    kmf_low.plot(ax=ax)

    results = logrank_test(all_times_high, all_times_low, event_observed_A=all_events_high, event_observed_B=all_events_low)
    p_value = results.p_value

    plt.text(0.05, 0.1, f'P-value = {p_value:.4f}', transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

    plt.title(f'Kaplan-Meier Survival Curves', fontsize=12)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Survival Probability', fontsize=10)
    plt.legend(loc='best')
    plt.savefig(r"../Prognostic Model/km.png", dpi=300)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    root = r"../Community detection/graph_with_group"
    metadata = pd.read_csv(r'../IDC_OS.csv')

    os_labels = metadata['vital_status'].values
    time_labels = metadata['os_time'].values
    barcodes = metadata['bcr_patient_barcode'].values

    le = LabelEncoder()
    encoded_os_labels = le.fit_transform(os_labels)

    os_dict = dict(zip(barcodes, encoded_os_labels))
    time_dict = dict(zip(barcodes, time_labels))

    dataset = CustomDataset(root, os_dict, time_dict)
    all_labels = np.array(dataset.labels)

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    num_epochs = 50
    best_model_path = r'../Prognostic Model/'

    # 用于存储所有折的结果
    fprs, tprs, aucs, c_indexs = [], [], [], []
    fold_times, fold_events, fold_risk_scores = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"Training fold {fold + 1}/{skf.get_n_splits()}")

        model = GT(in_channels=138, hidden1_channels=64, hidden2_channels=64, hidden3_channels=23)  ## GAT / GATv2
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        cox_fitter = CoxPHFitter()

        # 训练集和测试集的DataLoader
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=700, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=700, shuffle=False)

        best_auc, best_c_index = 0, 0
        best_fpr, best_tpr = None, None

        try:
            for epoch in range(num_epochs):
                model.train()
                for data, os_, time in train_loader:
                    optimizer.zero_grad()
                    total_loss, train_c_index = train(model, data, os_, time, optimizer, cox_fitter)

                with torch.no_grad():
                    for data, os_, time in test_loader:
                        c_index, auc, fpr, tpr, risk_score = val(model, data, os_, time, cox_fitter)

                        if c_index > best_c_index:
                            best_auc = auc
                            best_fpr, best_tpr = fpr, tpr
                            best_c_index = c_index
                            torch.save(model.state_dict(),
                                       os.path.join(best_model_path,
                                                    f'best_model_fold_{fold + 1}_{auc:.4f}_{c_index:.4f}.pth'))
                            # 保存CoxPHFitter模型
                            cox_fitter_save_path = os.path.join(best_model_path,
                                                                f'cox_fitter_fold_{fold + 1}_{auc:.4f}_{c_index:.4f}.pkl')
                            joblib.dump(cox_fitter, cox_fitter_save_path)

                    print(f"Fold {fold + 1}, Train Loss: {total_loss}, Train c_index: {train_c_index}, "
                          f"Test c_index: {c_index}, Test AUC: {auc}")

            fprs.append(best_fpr)
            tprs.append(best_tpr)
            aucs.append(best_auc)
            c_indexs.append(best_c_index)

            fold_times.append(time)
            fold_events.append(os_)
            fold_risk_scores.append(risk_score)

        except Exception as e:
            print(f"Error in fold {fold + 1}: {str(e)}")

    mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, mean_c_index, std_c_index = aggregate_roc_results(fprs, tprs, aucs,c_indexs)
    plot_roc_with_folds(mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, mean_c_index, std_c_index)

    aggregate_km_results(fold_times, fold_events, fold_risk_scores)






