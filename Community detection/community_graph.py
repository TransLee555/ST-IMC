import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json
from scipy.sparse import csr_matrix

def convert_int64_to_int(d):
    """
    Convert np.int64 types in the dictionary to regular int to ensure JSON compatibility.
    """
    for node in d['nodes']:
        for key, value in node.items():
            if isinstance(value, np.int64):
                node[key] = int(value)
    for link in d['links']:
        for key, value in link.items():
            if isinstance(value, np.int64):
                link[key] = int(value)
    return d

def process_svs_file(file_path, root, gene_name, n_neighbors=50, resolution=0.5):
    """
    Process a single SVS file and perform the following steps:
    1. Read the gene expression data and compute nearest neighbors
    2. Compute the custom adjacency matrix considering spatial distance
    3. Perform Leiden clustering
    4. Compute community similarities
    5. Build and save the network graph
    """

    # Step 1: Read gene expression data from CSV file
    csv_data = pd.read_csv(file_path, index_col=0)
    csv_data.columns = gene_name.index  # Ensure column names align with gene_name index

    # Create AnnData object and compute the spatial coordinates
    adata = sc.AnnData(csv_data)
    spatial_coords = np.array([list(map(float, name.split('_'))) for name in adata.obs_names])

    # Step 2: Compute neighbors using Scanpy's built-in function
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    # Step 3: Use scikit-learn's NearestNeighbors to compute the KNN graph based on PCA space
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(adata.obsm['X_pca'])  # Assuming PCA has been precomputed
    _, indices = knn.kneighbors(adata.obsm['X_pca'])

    # Step 4: Construct the adjacency matrix considering spatial distance
    adj_matrix = np.zeros((adata.n_obs, adata.n_obs))
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                distance = np.linalg.norm(spatial_coords[i] - spatial_coords[neighbor])
                adj_matrix[i, neighbor] = 1 / (distance + 1e-10)  # Prevent division by zero

    # Convert to sparse matrix format for efficiency
    custom_adj_matrix = csr_matrix(adj_matrix)

    # Step 5: Add custom adjacency matrix to AnnData
    adata.uns['neighbors']['connectivities'] = custom_adj_matrix
    adata.uns['neighbors']['distances'] = custom_adj_matrix

    # Step 6: Perform Leiden clustering on the AnnData object
    sc.tl.leiden(adata, resolution=resolution)

    # Step 7: Group cells by their Leiden cluster labels
    community_data = {}
    for cluster_label in set(adata.obs['leiden']):
        community_data[cluster_label] = []

    for i, label in enumerate(adata.obs['leiden']):
        community_data[label].append(adata.obs_names[i])

    # Save community data as a DataFrame
    community_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in community_data.items()]))
    community_df.to_csv(os.path.join(root, "graph_leiden_group", file_path.split("/")[-1].replace(".csv", "_leiden.csv")), index=False)

    # Step 8: Normalize community feature vectors
    community_data_int = {}
    for community, indices in community_data.items():
        community_data_int[community] = [adata.obs_names.get_loc(idx) for idx in indices]

    # Initialize dictionaries for storing community sums and counts
    community_sum = {community: np.zeros(adata.X.shape[1]) for community in community_data.keys()}
    community_counts = {community: 0 for community in community_data_int.keys()}

    # Sum the features for each community
    for community, indices in community_data_int.items():
        for i in indices:
            community_sum[community] += adata.X[i, :].flatten()
            community_counts[community] += 1

    # Normalize community features by the number of cells in each community
    community_normalized = {community: community_sum[community] / community_counts[community] for community in community_sum.keys()}

    # Step 9: Calculate cosine similarity between communities
    community_features = np.array(list(community_normalized.values()))
    similarity_matrix = cosine_similarity(community_features)

    # Step 10: Build a graph where nodes represent communities and edges are weighted by similarity
    G = nx.Graph()

    # Add nodes with their normalized features
    for community, norm_features in community_normalized.items():
        G.add_node(int(community), data=norm_features.tolist())

    # Add edges based on similarity threshold
    for i in range(len(community_normalized)):
        for j in range(i + 1, len(community_normalized)):
            if similarity_matrix[i, j] > 0:  # Skip if communities are not similar
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    # Convert graph data to a format suitable for JSON export
    data = nx.node_link_data(G)
    data = convert_int64_to_int(data)

    # Step 11: Save the graph as a JSON file
    with open(os.path.join(root, "graph_with_group", file_path.split("/")[-1].replace(".csv", "_graph.json")), "w") as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    # Path configurations
    root = os.path.dirname(os.path.abspath(__file__))
    gene_name = pd.read_csv(r"../Gene prediction/inferred_svg/immune_svg.csv", index_col=0)

    # Process all SVS files in the directory
    svs_root = r"../Gene prediction/inferred_svg/"
    for file_name in os.listdir(svs_root):
        csv_path = os.path.join(svs_root, file_name)
        if os.path.isfile(csv_path):
            process_svs_file(csv_path, root, gene_name)






