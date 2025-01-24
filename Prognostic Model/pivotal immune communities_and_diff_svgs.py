import os
import pandas as pd
import numpy as np
import scipy.stats as stats

def load_data(root_dir):
    """
    Load and process gene expression data from CSV files in the given directory.
    """
    data = pd.DataFrame([])
    data_index = []

    # Iterate through all directories
    for file in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, file)):
            data_index.append(file)
            temp_data = None

            # Process each CSV file in the subdirectories
            for csv_path in os.listdir(os.path.join(root_dir, file)):
                if csv_path.endswith(".csv"):
                    # leiden = int(csv_path[6])
                    score = float(csv_path[:-4].split("_")[1])

                    if score > 1.4:  # 1.4 / 1.8 threshold for pivotal immune communities
                        temp_data = pd.read_csv(os.path.join(root_dir, file, csv_path), index_col=0) \
                            if temp_data is None else temp_data + pd.read_csv(os.path.join(root_dir, file, csv_path), index_col=0)

            data = pd.concat([data, temp_data], axis=1)

    return data.T, data_index


def load_metadata(metadata_path, file_type="csv"):
    """
    Load metadata and create a label dictionary based on 'Responder' column for CSV or 'pCR' column for Excel.
    """
    if file_type == "csv":
        metadata = pd.read_csv(metadata_path)
        label_dict = {row['Patient']: 1 if row['Responder'] == 'responder' else 0 for _, row in metadata.iterrows()}
    elif file_type == "excel":
        metadata = pd.read_excel(metadata_path, sheet_name=0)
        label_dict = {row['patients']: row['pCR'] for _, row in metadata.iterrows()}
    else:
        raise ValueError("Unsupported file type. Please choose 'csv' or 'excel'.")

    return label_dict

def perform_differential_expression(data):
    """
    Perform t-test and calculate log2 fold change (logFC) to identify differentially expressed genes.
    """
    group_0 = data[data['pCR'] == 0].drop(columns=['pCR'])
    group_1 = data[data['pCR'] == 1].drop(columns=['pCR'])

    results = []
    for gene in group_0.columns:
        t_stat, p_value = stats.ttest_ind(group_0[gene], group_1[gene], equal_var=False)
        mean_0 = np.mean(group_0[gene])
        mean_1 = np.mean(group_1[gene])
        logFC = np.log2(mean_1 + 1) - np.log2(mean_0 + 1)  # Adding 1 to avoid log(0)
        results.append({'gene': gene, 'logFC': logFC, 'p_value': p_value})

    # Convert results to DataFrame and filter svgs
    results_df = pd.DataFrame(results)
    significant_genes_df = results_df[(results_df['p_value'] < 0.05) & (abs(results_df['logFC']) > 0)]

    return results_df, significant_genes_df

def save_results(significant_genes_df, output_dir):
    """
    Save results and significant genes to CSV.
    """
    significant_genes_df.to_csv(os.path.join(output_dir, 'YALE_HER2_svgs_for_enrichment.csv'), index=False)
    print("Significant Genes for Enrichment:\n", significant_genes_df)

if __name__ == '__main__':
    root = r"../Community detection/graph_leiden_group/YALE-HER2/"
    metadata_path = r'../Community detection/graph_with_group/Yale_trastuzumab_response_cohort_metadata_clean.csv'

    # Load data and metadata
    data, data_index = load_data(root)
    label_dict = load_metadata(metadata_path)

    # Add 'pCR' column based on labels from metadata
    data["pCR"] = [label_dict[i] for i in data_index]

    # Sort data by 'pCR' values
    data_sorted = data.sort_values(by='pCR')

    # Perform differential expression analysis
    results_df, significant_genes_df = perform_differential_expression(data_sorted)

    # Save results
    save_results(significant_genes_df, r"../Prognostic Model/")
