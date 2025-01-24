import pandas as pd
import numpy as np
import scanpy as sc
import os

# Get the path of the current script
root_ = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(root_, "st_data")  # Folder containing spatial transcriptomics data. Please download the dataset from the provided link in the 'Data Availability' section.
out_path = os.path.join(root_, "sample_svg")  # Folder to save results of selected highly variable genes
immune_geneset_path = r"D:\test\2499_IRG.xlsx"  # Path to the immune geneset file

# 1. Process each spatial transcriptomics sample and extract highly variable genes
for sample_name in os.listdir(data_path):
    # Read each sample's CSV file and transpose (swap rows and columns)
    csv_data = pd.read_csv(os.path.join(data_path, sample_name), index_col=0).T

    # Create an AnnData object (standard format for single-cell RNA-seq data)
    adata = sc.AnnData(csv_data)

    # Preprocessing: filter low-expression genes, normalize, and log-transform
    sc.pp.filter_genes(adata, min_cells=20)  # Filter genes expressed in fewer than 20 cells
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize gene counts across cells to a target sum
    sc.pp.log1p(adata)  # Log-transform the gene expression data (log(x+1))
    sc.pp.highly_variable_genes(adata)  # Identify highly variable genes

    # Extract names of highly variable genes
    highly_variable_genes = adata.var['highly_variable']  # Boolean mask of highly variable genes
    selected_genes = adata.var_names[highly_variable_genes]  # Get the gene names of highly variable genes

    # Save the selected highly variable genes to a CSV file for each sample
    pd.DataFrame(selected_genes, columns=['Gene']).to_csv(os.path.join(out_path, sample_name + ".csv"), index=False)

# 2. Load the immune geneset and process it
geneset = pd.read_excel(immune_geneset_path, skiprows=2)  # Load the immune gene set from Excel file
immune_geneset = geneset.Symbo  # Extract the gene symbols from the "Symbo" column

# Get unique immune genes and their synonyms
unique_immune_geneset = list(immune_geneset.unique())  # Remove duplicates
Synonyms_set = geneset.Synonyms  # Get the synonyms of immune genes

# Add synonyms to the unique immune gene set
for s in Synonyms_set:
    if s == "-":
        continue  # Skip empty or invalid synonym entries
    else:
        names = s.split("|")  # Split by "|" if multiple synonyms exist
        for name in names:
            unique_immune_geneset.append(name)

# Remove duplicates from the immune geneset
unique_immune_geneset = set(unique_immune_geneset)  # Convert to set to ensure uniqueness

# 3. Count the frequency of highly variable genes across all samples
high_var_gene = {}

# Iterate over each file in the output path where high variable genes are stored
for file in os.listdir(out_path):
    # Read the CSV file of selected highly variable genes
    gene = pd.read_csv(os.path.join(out_path, file))

    # Count the occurrence of each gene across all files
    for g in gene.Gene:
        if g in high_var_gene:
            high_var_gene[g] += 1  # Increment count if the gene is already in the dictionary
        else:
            high_var_gene[g] = 1  # Initialize count if the gene is not in the dictionary

# 4. Filter genes that appear in more than 30 samples
filtered_dict = {key: value for key, value in high_var_gene.items() if value > 30}  # Keep genes in more than 30 samples

# 5. Identify immune-related genes from the filtered high variable genes
immune_dict = {key: value for key, value in filtered_dict.items() if key in unique_immune_geneset}

# Get the list of immune-related genes
output_immune_gene = [key for key, value in immune_dict.items()]

# Save the list of immune-related genes to a CSV file
pd.DataFrame(output_immune_gene, columns=["Immune_Gene"]).to_csv(os.path.join(root_, "immune_svg.csv"), index=False)
