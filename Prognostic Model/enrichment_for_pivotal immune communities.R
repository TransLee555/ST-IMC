library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(ggplot2)

# Read differential gene expression data
diff_genes <- read.csv("../Prognostic Model/YALE_HER2_svgs_for_enrichment.csv")
diff_genes$gene <- as.character(diff_genes$gene)

# Perform Gene Ontology (GO) enrichment analysis (Biological Process)
ego_bp <- enrichGO(
  gene = diff_genes$gene,          
  OrgDb = org.Hs.eg.db,            
  keyType = "SYMBOL",              
  ont = "BP",                     
  pAdjustMethod = "BH",            
  pvalueCutoff = 0.05,             
  qvalueCutoff = 0.05              
)

# Create and save dotplot for GO enrichment analysis (Biological Process)
go_plot <- dotplot(ego_bp, showCategory = 10) + 
  ggtitle("GO Enrichment Analysis - Biological Process")
ggsave("../Prognostic Model/GO_Enrichment_Analysis_BP.png", plot = go_plot, width = 6.1, height = 6, dpi = 300)

# Map gene symbols to Entrez IDs
diff_genes$entrezid <- mapIds(
  org.Hs.eg.db,                   
  keys = diff_genes$gene,         
  column = "ENTREZID",            
  keytype = "SYMBOL",             
  multiVals = "first"             
)

# Remove rows with missing Entrez IDs
diff_genes <- na.omit(diff_genes)

# Perform KEGG pathway enrichment analysis
kegg <- enrichKEGG(
  gene = diff_genes$entrezid,     
  organism = 'hsa',               
  pvalueCutoff = 0.05             
)

# Create and save dotplot for KEGG pathway enrichment analysis
kegg_plot <- dotplot(kegg, showCategory = 10) + 
  ggtitle("KEGG Pathway Enrichment Analysis")
ggsave("../Prognostic Model/KEGG_Enrichment_Analysis.png", plot = kegg_plot, width = 6.1, height = 6, dpi = 300)

