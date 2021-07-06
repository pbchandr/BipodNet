# BipodNet/BipodNN

BipodNet/BipodNN is a multi-view **B**iology **I**nduced Neural **Net**work for **P**rediction **o**f **D**isease Phenotypes. It uses multi-modal data in the form of rna-seq and genotype data to predict Schizophrenia samples. The intermediate layer is a biologival masking gene layer in the form of EQTL and Gene Regulatory Network (GRN). We hypothesise that introducting biology will help in better phenotypic prediction and also helps in understanding the disease mechanisms better.

## Dependencies
The script is based on python 3.4 above and requires the following packages:
1. pytorch
2. scipy
3. numpy
4. scikit-learn

## Data
All sample data can be accessed [here](https://uwmadison.box.com/s/as518bcuttpkdonads64iriqfdd1aixl)

If you have your own data, please use the following guide to prepare the data.

### Preparing RNA-seq data
The cript requires a csv file that contains gene expression data where the rows are the samples and the columns are the genes

### Preparing Genotype data
The model uses the dosage information for SNP coordinates. The rows are the samples  and the columns are the SNP dosage information.

### Preparing intermediate layer
Gene Regulatory Network (GRN) and eQTL-gene linkage are used as the biologial masking intermediate layer that guides the activation units in the neural network model. To set up the GRN masking layer, an adjacency matrix is created where the rows are source genes and the columns are target genes. The gene names and the order must match the rna-seq input genes. Similarly, eqtl-gene adjacency is created. The eqtl ids and order should match the snp id from the genotype data. The script accepts an .npz format which contains the sprse matrix for both sources of data.



