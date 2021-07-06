# BipodNet/BipodNN

BipodNet/BipodNN is a multi-view **B**iology **I**nduced Neural **Net**work for **P**rediction **o**f **D**isease Phenotypes. It uses multi-modal data in the form of rna-seq and genotype data to predict Schizophrenia samples. The intermediate layer is a biologival masking gene layer in the form of EQTL and Gene Regulatory Network (GRN). We hypothesise that introducting biology will help in better phenotypic prediction and also helps in understanding the disease mechanisms better.

![BipodNet Architecture](https://user-images.githubusercontent.com/18314073/124612169-bc616880-de37-11eb-969a-16dc36ca0767.png)

In the above architecture, the link between the input layer and the gene layer is controleld by activations from GRN and eQTL. This is depicted by the red and green lines where the green lines depict activations and red lines represents no activation.

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

## Usage
BipodNet can be trained by running the following command:

```
python bipodnet_train.py --gex_obs='/path_to_rna_seq_csv_file' --grn_adj='/path_to_grn_npz_file' --snp_obs='/path_to_snp_dosage__csv_file' --eqtl_adj='/path_to_eqtl_npz_file'
```

The above code runs the default settings for training. Additional parameters can be given to the above code. The parameters involved are:
* **--num_fc_layers** = Number of fully conencted network layers. Default is 2 layers.
* **--num_fc_neurons** = Number of hidden units in each layer. Comma separated values in the form os string needs to be given. Default is '1000,500'
* **--dropout_keep_probability** = This is used to handle overfitting. Default is 0.5
