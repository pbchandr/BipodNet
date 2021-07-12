# BipodNet/BipodNN

BipodNet/BipodNN is a multi-view **B**iology **I**nduced Neural **Net**work for **P**rediction **o**f **D**isease Phenotypes. It uses multi-modal data to form a conenctino with intermediate phenotypes using activations guided by biology which is then used for prediction various diseases phenotypes. We hypothesize that introducing biology will help in better phenotypic prediction and also helps in understanding the disease mechanisms better. The intermeidate phenotypes can be shared across diffent modalities of the data (genes, cell-types, etc) or can have each modal-specific phenotype (one mode can have genes as the intermediate phenotypes and another can have cell types).

<!--![BipodNet Architecture](https://user-images.githubusercontent.com/18314073/124612169-bc616880-de37-11eb-969a-16dc36ca0767.png)-->
![bipodnet_architecture](https://user-images.githubusercontent.com/18314073/125311395-fed3eb00-e2f8-11eb-9719-289f396bd496.png)

In the above architecture, the link between the feature extraction layer and the drop-conenct layer is controlled by activations from biological prior. This is depicted by the red and green lines where the green lines depict activations and the red lines represent no activation.

For example, we can ghave RNA-seq and genotype data as inputs. The intermediate phenotypes can be a gene layer where the RNA-seq to gene layer can be linked using Gene Regulatory Network(GRN) and the genotype to gene can be linked using eQTL linkages from gtex.

## Dependencies
The script is based on python 3.4 above and requires the following packages:
1. pytorch
2. scipy
3. numpy
4. scikit-learn

## Data
All sample data can be accessed [here](http://resource.psychencode.org)

If you have your own data, please prepare the data in a .csv format and provide the link to the file location while training the model. Please note that the number and order of features form a module in the input layer should match that in the drop connect layer.

## Usage
BipodNet can be trained by running the following command:

```
python bipodnet_train.py --gex_obs='/path_to_rna_seq_csv_file' --grn_adj='/path_to_grn_npz_file' --snp_obs='/path_to_snp_dosage__csv_file' --eqtl_adj='/path_to_eqtl_npz_file'
```

The above code runs the default settings for training. Additional parameters can be given to the above code. The parameters involved are:
* **--num_fc_layers** = number of fully conencted network layers. Default is 2 layers.
* **--num_fc_neurons** = number of hidden units in each layer. Comma separated values in the form os string needs to be given. Default is '1000,500'
* **--dropout_keep_probability** = This is used to handle overfitting. Default is 0.5
