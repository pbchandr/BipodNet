# MVNet/DeepDICE

DeepDICE is a multi-view multi-modal **D**eep learning approach for **D**isese phenotype prediction by using **I**nternal **C**ross-modal **E**stimation. Our contributions are three-fold. Firstly, we present a deep learning framework that integrates genotype and gene expression data guided by prior biological knowledge in terms of QTLs and GRNs. Secondly, our framework can take in only one modality data and performs internal cross-modal estimation by learning linear transformations and uses the estimated values for disease prediction. Thirdly, we decipher the black box nature of the neural network architecture to identify and prioritize genes and SNPs that contribute towards disease onset. 


<!--![DeepDICE Architecture](https://user-images.githubusercontent.com/18314073/124612169-bc616880-de37-11eb-969a-16dc36ca0767.png)-->
<!--![bipodnet_architecture](https://user-images.githubusercontent.com/18314073/125311395-fed3eb00-e2f8-11eb-9719-289f396bd496.png)-->
<!--![mvnet_arch](https://user-images.githubusercontent.com/18314073/141710663-6184ebd6-90e3-49cf-81a8-1c5d99f1a055.png) -->
![deepdice_architecture](https://user-images.githubusercontent.com/18314073/153448583-004a6414-00c1-4e5a-8296-96d65b312583.png)



## Dependencies
The script is based on python 3.4 above and requires the following packages:
1. tensorflow: v1.14 (cpu) or v1.10(gpu)
2. scipy
3. numpy
4. scikit-learn

## Data
The model requires 5 inputs:
1. Gene Expression data (p samples by m genes)
2. Genotype data (p samples by n snps)
3. Gene Regulatory Networks (GRNs)
4. expression Quantitavie Trait Loci (eQTLs)
5. Disease phenotype

All sample data can be accessed [here](http://resource.psychencode.org)

If you have your own data, please prepare the data in a .csv format and use the MVNetPreprocess.py to extract the required files for training.

## Usage
BipodNet can be trained by running the following command:

```
python MVNetTrain.py --num_data_modal=2 --input_files='/path_to_gene_exp_csv_file, /path_to_genotype_csv_file' --intermediate_phenotype_files='/path_to_grn_npz_file, /path_to_eqtl_npz_file' --disease_label_file='path_to_class_labels_csv_file' --save='/path_to_save_model'
```

The above code runs the default settings for training. Additional settings that can ve included withthe above code are:
* **--num_fc_layers** = number of fully conencted network layers. Default is 2 layers.
* **--num_fc_neurons** = number of hidden units in each layer. Comma separated values in the form os string needs to be given. Default is '500,50'
* **--fc_dropout_prob** = This is used to handle overfitting. Default is 0.5.
* **--batch_size** = Batch size for training.
* **--out_reg_lambda** = L2 regularization parameter
* **--learn_rate** = Learning rate for the model.
* **--corr_reg_lambda** = Regularization parameter for the cross-modal estimation loss.
* **--cross-validate** = This is a flag which performs 5-fold CV when enabled
