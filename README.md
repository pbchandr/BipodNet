# DeepDICE

DeepDICE is a multi-view multi-modal **D**eep learning approach for **D**isese phenotype prediction by using **I**nternal **C**ross-modal **E**stimation. Our contributions are three-fold. Firstly, we present a deep learning framework that integrates genotype and gene expression data guided by prior biological knowledge in terms of QTLs and GRNs. Secondly, our framework can take in only one modality data and performs internal cross-modal estimation by learning linear transformations and uses the estimated values for disease prediction. Thirdly, we decipher the black box nature of the neural network architecture to prioritize genes and SNPs that contribute towards disease onset. 


<!--![DeepDICE Architecture](https://user-images.githubusercontent.com/18314073/124612169-bc616880-de37-11eb-969a-16dc36ca0767.png)-->
<!--![bipodnet_architecture](https://user-images.githubusercontent.com/18314073/125311395-fed3eb00-e2f8-11eb-9719-289f396bd496.png)-->
<!--![mvnet_arch](https://user-images.githubusercontent.com/18314073/141710663-6184ebd6-90e3-49cf-81a8-1c5d99f1a055.png) -->
![deepdice_architecture](https://user-images.githubusercontent.com/18314073/153448583-004a6414-00c1-4e5a-8296-96d65b312583.png)



## Dependencies
The script is based on python 3.4 above and requires the following packages:
1. pytorch: v1.4.0  (cpu) or v1.10.0(gpu)
2. scipy
3. numpy
4. scikit-learn

## Data
The model requires 5 inputs:
1. Gene Expression data (p samples by m genes)
2. Genotype data (p samples by n snps)
3. Gene Regulatory Networks (GRNs, m genes by k genes)
4. expression Quantitavie Trait Loci (eQTLs, m snps by k genes)
5. Disease phenotype

All sample data can be accessed [here](http://resource.psychencode.org)

If you have your own data, please prepare the data in a .csv format and use the DeepDicePreprocess.py to extract the required files for training.

## Usage
DeepDice has two versions of the code:
1. DeepDiceMVTrain - This version takes in two modalities as input for disease prediction and can be trained using the following command:

```
python -u DeepDiceMVTrain.py --input_files='/path_to_gene_exp_csv_file, /path_to_genotype_csv_file' --intermediate_phenotype_files='/path_to_grn_npz_file, /path_to_eqtl_npz_file' --disease_label_file='path_to_class_labels_csv_file' --save= '/path_to_save_model' > '/path_to_output.txt'
```

The above code runs the default settings for training. Additional settings that can be included along with the above code are:
* **--model_type** = This parameter determines whether to use biological drop connection for the first transparent layer or fully connected network. (default = 'drop_connect').
* **--latent_dim** = This parameter is used to specify the number of hidden nodes in the transparent layer if the model type is fully conencted network (default = 500).
* **--num_fc_layers** = Number of fully conencted network layers (default = 2).
* **--num_fc_neurons** = Number of hidden units in each layer following the transparent layer. Comma separated values needs to be given (default = '500,50').
* **--dropout_keep_prob** = This is used to handle overfitting (default = 0.5).
* **--normalize** = Feature normalization versus sample normalization (default = 'features').
* **--type_of_norm** = CHoose between standard and minmax normalization (default = 'standard').
* **--batch_size** = Batch size for training (default = 30).
* **--learn_rate** = Learning rate for the model (default = 0.001). 
* **--out_reg** = L2 regularization parameter (default = 0.005).
* **--corr_reg** = Regularization parameter for the cross-modal estimation loss (default = 0.5).
* **--cross-validate** = This is a flag which performs 5-fold CV when enabled ((default = False).


2. DeepDiceSVTrain - This version takes in only one input modality for disease prediction and can be trained using the following command:

```
python -u DeepDiceSVTrain.py --obs='/path_to_input_csv_file' --adj_file='/path_to_adjacency_npz_file' --label_file='path_to_class_labels_csv_file' --save= '/path_to_save_model' > '/path_to_output.txt'
```
 All additional parameters are the same as the DeepDiceMVTrain version except for --corr_reg which doesn't exist in this version.
