#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pbchandr
"""

import argparse
import os
import time
import random
import torch
from torch import nn
import numpy as np
import scipy as sp
import pandas as pd
import sklearn.metrics as skm
from sklearn.model_selection import  StratifiedKFold, train_test_split
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
import DeepDiceUtils as ut
from DeepDiceModel import DeepDiceSV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(inp, oup):
    """ Function to direct the input and ouput to CPU vs GPU"""    
    return inp.float().to(device), oup.int().reshape(-1, 1).to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def loss_batch(model, loss_fn, data_dl, l1_reg, opt=None):
    """ Function to compute loss per batch """

    tot_loss = 0
    predictions, truth = [], []

    # Loop through batches
    for xb, yb in data_dl:
        loss = 0.0

        # Get the predictions from the model
        yhat = model(xb)
        
        # Compute prediction and estimation loss
        pred_loss = loss_fn(yhat, yb.float())
        loss = pred_loss    
        for param in model.parameters():
            loss += l1_reg * torch.sum(torch.abs(param))
        
        # Back propogation
        if opt is not None:
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        predictions.extend(yhat.detach().cpu().numpy())
        truth.extend(yb.detach().cpu().numpy())
        tot_loss += loss.item()
        
    predictions = np.asarray(predictions)
    truth = np.asarray(truth)

    return tot_loss/len(data_dl), predictions, truth

def get_binary_performance(y_true, y_score):
    """Function to return various performance metrics"""

    y_pred = np.where(y_score<0.5, 0, 1)

    auc = skm.roc_auc_score(y_true, y_score)
    acc = skm.accuracy_score(y_true, y_pred)
    bacc = skm.balanced_accuracy_score(y_true, y_pred)
    return acc, bacc, auc

def fit(epochs, model, loss_fn, opt, train_dl, val_dl, l1_reg, save_dir, cv_cntr=1):
    """ Function to fit the model """
    max_tr_acc, max_val_acc = 0, 0
    max_tr_auc, max_val_auc = 0, 0
    stagnant, best_epoch = 0, 0
    
    # Iterate over several epochs. Ealry stopping criterira is applied
    for epoch in range(epochs):

        # Trainign phase - All modalities are given to the model
        model.train()
        tr_loss, tr_pred, tr_truth = loss_batch(model, loss_fn, train_dl, l1_reg, opt)
        
        # Evaluataion phase
        model.eval()
        # Input is all modalities
        val_loss, val_pred, val_truth = loss_batch(model, loss_fn, val_dl, l1_reg)
        
        tr_pred_bin = np.where(tr_pred<0.5, 0, 1)
        val_pred_bin = np.where(val_pred<0.5, 0, 1)

        tr_acc, tr_bacc, tr_auc = get_binary_performance(tr_truth, tr_pred)        
        val_acc, val_bacc, val_auc = get_binary_performance(val_truth, val_pred)

        print("\n*** Epoch = %d ***"%(epoch))
        print("Training: Loss - %.4f, ACC - %.4f, BACC - %.4f, AUC - %.4f"%(tr_loss, tr_acc,
                                                                            tr_bacc, tr_auc))
        print(skm.confusion_matrix(tr_truth, tr_pred_bin))
        print("Validation: Loss - %.4f, ACC - %.4f, BACC - %.4f, AUC - %.4f"%(val_loss, val_acc,
                                                                              val_bacc, val_auc))
        print(skm.confusion_matrix(val_truth, val_pred_bin))

        if epoch == 0:
            max_tr_acc, max_val_acc = tr_bacc, val_bacc
            max_tr_auc, max_val_auc = tr_auc, val_auc

            best_epoch = epoch
            torch.save(model, os.path.join(save_dir, 'run_' + str(cv_cntr) + '_best_model.pth'))

        else:
            if (tr_bacc >= max_tr_acc) and (val_bacc > max_val_acc):
            #if (val_cg_bacc > max_val_cg_acc):
                max_tr_acc, max_val_acc = tr_bacc, val_bacc
    
                max_tr_auc, max_val_auc = tr_auc, val_auc
    
                best_epoch = epoch
                torch.save(model, os.path.join(save_dir, 'run_' + str(cv_cntr) + '_best_model.pth'))

                print("saving model")
                stagnant = 0
            else:
                stagnant += 1
        if stagnant == 40:
            break

    return best_epoch, max_tr_acc, max_tr_auc, max_val_acc, max_val_auc


def run_split_train(obs_data, labels, args):
    #obs_data = tr_obs.copy()
    #labels = label_train.copy()
    
    """ Function to run one traditional training"""
    st_time = time.perf_counter()

    # Balanced vs non-balanced data splitting    
    pos_idx = list(np.where(labels == 1)[0])
    neg_idx = list(np.where(labels == 0)[0])

    if args.need_balance:
        if len(pos_idx) > len(neg_idx):
            pos_tr_idx = random.sample(pos_idx, round(args.train_percent*len(neg_idx)))
            neg_tr_idx = random.sample(neg_idx, round(args.train_percent*len(neg_idx)))
        else:
            pos_tr_idx = random.sample(pos_idx, round(args.train_percent*len(pos_idx)))
            neg_tr_idx = random.sample(neg_idx, round(args.train_percent*len(pos_idx)))
        tridx = pos_tr_idx + neg_tr_idx
    else:
        tridx = pos_idx + neg_idx

    tridx.sort()
    teidx = list(set(range(labels.shape[0])) - set(tridx))

    obs_tr, obs_val = obs_data.values[tridx, :], obs_data.values[teidx, :]
        
    # Data normalization
    scaler = None
    if args.type_of_norm == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = preprocessing.StandardScaler()

    if args.normalize == 'features':
        obs_tr = scaler.fit_transform(obs_tr)
        obs_val = scaler.fit_transform(obs_val)
    else:
        obs_tr_t = scaler.fit_transform(obs_tr.T)
        obs_val_t = scaler.fit_transform(obs_val.T)
        
        obs_tr, obs_val = obs_tr_t.T, obs_val_t.T

    y_tr, y_val = labels[tridx] , labels[teidx]
        
    # Make data iterable with batches
    obs_tr, y_tr = map(torch.tensor, (obs_tr, y_tr))

    train_ds = TensorDataset(obs_tr, y_tr)
    val_ds = TensorDataset(obs_val, y_val)

    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size*2, shuffle=False)

    train_dl = WrappedDataLoader(train_dl, preprocess)
    val_dl = WrappedDataLoader(val_dl, preprocess)
        
    # Get model parameters and create model
    fc_num_neurons = [int(x) for x in args.fcn_num_neurons.split(',')]
    args.num_fc_neurons = fc_num_neurons
    model = DeepDiceSV(args).to(device)
    
    if model is not None:
        for name, param in model.named_parameters():
            print(name, param.size())
        print(model)
        
    # Define the loss function snd initialize optimizer
    loss_fn = nn.BCELoss()
    args.loss_fn = loss_fn
    opt = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    
    # Train the model
    _, tr_acc, tr_auc, val_acc, val_auc = fit(args.epochs, model, loss_fn, opt, train_dl, val_dl,
                                              args.out_reg, args.save, cv_cntr=1)

    print('***Run Split train***')
    out_file = args.cell_type + '_perf.txt'
    header_str = "Model\tTr ACC\tTr AUC\tVal ACC\tVal AUC\n"
    
    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write(header_str)
            write_fl.close()

    wr_str = "%s\t%.5f\t%5f" %(args.save, tr_acc, tr_auc)
    wr_str += "\t%.5f\t%5f" %(val_acc, val_auc)

    with open(out_file, 'a') as write_fl:
        write_fl.write(wr_str)
        write_fl.close()

    print("Train - ACC: %.5f, AUC: %.5f" %(tr_acc, tr_auc))
    print("Val - ACC: %.5f, AUC: %.5f" %(val_acc, val_auc))    

    end_time = time.perf_counter()
    print("Run complete in %.3f minutes"%((end_time - st_time)/60.00))
    print("Optimization Finished!")



def run_cv_train(obs_data, labels, args):
    """ Function to run cross validation modelling"""
    # inp1, inp2 = tr_inp1, tr_inp2
    # labels = label_train

    # Split dalta into 5 fold CV splits
    cv_k = 5
    rnd_seed = random.randint(1, 9999999)
    kfl = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=rnd_seed)
    
    cntr = 1
    tr_acc_sc, tr_auc_sc, val_acc_sc, val_auc_sc = [], [], [], []
    
    print("Random Seed = %d"%(rnd_seed))
    st_time = time.perf_counter()
    
    for tridx, teidx in kfl.split(obs_data, labels):
        print("********** Run %d **********"%(cntr))

        obs_tr, obs_val = obs_data.values[tridx, :], obs_data.values[teidx, :]
            
        # Data normalization
        scaler = None
        if args.type_of_norm == 'minmax':
            scaler = preprocessing.MinMaxScaler()
        else:
            scaler = preprocessing.StandardScaler()
     
        if args.normalize == 'features':
            obs_tr = scaler.fit_transform(obs_tr)
            obs_val = scaler.fit_transform(obs_val)
        else:
            obs_tr_t = scaler.fit_transform(obs_tr.T)
            obs_val_t = scaler.fit_transform(obs_val.T)
            
            obs_tr, obs_val = obs_tr_t.T, obs_val_t.T
     
        y_tr, y_val = labels[tridx] , labels[teidx]
               
        # Make data iterable with batches
        obs_tr, y_tr = map(torch.tensor, (obs_tr, y_tr))
        obs_val, y_val = map(torch.tensor, (obs_val, y_val))
     
        train_ds = TensorDataset(obs_tr, y_tr)
        val_ds = TensorDataset(obs_val, y_val)
     
        train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size*2, shuffle=False)
     
        train_dl = WrappedDataLoader(train_dl, preprocess)
        val_dl = WrappedDataLoader(val_dl, preprocess)            

        # Get model parameters and create model
        fc_num_neurons = [int(x) for x in args.fcn_num_neurons.split(',')]
        args.num_fc_neurons = fc_num_neurons
        model = DeepDiceSV(args).to(device)
        
        if model is not None:
            for name, param in model.named_parameters():
                print(name, param.size())
            print(model)
            
        # Define the loss function snd initialize optimizer
        loss_fn = nn.BCELoss()
        args.loss_fn = loss_fn
        opt = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
        
        # Train the model
        _, tr_acc, tr_auc, val_acc, val_auc = fit(args.epochs, model, loss_fn, opt, train_dl, val_dl,
                                                  args.out_reg, args.save, cv_cntr=1)
        tr_acc_sc.append(tr_acc)
        tr_auc_sc.append(tr_auc)
        val_acc_sc.append(val_acc)
        val_auc_sc.append(val_auc)
            
        cntr += 1
        print("")

    print('***Cross Validation***')
    out_file = args.cell_type + '_perf.txt'
    header_str = "Model\tTr ACC\tTr AUC\tVal ACC\tVal AUC\n"
    
    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write(header_str)
            write_fl.close()

    wr_str = "%s\t%.5f +/- %.5f\t%5f +/- %.5f" %(args.save, np.mean(tr_acc_sc),
                                                 np.std(tr_acc_sc), np.mean(tr_auc_sc),
                                                 np.std(tr_auc_sc))

    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f" %(np.mean(val_acc_sc), np.std(val_acc_sc),
                                                 np.mean(val_auc_sc), np.std(val_auc_sc))

    with open(out_file, 'a') as write_fl:
        write_fl.write(wr_str)
        write_fl.close()
        
    print('Train perf on each fold', tr_acc_sc)
    print('Val perf on each fold', val_acc_sc)

    print("Train - ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(tr_acc_sc),
                                                             np.std(tr_acc_sc),
                                                             np.mean(tr_auc_sc),
                                                             np.std(tr_auc_sc)))
    print("Val - ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(val_acc_sc),
                                                           np.std(val_acc_sc),
                                                           np.mean(val_auc_sc),
                                                           np.std(val_auc_sc)))

    #Keep the best model and remove the other folders
    fls = os.listdir(args.save)
    model_fls = [f for f in fls if f.startswith('run')]
    print(model_fls)
    keep_model = args.save + '/run_' + str(val_acc_sc.index(max(val_acc_sc))+1) + '_best_model.pth'
    for fls in model_fls:
        if (args.save + '/'+ fls) != keep_model:
            os.remove((args.save + '/' + fls))

    end_time = time.perf_counter()
    print("Five fold CV complete in %.3f minutes"%((end_time - st_time)/60.00))


def train_deepdice_sv(args):
    """ Method to fetch the data and perfrom training """
    print("hello")
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    #-- Load and preprocess data -- #
    # Fetch required inputs
    st_time = time.perf_counter()

    inp = pd.read_csv(args.obs_file, header=0)#.drop(columns=['Unnamed: 0'])
    inp = inp.set_index(inp.columns[0])
    inp = inp.T

    lbls = pd.read_csv(args.label_file, header=0)#.drop(columns=['Unnamed: 0'])
    labels = lbls['label'].values
    print(labels.shape)

    # Split data into training and testing
    if args.split_sample_ids == "None":
        tr_idx, te_idx = train_test_split(inp.index.values, test_size=0.10, random_state=1)
        
        with open(args.save+'train_samples.out', 'w') as output:
            for row in tr_idx:
                output.write(str(row) + '\n')
        
        with open(args.save+'test_samples.out', 'w') as output:
            for row in te_idx:
                output.write(str(row) + '\n')
    else:
        samp_ids = pd.read_csv(args.split_sample_ids, header=None)
        tr_idx = samp_ids.iloc[:, 0].values
        tr_idx = tr_idx.astype(str)

    idx = np.where(inp.index.isin(tr_idx))[0]
    tr_obs = inp.loc[inp.index.isin(tr_idx), ]
 
    label_train = labels[idx, ]
    args.n_out = 1
    tru, trc = np.unique(label_train, return_counts=True)
    print("Train label split ---", dict(zip(tru, trc)))
    
    print('tr obs', tr_obs.shape)

    if args.adj_file != 'None':
        adj = sp.sparse.load_npz(args.adj_file)
        adj = adj.todense()
        adj[adj == 0] = np.max(adj)/10.0
        args.adj = torch.from_numpy(adj).float().to(device)
        args.p_feat, args.n_genes = adj.shape
        print('adj', adj.shape)
    else:
        args.p_feat, args.n_genes = tr_obs.shape[1], int(tr_obs.shape[1]/2)

    end_time = time.perf_counter()
    
    print("Data fetch & split completed in %.3f mins\n"%((end_time - st_time)/60.00))
    print(args)

    # Cross Validation
    if args.cross_validate:
        run_cv_train(tr_obs, label_train, args)
    else:
        run_split_train(tr_obs, label_train, args)     

def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--obs_file', type=str,
                        default="bulk_dlpfc_pval001_5snps/bulk_dlpfc_geno_001.csv",
                        help='Link to data file')
    parser.add_argument('--adj_file', type=str,
                        default="bulk_dlpfc_pval001_5snps/bulk_dlpfc_eqtl_adj_001.npz",
                        help='Path to prior knowledge matrix')
    parser.add_argument('--label_file', type=str,
                        default="bulk_dlpfc_pval001_5snps/bulk_dlpfc_labels_001.csv",
                        help='Path to Output labels file - Disease phenotypes')

    # Hyper parameters
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--out_reg', type=float, default=0.005, help='l2_reg_lambda')

    # First transparent layer
    parser.add_argument('--model_type', type=str, default='fully_connect',
                        help='Drop Connect vs FCN vs both for the first transparent layer')
    parser.add_argument('--latent_dim', type=int, default=500,
                        help='Number of dimensions for the latent space to be reduced.')
    # FCN
    parser.add_argument('--num_fc_layers', type=int, default=2,
                        help='Number of hidden layers.')
    parser.add_argument('--fcn_num_neurons', type=str, default='50',
                        help='Number of kernels for fully connected layers, comma delimited.')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Droupout % for handling overfitting. 1 to keep all & 0 to keep none')


    # Settings
    # Data Normalization
    parser.add_argument('--normalize', type=str, default='features',
                        help='Feature vs sample normalization')
    parser.add_argument('--type_of_norm', type=str, default='standard',
                        help='Standard Normalization vs min-max')
    # Data split    
    parser.add_argument('--train_percent', type=float, default=0.8,
                        help='Choose how the tain and testvalidation split to occur.')
    parser.add_argument('--need_balance', type=bool, default=False, help='balanced_training')    
    parser.add_argument('--cross_validate', type=bool, default=True,
                        help='Choose normal validation vs cross-validation')
    #Model training
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--stagnant', type=int, default=50, help='Early stop criteria')

    # Model save paths and others
    parser.add_argument('--save', type=str, default="model/try/",
                        help="path to save model")
    parser.add_argument('--cell_type', type=str, default='bulk', help='Cell type')
    
    # Remove these parameters later. This is for our convenience
    parser.add_argument('--split_sample_ids', type=str, help="training and testing splits",
                        default="None")
    parser.add_argument('--filtered_snp_file', type=str, help="File that contains SNPs filtered",
                        default="None")


    args = parser.parse_args()
    train_deepdice_sv(args)

if __name__ == '__main__':
    main()
