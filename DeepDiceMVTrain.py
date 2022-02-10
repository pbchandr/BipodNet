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
import pandas as pd
import sklearn.metrics as skm
from sklearn.model_selection import  StratifiedKFold, train_test_split
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
import DeepDiceUtils as ut
from DeepDiceModel import DeepDiceMV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(inp1, inp2, oup):
    """ Function to direct the input and ouput to CPU vs GPU"""    
    return inp1.float().to(device), inp2.float().to(device), oup.int().reshape(-1, 1).to(device)

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

def loss_batch(model, loss_fn, data_dl, l1_reg, corr_reg, estimate, opt=None):
    """ Function to compute loss per batch """

    tot_loss = 0
    predictions, truth = [], []
    est_loss_fn = nn.MSELoss()

    # Loop through batches
    for snps, gex, yb in data_dl:
        loss = 0.0

        # Get the predictions from the model
        yhat = model(snps, gex, estimate)
        
        # Get the intermediate layer outputs
        Cs, Cs_est, Cg, Cg_est = model.get_intermediate_layers(snps, gex)

        # Compute prediction and estimation loss
        pred_loss = loss_fn(yhat, yb.float())
        est_loss = est_loss_fn(Cg, Cg_est)
        loss += pred_loss + corr_reg*est_loss
        
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

def fit(epochs, model, loss_fn, opt, train_dl, val_dl, l1_reg, corr_reg, save_dir, cv_cntr=1):
    """ Function to fit the model """
    max_tr_acc, max_val_acc, max_val_cg_acc, max_val_cs_auc = 0, 0, 0, 0
    max_tr_auc, max_val_auc, max_val_cg_auc, max_val_cs_auc = 0, 0, 0, 0
    stagnant, best_epoch = 0, 0
    
    # Iterate over several epochs. Ealry stopping criterira is applied
    for epoch in range(epochs):

        # Trainign phase - All modalities are given to the model
        model.train()
        estimate = 'None'
        tr_loss, tr_pred, tr_truth = loss_batch(model, loss_fn, train_dl,
                                                l1_reg, corr_reg, estimate, opt)
        
        # Evaluataion phase
        model.eval()
        # Input is all modalities
        val_loss, val_pred, val_truth = loss_batch(model, loss_fn, val_dl, l1_reg,
                                                   corr_reg, estimate)
        # Single modality input with estimating Cg from Cs
        estimate = 'cg'
        val_cg_loss, val_cg_pred, val_cg_truth = loss_batch(model, loss_fn, val_dl,
                                                            l1_reg, corr_reg, estimate)
        # Single modality input with estimating Cs from Cg        
        estimate = 'cs'
        val_cs_loss, val_cs_pred, val_cs_truth = loss_batch(model, loss_fn, val_dl,
                                                            l1_reg, corr_reg, estimate)
        
        tr_pred_bin = np.where(tr_pred<0.5, 0, 1)
        val_pred_bin = np.where(val_pred<0.5, 0, 1)
        val_cg_pred_bin = np.where(val_cg_pred<0.5, 0, 1)
        val_cs_pred_bin = np.where(val_cs_pred<0.5, 0, 1)

        tr_acc, tr_bacc, tr_auc = get_binary_performance(tr_truth, tr_pred)        
        val_acc, val_bacc, val_auc = get_binary_performance(val_truth, val_pred)
        val_cg_acc, val_cg_bacc, val_cg_auc = get_binary_performance(val_truth, val_cg_pred)
        val_cs_acc, val_cs_bacc, val_cs_auc = get_binary_performance(val_truth, val_cs_pred)

        print("\n*** Epoch = %d ***"%(epoch))
        print("Training: Loss - %.4f, ACC - %.4f, BACC - %.4f, AUC - %.4f"%(tr_loss, tr_acc,
                                                                            tr_bacc, tr_auc))
        print(skm.confusion_matrix(tr_truth, tr_pred_bin))
        print("Validation: Loss - %.4f, ACC - %.4f, BACC - %.4f, AUC - %.4f"%(val_loss, val_acc,
                                                                              val_bacc, val_auc))
        print(skm.confusion_matrix(val_truth, val_pred_bin))

        print("Val Cs->Cg: Loss - %.4f, ACC - %.4f, BACC - %.4f, AUC - %.4f"%(val_cg_loss,
                                                                              val_cg_acc,
                                                                              val_cg_bacc,
                                                                              val_cg_auc))
        print(skm.confusion_matrix(val_truth, val_cg_pred_bin))
        


        print("Val Cg->Cs: Loss - %.4f, ACC - %.4f, BACC - %.4f, AUC - %.4f"%(val_cs_loss,
                                                                              val_cs_acc,
                                                                              val_cs_bacc,
                                                                              val_cs_auc))
        print(skm.confusion_matrix(val_truth, val_cs_pred_bin))

        if epoch == 0:
            max_tr_acc, max_val_acc = tr_bacc, val_bacc
            max_val_cg_acc, max_val_cs_acc = val_cg_bacc, val_cs_bacc

            max_tr_auc, max_val_auc = tr_auc, val_auc
            max_val_cg_auc, max_val_cs_auc = val_cg_auc, val_cs_auc

            best_epoch = epoch
            torch.save(model, os.path.join(save_dir, 'run_' + str(cv_cntr) + '_best_model.pth'))

        else:
            if (tr_bacc >= max_tr_acc) and (val_bacc > max_val_acc):
            #if (val_cg_bacc > max_val_cg_acc):
                max_tr_acc, max_val_acc = tr_bacc, val_bacc
                max_val_cg_acc, max_val_cs_acc = val_cg_bacc, val_cs_bacc
    
                max_tr_auc, max_val_auc = tr_auc, val_auc
                max_val_cg_auc, max_val_cs_auc = val_cg_auc, val_cs_auc
    
                best_epoch = epoch
                torch.save(model, os.path.join(save_dir, 'run_' + str(cv_cntr) + '_best_model.pth'))

                print("saving model")
                stagnant = 0
                
            # elif(ve1_bacc > max_ve1_acc):
            #     max_tr_acc, max_val_acc, max_ve1_acc  = tr_bacc, val_bacc, ve1_bacc
            #     max_tr_auc, max_val_auc, max_ve1_auc = tr_auc, val_auc, ve1_auc
            #     best_epoch = epoch
            #     torch.save(model, os.path.join(save_dir, 'run_' + str(epoch) + '_best_model.pth'))
            #     print("saving model")
            #     stagnant = 0
            else:
                stagnant += 1
        if stagnant == 40:
            break

    return best_epoch, max_tr_acc, max_tr_auc, max_val_acc, max_val_auc, max_val_cg_acc, max_val_cg_auc, max_val_cs_acc, max_val_cs_auc


def run_split_train(snp_data, gex_data, labels, args):
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

    snps_tr, snps_val = snp_data.values[tridx, :], snp_data.values[teidx, :]
    gex_tr, gex_val = gex_data.values[tridx, :], gex_data.values[teidx, :]
        
    # Data normalization
    scaler = None
    if args.type_of_norm == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = preprocessing.StandardScaler()

    if args.normalize == 'features':
        snps_tr = scaler.fit_transform(snps_tr)
        gex_tr = scaler.fit_transform(gex_tr)
        snps_val = scaler.fit_transform(snps_val)
        gex_val = scaler.fit_transform(gex_val)
    else:
        snps_tr_t = scaler.fit_transform(snps_tr.T)
        gex_tr_t = scaler.fit_transform(gex_tr.T)
        snps_val_t = scaler.fit_transform(snps_val.T)
        gex_val_t = scaler.fit_transform(gex_val.T)
        
        snps_tr, gex_tr = snps_tr_t.T, gex_tr_t.T
        snps_val, gex_val = snps_val_t.T, gex_val_t.T        

    y_tr, y_val = labels[tridx] , labels[teidx]
        
    # Make data iterable with batches
    snps_tr, gex_tr, y_tr = map(torch.tensor, (snps_tr, gex_tr, y_tr))
    snps_val, gex_val, y_val = map(torch.tensor, (snps_val, gex_val, y_val))

    train_ds = TensorDataset(snps_tr, gex_tr, y_tr)
    val_ds = TensorDataset(snps_val, gex_val, y_val)

    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size*2, shuffle=False)

    train_dl = WrappedDataLoader(train_dl, preprocess)
    val_dl = WrappedDataLoader(val_dl, preprocess)
        
    # Get model parameters and create model
    fc_num_neurons = [int(x) for x in args.fcn_num_neurons.split(',')]
    args.num_fc_neurons = fc_num_neurons
    model = DeepDiceMV(args).to(device)
    
    if model is not None:
        for name, param in model.named_parameters():
            print(name, param.size())
        print(model)
        
    # Define the loss function snd initialize optimizer
    loss_fn = nn.BCELoss()
    args.loss_fn = loss_fn
    opt = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    
    # Train the model
    _, tr_acc, tr_auc, val_acc, val_auc, val_cg_acc, val_cg_auc, val_cs_acc, val_cs_auc = fit(args.epochs, model, loss_fn,
                                                                                              opt, train_dl, val_dl,
                                                                                              args.out_reg, args.corr_reg,
                                                                                              args.save, cv_cntr=1)

    print('***Run Split train***')
    out_file = args.cell_type + '_perf.txt'
    header_str = "Model\tTr ACC\tTr AUC\tVal ACC\tVal AUC\tVal Cs>Cg Est ACC\tVal Cs->Cg Est AUC"
    header_str += "\tVal Cg->Cs Est ACC\tVal Cg->Cs Est AUC\n"
    
    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write(header_str)
            write_fl.close()

    wr_str = "%s\t%.5f\t%5f" %(args.save, tr_acc, tr_auc)
    wr_str += "\t%.5f\t%5f" %(val_acc, val_auc)
    wr_str += "\t%.5f\t%5f" %(val_cg_acc, val_cg_auc)
    wr_str += "\t%.5f\t%5f\n" %(val_cs_acc, val_cs_auc)

    with open(out_file, 'a') as write_fl:
        write_fl.write(wr_str)
        write_fl.close()

    print("Train - ACC: %.5f, AUC: %.5f" %(tr_acc, tr_auc))
    print("Val - ACC: %.5f, AUC: %.5f" %(val_acc, val_auc))    
    print("Val Cs->Cg Est - ACC: %.5f, AUC: %.5f" %(val_cg_acc, val_cg_auc))
    print("Val Cg->Cs Est - ACC: %.5f, AUC: %.5f" %(val_cs_acc, val_cs_auc))

    end_time = time.perf_counter()
    print("Five fold CV complete in %.3f minutes"%((end_time - st_time)/60.00))
    print("Optimization Finished!")



def run_cv_train(snp_data, gex_data, labels, args):
    """ Function to run cross validation modelling"""
    # inp1, inp2 = tr_inp1, tr_inp2
    # labels = label_train

    # Split dalta into 5 fold CV splits
    cv_k = 5
    rnd_seed = random.randint(1, 9999999)
    kfl = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=rnd_seed)
    
    cntr = 1
    tr_acc_sc, tr_auc_sc, val_acc_sc, val_auc_sc = [], [], [], []
    val_cg_acc_sc, val_cg_auc_sc = [], []
    val_cs_acc_sc, val_cs_auc_sc = [], []
    
    print("Random Seed = %d"%(rnd_seed))
    st_time = time.perf_counter()
    
    for tridx, teidx in kfl.split(snp_data, labels):
        print("********** Run %d **********"%(cntr))

        snps_tr, snps_val = snp_data.values[tridx, :], snp_data.values[teidx, :]
        gex_tr, gex_val = gex_data.values[tridx, :], gex_data.values[teidx, :]
            
        # Data normalization
        scaler = None
        if args.type_of_norm == 'minmax':
            scaler = preprocessing.MinMaxScaler()
        else:
            scaler = preprocessing.StandardScaler()
    
        if args.normalize == 'features':
            snps_tr = scaler.fit_transform(snps_tr)
            gex_tr = scaler.fit_transform(gex_tr)
            snps_val = scaler.fit_transform(snps_val)
            gex_val = scaler.fit_transform(gex_val)
        else:
            snps_tr_t = scaler.fit_transform(snps_tr.T)
            gex_tr_t = scaler.fit_transform(gex_tr.T)
            snps_val_t = scaler.fit_transform(snps_val.T)
            gex_val_t = scaler.fit_transform(gex_val.T)
            
            snps_tr, gex_tr = snps_tr_t.T, gex_tr_t.T
            snps_val, gex_val = snps_val_t.T, gex_val_t.T        
    
        y_tr, y_val = labels[tridx] , labels[teidx]
            
        # Make data iterable with batches
        snps_tr, gex_tr, y_tr = map(torch.tensor, (snps_tr, gex_tr, y_tr))
        snps_val, gex_val, y_val = map(torch.tensor, (snps_val, gex_val, y_val))
    
        train_ds = TensorDataset(snps_tr, gex_tr, y_tr)
        val_ds = TensorDataset(snps_val, gex_val, y_val)
    
        train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size*2, shuffle=False)
    
        train_dl = WrappedDataLoader(train_dl, preprocess)
        val_dl = WrappedDataLoader(val_dl, preprocess)
            
        # Get model parameters and create model
        fc_num_neurons = [int(x) for x in args.fcn_num_neurons.split(',')]
        args.num_fc_neurons = fc_num_neurons
        model = DeepDiceMV(args).to(device)
        
        if model is not None:
            for name, param in model.named_parameters():
                print(name, param.size())
            print(model)
            
        # Define the loss function snd initialize optimizer
        loss_fn = nn.BCELoss()
        args.loss_fn = loss_fn
        opt = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
        
        # Train the model
        _, tr_acc, tr_auc, val_acc, val_auc, val_cg_acc, val_cg_auc, val_cs_acc, val_cs_auc = fit(args.epochs, model, loss_fn,
                                                                                                  opt, train_dl, val_dl,
                                                                                                  args.out_reg, args.corr_reg,
                                                                                                  args.save, cv_cntr=1)
        tr_acc_sc.append(tr_acc)
        tr_auc_sc.append(tr_auc)
        val_acc_sc.append(val_acc)
        val_auc_sc.append(val_auc)
        val_cg_acc_sc.append(val_cg_acc)
        val_cg_auc_sc.append(val_cg_auc)
        val_cs_acc_sc.append(val_cs_acc)
        val_cs_auc_sc.append(val_cs_auc)
            
        cntr += 1
        print("")

    print('***Cross Validation***')
    out_file = args.cell_type + '_perf.txt'
    header_str = "Model\tTr ACC\tTr AUC\tVal ACC\tVal AUC\tVal Cs->Cg Est ACC\tVal Cs->Cg Est AUC"
    header_str += "\tVal Cg->Cs Est ACC\tVal Cg->Cs Est AUC\n"
    
    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write(header_str)
            write_fl.close()

    wr_str = "%s\t%.5f +/- %.5f\t%5f +/- %.5f" %(args.save, np.mean(tr_acc_sc),
                                                 np.std(tr_acc_sc), np.mean(tr_auc_sc),
                                                 np.std(tr_auc_sc))

    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f" %(np.mean(val_acc_sc), np.std(val_acc_sc),
                                                 np.mean(val_auc_sc), np.std(val_auc_sc))

    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f" %(np.mean(val_cg_acc_sc), np.std(val_cg_acc_sc),
                                                 np.mean(val_cg_auc_sc), np.std(val_cg_auc_sc))

    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f\n" %(np.mean(val_cs_acc_sc), np.std(val_cs_acc_sc),
                                                   np.mean(val_cs_auc_sc), np.std(val_cs_auc_sc))

    with open(out_file, 'a') as write_fl:
        write_fl.write(wr_str)
        write_fl.close()
        
    print('Train perf on each fold', tr_acc_sc)
    print('2 Mode val perf on each fold', val_acc_sc)
    print('Cs->Cg val perf on each fold', val_cg_acc_sc)
    print('Cg - > Cs val perf on each fold', val_cs_acc_sc)

    print("Train - ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(tr_acc_sc),
                                                             np.std(tr_acc_sc),
                                                             np.mean(tr_auc_sc),
                                                             np.std(tr_auc_sc)))
    print("Val - ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(val_acc_sc),
                                                           np.std(val_acc_sc),
                                                           np.mean(val_auc_sc),
                                                           np.std(val_auc_sc)))
    print("Val Cs->Cg Est - ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(val_cg_acc_sc),
                                                                      np.std(val_cg_acc_sc),
                                                                      np.mean(val_cg_auc_sc),
                                                                      np.std(val_cg_auc_sc)))

    print("Val Cg->Cs Est - ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(val_cs_acc_sc),
                                                                      np.std(val_cs_acc_sc),
                                                                      np.mean(val_cs_auc_sc),
                                                                      np.std(val_cs_auc_sc)))
    
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


def train_deepdice_mv(args):
    """ Method to fetch the data and perfrom training """
    print("hello")
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    #-- Load and preprocess data -- #
    # Fetch required inputs
    st_time = time.perf_counter()

    inp, adj, labels = ut.get_csv_data(args.input_files, args.intermediate_phenotype_files,
                                       args.disease_label_file)
    
    # Split data into training and testing
    if args.split_sample_ids == "None":
        tr_idx, te_idx = train_test_split(inp[0].index.values, test_size=0.10, random_state=1)
        
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

    idx = np.where(inp[0].index.isin(tr_idx))[0]
    tr_snps = inp[0].loc[inp[0].index.isin(tr_idx), ]
    tr_gex = inp[1].loc[inp[1].index.isin(tr_idx), ]
 
    label_train = labels[idx]
    args.n_out = 1
    tru, trc = np.unique(label_train, return_counts=True)
    print("Train label split ---", dict(zip(tru, trc)))
    
    print('tr snps', tr_snps.shape)
    print('tr gex', tr_gex.shape)
    print('adj1', adj[0].shape)
    print('adj2', adj[1].shape)
    
    args.p_snps, args.n_genes = adj[0].shape
    args.m_tfs, args.n_genes = adj[1].shape
    args.eqtl = torch.from_numpy(adj[0]).float().to(device)
    args.grn = torch.from_numpy(adj[1]).float().to(device)

    end_time = time.perf_counter()
    
    print("Data fetch & split completed in %.3f mins\n"%((end_time - st_time)/60.00))
    print(args)

    # Cross Validation
    if args.cross_validate:
        run_cv_train(tr_snps, tr_gex, label_train, args)
    else:
        run_split_train(tr_snps, tr_gex, label_train, args)     

def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--num_data_modal', type=int, default=2,
                        help='Path to the input data file')
    parser.add_argument('--input_files', type=str,
                        default="bulk/pval_filt/001/bulk_dlpfc_geno_001.csv,bulk/pval_filt/001/bulk_dlpfc_gex_001.csv",
                        help='Comma separated input data paths')
    parser.add_argument('--intermediate_phenotype_files', type=str,
                        default="bulk/pval_filt/001/bulk_dlpfc_eqtl_adj_001.npz,bulk/pval_filt/001/bulk_dlpfc_grn_adj_001.npz",
                        help='Path to prior knowledge matrices for each data modality')
    parser.add_argument('--disease_label_file', type=str,
                        default="bulk/pval_filt/001/bulk_dlpfc_labels_001.csv",
                        help='Path to Output labels file - Disease phenotypes')
    
    # Remove these parameters later. This is for our convenience
    parser.add_argument('--split_sample_ids', type=str, help="training and testing splits",
                        default="None")
    parser.add_argument('--filtered_snp_file', type=str, help="File that contains SNPs filtered",
                        default="None")

    # Hyper parameters
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--out_reg', type=float, default=0.005, help='l2_reg_lambda')
    parser.add_argument('--corr_reg', type=float, default=0.5, help='l2_corr_lambda')

    # First transparent layer
    parser.add_argument('--model_type', type=str, default='drop_connect',
                        help='Drop Connect vs FCN vs both for the first transparent layer')
    parser.add_argument('--latent_dim', type=int, default=500,
                        help='Number of dimensions for the latent space to be reduced.')
    # FCN
    parser.add_argument('--num_fc_layers', type=int, default=1,
                        help='Number of fully connected layers to be used after convolution layer')
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
    parser.add_argument('--cross_validate', type=bool, default=False,
                        help='Choose normal validation vs cross-validation')
    #Model training
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--stagnant', type=int, default=50, help='Early stop criteria')
    parser.add_argument('--direction', type=str, default='genetosnp',
                        help='snptogene vs genetosnp')

    # Model save paths and others
    parser.add_argument('--save', type=str, default="model/try/",
                        help="path to save model")
    parser.add_argument('--cell_type', type=str, default='bulk', help='Cell type')

    args = parser.parse_args()
    train_deepdice_mv(args)

if __name__ == '__main__':
    main()
