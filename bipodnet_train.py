#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pbchandr
"""

import torch
import torch.nn as nn
import pandas as pd
import scipy.sparse as sp
import numpy as np
import argparse
import bipodnet_model as bm

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def load_data(args):
    """ Function to laod all required data """
    lbls = pd.read_pickle(args.label)
    label = lbls.values[4]
    
    if args.add_gene_expression:
        gex_obs = pd.read_csv(args.gex_obs).drop(columns=['Unnamed: 0'])
        grn_adj = sp.load_npz(args.grn_adj)
        if args.add_snp:
            snp_obs = pd.read_csv(args.snp_obs).drop(columns=['Unnamed: 0'])
            eqtl_adj = sp.load_npz(args.eqtl_adj)
            return gex_obs, grn_adj, snp_obs, eqtl_adj, label
        else:
            return gex_obs, grn_adj, label
    else:
        if args.add_snp:
            snp_obs = pd.read_csv(args.snp_obs).drop(columns=['Unnamed: 0'])
            eqtl_adj = sp.load_npz(args.eqtl_adj)
            return snp_obs, eqtl_adj, label

def preprocess(x, y):
    return x.float(), y.int().reshape(-1, 1)

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

def train_epoch(model, dl, args, optimizer):
    correct, total = 0.0, 0.0
    losses = 0.0
    for i, (x_tr, y_tr) in enumerate(dl):
        x_snp = x_tr[:, 0:args.n_snp]
        x_gex = x_tr[:, args.n_snp:]
        
        # Forward pass
        out = model(x_gex, x_snp)    
        loss = args.loss_fn(out, y_tr.float())
        
        # Compute accuracy
        total += y_tr.size(0)
        
        y_pred = out.reshape(-1).detach().numpy().round()
        y_truth = y_tr.detach().numpy().reshape(-1)

        correct += len(np.where(y_truth == y_pred)[0])
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses += loss.item()

    return losses/(i+1), correct/total



def train_bipodnet(args):
    """ Method to fetch the data and perfroming training """
    
    # Step 1: Get data
    gene_obs, grn_adj, snp_obs, eqtl_adj, label = load_data(args)
    args.n_snp, args.n_snp_genes = eqtl_adj.shape
    args.n_gex, args.n_gex_genes = grn_adj.shape
    
    grn_adj = torch.from_numpy(grn_adj.todense()).float()
    eqtl_adj = torch.from_numpy(eqtl_adj.todense()).float()
    args.grn_adj = grn_adj
    args.eqtl_adj = eqtl_adj
    
    obs = pd.concat([snp_obs, gene_obs], axis = 1)
    args.n_out = 1
    
    # Step 2: Make data iterable with batches
    X_train, X_test, y_train, y_test = train_test_split(obs.values, np.reshape(label, (-1, 1)),
                                                        test_size=0.20, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.20, random_state=73)
    X_train, y_train, X_val, y_val = map(torch.tensor, (X_train, y_train, X_val, y_val))
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size*2, shuffle=False)
    
    train_dl = WrappedDataLoader(train_dl, preprocess)
    val_dl = WrappedDataLoader(val_dl, preprocess)
    
    # Step 3: Get model parameters and create model
    fc_num_neurons = [int(x) for x in args.num_fc_neurons.split(',')]
    if args.num_fc_layers == 1:
        args.nfc1 = fc_num_neurons[0]
        model = bm.MVNet(args)
    elif args.num_fc_layers == 2:
        args.nfc1, args.nfc2 = fc_num_neurons[0], fc_num_neurons[1]
        model = bm.MVNet2(args)
    elif args.num_fc_layers == 3:
        args.nfc1, args.nfc2, args.nfc3 = fc_num_neurons[0], fc_num_neurons[1], fc_num_neurons[2]
        model = bm.MVNet3(args)
    elif args.num_fc_layers == 4:
        args.nfc1, args.nfc2 = fc_num_neurons[0], fc_num_neurons[1]
        args.nfc3, args.nfc4 = fc_num_neurons[2], fc_num_neurons[3]
        model = bm.MVNet4(args)
    elif args.num_fc_layers == 5:
        args.nfc1, args.nfc2, args.nfc3 = fc_num_neurons[0], fc_num_neurons[1], fc_num_neurons[2]
        args.nfc4, args.nfc5 = fc_num_neurons[3], fc_num_neurons[4]
        model = bm.MVNet5(args)

    for name, param in model.named_parameters():
        print(name, param.size())
    print(model)
        
    # Step 4: Define the loss function
    loss_fn = nn.BCELoss()
    args.loss_fn = loss_fn

    # Step 5: Initiate Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    
    # Step 6: Train the model
    for epoch in range(args.epochs):
        model.train()
        tr_loss, tr_acc = train_epoch(model, train_dl, args, opt)
        
        
        # Evaluate on test set
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for x_val, y_val in val_dl:
                x_snp = x_val[:, 0:args.n_snp]
                x_gex = x_val[:, args.n_snp:]
                outputs = model(x_gex, x_snp)
                y_pred = outputs.reshape(-1).detach().numpy().round()
                y_truth = y_val.detach().numpy().reshape(-1)
                total += y_val.size(0)
                correct += len(np.where(y_truth == y_pred)[0])

            print ('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}' 
                   .format(epoch+1, args.epochs, tr_loss, tr_acc, correct/total))

def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--add_gene_expression', type=bool, default=True,
                        help='Flag to include gene expression data')
    parser.add_argument('--gex_obs', type=str, default='data/gex_obs.csv',
                        help='Path to gene expression information file')
    parser.add_argument('--grn_adj', type=str, default='data/grn_adj.npz',
                        help='Path to GRN file')
    
    parser.add_argument('--add_snp', type=bool, default=True,
                        help='Flag to include SNP data')
    parser.add_argument('--snp_obs', type=str, default='data/snp_obs.csv',
                        help='Path to genotype dosage information file')
    parser.add_argument('--eqtl_adj', type=str, default='data/eqtl_adj.npz',
                        help='Path to EQTL linking SNP to gene file')
    
    parser.add_argument('--label', type=str, default='data/label_gtex.pkl',
                        help='Path to gen expresison file')
    

    # FCN
    parser.add_argument('--num_fc_layers', type=int, default=2,
                        help='Number of fully connected layers to be used after convolution layer')
    parser.add_argument('--num_fc_neurons', type=str, default='1000,500',
                        help='Number of kernels for fully connected layers, comma delimited.')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Droupout % for handling overfitting. 1 to keep all & 0 to keep none')

    # 
    parser.add_argument('--batch_size', type=int, default=60, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--l1_reg', type=int, default=0.0001, help='L regulationation')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='L2 regulationation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')

    args = parser.parse_args()
    print(args)
    train_bipodnet(args)

if __name__ == '__main__':
    main()


