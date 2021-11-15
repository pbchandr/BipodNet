#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:59:07 2021
@author: pramod
"""

import sys
import pandas as pd
import numpy as np
import scipy as sp

def file_check(args):
    """ Function to check if all required file locations are provided by the user"""
    input_files = [x for x in args.input_files.split(',')]
    adj_files = [x for x in args.intermediate_phenotype_files.split(',')]

    if any([args.num_data_modal != len(input_files), args.num_data_modal != len(adj_files)]):
        print("Error number of data modes and the corresponding files do not match")
        sys.exit(1)
    return True

def get_csv_data(input_files, intermediate_phenotype_files, label_file):
    """ Read and fetch the data from files"""
    inp_files = [x for x in input_files.split(',')]
    adj_files = [x for x in intermediate_phenotype_files.split(',')]

    inp, adj = [], []

    for i in range(len(inp_files)):
        dm_inp = pd.read_csv(inp_files[i], header=0)#.drop(columns=['Unnamed: 0'])
        dm_inp = dm_inp.set_index(dm_inp.columns[0])

        adj_sp = sp.sparse.load_npz(adj_files[i])
        dm_adj = adj_sp.todense()
        dm_adj[dm_adj == 0] = np.max(dm_adj)/10.0

        inp.append(dm_inp.T)
        adj.append(dm_adj)
        print("dm_inp_%d"%i, dm_inp.shape)
        print("dm_adj_%d"%i, dm_adj.shape)

    lbls = pd.read_csv(label_file, header=0)#.drop(columns=['Unnamed: 0'])
    labels = lbls['label'].values
    print("labels", labels.shape)

    return inp, adj, labels


def get_pickle_data(input_files, intermediate_phenotype_files, label_file):
    """ Read and fetch the data from files"""
    inp_files = [x for x in input_files.split(',')]
    adj_files = [x for x in intermediate_phenotype_files.split(',')]

    inp, adj = [], []

    for i in range(len(inp_files)):
        dm_inp = pd.read_pickle(inp_files[i])

        adj_sp = sp.sparse.load_npz(adj_files[i])
        dm_adj = adj_sp.todense()
        dm_adj[dm_adj == 0] = np.max(dm_adj)/10.0
        inp.append(dm_inp.T)
        adj.append(dm_adj)

        print("dm_inp_%d"%i, dm_inp.shape)
        print("dm_adj_%d"%i, dm_adj.shape)

    labels = pd.read_pickle(label_file)
    print("labels", labels.shape)

    return inp, adj, labels
