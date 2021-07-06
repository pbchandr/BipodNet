#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pbchandr
"""

import torch
import torch.nn as nn

class DropConenct(nn.Module):
    def __init__(self, in_features, out_features):
        super(DropConenct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.normal_(self.bias)
        
    def forward(self, input, adj):
        return input.matmul(self.weight.t() * adj) + self.bias

class MVNet(nn.Module):
    def __init__(self, args):
        super(MVNet, self).__init__()
        self.grn_adj = args.grn_adj
        self.eqtl_adj = args.eqtl_adj
        self.dc1 = DropConenct(args.n_gex, args.n_gex_genes)
        self.dc2 = DropConenct(args.n_snp, args.n_snp_genes)
        self.fc1 = nn.Linear(args.n_gex_genes+args.n_snp_genes, args.nfc1)
        self.pred = nn.Linear(args.nfc1, args.n_out)
        self.dropout = nn.Dropout(1-args.dropout_keep_prob)
    def forward(self, gex_inp, snp_inp):
        out1 = self.dc1(gex_inp, self.grn_adj).relu()
        out2 = self.dc2(snp_inp, self.eqtl_adj).relu()
        out = torch.cat([out1, out2], dim=1)
        out = self.fc1(out).relu()
        out = self.dropout(out)
        out = self.pred(out).sigmoid()
        return out

class MVNet2(nn.Module):
    def __init__(self, args):
        super(MVNet2, self).__init__()
        self.grn_adj = args.grn_adj
        self.eqtl_adj = args.eqtl_adj
        self.dc1 = DropConenct(args.n_gex, args.n_gex_genes)
        self.dc2 = DropConenct(args.n_snp, args.n_snp_genes)
        self.fc1 = nn.Linear(args.n_gex_genes+args.n_snp_genes, args.nfc1)
        self.fc2 = nn.Linear(args.nfc1, args.nfc2)
        self.pred = nn.Linear(args.nfc2, args.n_out)
        self.dropout = nn.Dropout(1-args.dropout_keep_prob)
    def forward(self, gex_inp, snp_inp):
        out1 = self.dc1(gex_inp, self.grn_adj).relu()
        out2 = self.dc2(snp_inp, self.eqtl_adj).relu()
        out = torch.cat([out1, out2], dim=1)
        out = self.fc1(out).relu()
        out = self.dropout(out)
        out = self.fc2(out).relu()
        out = self.dropout(out)
        out = self.pred(out).sigmoid()
        return out

class MVNet3(nn.Module):
    def __init__(self, args):
        super(MVNet3, self).__init__()
        self.grn_adj = args.grn_adj
        self.eqtl_adj = args.eqtl_adj
        self.dc1 = DropConenct(args.n_gex, args.n_gex_genes)
        self.dc2 = DropConenct(args.n_snp, args.n_snp_genes)
        self.fc1 = nn.Linear(args.n_gex_genes+args.n_snp_genes, args.nfc1)
        self.fc2 = nn.Linear(args.nfc1, args.nfc2)
        self.fc3 = nn.Linear(args.nfc2, args.nfc3)
        self.pred = nn.Linear(args.nfc3, args.n_out)
        self.dropout = nn.Dropout(1-args.dropout_keep_prob)
    def forward(self, gex_inp, snp_inp):
        out1 = self.dc1(gex_inp, self.grn_adj).relu()
        out2 = self.dc2(snp_inp, self.eqtl_adj).relu()
        out = torch.cat([out1, out2], dim=1)
        out = self.fc1(out).relu()
        out = self.dropout(out)
        out = self.fc2(out).relu()
        out = self.dropout(out)
        out = self.fc3(out).relu()
        out = self.dropout(out)
        out = self.pred(out).sigmoid()
        return out

class MVNet4(nn.Module):
    def __init__(self, args):
        super(MVNet4, self).__init__()
        self.grn_adj = args.grn_adj
        self.eqtl_adj = args.eqtl_adj
        self.dc1 = DropConenct(args.n_gex, args.n_gex_genes)
        self.dc2 = DropConenct(args.n_snp, args.n_snp_genes)
        self.fc1 = nn.Linear(args.n_gex_genes+args.n_snp_genes, args.nfc1)
        self.fc2 = nn.Linear(args.nfc1, args.nfc2)
        self.fc3 = nn.Linear(args.nfc2, args.nfc3)
        self.fc4 = nn.Linear(args.nfc3, args.nfc4)
        self.pred = nn.Linear(args.nfc4, args.n_out)
        self.dropout = nn.Dropout(1-args.dropout_keep_prob)
    def forward(self, gex_inp, snp_inp):
        out1 = self.dc1(gex_inp, self.grn_adj).relu()
        out2 = self.dc2(snp_inp, self.eqtl_adj).relu()
        out = torch.cat([out1, out2], dim=1)
        out = self.fc1(out).relu()
        out = self.dropout(out)
        out = self.fc2(out).relu()
        out = self.dropout(out)
        out = self.fc3(out).relu()
        out = self.dropout(out)
        out = self.fc4(out).relu()
        out = self.dropout(out)
        out = self.pred(out).sigmoid()
        return out
    
class MVNet5(nn.Module):
    def __init__(self, args):
        super(MVNet5, self).__init__()
        self.grn_adj = args.grn_adj
        self.eqtl_adj = args.eqtl_adj
        self.dc1 = DropConenct(args.n_gex, args.n_gex_genes)
        self.dc2 = DropConenct(args.n_snp, args.n_snp_genes)
        self.fc1 = nn.Linear(args.n_gex_genes+args.n_snp_genes, args.nfc1)
        self.fc2 = nn.Linear(args.nfc1, args.nfc2)
        self.fc3 = nn.Linear(args.nfc2, args.nfc3)
        self.fc4 = nn.Linear(args.nfc3, args.nfc4)
        self.fc5 = nn.Linear(args.nfc4, args.nfc5)
        self.pred = nn.Linear(args.nfc5, args.n_out)
        self.dropout = nn.Dropout(1-args.dropout_keep_prob)
    def forward(self, gex_inp, snp_inp):
        out1 = self.dc1(gex_inp, self.grn_adj).relu()
        out2 = self.dc2(snp_inp, self.eqtl_adj).relu()
        out = torch.cat([out1, out2], dim=1)
        out = self.fc1(out).relu()
        out = self.dropout(out)
        out = self.fc2(out).relu()
        out = self.dropout(out)
        out = self.fc3(out).relu()
        out = self.dropout(out)
        out = self.fc4(out).relu()
        out = self.dropout(out)
        out = self.fc5(out).relu()
        out = self.dropout(out)
        out = self.pred(out).sigmoid()
        return out