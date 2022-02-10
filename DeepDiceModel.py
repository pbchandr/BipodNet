#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:37:43 2022

@author: pramod
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    """ Class for Feed Forward Network"""
    def __init__(self, input_dim, hidden_dim_array, dropout_keep_prob):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_layers = len(hidden_dim_array)
        for idx in range(self.hidden_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim_array[idx]))
            if self.hidden_layers >= 1 and idx < (self.hidden_layers - 1):
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(1-dropout_keep_prob))
            input_dim = hidden_dim_array[idx]

    def forward(self, inp):                    
        for layer in self.layers:
            inp = layer(inp)
        return inp

class DropConnect(nn.Module):
    """ Class for performing drop-connection"""
    def __init__(self, in_features, out_features, adj):
        super(DropConnect, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        #nn.init.zeros_(self.weight)
        #nn.init.xavier_normal_(self.weight)
        nn.init.kaiming_uniform_(self.weight)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return input.matmul(self.weight.t() * self.adj) + self.bias
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

class DeepDiceMV(nn.Module):
    def __init__(self, args):
        super(DeepDiceMV, self).__init__()
        self.dc1, self.dc2 = None, None
        self.lc1, self.lc2 = None, None
        
        num_dc_out = 2*args.n_genes
        if args.model_type == 'drop_connect':
            self.dc1 = DropConnect(args.p_snps, args.n_genes, args.eqtl)
            self.dc2 = DropConnect(args.m_tfs, args.n_genes, args.grn)
        elif args.model_type == 'fully_connected':
            self.dc1 = nn.Linear(args.p_snps, args.n_genes)
            self.dc2 = nn.Linear(args.m_tfs, args.n_genes)
        elif args.model_type == 'combined':
            self.lc1 = DropConnect(args.p_snps, args.n_genes, args.eqtl)
            self.lc2 = DropConnect(args.m_tfs, args.n_genes, args.grn)
            self.dc1 = nn.Linear(args.n_genes, args.latent_dim)
            self.dc2 = nn.Linear(args.n_genes, args.latent_dim)
            num_dc_out = 2*args.latent_dim

        self.fc = FNN(num_dc_out, args.num_fc_neurons, args.dropout_keep_prob)
        self.pred = nn.Linear(args.num_fc_neurons[-1], args.n_out)
        self.dropout = nn.Dropout(1-args.dropout_keep_prob)
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.alpha)
        if self.beta is not None:
            nn.init.constant_(self.beta, 0.1)

    def forward(self, snp_dos, gene_exp, estimate):
        if self.lc1 is not None:
            Cs_int = self.lc1(snp_dos)
            Cg_int = self.lc2(gene_exp)
            Cs = self.dc1(Cs_int)
            Cg = self.dc2(Cg_int)
        else:
            Cs = self.dc1(snp_dos)
            Cg = self.dc2(gene_exp)

        Cg_est = (self.alpha * Cs) + self.beta
        Cs_est = (Cg - self.alpha)/self.beta
        
        Cs_out, Cg_out = 0, 0
        if estimate == 'None':
            #print("estimate")
            Cs_out, Cg_out = Cs, Cg
        elif estimate == 'cg':
            Cs_out = Cs
            Cg_out = Cg_est
        elif estimate == 'cs':
            Cs_out = Cs_est
            Cg_out = Cg

        dc_out = torch.cat([Cs_out, Cg_out], dim=1)
        dc_out = self.dropout(dc_out.relu())
        fc_out = self.fc(dc_out)
        pred = self.pred(fc_out).sigmoid()
        return pred
    
    def get_intermediate_layers(self, snp_dos, gene_exp):
        if self.lc1 is not None:
            Cs_int = self.lc1(snp_dos)
            Cg_int = self.lc2(gene_exp)
            Cs = self.dc1(Cs_int)
            Cg = self.dc2(Cg_int)
        else:
            Cs = self.dc1(snp_dos)
            Cg = self.dc2(gene_exp)

        Cg_est = (self.alpha * Cs) + self.beta
        Cs_est = (Cg - self.alpha)/self.beta
        return Cs, Cs_est, Cg, Cg_est
        

class DeepDiceSV(nn.Module):
    def __init__(self, args):
        super(DeepDiceSV, self).__init__()
        self.dc = None
        if args.model_type == 'drop_connect':
            self.dc = DropConnect(args.p_feat, args.n_genes, args.adj)
        else:
            self.dc = nn.Linear(args.p_feat, args.n_genes)

        self.fc = FNN(args.n_genes, args.num_fc_neurons, args.dropout_keep_prob)
        self.pred = nn.Linear(args.num_fc_neurons[-1], args.n_out)
        self.dropout = nn.Dropout(1-args.dropout_keep_prob)
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.alpha)
        if self.beta is not None:
            nn.init.constant_(self.beta, 0.1)

    def forward(self, inp_data):
        dc_out = self.dc(inp_data)
        dc_out = self.dropout(dc_out.relu())
        fc_out = self.fc(dc_out)
        pred = self.pred(fc_out).sigmoid()
        return pred
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:15:12 2022

@author: pramod
"""

