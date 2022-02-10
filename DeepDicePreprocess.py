#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pramod
"""

"""
 - Modify lines 19-27 to point to the locations of the input files and cell type.
 - Modify lines 120, 123, 126, 129, 134, 140, 143,146, 147 
   to point to the dir where the output files are to be written.
"""
import pandas as pd
import networkx as nx
import scipy as sp
import numpy as np
import pickle as pk

ctype = 'glu' # gaba vs mgas vs glu vs olig
feat_type = 'bio_feat' # all_feat vs bio_feat
grn_type = 'noopenchromatin' #openchromatin vs referenceGRN vs noopenchromatin

gex_file = 'processed/' + ctype + '/'+  ctype + '_dlpfc_gene_exp_' + feat_type + '.csv'
snp_file = 'processed/' + ctype + '/' + ctype + '_dlpfc_genotype_' + feat_type + '.csv'
phen_file = 'processed/' + ctype +'/' + ctype + '_dlpfc_labels_' + feat_type + '.csv' 
grn_file = 'processed/' + ctype + '/' + ctype + '_dlpfc_grn_' + feat_type + '.csv'
eqtl_file = 'processed/' + ctype + '/' + ctype + '_dlpfc_eqtl_' + feat_type + '.csv' 

# Read Gene Expression Data
gex = pd.read_csv(gex_file).set_index('gene_id')
print('gex', gex.shape)

# Read SNP information
snps = pd.read_csv(snp_file).set_index('snp_id')
print('snps', snps.shape)

# Read class labels
phen = pd.read_csv(phen_file)
labels = np.float32(np.copy(phen['label'].values))

# Read eqtl data
eqtl = pd.read_csv(eqtl_file).assign(weight=1)
eqtl = eqtl[['snp_id','gene_id','weight']]
eqtl.columns = ['source','target','weight']
print('eqtl', eqtl.shape)


# Read GRN data
grn = pd.read_csv(grn_file).assign(weight=1)
grn.columns = ['target','source','weight']
print('grn', grn.shape)

# Get the common gene list
grn_genes = list(set(grn.source.values).union(set(grn.target.values)))
eqtl_genes = list(set(eqtl.target.values))
gex_genes = list(gex.index.values)

len(set(gex_genes).intersection(set(eqtl_genes)))
len(set(gex_genes).intersection(set(grn_genes)))
len(set(grn_genes).intersection(set(eqtl_genes)))

# EQTL adjacency matrix
eqtl_idx = eqtl[['source','target']].stack().reset_index(level=[0], drop=True).drop_duplicates().reset_index()
col_idx = eqtl_idx[eqtl_idx['index']=='target'].index.values
row_idx = eqtl_idx[eqtl_idx['index']=='source'].index.values

gene_ls = eqtl_idx[eqtl_idx['index']=='target'][0].tolist()
snp_ls = eqtl_idx[eqtl_idx['index']=='source'][0].tolist()

G = nx.from_pandas_edgelist(eqtl, create_using=nx.DiGraph())
eqtl_adj = nx.adjacency_matrix(G)
eqtl_adj = sp.sparse.csr_matrix(eqtl_adj.tocsr()[row_idx,:][:,col_idx].todense())
eqtl_adj_d = pd.DataFrame(eqtl_adj.todense(), index=snp_ls, columns=gene_ls)
eqtl_adj_d = eqtl_adj_d.loc[:, eqtl_adj_d.columns.isin(gex_genes)]

add_mat_w = len(list(set(gex_genes) - set(eqtl_adj_d.columns)))
add_mat_l = eqtl_adj_d.shape[0]
new_mat = pd.DataFrame(np.zeros((add_mat_l, add_mat_w)), index = snp_ls)

eqtl_adj_final = pd.concat([eqtl_adj_d, new_mat], axis=1)
gene_ls = list(eqtl_adj_d.columns) + list(set(gex_genes) - set(eqtl_adj_d.columns))
eqtl_adj_final.columns  = gene_ls
print('eqtl_adj_final', eqtl_adj_final.shape)

# GRN adjacency matrix
G = nx.from_pandas_edgelist(grn, create_using=nx.DiGraph())
grn_adj = nx.adjacency_matrix(G)
grn_adj_v1 = pd.DataFrame(grn_adj.todense(), index = grn_genes, columns=grn_genes)
print('grn_adj_v1', grn_adj_v1.shape)

rem_genes = list(set(gex_genes) - set(grn_genes))
add_mat_w = len(rem_genes)
add_mat_l = grn_adj_v1.shape[0]
new_mat = pd.DataFrame(np.zeros((add_mat_l, add_mat_w)), index = grn_genes, columns=rem_genes)
grn_adj_v2 = pd.concat([grn_adj_v1, new_mat], axis=1)
print('grn_adj_v2', grn_adj_v2.shape)

add_mat_l = len(rem_genes)
add_mat_w = grn_adj_v2.shape[1]
new_mat = pd.DataFrame(np.zeros((add_mat_l, add_mat_w)),
                       index = rem_genes, columns=grn_adj_v2.columns)
grn_adj_final = pd.concat([grn_adj_v2, new_mat], axis=0)
print('grn_adj_final', grn_adj_final.shape)


gene_ls = list(grn_adj_final.index)
eqtl_adj_final = eqtl_adj_final.loc[:, gene_ls]

eqtl_adj_sparse_mat = sp.sparse.csr_matrix(eqtl_adj_final)
grn_adj_sparse_mat = sp.sparse.csr_matrix(grn_adj_final)
adj_sparse_mat = sp.sparse.vstack([eqtl_adj_sparse_mat, grn_adj_sparse_mat])

print('adj_sparse_mat', adj_sparse_mat.shape)
del eqtl_adj_final, grn_adj_final

# Sort all the data
gex = gex.reindex(gene_ls)
snps = snps.reindex(snp_ls)

input_data = pd.concat([snps, gex])
print('input_data', input_data.shape)

with open('processed/'+ctype+'/'+ctype+'_dlpfc_obs_'+feat_type+'.pkl', 'wb') as obs_fl:
    pk.dump(input_data, obs_fl)    
    
with open('processed/'+ctype+'/'+ctype+'_dlpfc_labels_'+feat_type+'.pkl', 'wb') as obs_fl:
    pk.dump(labels, obs_fl)

sp.sparse.save_npz('processed/'+ctype+'/'+ctype+'_dlpfc_adj_'+feat_type+'.npz', adj_sparse_mat)

# write out gene names
with open('processed/'+ctype+'/'+ctype+'_dlpfc_genes_'+feat_type+'.list','w') as glf:
    for g in list(gene_ls):
        glf.write(g+"\n") 

# write snp list
with open('processed/'+ctype+'/'+ctype+'_dlpfc_snps_'+feat_type+'.list','w') as glf:
    for g in list(snp_ls):
        glf.write(g+"\n") 


# Write each individual data types
gex.to_csv('processed/'+ctype+'/'+ctype+'_dlpfc_gex_'+feat_type+'.csv')
with open('processed/'+ctype+'/'+ctype+'_dlpfc_gex_'+feat_type+'.pkl', 'wb') as obs_fl:
    pk.dump(gex, obs_fl)    
        
snps.to_csv('processed/'+ctype+'/'+ctype+'_dlpfc_geno_'+feat_type+'.csv')
with open('processed/'+ctype+'/'+ctype+'_dlpfc_geno_'+feat_type+'.pkl', 'wb') as obs_fl:
    pk.dump(snps, obs_fl)    

sp.sparse.save_npz('processed/'+ctype+'/'+ctype+'_dlpfc_eqtl_adj_'+feat_type+'.npz', eqtl_adj_sparse_mat)
sp.sparse.save_npz('processed/'+ctype+'/'+ctype+'_dlpfc_grn_adj_'+feat_type+'.npz', grn_adj_sparse_mat)
