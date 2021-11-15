#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:07:26 2021

@author: pramod
"""
import argparse
import os
import time
import random
import tensorflow as tf
import numpy as np
import sklearn.metrics as skm
from sklearn.model_selection import  StratifiedKFold, train_test_split
from sklearn import preprocessing
import MVNetUtils as ut
from MVNetModel import MVNet


def get_binary_perfromance(y_true, y_pred):
    """Function to return the precision, recall, fscore, and accuracy"""
    precision, recall, fscore, _ = skm.precision_recall_fscore_support(y_true, y_pred,
                                                                       average="weighted")
    acc = skm.accuracy_score(y_true, y_pred)
    bacc = skm.balanced_accuracy_score(y_true, y_pred)
    return precision, recall, fscore, acc, bacc

def eval_step(y_true, y_pred_scores, verbose):
    """Function to evaluate the algorithm"""
    y_pred = np.argmax(y_pred_scores, axis=1)
    y_true = np.argmax(y_true, axis=1)

    precision, recall, fscore, acc, bacc = get_binary_perfromance(y_true, y_pred)
    auc = skm.roc_auc_score(y_true, y_pred_scores[:, 1])
    print("P: %.5f, R: %.5f, F1: %.5f, ACC: %.5f, AUC: %.5f, BACC: %.5f"%(precision, recall,
                                                                          fscore, acc, auc, bacc))
    print(skm.confusion_matrix(y_true, y_pred))
    return bacc, auc

def predict(sess, model, eval_inp1, eval_inp2, adj1, adj2, args, verbose, estimate=False):
    """Function to get the predictions based on the built model"""

    print("----------%s----------"%(verbose))

    predictions = []
    dc_out2_all = []
    avg_corr_err = 0.0
    total_batch = int(len(eval_inp1)/args.batch_size)

    flag = 'true' if estimate else 'false'
    #print(verbose, flag)
    for ptr in range(0, len(eval_inp1), args.batch_size):
        pred, dc_out2, corr_err = sess.run([model.scores, model.dc_out2, model.dc_corr_loss],
                                           feed_dict={model.dm1: eval_inp1[ptr:ptr+args.batch_size],
                                                      model.dm2: eval_inp2[ptr:ptr+args.batch_size],
                                                      model.adj1: adj1,
                                                      model.adj2: adj2,
                                                      model.dropout_keep_prob: 1.0,
                                                      model.estimate: flag})
        
        predictions.extend(pred)
        dc_out2_all.extend(dc_out2)
        if estimate:
            avg_corr_err += corr_err/total_batch

    predictions = np.asarray(predictions)
    dc_out2_all = np.asarray(dc_out2_all)
    if estimate:
        print("Corr Error: %.5f"%(avg_corr_err))

    return predictions

def train_step(sess, model, tr_inp1, tr_inp2, adj1, adj2, tr_lbl, args):
    """Training step involving optimizing the weights based on the data"""
    avg_cost, error, avg_corr_err = 0.0, 0.0, 0.0
    keep_prob = 1 - args.dropout_rate
    total_batch = int(len(tr_inp1)/args.batch_size)

    for ptr in range(0, len(tr_inp1), args.batch_size):
        cost, err, corr_err, _ = sess.run([model.loss, model.error, model.dc_corr_loss, model.optimizer],
                                          feed_dict={model.dm1: tr_inp1[ptr:ptr+args.batch_size],
                                                     model.dm2: tr_inp2[ptr:ptr+args.batch_size],
                                                     model.adj1: adj1,
                                                     model.adj2: adj2,
                                                     model.input_y: tr_lbl[ptr:ptr+args.batch_size],
                                                     model.dropout_keep_prob: keep_prob,
                                                     model.estimate: 'false'})
        # Compute average loss across batches
        avg_cost += cost/total_batch
        error += np.mean(err)/total_batch
        avg_corr_err += corr_err/total_batch
        

    #print ("Error: %.3f, "%(error))
    return avg_cost, avg_corr_err


def run_split_train(inp1, inp2, adj1, adj2, labels, args):
    """ Function to run normal training"""
    lbls = np.argmax(labels, 1)
    pos_idx = list(np.where(lbls == 1)[0])
    neg_idx = list(np.where(lbls == 0)[0])

    if len(pos_idx) > len(neg_idx):
        pos_tr_idx = random.sample(pos_idx, round(args.train_percent*len(neg_idx)))
        neg_tr_idx = random.sample(neg_idx, round(args.train_percent*len(neg_idx)))
    else:
        pos_tr_idx = random.sample(pos_idx, round(args.train_percent*len(pos_idx)))
        neg_tr_idx = random.sample(neg_idx, round(args.train_percent*len(pos_idx)))
    tridx = pos_tr_idx + neg_tr_idx
    tridx.sort()
    validx = list(set(range(labels.shape[0])) - set(tridx))

    X1_tr , X1_val = inp1.values[tridx,:], inp1.values[validx,:]
    X2_tr , X2_val = inp2.values[tridx,:], inp2.values[validx,:]
        
    X1_tr = preprocessing.StandardScaler().fit_transform(X1_tr)
    X2_tr = preprocessing.StandardScaler().fit_transform(X2_tr)
    X1_val = preprocessing.StandardScaler().fit_transform(X1_val)
    X2_val = preprocessing.StandardScaler().fit_transform(X2_val)
    
    y_tr, y_val = labels[tridx, :] , labels[validx, :]
    
    # Step 1: Create model
    st_time = time.time()
    model = MVNet(args)
    tot_time = round((time.time() - st_time)/60, 2)
    print("Step 1: Model object create completed in %.2f min\n"%(tot_time))
        
    # Step 2: Build the graph and train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # Train cycle
        max_tr_acc, max_val_acc, count_eval = 0.0, 0.0, 0
        max_tr_auc, max_val_auc = 0.0, 0.0
        max_val_est_acc, max_val_est_auc = 0.0, 0.0
            
        for epoch in range(args.train_epochs):
            output_str = "\nEpoch: %02d, "%(epoch+1)
            model_save = False

            st_time = time.time()
            avg_cost, avg_corr_err = train_step(sess, model, X1_tr, X2_tr, adj1, adj2, y_tr, args)
            tot_time = round((time.time() - st_time)/60, 2)

            output_str += "Total Train Time: %.2f"%(tot_time)
            print("%s, cost: %.5f, avg_corr_err: %.5f"%(output_str, avg_cost, avg_corr_err))

            tr_pred = predict(sess, model, X1_tr, X2_tr, adj1, adj2, args, verbose='training')
            tr_bacc, tr_auc = eval_step(y_tr, tr_pred, verbose='Training')

            val_pred = predict(sess, model, X1_val, X2_val, adj1,
                               adj2, args, verbose='validation')
            val_bacc, val_auc = eval_step(y_val, val_pred, verbose='Validation')

            val_pred_est = predict(sess, model, X1_val, X2_val, adj1, adj2,
                                   args, verbose='validation', estimate=True)
            val_est_bacc, val_est_auc = eval_step(y_val, val_pred_est,
                                                  verbose='Validation Estimation')
            if epoch == 0:
                max_tr_acc,  max_tr_auc = tr_bacc, tr_auc
                max_val_acc,  max_val_auc = val_bacc, val_auc
                max_val_est_acc, max_val_est_auc = val_est_bacc, val_est_auc
                model_save = True
                print('Max_tr: %.4f, Max_val: %.4f, Max_val_est: %.4f'%(max_tr_acc, max_val_acc,
                                                                        max_val_est_acc))
            else:
                val_imp =  val_bacc - max_val_acc
                tr_imp = tr_bacc - max_tr_acc

                #if tr_imp >= 0 and val_imp >= 0:
                if val_imp >= 0:
                    model_save = True
                    count_eval = 0
                    max_tr_acc,  max_tr_auc = tr_bacc, tr_auc
                    max_val_acc,  max_val_auc = val_bacc, val_auc 
                    max_val_est_acc, max_val_est_auc = val_est_bacc, val_est_auc
                    print('Max_tr: %.4f, Max_val: %.4f, Max_val_est: %.4f'%(max_tr_acc,
                                                                            max_val_acc,
                                                                            max_val_est_acc))
                else:
                    count_eval += 1
                    print('Max_tr: %.4f, Max_val: %.4f'%(max_tr_acc, max_val_acc))
                    print("Total Evaluation time: %.2f\n"%(round((time.time() - st_time)/60, 2)))

            if model_save is True and args.save is not None:    
                print("Total Evaluation time: %.2f"%(round((time.time() - st_time)/60, 2)))
                print("Saving model to {}\n".format(args.save))
                saver.save(sess, args.save+"/run_tr_val")

            if count_eval >= args.stagnant:
                if epoch < 15:
                    count_eval = 0
                else:
                    break
    
        out_file = args.cell_type + '_perf.txt'

        if not os.path.exists(out_file):
            with open(out_file, 'w') as write_fl:
                write_fl.write("Model\tTrain ACC\tTrain AUC\tVal ACC\tVal AUC\tVal Est ACC\tVal Est AUC\n")
                write_fl.close()

        wr_str = "%s\t%.5f\t%5f\t%5f\t%5f\t%5f\t%5f\n" %(args.save, max_tr_acc, max_tr_auc,
                                                        max_val_acc, max_val_auc,
                                                        max_val_est_acc, max_val_est_auc)
        with open(out_file, 'a') as write_fl:
            write_fl.write(wr_str)
            write_fl.close()

        print('*** Train Validation Split ***')

        print("Train - ACC: %.5f, AUC: %.5f" %(max_tr_acc, max_tr_auc))        
        print("Val- ACC: %.5f, AUC: %.5f" %(max_val_acc, max_val_auc))
        print("Val Est - ACC: %.5f, AUC: %.5f" %(max_val_est_acc, max_val_est_auc))

    print("Optimization Finished!")


def run_cv_train(inp1, inp2, adj1, adj2, labels, args):
    """ Function to run cross validation modelling"""
    cv_k = 5
    rnd_seed = random.randint(1, 9999999)
    kfl = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=rnd_seed)
    
    cntr = 1
    tr_acc_sc, tr_auc_sc, val_acc_sc, val_auc_sc = [], [], [], []
    val_acc_est_sc, val_auc_est_sc = [], []
    
    print("Random Seed = %d"%(rnd_seed))
    
    for tridx, teidx in kfl.split(inp1, np.argmax(labels,1)):
        print("********** Run %d **********"%(cntr))

        tf.reset_default_graph()

        X1_tr , X1_val = inp1.values[tridx,:], inp1.values[teidx,:]
        X2_tr , X2_val = inp2.values[tridx,:], inp2.values[teidx,:]
            
        X1_tr = preprocessing.StandardScaler().fit_transform(X1_tr)
        X2_tr = preprocessing.StandardScaler().fit_transform(X2_tr)
        X1_val = preprocessing.StandardScaler().fit_transform(X1_val)
        X2_val = preprocessing.StandardScaler().fit_transform(X2_val)
        
        y_tr, y_val = labels[tridx, :] , labels[teidx, :]
        
        # Step 1: Create model
        st_time = time.time()
        model = MVNet(args)
        tot_time = round((time.time() - st_time)/60, 2)
        print("Step 2: Model object create completed in %.2f min\n"%(tot_time))
        
        # Step 2: Build the graph and train the model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # Train cycle
            max_tr_acc, max_val_acc, count_eval = 0.0, 0.0, 0
            max_tr_auc, max_val_auc = 0.0, 0.0
            max_val_est_acc, max_val_est_auc = 0.0, 0.0
            
            for epoch in range(args.train_epochs):
                output_str = "\nEpoch: %02d, "%(epoch+1)
                model_save = False

                st_time = time.time()
                avg_cost, avg_corr_err = train_step(sess, model, X1_tr, X2_tr,
                                                    adj1, adj2, y_tr, args)
                tot_time = round((time.time() - st_time)/60, 2)

                output_str += "Total Train Time: %.2f"%(tot_time)
                print("%s, cost: %.5f, avg_corr_err: %.5f"%(output_str, avg_cost, avg_corr_err))
    
                tr_pred = predict(sess, model, X1_tr, X2_tr, adj1, adj2, args, verbose='training')
                tr_bacc, tr_auc = eval_step(y_tr, tr_pred, verbose='Training')

                val_pred = predict(sess, model, X1_val, X2_val, adj1,
                                   adj2, args, verbose='validation')
                val_bacc, val_auc = eval_step(y_val, val_pred, verbose='Validation')

                val_pred_est = predict(sess, model, X1_val, X2_val, adj1, adj2,
                                       args, verbose='validation', estimate=True)
                val_est_bacc, val_est_auc = eval_step(y_val, val_pred_est,
                                                      verbose='Validation Estimation')
                if epoch == 0:
                    max_tr_acc,  max_tr_auc = tr_bacc, tr_auc
                    max_val_acc,  max_val_auc = val_bacc, val_auc
                    max_val_est_acc, max_val_est_auc = val_est_bacc, val_est_auc
                    model_save = True
                    print('Max_tr:%.4f, Max_val:%.4f'%(max_tr_acc, max_val_acc))
                else:
                    val_imp =  val_bacc - max_val_acc
                    tr_imp = tr_bacc - max_tr_acc

                    #if tr_imp >= 0 and val_imp >= 0:
                    if val_imp >= 0:
                        model_save = True
                        count_eval = 0
                        max_tr_acc,  max_tr_auc = tr_bacc, tr_auc
                        max_val_acc,  max_val_auc = val_bacc, val_auc 
                        max_val_est_acc, max_val_est_auc = val_est_bacc, val_est_auc
                        print('Max_tr: %.4f, Max_val: %.4f'%(max_tr_acc, max_val_acc))
                    else:
                        count_eval += 1
                        print('Max_tr: %.4f, Max_val: %.4f'%(max_tr_acc, max_val_acc))
                        print("Total Evaluation time: %.2f\n"%(round((time.time() - st_time)/60, 2)))
    
                if model_save is True and args.save is not None:    
                    print("Total Evaluation time: %.2f"%(round((time.time() - st_time)/60, 2)))
                    print("Saving model to {}\n".format(args.save))
                    saver.save(sess, args.save+"/run_%d"%cntr)

                if count_eval >= args.stagnant:
                    if epoch < 15:
                        count_eval = 0
                    else:
                        break
            
            tr_acc_sc.append(max_tr_acc)
            tr_auc_sc.append(max_tr_auc)
            val_acc_sc.append(max_val_acc)
            val_auc_sc.append(max_val_auc)
            val_acc_est_sc.append(max_val_est_acc)
            val_auc_est_sc.append(max_val_est_auc)
            
            cntr += 1
            print("Run %d complete"%cntr)
    
    print('***Cross Validation***')
    out_file = args.cell_type + '_perf.txt'

    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write("Model\tTrain ACC\tTrain AUC\tVal ACC\tVal AUC\tVal Est ACC\tVal Est AUC\n")
            write_fl.close()

    wr_str = "%s\t%.5f +/- %.5f\t%5f +/- %.5f" %(args.save, np.mean(tr_acc_sc), np.std(tr_acc_sc),
                                                 np.mean(tr_auc_sc), np.std(tr_auc_sc))

    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f" %(np.mean(val_acc_sc), np.std(val_acc_sc),
                                                 np.mean(val_auc_sc), np.std(val_auc_sc))
                                                   
    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f\n" %(np.mean(val_acc_est_sc), np.std(val_acc_est_sc),
                                                   np.mean(val_auc_est_sc), np.std(val_auc_est_sc))
                                                   

    with open(out_file, 'a') as write_fl:
        write_fl.write(wr_str)
        write_fl.close()

    print("Train- ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(tr_acc_sc), np.std(tr_acc_sc),
                                                            np.mean(tr_auc_sc), np.std(tr_auc_sc)))
    print("Val- ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(val_acc_sc), np.std(val_acc_sc),
                                                          np.mean(val_auc_sc), np.std(val_auc_sc)))
    print("Val Est- ACC: %.5f +/- %.5f, AUC: %.5f +/- %.5f" %(np.mean(val_acc_est_sc),
                                                              np.std(val_acc_est_sc),
                                                              np.mean(val_auc_est_sc),
                                                              np.std(val_auc_est_sc)))

def train_data(args):
    """ Training method"""

    if ut.file_check(args):
        print("File check passed")

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Load and preprocess data
    st_time = time.time()
    inp, adj, labels = ut.get_csv_data(args.input_files, args.intermediate_phenotype_files,
                                          args.disease_label_file)

    tr_idx, te_idx = train_test_split(inp[0].index.values, test_size=0.10, random_state=1)
    
    
    #te_idx.tofile(args.save+'/test_samples.txt', '\n', '%s')    

    idx = np.where(inp[0].index.isin(tr_idx))[0]
    tr_inp1 = inp[0].loc[inp[0].index.isin(tr_idx), ]
    tr_inp2 = inp[1].loc[inp[1].index.isin(tr_idx), ]
    
    
    with open(args.save+'/train_samples.out', 'w') as output:
        for row in tr_idx:
            output.write(str(row) + '\n')
    
    with open(args.save+'/test_samples.out', 'w') as output:
        for row in te_idx:
            output.write(str(row) + '\n')
    
    
    

    args.dm_nrow1, args.dm_ncol1 = adj[0].shape
    args.dm_nrow2, args.dm_ncol2 = adj[1].shape

    args.num_classes = 2
    if args.task == 'classification':
        label_train = np.eye(len(np.unique(labels[idx])))[np.int32(labels[idx])]
        tru, trc = np.unique(np.argmax(label_train, 1), return_counts=True)
        print("Train label split ---", dict(zip(tru, trc)))
        args.num_classes = label_train.shape[1]

    elif args.task == 'regression':
        label_train = labels[idx]
        args.num_classes = 1

    tot_time = round((time.time() - st_time)/60, 2)
    print("Data fetch & split completed in %.2f min\n"%(tot_time))

    # Cross Validation
    if args.cross_validate:
        run_cv_train(tr_inp1, tr_inp2, adj[0], adj[1], label_train, args)
    else:
        run_split_train(tr_inp1, tr_inp2, adj[0], adj[1], label_train, args)



def main():
    """ Main: This method is used to parse all the arguements and call train function """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--num_data_modal', type=int, default=2,
                        help='Path to the input data file')
    parser.add_argument('--input_files', type=str,
                        default='bulk/bulk_dlpfc_gex_bio_feat.csv,bulk/bulk_dlpfc_geno_bio_feat.csv',
                        help='Comma separated input data paths')
    parser.add_argument('--intermediate_phenotype_files', type=str,
                        default='bulk/bulk_dlpfc_grn_adj_bio_feat.npz,bulk/bulk_dlpfc_eqtl_adj_bio_feat.npz',
                        help='Path to transparent layer adjacency matrix')
    parser.add_argument('--disease_label_file', type=str,
                        default='bulk/bulk_dlpfc_labels_bio_feat.csv',
                        help='Path to Output labels file - Disease phenotypes')
    
    #parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--task', type=str, default='classification', help='Choose task type')
    parser.add_argument('--cross_validate', type=bool, default=False, help='Choose normal validation vs cross-validation')


    parser.add_argument('--need_balance', type=bool, default=False, help='balanced_training')

    parser.add_argument('--train_percent', type=float, default=0.8,
                        help='Choose how the tain and testvalidation split to occur.')
    parser.add_argument('--cell_type', type=str, default='bulk', help='Cell type')

    # 2 - FCN
    parser.add_argument('--num_fc_layers', type=int, default=2,
                        help='Number of fully connected layers to be used after convolution layer')
    parser.add_argument('--num_fc_neurons', type=str, default='500,100',
                        help='Number of kernels for fully connected layers, comma delimited.')
    parser.add_argument('--fc_dropout_prob', type=float, default=0.5,
                        help='Droupout % for handling overfitting. 1 to keep all & 0 to keep none')

    # Settings
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size of training')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--out_reg_lambda', type=float, default=0.005, help='l2_reg_lambda')
    parser.add_argument('--corr_reg_lambda', type=float, default=0.5, help='l2_reg_lambda')

    parser.add_argument('--stagnant', type=int, default=50, help='Early stop criteria')
    parser.add_argument('--dropout_rate', type=float, default=0.75,
                        help='Dropout % for handling overfit. 1 to keep all & 0 to keep none')
    # Model save paths
    parser.add_argument('--save', type=str, default="model/bulk_try",
                        help="path to save model")

    args = parser.parse_args()
    print(args)
    train_data(args)

if __name__ == '__main__':
    main()
