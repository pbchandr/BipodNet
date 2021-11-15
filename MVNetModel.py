#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:05:21 2021
@author: pramod
"""

import sys
import tensorflow as tf

class MVNet(object):
    """DropConenct Deep Learning integrative model"""

    @classmethod
    def weight_variable(cls, shape, var_name, random=False):
        """Function to create a weights"""
        if random:
            initial = tf.truncated_normal(shape, stddev=0.1)
        else:
            initial = tf.zeros(shape)
        return tf.Variable(initial, name=var_name+'_wt')

    @classmethod
    def bias_variable(cls, shape, var_name, random=False):
        """Function to create a bias based on the shape"""
        if random:
            initial = tf.random_normal(shape)
        else:
            initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=var_name+'_bias')

    @classmethod
    def fc_layer(cls, inp, shape, var_name, random=False):
        """Function for creating fully connected network"""
        fc_weight = cls.weight_variable(shape, var_name, random)
        #fc_weight = tf.Print(fc_weight, [fc_weight], var_name + "fc_weight: ")
        fc_bias = cls.bias_variable([shape[1]], var_name, random)
        #fc_bias = tf.Print(fc_bias, [fc_bias], var_name + "fc_bias: ")
        fc_mul_wb = tf.add(tf.matmul(inp, fc_weight), fc_bias)
        return fc_mul_wb, fc_weight

    @classmethod
    def dc_layer(cls, inp, adj, shape, var_name, random=False):
        """Function for creating Drop Connect layer"""
        dc_weight = cls.weight_variable(shape, var_name, random)
        #dc_weight = tf.Print(dc_weight, [dc_weight], var_name + "_dc_weight: ")
        dc_bias = cls.bias_variable([shape[1]], var_name, random)
        #dc_bias = tf.Print(dc_bias, [dc_bias], var_name + "_dc_bias: ")
        dc_mul_wb = tf.add(tf.matmul(inp, tf.multiply(dc_weight, adj)), dc_bias)
        return dc_mul_wb

    @classmethod
    def parameter_checks(cls, args):
        """Function to check parameter dimension checks"""

        # FCN dimenstion check
        fc_num_layers = int(args.num_fc_layers)
        fc_num_neurons = [int(x) for x in args.num_fc_neurons.split(',')]

        if len(fc_num_neurons) != fc_num_layers:
            print("FC ERROR: Number of layers and number of neurons do not match")
            sys.exit(1)
        print("FFN dimenston check passed")

        return True

    def __init__(self, args):
        """Main CNN model function"""
        # Parse hyper-parameters
        if self.parameter_checks(args):
            fc_num_layers = int(args.num_fc_layers)
            fc_num_neurons = [int(x) for x in args.num_fc_neurons.split(',')]
        else:
            sys.exit(1)

        # Input layer: Input data & droupouts
        self.dm1 = tf.placeholder(tf.float32, [None, args.dm_nrow1], name="input_data_1")
        self.dm2 = tf.placeholder(tf.float32, [None, args.dm_nrow2], name="input_data_2")
        self.adj1 = tf.placeholder(tf.float32, [args.dm_nrow1, args.dm_ncol1], name="adj_data_1")
        self.adj2 = tf.placeholder(tf.float32, [args.dm_nrow2, args.dm_ncol2], name="adj_data_2")
        self.input_y = tf.placeholder(tf.float32, [None, args.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.estimate = tf.placeholder(tf.string, shape=(None), name="modal_estimation")
        

        # Dropconenct masking layer
        self.dc_out1 = self.dc_layer(self.dm1, self.adj1,
                                     [args.dm_nrow1, args.dm_ncol1], 'dm1')
        #dc_out1 = tf.Print(dc_out1, [dc_out1], "dc_out1: ")
        
        # We want dc_out1 in terms of dc_out2
        dc_corr_wt = self.weight_variable(shape=[self.dc_out1.shape[1]], var_name="dc_corr")
        dc_corr_bias = self.bias_variable(shape=[self.dc_out1.shape[1]], var_name="dc_corr")
        dc_corr_out = tf.multiply(dc_corr_wt, self.dc_out1) + dc_corr_bias
        
        dc_corr_out1 = self.dc_layer(self.dm2, self.adj2, [args.dm_nrow2, args.dm_ncol2], 'dm2')
        self.dc_out2 = tf.cond(tf.equal(self.estimate, 'true'),
                               lambda: dc_corr_out,
                               lambda: dc_corr_out1)

        #dc_out2 = tf.Print(dc_out2, [dc_out2], "dc_out2: ")
        dc_output = tf.concat([self.dc_out1, self.dc_out2], 1)
        dc_output = tf.nn.relu(dc_output)
        #dc_output = tf.Print(dc_output, [dc_output], "dc_output: ")
        self.dc_output = dc_output

        dc_output_drop = tf.nn.dropout(dc_output, keep_prob=self.dropout_keep_prob,
                                         name='dc_output_drop')
        print("dc_output_drop", dc_output_drop)

        # Dense fully connected layer
        self.out_wt_loss = 0.0
        fc_out = []
        for i in range(fc_num_layers):
            with tf.variable_scope('fc-layer-%d'%i):
                if i == 0:
                    n_1 = int(dc_output_drop.shape[1])
                    n_2 = fc_num_neurons[i]
                    fcn_ip_layer = dc_output_drop
                else:
                    n_1 = fc_num_neurons[(i-1)]
                    n_2 = fc_num_neurons[i]
                    fcn_ip_layer = fc_out
                fc_shape = [n_1, n_2]
                print('fc_shape-%d'%(i), fc_shape)
                fc_out, fc_weight = self.fc_layer(fcn_ip_layer, fc_shape, 'fc-%d-'%i, False)
                fc_out = tf.nn.dropout(fc_out, keep_prob=self.dropout_keep_prob, name='fc_dropout_%d'%i)
                #fc_out = tf.Print(fc_out, [fc_out], "fc_out_%d: "%i)
                #fc_weight = tf.Print(fc_weight, [fc_weight], "fc_weight_%d: "%i)
                self.out_wt_loss += tf.nn.l2_loss(fc_weight)
                print('fc_layer', fc_out)
                print('fc_weight_shape', fc_weight.shape)

                if fc_num_layers >= 1 and i < (fc_num_layers - 1):
                    fc_out = tf.nn.relu(fc_out)
                #fc_out = tf.Print(fc_out, [fc_out], "fc_out AFTER: ")

        self.fc_out_layer = fc_out

        # Final prediction scores
        with tf.variable_scope("output"):
            output_weight = tf.get_variable(shape=[fc_num_neurons[-1], args.num_classes],
                                            initializer=tf.contrib.layers.xavier_initializer(),
                                            name="out_wt")
            print("fc_out before: ", fc_out)
            #output_weight = tf.Print(output_weight, [output_weight], "output_weight: ")
            output_bias = tf.Variable(tf.constant(0.1, shape=[args.num_classes]), name="out_b")
            #output_bias = tf.Print(output_bias, [output_bias], "output_bias: ")
            matmul = tf.nn.xw_plus_b(fc_out, output_weight, output_bias, name="matmul")
            #matmul = tf.Print(matmul, [matmul], "matmul: ")

            self.scores = matmul

            print('output_weight_shape', output_weight)
            print(self.scores.shape)

        self.out_wt_loss += tf.nn.l2_loss(output_weight)
        self.out_wt_loss = args.out_reg_lambda * self.out_wt_loss
        self.dc_corr_loss = tf.losses.mean_squared_error(self.dc_out2, dc_corr_out)

        # Calculate losses - cross-entropy loss for classification/ mse for regression
        with tf.variable_scope("loss"):
            if args.task == 'regression':
                self.error = tf.losses.mean_squared_error(self.input_y, self.scores)
            else:
                self.error = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                     labels=self.input_y)
        self.loss = tf.reduce_mean(self.error) + self.out_wt_loss
        self.loss +=  args.corr_reg_lambda * self.dc_corr_loss

        # Optimization
        with tf.variable_scope("optimize"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learn_rate).minimize(self.loss)
        print("optimizer", self.optimizer)
        
        # Compute Gradient
        with tf.variable_scope('gradient'):
            self.gradient_step =  tf.train.AdamOptimizer(learning_rate=args.learn_rate).compute_gradients(self.scores,
                                                                                                          [self.dm1,self.dm2])
        print('gradient',self.gradient_step)
        
