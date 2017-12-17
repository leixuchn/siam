#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Contains definitions of the network in [1].

  [1] Bertinetto, L., et al. (2016).
      "Fully-Convolutional Siamese Networks for Object Tracking."
      arXiv preprint arXiv:1606.09549.

Typical use:

   import siamese_fc
   with slim.arg_scope(siamese_fc.siamese_fc_arg_scope()):
      net, end_points = siamese_fc.siamese_fc(inputs, is_training=False)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
import sys
sys.path.append('/home/v-chaoqw/MYSFC-ORI/embeddings/')
import libs.deform_conv_op as deform_conv_op




def siamese_fc_arg_scope(weight_decay=5e-4,
                         dropout_keep_prob=0.8,
                         batch_norm_decay=0.9997,
                         batch_norm_epsilon=1e-6,
                         trainable=True,
                         is_training=False,
                         batch_norm_scale=True,
                         init_method=None):
  """Defines the default arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    dropout_keep_prob: The dropout keep probability used after each convolutional
      layer. It is used for three datasets without data augmentation: CIFAR10,
      CIFAR 100, and SVHN.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the siamese_fc models.
  """
  # Only consider the model to be in training mode if it's trainable.
  # This is vital for batch_norm since moving_mean and moving_variance
  # will get updated even if not trainable.
  is_model_training = trainable and is_training
  batch_norm_params = {
    # The original implementation use scaling
    "scale": batch_norm_scale,
    # Decay for the moving averages.
    "decay": batch_norm_decay,
    # Epsilon to prevent 0s in variance.
    "epsilon": batch_norm_epsilon,
    "trainable": trainable,
    "is_training": is_model_training,
    # Collection containing the moving mean and moving variance.
    "variables_collections": {
      "beta": None,
      "gamma": None,
      "moving_mean": ["moving_vars"],
      "moving_variance": ["moving_vars"],
    },
    'updates_collections': None, # Ensure that updates are done within a frame
  }

  if trainable:
    weights_regularizer = slim.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  if init_method == 'kaiming_normal':
    initializer = slim.variance_scaling_initializer()
  else:
    initializer = slim.xavier_initializer()

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=weights_regularizer,
      weights_initializer=initializer,#tf.zeros_initializer(),#
      padding='VALID',
      trainable=trainable,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.dropout], keep_prob=dropout_keep_prob) as arg_sc:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_model_training):#False):#
          return arg_sc


def make_var( name, shape, initializer=None):
  return tf.get_variable(name, shape, initializer=initializer)
def conv(a, k_h, k_w, c_o, s_h, s_w, name, rate=1, biased=True, relu=True, padding='zeros', initializer=None):
  """ contribution by miraclebiu, and biased option"""

  c_i = a.get_shape()[-1]
  convolve = lambda i, k: tf.nn.convolution(i, k, padding=padding, strides=[s_h, s_w], dilation_rate=[rate, rate])
  with tf.variable_scope(name) as scope:

    # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
    init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
    init_biases = tf.constant_initializer(0.0)
    kernel = make_var('weights', [k_h, k_w, c_i, c_o], init_weights)
    if biased:
      biases = make_var('biases', [c_o], init_biases)
      conv = convolve(a, kernel)
      if relu:
        bias = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(bias)
      return tf.nn.bias_add(conv, biases)
    else:
      conv = convolve(a, kernel)
      if relu:
        return tf.nn.relu(conv)
      return conv

def deform_conv(a,b, k_h, k_w, c_o, s_h, s_w, num_deform_group, name, num_groups = 1, rate = 1, biased=True, relu=True,  padding='zeros', initializer=None):


  data = a
  offset = b
  c_i = data.get_shape()[-1]
  trans2NCHW = lambda x:tf.transpose(x, [0, 3 ,1 ,2])
  trans2NHWC = lambda x:tf.transpose(x, [0, 2 ,3, 1])
        # deform conv only supports NCHW
  data = trans2NCHW(data)
  offset = trans2NCHW(offset)
  dconvolve = lambda i, k, o: deform_conv_op.deform_conv_op(i, k, o, strides = [1, 1, s_h, s_w], rates=[1, 1, rate, rate], padding=padding, num_groups=num_groups, deformable_group=num_deform_group)
  with tf.variable_scope(name) as scope:

   # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
    init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
    init_biases = tf.constant_initializer(0.0)
    kernel = make_var('weights', [c_o, c_i, k_h, k_w], init_weights)
    print(data, kernel, offset)
    dconv = trans2NHWC(dconvolve(data, kernel, offset))
    if biased:
      biases = make_var('biases', [c_o], init_biases)
      if relu:
        bias = tf.nn.bias_add(dconv, biases)
        return tf.nn.relu(bias)
      return tf.nn.bias_add(dconv, biases)
    else:
      if relu:
        return tf.nn.relu(dconv)
      return dconv





def siamese_fc(inputs,
               reuse=None,
               scope='siamese_fc',deform=False):
  with tf.variable_scope(scope, 'siamese_fc', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = inputs
      print('inputs',inputs)
      net = slim.conv2d(net, 96, [11, 11], 2, scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      print('conv1',net)
      with tf.variable_scope('conv2'):
        b1, b2 = tf.split(net, 2, 3)

        # The original implementation has bias terms for all convolution, but
        # it actually isn't necessary if convolution layer follows a batch
        # normalization layer since batch norm will subtract the mean.

#        with tf.variable_scope('def'):
#          
#
#
#            offset1=conv(b1,5,5,200, 1, 1, biased=True, rate=2, relu=False, name='offset1', padding='SAME', initializer='zeros')
#            
#            print("sdiufhasudf",b1)
#           # print("sdfah",offset2)
#            b1 = deform_conv(b1,offset1,5,5, 128, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=4, name='b1')#slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
#
#            offset2=conv(b2,5,5,200, 1, 1, biased=True, rate=2, relu=False, name='offset2', padding='SAME', initializer='zeros')
#            b2 =deform_conv(b2,offset2,5,5, 128, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=4, name='b2')# slim.conv2d(b2, 128, [3, 3], 1, scope='b2')# 
#        net = tf.concat([b1, b2], 3)




        b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
        b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
        net = tf.concat([b1, b2], 3)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], 1, scope='conv3')
      with tf.variable_scope('conv4'):
        b1, b2 = tf.split(net, 2, 3)
        b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
        b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
        
        net = tf.concat([b1, b2], 3)
        
        n1=net
      # Conv 5 with only convolution
      with tf.variable_scope('conv5'):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=None, normalizer_fn=None):



##############################################################################################################shortcut+deformonlyinstance
#          b1, b2 = tf.split(net, 2, 3)
#          b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
#          b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
#          net = tf.concat([b1, b2], 3)
#
#          with tf.variable_scope('def'):
#            if deform==False:
#              i=0.0
#            else:
#              i=1.0
#            print("111111111",net,n1)
#            offset1=conv(net,3, 3,18, 1, 1, biased=False, rate=2, relu=False, name='offset1', padding='SAME', initializer='zeros')
#            net = net+i*deform_conv(n1,offset1,3, 3, 256, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=1, name='db1',initializer='zeros')#4, name='b1')#slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
 

##############################################################################################################nosplit+clip
          with tf.variable_scope('def'):    
            print('net',net)      
            offset1=conv(net,3, 3,18, 1, 1, biased=False, rate=2, relu=False, name='offset1', padding='SAME', initializer='zeros')
            offset1=tf.clip_by_value(offset1,-10,10,name='clip')
            net=deform_conv(net,offset1,3, 3, 256, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=1, name='b1')#slim.conv2d(b1, 128, [3, 3], 1, scope='b1')



##############################################################################################################nosplit
#          with tf.variable_scope('def'):    
#            print('net',net)      
#            offset1=conv(net,3, 3,18, 1, 1, biased=False, rate=2, relu=False, name='offset1', padding='SAME', initializer='zeros')   
#            net=deform_conv(net,offset1,3, 3, 256, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=1, name='b1')#slim.conv2d(b1, 128, [3, 3], 1, scope='b1')

######################################################################################################################SFC-only
#          b1, b2 = tf.split(net, 2, 3)
#          b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
#          b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
#        net = tf.concat([b1, b2], 3)
#        offset1=net

##################################################################################################################################


#            offset2=conv(b2,3, 3,72, 1, 1, biased=False, rate=2, relu=False, name='offset2', padding='SAME', initializer='zeros')

           # print("sdfah",offset2)
#            b1 = deform_conv(b1,offset1,3, 3, 256, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=1, name='b1',)#4, name='b1')#slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
#            print("ddd",b1)
#            b2 =deform_conv(b2,offset2,3, 3, 128, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=4, name='b2')#4, name='b2')# slim.conv2d(b2, 128, [3, 3], 1, scope='b2')# 
#        net = tf.concat([b1, b2], 3)
#      # Convert end_points_collection into a dictionary of end_points.
#  
#
##          b1, b2 = tf.split(net, 2, 3)
##          b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
##          b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
##        net = tf.concat([b1, b2], 3)
#      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#      return net, end_points

#          with tf.variable_scope('def'):
#        
#            offset1=conv(b1,3, 3, 72, 1, 1, biased=True, rate=2, relu=False, name='offset1', padding='SAME', initializer='zeros')
#          #  offset2=conv(b2,3, 3, 72, 1, 1, biased=True, rate=2, relu=False, name='offset2', padding='SAME', initializer='zeros')
#            print("sdiufhasudf",b1)
#           # print("sdfah",offset2)
#            b1 = deform_conv(b1,offset1,3, 3, 128, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=4, name='b1')#slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
#            print("ddd",b1)
#          b2 =slim.conv2d(b2, 128, [3, 3], 1, scope='b2')# deform_conv(b2,offset2,3, 3, 128, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=4, name='b2')# 
#        net = tf.concat([b1, b2], 3)
#      # Convert end_points_collection into a dictionary of end_points.
#
##      with tf.variable_scope('conv5'):
##        with slim.arg_scope([slim.conv2d],
##                            activation_fn=None, normalizer_fn=None):

#      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#      return net, end_points

#          with tf.variable_scope('def'):
#          
#
#
#            offset1=conv(b1,3, 3, 72, 1, 1, biased=True, rate=2, relu=False, name='offset1', padding='SAME', initializer='zeros')
#            
#            print("sdiufhasudf",b1)
#           # print("sdfah",offset2)
#            b1 = deform_conv(b1,offset1,3, 3, 128, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=4, name='b1')#slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
#            print("ddd",b1)
#            with tf.variable_scope('def1'):
#              offset2=conv(b2,3, 3, 72, 1, 1, biased=True, rate=2, relu=False, name='offset2', padding='SAME', initializer='zeros')
#              b2 =deform_conv(b2,offset2,3, 3, 128, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=4, name='b2')# slim.conv2d(b2, 128, [3, 3], 1, scope='b2')# 
#        net = tf.concat([b1, b2], 3)
      # Convert end_points_collection into a dictionary of end_points.
#
##      with tf.variable_scope('conv5'):
##        with slim.arg_scope([slim.conv2d],
##                            activation_fn=None, normalizer_fn=None):
#          b1, b2 = tf.split(net, 2, 3)
#          b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
#          b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
#        net = tf.concat([b1, b2], 3)
  
  
  
#        with tf.variable_scope('def'):        
#          offset=conv(net,3, 3, 72, 1, 1, biased=True, rate=2, relu=False, name='offset1', padding='SAME', initializer='zeros')
#       
#           # print("sdfah",offset2)
#          net= deform_conv(net,offset,3, 3, 256, 1, 1, biased=False, rate=1, relu=False, padding="VALID",num_deform_group=4, name='b1')#slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
##     
#        
#        
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net,offset1#    end_points #offset1#         

siamese_fc.stride = 8
