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


def siamese_obj_fc_arg_scope(weight_decay=5e-4,
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
      weights_initializer=initializer,
      padding='VALID',
      trainable=trainable,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.dropout], keep_prob=dropout_keep_prob) as arg_sc:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_model_training):
          return arg_sc
def _ori_siam(_net, n_out=128):
  _net = slim.conv2d(_net, 96, [11, 11], 2, scope='conv1')
  _net = slim.max_pool2d(_net, [3, 3], 2, scope='pool1')
  with tf.variable_scope('conv2'):
    b1, b2 = tf.split(_net, 2, 3)
    b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
    # The original implementation has bias terms for all convolution, but
    # it actually isn't necessary if convolution layer follows a batch
    # normalization layer since batch norm will subtract the mean.
    b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
    _net = tf.concat([b1, b2], 3)
  _net = slim.max_pool2d(_net, [3, 3], 2, scope='pool2')
  _net = slim.conv2d(_net, 384, [3, 3], 1, scope='conv3')
  with tf.variable_scope('conv4'):
    b1, b2 = tf.split(_net, 2, 3)
    b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
    b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
    _net = tf.concat([b1, b2], 3)
  # Conv 5 with only convolution
  with tf.variable_scope('conv5'):
    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
      b1, b2 = tf.split(_net, 2, 3)
      b1 = slim.conv2d(b1, n_out/2, [3, 3], 1, scope='b1')
      b2 = slim.conv2d(b2, n_out/2, [3, 3], 1, scope='b2')
    _net = tf.concat([b1, b2], 3)
  return _net

def _obj_siam_alex(_net, n_out=128, train_1x1=True):
  with slim.arg_scope([slim.conv2d], normalizer_fn=None, trainable=False, normalizer_params=False):
    _net = _net - [123.0,117.0,104.0]# RGB
    _net = tf.reverse(_net,[3])# convert img to BGR
    _net = slim.conv2d(_net, 96, [11, 11], 2, scope='conv1')
    _net = slim.max_pool2d(_net, [3, 3], 2, scope='pool1')
    _net = tf.nn.local_response_normalization(_net,depth_radius=2,alpha=2e-5,beta=0.75,bias=1.0,name='norm1')
    with tf.variable_scope('conv2'):
      b1, b2 = tf.split(_net, 2, 3)
      b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
      b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
      _net = tf.concat([b1, b2], 3)
    _net = slim.max_pool2d(_net, [3, 3], 2, scope='pool2')
    _net = tf.nn.local_response_normalization(_net,depth_radius=2,alpha=2e-5,beta=0.75,bias=1.0,name='norm2')
    _net = slim.conv2d(_net, 384, [3, 3], 1, scope='conv3')
    with tf.variable_scope('conv4'):
      b1, b2 = tf.split(_net, 2, 3)
      b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
      b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
      _net = tf.concat([b1, b2], 3)
    # Conv 5 with only convolution
    with tf.variable_scope('conv5'):
      with slim.arg_scope([slim.conv2d],activation_fn=None, normalizer_fn=None):
        b1, b2 = tf.split(_net, 2, 3)
        b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
        b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
      _net = tf.concat([b1, b2], 3)
  with slim.arg_scope([slim.conv2d],activation_fn=None, normalizer_fn=None,trainable=train_1x1):
    _net = slim.conv2d(_net, n_out, [1,1], 1, scope='c1x1')
  return _net
def _combine_2_net(net1,net2,fusion_type):
  if   fusion_type == 'concact': _net = tf.concat([net1, net2], 3)
  elif fusion_type == 'ave': _net = (net1 + net2)/2
  elif fusion_type == 'max': _net = tf.maximum(net1,net2)
  elif fusion_type == 'mul': _net = net1 * net2
  elif fusion_type == 'max_l2': 
    net1_l2 = net1 * net1
    net2_l2 = net2 * net2
    energy_ori_siam = tf.reduce_mean(net1_l2, [1,2],keep_dims=True)
    energy_obj_net = tf.reduce_mean(net2_l2, [1,2],keep_dims=True)
    compare_energy = tf.to_float(tf.greater(energy_ori_siam,energy_obj_net))
    _net = net1 * compare_energy + net2 * (1-compare_energy)
  else: raise ValueError('fusion_type not difined!')
  return _net
def siamese_obj_fc(inputs,
               reuse=None,
               scope='siamese_fc',
               en_siam=True,
               en_obj=True,
               fusion_type='concact',
               obj_net='alexnet'):
  with tf.variable_scope(scope, 'siamese_fc', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = inputs
      if en_siam:
        with tf.variable_scope('ori_siam'):
          net_siam = _ori_siam(net)
      if en_obj:
        if obj_net == 'alexnet':
          with tf.variable_scope('alexnet'):
            net_obj = _obj_siam_alex(net)
        elif obj_net == 'siam':
          with tf.variable_scope('ori_siam_2'):
            net_obj = _ori_siam(net)
        else: raise ValueError('obj_net not defined!')
      if en_siam and en_obj:
        net = _combine_2_net(net_siam, net_obj, fusion_type)
      elif en_siam:net = net_siam
      elif en_obj:net = net_obj
      else: raise ValueError('obj or siam must enable one!')
      # Convert end_points_collection into a dictionary of end_points.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points

siamese_obj_fc.stride = 8
