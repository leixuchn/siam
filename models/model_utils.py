#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Utilities for model construction"""
import re

import numpy as np
import tensorflow as tf
from scipy import io as sio

from utils.misc import get_center
import logging

def _construct_gt_response(response_size, batch_size, stride, gt_config=None):
  """Construct a batch of 2D ground truth response

  Args:
    response_size: a list or tuple with two elements [ho, wo]
    batch_size: an integer e.g. 16
    stride: embedding stride e.g. 8
    gt_config: configurations for ground truth generation

  return:
    a float tensor of shape [batch_size] +  response_size
  """
  with tf.variable_scope('construct_gt') as ct_scope:
    ho = response_size[0]
    wo = response_size[1]
    y = tf.cast(tf.range(0, ho), dtype=tf.float32) - get_center(ho)
    x = tf.cast(tf.range(0, wo), dtype=tf.float32) - get_center(wo)
    [Y, X] = tf.meshgrid(y, x)

    gt_type = gt_config['type']
    if gt_type == 'gaussian':
      def _gaussian_2d(X, Y, sigma):
        x0, y0 = 0, 0  # the target position, i.e. the center
        return tf.exp(-0.5 * (((X - x0) / sigma) ** 2 + ((Y - y0) / sigma) ** 2))

      sigma = gt_config['rPos'] / stride / 3.0
      gt = _gaussian_2d(X, Y, sigma)
    elif gt_type == 'overlap':
      def _overlap_score(X, Y, stride, area):
        area_x, area_y = [tf.to_float(a) / stride for a in area]
        x_diff = (area_x - tf.abs(X))
        y_diff = (area_y - tf.abs(Y))

        # Intersection over union
        Z = x_diff * y_diff / (2 * area_x * area_y - x_diff * y_diff)

        # Remove negative intersections
        Z = tf.where(x_diff > 0, Z, tf.zeros_like(Z))
        Z = tf.where(y_diff > 0, Z, tf.zeros_like(Z))
        return Z

      area = [64, 64]
      logging.info('area are fixed for overlap gt type')
      gt = _overlap_score(X, Y, stride, area)
    elif gt_type == 'logistic':
      def _logistic_label(X, Y, rPos, rNeg):
        # dist_to_center = tf.sqrt(tf.square(X) + tf.square(Y))  # L2 dist
        dist_to_center = tf.abs(X) + tf.abs(Y)  # Block dist
        Z = tf.where(dist_to_center <= rPos,
                     tf.ones_like(X),
                     tf.where(dist_to_center < rNeg,
                              0.5 * tf.ones_like(X),
                              tf.zeros_like(X)))
        return Z

      rPos = gt_config['rPos'] / stride
      rNeg = gt_config['rNeg'] / stride
      gt = _logistic_label(X, Y, rPos, rNeg)
    else:
      raise NotImplementedError

      # Create a batch of ground truth response
    gt_expand = tf.reshape(gt, [1] + response_size)
    gt = tf.tile(gt_expand, [batch_size, 1, 1])
    return gt


def get_exemplar_images(images, exemplar_size, targets_pos=None):
  """Get exemplar image from input images

  args:
    images: images of shape [batch, height, width, 3]
    exemplar_size: [height, width]
    targets_pos: target center positions in the input images, of shape [batch, 2]

  return:
    exemplar images of shape [batch, height, width, 3]
  """
  with tf.name_scope('get_exemplar_image'):
    batch_size, x_height, x_width = images.get_shape().as_list()[:3]
    z_height, z_width = exemplar_size

    if targets_pos is None:
      target_pos_single = [[get_center(x_height), get_center(x_width)]]
      targets_pos_ = tf.tile(target_pos_single, [batch_size, 1])
    else:
      targets_pos_ = targets_pos

    # convert to top-left corner based coordinates
    top = tf.to_int32(tf.round(targets_pos_[:, 0] - get_center(z_height)))
    bottom = tf.to_int32(top + z_height)
    left = tf.to_int32(tf.round(targets_pos_[:, 1] - get_center(z_width)))
    right = tf.to_int32(left + z_width)

    def _slice(x):
      f, t, l, b, r = x
      c = f[t:b, l:r]
      return c

    exemplar_img = tf.map_fn(_slice, (images, top, left, bottom, right), dtype=images.dtype)
    exemplar_img.set_shape([batch_size, z_height, z_width, 3])
    print("batch",batch_size)
    return exemplar_img



def extract_patch(inputs, patch_size, top_left=None):
  """Extract patch from inputs Tensor

  args:
    inputs: Tensor of shape [batch, height, width, feature_num]
    patch_size: [height, width]
    top_left: patch top_left positions in the input tensor, of shape [batch, 2]

  return:
    patches of shape [batch, height, width, feature_num]
  """
  with tf.name_scope('extract_patch'):
    batch_size, x_height, x_width, feat_num = inputs.get_shape().as_list()
    z_height, z_width = patch_size

    if top_left is None:
      pos_single = [[get_center(x_height), get_center(x_width)]]
      patch_center_ = tf.tile(pos_single, [batch_size, 1])

      # convert to top-left corner based coordinates
      top = tf.to_int32(tf.round(patch_center_[:, 0] - get_center(z_height)))
      left = tf.to_int32(tf.round(patch_center_[:, 1] - get_center(z_width)))
    else:
      top = tf.to_int32(top_left[:, 0])
      left = tf.to_int32(top_left[:, 1])

    bottom = tf.to_int32(top + z_height)
    right = tf.to_int32(left + z_width)

    def _slice(x):
      f, t, l, b, r = x
      c = f[t:b, l:r]
      return c

    patch = tf.map_fn(_slice, (inputs, top, left, bottom, right), dtype=inputs.dtype)

    # Restore some shape
    patch.set_shape([batch_size, z_height, z_width, feat_num])
    return patch


def get_params_from_mat(matpath):
  """Get parameter from .mat file into parms(dict)"""

  def squeeze(vars_):
    # Matlab save some params with shape (*, 1)
    # while in tensorflow, we don't need the trailing dimension.
    if isinstance(vars_, (list, tuple)):
      return [np.squeeze(v, 1) for v in vars_]
    else:
      return np.squeeze(vars_, 1)

  netparams = sio.loadmat(matpath)["net"]["params"][0][0]
  params = dict()

  for i in range(netparams.size):
    param = netparams[0][i]
    name = param["name"][0]
    value = param["value"]
    value_size = param["value"].shape[0]

    match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name, re.I)
    if match:
      items = match.groups()
    elif name == 'adjust_f':
      params['detection/weights'] = squeeze(value)
      continue
    elif name == 'adjust_b':
      params['detection/biases'] = squeeze(value)
      continue
    else:
      raise Exception('unrecognized layer params')

    op, layer, types = items
    layer = int(layer)
    if layer in [1, 3]:
      if op == 'conv':  # convolution
        if types == 'f':
          params['conv%d/weights' % layer] = value
        elif types == 'b':
          value = squeeze(value)
          params['conv%d/biases' % layer] = value
      elif op == 'bn':  # batch normalization
        if types == 'x':
          m, v = squeeze(np.split(value, 2, 1))
          params['conv%d/BatchNorm/moving_mean' % layer] = m
          params['conv%d/BatchNorm/moving_variance' % layer] = np.square(v)
        elif types == 'm':
          value = squeeze(value)
          params['conv%d/BatchNorm/gamma' % layer] = value
        elif types == 'b':
          value = squeeze(value)
          params['conv%d/BatchNorm/beta' % layer] = value
      else:
        raise Exception
    elif layer in [2, 4]:
      if op == 'conv' and types == 'f':
        b1, b2 = np.split(value, 2, 3)
      else:
        b1, b2 = np.split(value, 2, 0)
      if op == 'conv':
        if types == 'f':
          params['conv%d/b1/weights' % layer] = b1
          params['conv%d/b2/weights' % layer] = b2
        elif types == 'b':
          b1, b2 = squeeze(np.split(value, 2, 0))
          params['conv%d/b1/biases' % layer] = b1
          params['conv%d/b2/biases' % layer] = b2
      elif op == 'bn':
        if types == 'x':
          m1, v1 = squeeze(np.split(b1, 2, 1))
          m2, v2 = squeeze(np.split(b2, 2, 1))
          params['conv%d/b1/BatchNorm/moving_mean' % layer] = m1
          params['conv%d/b2/BatchNorm/moving_mean' % layer] = m2
          params['conv%d/b1/BatchNorm/moving_variance' % layer] = np.square(v1)
          params['conv%d/b2/BatchNorm/moving_variance' % layer] = np.square(v2)
        elif types == 'm':
          params['conv%d/b1/BatchNorm/gamma' % layer] = squeeze(b1)
          params['conv%d/b2/BatchNorm/gamma' % layer] = squeeze(b2)
        elif types == 'b':
          params['conv%d/b1/BatchNorm/beta' % layer] = squeeze(b1)
          params['conv%d/b2/BatchNorm/beta' % layer] = squeeze(b2)
      else:
        raise Exception

    elif layer in [5]:
      if op == 'conv' and types == 'f':
        b1, b2 = np.split(value, 2, 3)
      else:
        b1, b2 = squeeze(np.split(value, 2, 0))
      assert op == 'conv', 'layer5 contains only convolution'
      if types == 'f':
        params['conv%d/b1/weights' % layer] = b1
        params['conv%d/b2/weights' % layer] = b2
      elif types == 'b':
        params['conv%d/b1/biases' % layer] = b1
        params['conv%d/b2/biases' % layer] = b2

  return params


def load_mat_model(matpath, embed_scope, detection_scope=None):
  # Restore SiameseFC models from .mat model files
  params = get_params_from_mat(matpath)

  assign_ops = []

  def _assign(ref_name, params, scope=embed_scope):
    var_in_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope + ref_name)[0]
    var_in_mat = params[ref_name]
    op = tf.assign(var_in_model, var_in_mat)
    assign_ops.append(op)

  for l in range(1, 6):
    if l in [1, 3]:
      _assign('conv%d/weights' % l, params)
      # _assign('conv%d/biases' % l, params)
      _assign('conv%d/BatchNorm/beta' % l, params)
      _assign('conv%d/BatchNorm/gamma' % l, params)
      _assign('conv%d/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/BatchNorm/moving_variance' % l, params)
    elif l in [2, 4]:
      # Branch 1
      _assign('conv%d/b1/weights' % l, params)
      # _assign('conv%d/b1/biases' % l, params)
      _assign('conv%d/b1/BatchNorm/beta' % l, params)
      _assign('conv%d/b1/BatchNorm/gamma' % l, params)
      _assign('conv%d/b1/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/b1/BatchNorm/moving_variance' % l, params)
      # Branch 2
      _assign('conv%d/b2/weights' % l, params)
      # _assign('conv%d/b2/biases' % l, params)
      _assign('conv%d/b2/BatchNorm/beta' % l, params)
      _assign('conv%d/b2/BatchNorm/gamma' % l, params)
      _assign('conv%d/b2/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/b2/BatchNorm/moving_variance' % l, params)
    elif l in [5]:
      # Branch 1
      _assign('conv%d/b1/weights' % l, params)
      _assign('conv%d/b1/biases' % l, params)
      # Branch 2
      _assign('conv%d/b2/weights' % l, params)
      _assign('conv%d/b2/biases' % l, params)
    else:
      raise Exception('layer number must below 5')

  if detection_scope:
    _assign(detection_scope + 'biases', params, scope='')

  initialize = tf.group(*assign_ops)
  return initialize
def load_caffenet(path_caffenet):
  logging.info('Load object model from ' + path_caffenet)
  data_dict = np.load(path_caffenet).item()
  for op_name in data_dict:
    if op_name.find('fc')!=-1:
      continue
    full_op_name = 'siamese_fc/alexnet/'+op_name
    with tf.variable_scope(full_op_name, reuse=True):
      if op_name in ['conv2','conv4','conv5']:
        for param_name, data in data_dict[op_name].iteritems():
          d1, d2 = tf.split(data, 2, -1+len(data.shape)) # Last dim is selected to split
          for [d_, b_] in [[d1,'b1'],[d2,'b2']]:
            with tf.variable_scope(b_, reuse=True):
              var = tf.get_variable(param_name)
              sess.run(var.assign(d_))
      else:
        for param_name, data in data_dict[op_name].iteritems():
          var = tf.get_variable(param_name)
          sess.run(var.assign(data))