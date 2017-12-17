#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.


import tensorflow as tf

from metrics import metric_ops


def center_score_error(response):
  """Center score error
  Args:
  - response: 3-D float Tensor of shape [batch, time, height, width]
  """
  with tf.name_scope('CS-err'):
    r, c = get_center_index(response)
    center_score = response[:, r, c]
    return tf.reduce_mean(tf.to_float(center_score < 0))


def center_dist_error(response):
  with tf.name_scope('CD-err'):
    radius_in_pixel = 50.
    total_stride = 8.
    num_thresholds = 100
    radius_in_response = radius_in_pixel / total_stride

    gt_r, gt_c = get_center_index(response)
    max_r, max_c = get_maximum_response_ind(response)
    gt_r = tf.to_float(gt_r)
    gt_c = tf.to_float(gt_c)
    max_r = tf.to_float(max_r)
    max_c = tf.to_float(max_c)
    distances = tf.sqrt((gt_r - max_r) ** 2 + (gt_c - max_c) ** 2)

    # We cast distances as predictions in the range [0, 1] where 0 means false and
    # 1 means true. In this way, we can readily use streaming_auc to compute area
    # under curve.
    dist_norm = distances / radius_in_response
    dist_norm = tf.minimum(dist_norm, 1.)
    predictions = 1. - dist_norm
    # predictions = tf.Print(predictions, [predictions], summarize=8)
    labels = tf.ones_like(predictions)
    auc, update_op = metric_ops.streaming_auc(predictions, labels,
                                              num_thresholds=100,
                                              curve='R')
    with tf.control_dependencies([update_op]):
      err = 1. - auc
    return err


def get_center_index(response):
  """
  Args:
    response: 3-D float tensor of shape [batch, height, width]
  """
  shape = tf.shape(response)
  c1 = (shape[1] - 1) / 2
  c2 = (shape[2] - 1) / 2
  return c1, c2


def get_maximum_response_ind(response):
  """

  Args:
    response: a 3-D Tensor of shape [batch, ho, wo]

  returns:
  a tuple of
      ind_row: a 1-D Tensor of shape [batch]
      ind_col: a 1-D Tensor of shape [batch]
  """
  response_shape = response.get_shape().as_list()
  response_spatial_size = response_shape[-2:]  # e.g. [29, 29]
  length = response_spatial_size[0] * response_spatial_size[1]

  # Get maximum response index
  # note index starts from zero
  ind_max = tf.argmax(tf.reshape(response, [-1, length]), 1)
  ind_row = tf.div(ind_max, response_spatial_size[1])
  ind_col = tf.mod(ind_max, response_spatial_size[1])
  return ind_row, ind_col
