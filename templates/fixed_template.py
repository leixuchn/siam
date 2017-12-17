#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""
Contains template generator without learnable parameters, therefore called fixed template.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def dumb_arg_scope(weight_decay=None, trainable=None, is_training=None):
  """Dumb arg_scope doing nothing"""
  with slim.arg_scope([]) as arg_sc:
    return arg_sc


def identity(embeds, reuse=None):
  """Use the original embeddings as templates. This is the approach used in siamese fc"""
  with tf.name_scope('identity_template'):
    return tf.identity(embeds, name='identity'), None
