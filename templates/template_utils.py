#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Template utilities"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def crop_tensor(inputs, offset_height, offset_width, target_height, target_width):
  """Crop a 4-D Tensor to a specified bounding box

  Args:
    inputs: a 4-D tensor.
    offset_height: vertical coordinate of the top-left corner of the result in the input.
    offset_width: horizontal coordinate of the top-left corner of the result in the input.
    target_height: height of the result.
    target_width: width of the result.

  Return:
    A 4-D tensor croped from inputs with specified height and width.
  """
  with tf.name_scope('crop_tensor'):
    return inputs[:, offset_height : offset_height + target_height,
                  offset_width : offset_width + target_width, :]
