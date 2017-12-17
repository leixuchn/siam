#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""
Inference Utilities
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imresize
from inference.vot import Rectangle
from utils.misc import get_center


def im2rgb(im):
  if len(im.shape) != 3:
    im = np.stack([im, im, im], -1)
  return im


def convert_bbox(bbox, to, offsetx,offsety):
  x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
  if to == 'top-left-based':
    x -= get_center(target_width)
    y -= get_center(target_height)
  elif to == 'center-based':
    y += get_center(target_height)
    x += get_center(target_width)
    x+=offsetx
    y+=offsety
    
    
  else:
    raise NotImplementedError
  return Rectangle(x, y, target_width, target_height)


def get_crops(im, bbox, size_z, size_x, context_amount):
  """Obtain image sub-window, padding with avg channel if area goes outside of border

  Adapted from https://github.com/bertinetto/siamese-fc/blob/master/ILSVRC15-curation/save_crops.m#L46

  :param im: image ndarray
  :param bbox: named tuple (x, y, width, height) x, y corresponds to the crops center
  :param size_z: target + context size
  :param size_x: the resultant crop size
  :param context_amount: the amount of context
  :return: image crop
  """
  cy, cx, h, w = bbox.y, bbox.x, bbox.height, bbox.width
  wc_z = w + context_amount * (w + h)
  hc_z = h + context_amount * (w + h)
  s_z = np.sqrt(wc_z * hc_z)
  scale_z = size_z / s_z

  d_search = (size_x - size_z) / 2
  pad = d_search / scale_z
  s_x = s_z + 2 * pad
  scale_x = size_x / s_x

  image_crop_x, _, _, _, _ = get_subwindow_avg(im, [cy, cx],
                                               [size_x, size_x],
                                               [np.round(s_x), np.round(s_x)])

  # Size of object within the crops
  # ws_x = w * scale_x
  # hs_x = h * scale_x
  # bbox_x = [size_x / 2, size_x / 2, ws_x, hs_x]
  return image_crop_x, scale_x


def get_subwindow_avg(im, pos, model_sz, original_sz):
  # avg_chans = np.mean(im, axis=(0, 1)) # This version is 3x slower
  avg_chans = [np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])]
  if not original_sz:
    original_sz = model_sz
  sz = original_sz
  im_sz = im.shape
  # make sure the size is not too small
  assert im_sz[0] > 2 and im_sz[1] > 2
  c = [get_center(s) for s in sz]

  # check out-of-bounds coordinates, and set them to avg_chans
  context_xmin = np.int(np.round(pos[1] - c[1]))
  context_xmax = np.int(context_xmin + sz[1] - 1)
  context_ymin = np.int(np.round(pos[0] - c[0]))
  context_ymax = np.int(context_ymin + sz[0] - 1)
  left_pad = np.int(np.maximum(0, -context_xmin))
  top_pad = np.int(np.maximum(0, -context_ymin))
  right_pad = np.int(np.maximum(0, context_xmax - im_sz[1] + 1))
  bottom_pad = np.int(np.maximum(0, context_ymax - im_sz[0] + 1))

  context_xmin = context_xmin + left_pad
  context_xmax = context_xmax + left_pad
  context_ymin = context_ymin + top_pad
  context_ymax = context_ymax + top_pad
  if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
    R = np.pad(im[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[0]))
    G = np.pad(im[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[1]))
    B = np.pad(im[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[2]))

    im = np.stack((R, G, B), axis=2)

  im_patch_original = im[context_ymin:context_ymax + 1,
                      context_xmin:context_xmax + 1, :]
  if not np.array_equal(model_sz, original_sz):
    im_patch = imresize(im_patch_original, model_sz, interp='bilinear')
  else:
    im_patch = im_patch_original
  return im_patch, left_pad, top_pad, right_pad, bottom_pad


class Sequence:
  """Mimick VOT class"""

  def __init__(self, filenames, initial_bbox):
    self._files = filenames
    self._frame = 0
    self._region = initial_bbox
    self._result = []

  def frame(self):
    """
    Get a frame (image path) from client

    Returns:
        absolute path of the image
    """
    if self._frame >= len(self._files):
      return None
    return self._files[self._frame]

  def region(self):
    """
    Send configuration message to the client and receive the initialization
    region and the path of the first image

    Returns:
        initialization region
    """

    return self._region

  def report(self, region):
    """
    Report the tracking results to the client

    Arguments:
        region: region for the frame
    """
    self._result.append(region)
    self._frame += 1

  def quit(self):
    return self._result
