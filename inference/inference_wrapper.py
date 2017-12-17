#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Model Wrapper class for performing inference with a TrackModel"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from embeddings import embedding_factory
from inference import inference_wrapper_base
from models.model_utils import get_exemplar_images
from templates import template_factory
from utils.misc import get_center, get
import logging
from pprint import pprint
class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
  """Model wrapper class for performing inference with a track model."""

  def __init__(self):
    super(InferenceWrapper, self).__init__()
    self.image = None
    self.target_bbox_feed = None
    self.images = None
    self.embeds = None
    self.templates = None
    self.init = None
    self.model_config = None
    self.track_config = None
    self.response_up = None

  def build_model(self, model_config, track_config):
    self.model_config = model_config
    self.track_config = track_config

    self.build_inputs()
    self.build_extract_crops()
    self.build_template()
    self.build_detection()
    self.build_upsample()

  def build_inputs(self):
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.to_float(image)

    if self.model_config['normalize_image']:
      # Normalize to the range [-1, 1]
      image = tf.multiply(2.0, tf.subtract(image, 0.5))
    self.image = image

    self.target_bbox_feed = tf.placeholder(dtype=tf.float32,
                                           shape=[4],
                                           name='target_bbox_feed') # center's y, x, height, width

  def build_extract_crops(self):
    model_config = self.model_config
    track_config = self.track_config
    context_amount = 0.5

    size_z = model_config['z_image_size']
    size_x = model_config['x_image_size']

    num_scales = track_config['num_scales']

    scales = np.arange(num_scales) - get_center(num_scales)
    assert np.sum(scales) == 0, 'scales should be symmetric'
    assert  track_config['scale_step'] >= 1.0, 'scale step should be >= 1.0'
    search_factors = [track_config['scale_step'] ** x for x in scales]

    frame_sz = tf.shape(self.image)
    target_yx = self.target_bbox_feed[0:2]
    target_size = self.target_bbox_feed[2:4]
    avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')

    # Compute base values
    base_z_size = target_size
    base_z_context_size = base_z_size  + context_amount * tf.reduce_sum(base_z_size)
    base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canoical size
    base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
    d_search = (size_x - size_z) / 2.0
    base_pad = tf.div(d_search, base_scale_z)
    base_s_x = base_s_z + 2 * base_pad
    base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

    boxes = []
    for factor in search_factors:
      s_x = factor * base_s_x
      frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
      topleft = tf.div(target_yx - get_center(s_x) , frame_sz_1)
      bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
      box = tf.concat([topleft, bottomright], axis=0)
      boxes.append(box)
    boxes = tf.stack(boxes)

    scale_xs = []
    for factor in search_factors:
      scale_x = base_scale_x / factor
      scale_xs.append(scale_x)
    self.scale_xs = tf.stack(scale_xs)

    image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
    image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                             box_ind=tf.zeros((track_config['num_scales']), tf.int32),
                                             crop_size=[size_x, size_x])
    self.images = image_cropped + avg_chan

  def get_image_embedding(self, images, reuse=None,deform=False):
    config = self.model_config['embed_config']
    embedding_fn = embedding_factory.get_network_fn(
      config['embedding_name'],
      weight_decay=config['weight_decay'],
      trainable=config['train_embedding'],
      is_training=False,
      init_method=get(config, 'init_method', None),
      bn_momentum=get(config, 'bn_momentum', 3e-4),
      bn_epsilon=get(config, 'bn_epsilon', 1e-6), )
    embed, _ = embedding_fn(images, reuse,deform)

    return embed

  def build_template(self):
    model_config = self.model_config
    track_config = self.track_config
    examplar_images = get_exemplar_images(self.images, [model_config['z_image_size'],
                                                        model_config['z_image_size']])
    templates = self.get_image_embedding(examplar_images,deform=False)
    center_scale = int(get_center(track_config['num_scales']))
    center_template = tf.identity(templates[center_scale])
    templates = tf.stack([center_template for _ in range(model_config['batch_size'])])

    with tf.variable_scope('target_template'):
      template_fn = template_factory.get_network_fn(
        model_config['template_name'],
        weight_decay=model_config['weight_decay'],
        is_training=False)
      templates, _ = template_fn(templates)

      # Store template in Variable such that we don't have to feed this template.
      with tf.variable_scope('State'):
        state = tf.get_variable('exemplar',
                                initializer=tf.zeros_like(templates),
                                trainable=False)
        with tf.control_dependencies([templates]):
          self.init = tf.assign(state, templates, validate_shape=True)
        self.templates = state

  def build_detection(self):
    self.embeds = self.get_image_embedding(self.images, reuse=True,deform=True)
    with tf.variable_scope('detection'):
      def _translation_match(x, z):
        x = tf.expand_dims(x, 0)  # [batch, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, out_channels]
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')
      print("awjksfehawkjfh",self.embeds,self.templates)
      output = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds, self.templates), dtype=self.embeds.dtype)  # of shape [16, 1, 17, 17, 1]
      output = tf.squeeze(output, [1, 4])  # of shape e.g. [16, 17, 17]

      # Adjust score, this is required to make training possible.
      bias = tf.get_variable('biases', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      response = self.model_config['adjust_response_config']['scale'] * output + bias
      self.response = response


             
      

  def build_upsample(self):
    with tf.variable_scope('upsample'):
      response = tf.expand_dims(self.response, 3)
      up_method = get(self.model_config['adjust_response_config'], 'upsample_method', 'bicubic')
      align = get(self.model_config['adjust_response_config'], 'align_cornor', True)
      logging.info('Upsample method -- {}'.format(up_method))
      logging.info('Upsample response align cornor -- {}'.format(align))
      logging.info('Upsampling size -- {}'.format(self.model_config['u_image_size']))
      methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                 'bicubic': tf.image.ResizeMethod.BICUBIC}
      up_method = methods[up_method]
      response_up = tf.image.resize_images(response,
                                           [self.model_config['u_image_size'],
                                            self.model_config['u_image_size']],
                                           method=up_method,
                                           align_corners=align)
      response_up = tf.squeeze(response_up, [3])
      self.response_up = response_up

  def initialize(self, sess, input_feed):
    image_path, target_bbox = input_feed
    scale_xs, _ = sess.run([self.scale_xs, self.init],
                           feed_dict={'filename:0': image_path,
                                      "target_bbox_feed:0": target_bbox,
                                      })
                                      
                              
    return scale_xs

  def inference_step(self, sess, input_feed):
    image_path, target_bbox = input_feed
    scale_xs, response_output = sess.run(
      fetches=[self.scale_xs, self.response_up],
      feed_dict={
        "filename:0": image_path,
        "target_bbox_feed:0": target_bbox,})
    
         




    output = {
      'scale_xs': scale_xs,
      'response': response_output}
    return output, None
