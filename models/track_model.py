#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""
The main body of track Model, including feature extraction, matching, loss computation and more.
"""

import tensorflow as tf

from datasets.dataloader import DataLoader
from embeddings import embedding_factory
from metrics.track_metrics import center_dist_error, center_score_error
from models.model_utils import _construct_gt_response, load_mat_model
from preprocessing import preprocessing_factory
from templates import template_factory
from utils.misc import get
import random
slim = tf.contrib.slim
import logging
#
#class TrackModel:
#  def __init__(self, model_config, mode):
#    assert mode in ['train', 'val', 'inference']
#    self.config = model_config
#    self.mode = mode
#
#    # DataLoader instance
#    self.dataloader = None
#
#    # An float32 Tensor with shape [batch_size, height, width, 3]
#    self.examplar_image = None
#
#    # A float32 Tensor with shape [batch_size, time_steps - 1, height, width, 3]
#    self.instance_image = None
#
#    # A float32 Tensor with shape [batch_size, time_steps, height, width]
#    self.response = None
#
#    # A float32 scalar Tensor; the batch loss without weight decay loss
#    self.batch_loss = None
#
#    # A float32 scalar Tensor; the total loss for the trainer to optimize
#    self.total_loss = None
#
#    # Collection of variables from the inception submodel
#    self.inception_variables = []
#
#    # Funtion to restore the inception submodel from checkpoint
#    self.init_fn = None
#
#    # Global step Tensor
#    self.global_step = None
#
#  def is_training(self):
#    """Returns true if the model is built for training mode"""
#    return self.mode == 'train'
#
#  def build_inputs(self):
#    """Input prefetching, preprocessing and batching
#
#    Outputs:
#      self.examplars: image batch of shape [batch, image_size, image_size, 3]
#      self.instances: image batch of shape [batch, image_size, image_size, 3]
#    """
#    config = self.config
#
#    # Prefetch image sequences
#    with tf.device("/cpu:0"):  # Put data loading and preprocessing in CPU is substantially faster
#      
#      if self.mode == 'train':
#        self.dataloader = DataLoader(self.config['prefetch_config_train'], self.is_training())
#      if  self.mode == 'val':
#        self.dataloader = DataLoader(self.config['prefetch_config_val'], self.is_training())      
#      prefetch_queue = self.dataloader.get_queue()
#      video = prefetch_queue.dequeue()
#      examplar_image = video[0, :]
#      instance_image = video[1, :]
#
#      # Preprocess the examplar image and instance image
#      self.preprocessing_name = self.config['preprocessing_name'] or self.config['embedding_name']
#      image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#        self.preprocessing_name, is_training=self.is_training())
#      with tf.name_scope('Examplar'):
#        # Crop an image slightly larger than examplar for bbox distortion.
#        examplar_image = tf.image.resize_image_with_crop_or_pad(examplar_image,
#                                                                self.config['z_image_size'] + 8,
#                                                                self.config['z_image_size'] + 8)
#        examplar_image = image_preprocessing_fn(examplar_image,
#                                                self.config['z_image_size'],
#                                                self.config['z_image_size'],
#                                                fast_mode=self.config['fast_mode'],
#                                                normalize=self.config['normalize_image'])
#      with tf.name_scope('Instance'):
#        # Note we use slightly smaller instance image to enable bbox distortion.
#        instance_image = image_preprocessing_fn(instance_image,
#                                                self.config['x_image_size'] - 2 * 8,
#                                                self.config['x_image_size'] - 2 * 8,
#                                                fast_mode=self.config['fast_mode'],
#                                                normalize=self.config['normalize_image'])
#        logging.info('For training, we use instance size {} - 2 * 8 ...'.format(self.config['x_image_size']))
#
#      # Batch inputs.
#      examplars, instances = tf.train.batch(
#        [examplar_image, instance_image],
#        batch_size=self.config['batch_size'],
#        num_threads=2,
#        capacity=10 * self.config['batch_size'])
#
#    self.examplars = examplars
#    self.instances = instances
#
#  def build_image_embeddings(self,reuse=False):
#    """Builds the image model subgraph and generates image embeddings
#
#    Inputs:
#      self.examplars: A tensor of shape [batch, hz, wz, 3]
#      self.instances: A tensor of shape [batch, hx, wx, 3]
#    
#    Outputs:
#      self.examplar_embeds
#      self.instance_embeds
#    """
#    # Select embedding network
#    config = self.config['embed_config']
#    embedding_fn = embedding_factory.get_network_fn(
#      config['embedding_name'],
#      weight_decay=config['weight_decay'],
#      trainable=config['train_embedding'],
#      is_training=self.is_training(),
#      init_method=get(config, 'init_method', None),
#      bn_momentum=get(config, 'bn_momentum', 3e-4),
#      bn_epsilon=get(config, 'bn_epsilon', 1e-6),)
#    self.stride = embedding_fn.stride
#
#    # Embedding for examplar images
##    self.examplar_embeds, _ = embedding_fn(self.examplars)
##    self.instance_embeds, _ = embedding_fn(self.instances, reuse=True)
#    self.examplar_embeds,self.offset1_exam = embedding_fn(self.examplars,reuse=reuse,deform=False)
#    self.instance_embeds, self.offset1_inst = embedding_fn(self.instances, reuse=True,deform=True)
#  def build_template(self):
#    with tf.variable_scope('target_template'):
#      template_fn = template_factory.get_network_fn(
#        self.config['template_name'],
#        weight_decay=self.config['weight_decay'],
#        is_training=self.is_training())
#
#      self.templates, _ = template_fn(self.examplar_embeds)
#
#  def build_detection(self,reuse=False):
#    with tf.variable_scope('detection',reuse=reuse):
#      def _translation_match(x, z):
#        x = tf.expand_dims(x, 0)  # [batch, in_height, in_width, in_channels]
#        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, out_channels]
#        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')
#
#      output = tf.map_fn(
#        lambda x: _translation_match(x[0], x[1]),
#        (self.instance_embeds, self.templates), dtype=self.instance_embeds.dtype)  # of shape [16, 1, 17, 17, 1]
#      output = tf.squeeze(output, [1, 4])  # of shape e.g. [16, 17, 17]
#
#      config = self.config['adjust_response_config']
#      # Adjust score, this is required to make training possible.
#      bias = tf.get_variable('biases', [1],
#                             dtype=tf.float32,
#                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
#                             trainable=config['train_bias'])
#      response = config['scale'] * output + bias
#      self.response = response
#
#  def build_loss(self):
#    """Build the optimization loss
#    args:
#    - self.response of shape [batch, ho, wo]
#    - self.z_target of shape [batch, ho, wo]
#
#    return:
#      a scalar
#    """
#    response = self.response
#    print(response)
#    response_got_shape = response.get_shape().as_list()  # static shape
#    response_size = response_got_shape[1:3]  # [height, width]
#
#    # Construct ground truth output
#    gt = _construct_gt_response(response_size,
#                                self.config['batch_size'],
#                                self.stride,
#                                self.config['gt_config'])
#
#    with tf.name_scope('Compute_loss'):
#      # loss computation
#      if self.config['loss_type'] == 'Cross_entropy':
#        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=response,
#                                                       labels=gt)
#      elif self.config['loss_type'] == 'L2':
#        loss = tf.square(response - gt)
#
#      # Balancing class weight
#      n_pos = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 1)))
#      n_neg = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 0)))
#      w_pos = 0.5 / n_pos
#      w_neg = 0.5 / n_neg
#      class_weights = tf.where(tf.equal(gt, 1),
#                               w_pos * tf.ones_like(gt),
#                               tf.ones_like(gt))
#      class_weights = tf.where(tf.equal(gt, 0),
#                               w_neg * tf.ones_like(gt),
#                               class_weights)
#      loss = loss * class_weights
#
#      # Note that we use reduce_sum instead of reduce_mean since the loss has
#      # already been normalized by class_weights in spatial dimension.
#      loss = tf.reduce_sum(loss, [1, 2])  # new shape: [batch]
#
#      batch_loss = tf.reduce_mean(loss, name='batch_loss')
#      tf.losses.add_loss(batch_loss)
#
#      total_loss = tf.losses.get_total_loss()
#      self.batch_loss = batch_loss
#      self.total_loss = total_loss
#
#    with tf.name_scope(self.mode):
#      tf.summary.image('exemplar', self.examplars)
#      tf.summary.image('instance', self.instances)
#
#      # Add summaries
#      tf.summary.scalar('batch_loss', batch_loss)
#      tf.summary.scalar('total_loss', total_loss)
#      if self.mode == 'train':
#        tf.summary.image('GT', tf.reshape(gt[0], [1] + response_size + [1]))
#
#
#      # summary for response
#      tf.summary.histogram('response', self.response)
#      tf.summary.histogram('offset1_exam', self.offset1_exam)
#      tf.summary.histogram('offset1_inst', self.offset1_inst)      
#      if self.config['loss_type'] == 'Cross_entropy':
#        tf.summary.image('Response', tf.expand_dims(tf.sigmoid(response), -1), 5)
##      elif self.config['loss_type'] == 'L2':
##        tf.summary.image('Response', tf.expand_dims(response[0], -1), 5)
#
#      # Two more metrics to monitor the performance of training
#      tf.summary.scalar('center_score_error', center_score_error(response))
#      tf.summary.scalar('center_dist_error', center_dist_error(response))
#
#  def setup_global_step(self):
#    """Sets up the global step Tensor"""
#    global_step = tf.Variable(
#      initial_value=0,
#      name='global_step',
#      trainable=False,
#      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
#
#    self.global_step = global_step
#
#  def setup_embedding_initializer(self):
#    """Sets up the function to restore embedding variables from checkpoint."""
#    embed_config = self.config['embed_config']
#    if self.mode != "inference" and embed_config['embedding_checkpoint_file']:
#      if embed_config['embedding_name'] == 'siamese_fc':
#        # Restore Siamese FC models from .mat model files
#        initialize = load_mat_model(embed_config['embedding_checkpoint_file'], 'siamese_fc/', 'detection/')
#
#        def restore_fn(sess):
#          logging.info("Restoring embedding variables from checkpoint file %s",
#                          embed_config['embedding_checkpoint_file'])
#          sess.run([initialize])
#
#      self.init_fn = restore_fn
#
#  def build(self,reuse=False):
#    """Creates all ops for training and evaluation"""
#    self.build_inputs()
#    self.build_image_embeddings(reuse=reuse)
#    self.build_template()
#    self.build_detection(reuse=reuse)
#    self.build_loss()
#    self.setup_embedding_initializer()
#    if self.is_training():
#      self.setup_global_step()
#    return self.response









class SiameseModel:
  def __init__(self, model_config, train_config, mode='train'):
    self.model_config = model_config
    self.train_config = train_config
    self.mode = mode
    assert mode in ['train', 'val', 'inference']
    self.config = model_config

    self.gt=None
    self.dataloader = None
    self.examplars = None
    self.instances = None
    self.response = None
    self.batch_loss = None
    self.total_loss = None
    self.init_fn = None
    self.global_step = None

  def is_training(self):
    """Returns true if the model is built for training mode"""
    return self.mode == 'train'
    
  def build_inputs(self):
    """Input prefetching, preprocessing and batching

    Outputs:
      self.examplars: image batch of shape [batch, image_size, image_size, 3]
      self.instances: image batch of shape [batch, image_size, image_size, 3]
    """
    config = self.config

    # Prefetch image sequences
    with tf.device("/cpu:0"):  # Put data loading and preprocessing in CPU is substantially faster
      
      if self.mode == 'train':
        self.dataloader = DataLoader(self.config['prefetch_config_train'], self.is_training())
      if  self.mode == 'val':
        self.dataloader = DataLoader(self.config['prefetch_config_val'], self.is_training())      
      prefetch_queue = self.dataloader.get_queue()
      video = prefetch_queue.dequeue()
      examplar_image = video[0, :]
      instance_image = video[1, :]

      # Preprocess the examplar image and instance image
      self.preprocessing_name = self.config['preprocessing_name'] or self.config['embedding_name']
      image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        self.preprocessing_name, is_training=self.is_training())
      with tf.name_scope('Examplar'):
        # Crop an image slightly larger than examplar for bbox distortion.
        examplar_image = tf.image.resize_image_with_crop_or_pad(examplar_image,
                                                                self.config['z_image_size'] + 8,
                                                                self.config['z_image_size'] + 8)
        examplar_image = image_preprocessing_fn(examplar_image,
                                                self.config['z_image_size'],
                                                self.config['z_image_size'],
                                                fast_mode=self.config['fast_mode'],
                                                normalize=self.config['normalize_image'])
      with tf.name_scope('Instance'):
        # Note we use slightly smaller instance image to enable bbox distortion.
        instance_image = image_preprocessing_fn(instance_image,
                                                self.config['x_image_size'] - 2 * 8,
                                                self.config['x_image_size'] - 2 * 8,
                                                fast_mode=self.config['fast_mode'],
                                                normalize=self.config['normalize_image'])
        logging.info('For training, we use instance size {} - 2 * 8 ...'.format(self.config['x_image_size']))

      # Batch inputs.
      examplars, instances = tf.train.batch(
        [examplar_image, instance_image],
        batch_size=self.config['batch_size'],
        num_threads=2,
        capacity=10 * self.config['batch_size'])

    self.examplars = examplars
    self.instances = instances

  def build_image_embeddings(self,reuse=False):
    """Builds the image model subgraph and generates image embeddings

    Inputs:
      self.examplars: A tensor of shape [batch, hz, wz, 3]
      self.instances: A tensor of shape [batch, hx, wx, 3]
    
    Outputs:
      self.examplar_embeds
      self.instance_embeds
    """
    # Select embedding network
    config = self.config['embed_config']
    embedding_fn = embedding_factory.get_network_fn(
      config['embedding_name'],
      weight_decay=config['weight_decay'],
      trainable=config['train_embedding'],
      is_training=self.is_training(),
      init_method=get(config, 'init_method', None),
      bn_momentum=get(config, 'bn_momentum', 3e-4),
      bn_epsilon=get(config, 'bn_epsilon', 1e-6),)
    self.stride = embedding_fn.stride

    # Embedding for examplar images
#    self.examplar_embeds, _ = embedding_fn(self.examplars)
#    self.instance_embeds, _ = embedding_fn(self.instances, reuse=True)
    self.examplar_embeds,self.offset1_exam = embedding_fn(self.examplars,reuse=reuse,deform=False)
    self.instance_embeds, self.offset1_inst = embedding_fn(self.instances, reuse=True,deform=True)
  def build_template(self):
    with tf.variable_scope('target_template'):
      template_fn = template_factory.get_network_fn(
        self.config['template_name'],
        weight_decay=self.config['weight_decay'],
        is_training=self.is_training())

      self.templates, _ = template_fn(self.examplar_embeds)

  def build_detection(self, reuse=False):
    with tf.variable_scope('detection', reuse=reuse):
      def _translation_match(x, z):  # translation match for one example within a batch
        x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

      output = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                         (self.instance_embeds, self.templates),
                         dtype=self.instance_embeds.dtype)
      output = tf.squeeze(output, [1, 4])  # of shape e.g., [8, 15, 15]

      # Adjust score, this is required to make training possible.
      config = self.model_config['adjust_response_config']
      bias = tf.get_variable('biases', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=config['train_bias'])
      response = config['scale'] * output + bias
      self.response = response

  def build_loss(self):
    response = self.response
    response_size = response.get_shape().as_list()[1:3]  # [height, width]

    gt = _construct_gt_response(response_size,
                                self.config['batch_size'],
                                self.stride,
                                self.config['gt_config'])
    self.gt=gt

    with tf.name_scope('Loss'):
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=response,
                                                     labels=gt)

      with tf.name_scope('Balance_weights'):
        n_pos = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 1)))
        n_neg = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 0)))
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        class_weights = tf.where(tf.equal(gt, 1),
                                 w_pos * tf.ones_like(gt),
                                 tf.ones_like(gt))
        class_weights = tf.where(tf.equal(gt, 0),
                                 w_neg * tf.ones_like(gt),
                                 class_weights)
        loss = loss * class_weights

      # Note that we use reduce_sum instead of reduce_mean since the loss has
      # already been normalized by class_weights in spatial dimension.
      loss = tf.reduce_sum(loss, [1, 2])

      batch_loss = tf.reduce_mean(loss, name='batch_loss')
      tf.losses.add_loss(batch_loss)

      total_loss = tf.losses.get_total_loss()
      self.batch_loss = batch_loss
      self.total_loss = total_loss


      tf.summary.image(self.mode+'exemplar', self.examplars)
      tf.summary.image(self.mode+'instance', self.instances)

      mean_batch_loss, update_op1 = tf.metrics.mean(batch_loss)
      mean_total_loss, update_op2 = tf.metrics.mean(total_loss)
      with tf.control_dependencies([update_op1, update_op2]):

        tf.summary.scalar(self.mode+'batch_loss', mean_batch_loss)
        tf.summary.scalar(self.mode+'total_loss', mean_total_loss)

      if self.mode == 'train':
        with tf.name_scope("GT"):
          tf.summary.image('GT', tf.reshape(gt[0], [1] + response_size + [1]))
     
      tf.summary.image(self.mode+'Response', tf.expand_dims(tf.sigmoid(response), -1))
      tf.summary.histogram(self.mode+'Response', self.response)

        # Two more metrics to monitor the performance of training
      tf.summary.scalar(self.mode+'center_score_error', center_score_error(response))
      tf.summary.scalar(self.mode+'center_dist_error', center_dist_error(response))

  def setup_global_step(self):
    global_step = tf.Variable(
      initial_value=0,
      name='global_step',
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def setup_embedding_initializer(self):
    """Sets up the function to restore embedding variables from checkpoint."""
    embed_config = self.model_config['embed_config']
    if embed_config['embedding_checkpoint_file']:
      # Restore Siamese FC models from .mat model files
      initialize = load_mat_model(embed_config['embedding_checkpoint_file'],
                                  'convolutional_alexnet/', 'detection/')

      def restore_fn(sess):
        tf.logging.info("Restoring embedding variables from checkpoint file %s",
                        embed_config['embedding_checkpoint_file'])
        sess.run([initialize])

      self.init_fn = restore_fn

  def build(self, reuse=False):
    """Creates all ops for training and evaluation"""
    with tf.name_scope(self.mode):
      self.build_inputs()
      self.build_image_embeddings(reuse=reuse)
      self.build_template()
      self.build_detection(reuse=reuse)
      self.setup_embedding_initializer()

      
      self.build_loss()

      if self.is_training():
        self.setup_global_step()
