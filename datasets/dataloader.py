#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import threading
import tensorflow as tf

from datasets.transforms import Compose, Map, RandomGray, RandomCrop, RandomSizedCrop, CenterCrop, RandomStretch, \
  RandomResolution, IMap, FixGray
from datasets.vid import VID
from datasets.sampler import Sampler
from utils.misc import get


class DataLoader(object):
  def __init__(self, config, is_training):
    self.config = config
    self.threads = None
    self.queue = None

    preprocess_name = get(config, 'preprocessing_name', None)
    if preprocess_name == 'siamese_fc_color':
      logging.info('preproces -- siamese_fc_color')
      transform = Compose([Map(RandomStretch()),
                           Map(CenterCrop((255 - 8, 255 - 8))),
                           Map(RandomCrop(255 - 2 * 8)),])
      self.image_shape = (255 - 2 * 8, 255 - 2 * 8, 3)
    elif preprocess_name == 'siamese_fc_gray':
      logging.info('preproces -- siamese_fc_gray')
      transform = Compose([Map(RandomStretch()),
                           Map(CenterCrop((255 - 8, 255 - 8))),
                           Map(RandomCrop(255 - 2 * 8)),
                           IMap(RandomGray()),])
      self.image_shape = (255 - 2 * 8, 255 - 2 * 8, 3)
    elif preprocess_name == 'siamese_fc_gray_fixed':
      logging.info('preproces -- siamese_fc_gray_fixed')
      transform = Compose([Map(RandomStretch()),
                           Map(CenterCrop((255 - 8, 255 - 8))),
                           Map(RandomCrop(255 - 2 * 8)),
                           IMap(FixGray(batch_size=config['batch_size'])),])
      self.image_shape = (255 - 2 * 8, 255 - 2 * 8, 3)
    elif preprocess_name == 'translate':
      logging.info('preproces -- translate')
      transform = Compose([Map(CenterCrop((255 - 8, 255 - 8))),
                           Map(RandomCrop(255 - 2 * 8))])
      self.image_shape = (255 - 2 * 8, 255 - 2 * 8, 3)
    elif preprocess_name == 'model_updater':
      logging.info('preproces -- model_updater')
      transform = None
      self.image_shape = (255 - 2 * 8, 255 - 2 * 8, 3)
    elif preprocess_name == 'mine':
      logging.info('preproces -- my data augmentation')
      transform = Compose([Map(RandomResolution(get(config, 'min_downsample', 0.5))),
                           Map(RandomStretch()),
                           Map(CenterCrop((255 - 8, 255 - 8))),
                           Map(RandomCrop(255 - 2 * 8))])
      self.image_shape = (255 - 2 * 8, 255 - 2 * 8, 3)
    else:
      logging.info('preproces -- None')
      transform = None
      self.image_shape = (255, 255, 3)
    print(config['input_imdb'])

    root = get(config, 'root', None)
    self.dataset = VID(root, config['input_imdb'], config['time_steps'], config['max_frame_dist'], transform)
    self.sampler = Sampler(self.dataset, shuffle=is_training)

    self.construct_queue()

  def construct_queue(self):
    config = self.config
    with tf.name_scope('Prefetcher'):
      image_shape = self.image_shape
      video_shape = (config['time_steps'],) + image_shape
      self.video = tf.placeholder(dtype=tf.uint8, shape=video_shape)

      # We use FIFOQueue since videos will be shuffled before entering queue
      capacity = config['values_per_shard'] + 10 * config['batch_size']
      self.queue = tf.FIFOQueue(
        shapes=[video_shape],
        capacity=capacity,
        dtypes=[tf.uint8], name="fifo_input_queue")

      self.enqueue_op = self.queue.enqueue([self.video])

      tf.summary.scalar(
        "queue/input_queue/fraction_of_%d_full" % (capacity),
        tf.cast(self.queue.size(), tf.float32) * (1. / capacity))

  def get_queue(self):
    return self.queue

  def thread_main(self, sess):
    """
    Function run on alternate thread. Basically, keep adding data to the queue.
    """
    t = threading.currentThread()
    for video_id in self.sampler:
      video = self.dataset[video_id]
      if getattr(t, "do_run", True):
        sess.run(self.enqueue_op, feed_dict={self.video: video})
      else:
        break

  def request_stop(self):
    for t in self.threads:
      t.do_run = False

  def join(self):
    for t in self.threads:
      t.join()

  def start_threads(self, sess):
    """Start background theads to feed queue"""
    threads = []
    num_reader_threads = 1
    for n in range(num_reader_threads):
      logging.info('Starting threads {} ...\n'.format(n))
      t = threading.Thread(target=self.thread_main, args=(sess,))
      t.daemon = True  # thread will close when parent quits
      t.start()
      threads.append(t)
    self.threads = threads
    return threads