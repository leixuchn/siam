#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Base wrapper class for performing inference with a track model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
import logging

# pylint: disable=unused-argument
from utils.misc import get


class InferenceWrapperBase(object):
  """Base wrapper class for performing inference with TrackModel"""

  def __init__(self):
    pass

  def build_model(self, model_config, track_config):
    """Builds the model for inference

    Args:
    - model_config: Object containing configuration for building the model.

    Returns:
    - model: the model object
    """
    tf.logging.fatal("Please implement build_model in subclass")

  def _create_restore_fn(self, checkpoint_path, saver):
    """Creates a function that restores a model from checkpoint.

    Args:
    - checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.
    - saver: Saver for restoring variables from the checkpoint file.
    Returns:
    - restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    Raises:
      ValueError: If checkpoint_path does not refer to a checkpoint file or a
        directory containing a checkpoint file.
    """

    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_path:
        raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

    def _restore_fn(sess):
      logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      logging.info("Successfully loaded checkpoint: %s",
                      os.path.basename(checkpoint_path))

    return _restore_fn

  def build_graph_from_config(self, model_config, track_config, checkpoint_path):
    """Builds the inference graph from a configuration object.

    Args:
    - model_config: Object containing configuration for building the model.
    - checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
    - restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    logging.info("Building model.")
    self.build_model(model_config, track_config)
    ema = tf.train.ExponentialMovingAverage(model_config['moving_average'])
    if model_config['moving_average'] > 0:
      variables_to_restore = ema.variables_to_restore()
    else:
      variables_to_restore = ema.variables_to_restore(moving_avg_variables=[]) 

    # Filter out State variables
    variables_to_restore_filterd = {}
    for key, value in variables_to_restore.items():
      if key.split('/')[1] != 'State':
        variables_to_restore_filterd[key] = value

    saver = tf.train.Saver(variables_to_restore_filterd)

    return self._create_restore_fn(checkpoint_path, saver)

  def build_graph_from_proto(self, graph_def_file, saver_def_file,
                             checkpoint_path):
    """Builds the inference graph from serialized GraphDef and SaverDef protos.

    Args:
    - graph_def_file: File containing a serialized GraphDef proto.
    - saver_def_file: File containing a serialized SaverDef proto.
    - checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.
    Returns:
    - restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    # Load the Graph.
    logging.info("Loading GraphDef from file: %s", graph_def_file)
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(graph_def_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    # Load the Saver.
    logging.info("Loading SaverDef from file: %s", saver_def_file)
    saver_def = tf.train.SaverDef()
    with tf.gfile.FastGFile(saver_def_file, "rb") as f:
      saver_def.ParseFromString(f.read())
    saver = tf.train.Saver(saver_def=saver_def)

    return self._create_restore_fn(checkpoint_path, saver)

  def inference_step(self, sess, input_feed):
    """Runs one step of inference."""
    tf.logging.fatal("Please implement inference_step in subclass")

# pylint: enable=unused-argument
