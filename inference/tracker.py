#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.
"""Class for tracking using a track model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path as osp
import numpy as np
from scipy.misc import imsave
from scipy.ndimage import imread

from inference.inference_utils import get_crops, im2rgb, convert_bbox
from inference.vot import Rectangle
from utils.misc import get_center, get


class TargetState(object):
  """Represents target state."""

  def __init__(self, bbox, search_pos, scale_idx):
    self.bbox = bbox
    self.search_pos = search_pos  # target center position in search image
    self.scale_idx = scale_idx


class Tracker(object):
  """Tracker based on a track model."""

  def __init__(self, model, model_config, track_config):
    """Initializes the tracker.

    Args:
      model: Object encapsulating a trained track model. Must have
        methods inference_step(). For example, an instance of
        InferenceWrapperBase.
      model_config: track model configurations.
      track_config: tracking configurations.
    """
    self.model = model
    self.model_config = model_config
    self.track_config = track_config

    self.z_image_size = model_config['z_image_size']
    self.x_image_size = model_config['x_image_size']
    self.r_embed_size = model_config['r_embed_size']
    self.r_image_size = model_config['u_image_size']

    self.num_scales = track_config['num_scales']
    self.log_level = track_config['log_level']
    logging.info('track num scales -- {}'.format(track_config['num_scales']))
    scales = np.arange(self.num_scales) - get_center(self.num_scales)
    self.search_factors = [self.track_config['scale_step'] ** x for x in scales]

    # Cosine window
    window = np.dot(np.expand_dims(np.hanning(self.r_image_size), 1),
                    np.expand_dims(np.hanning(self.r_image_size), 0))
    self.window = window / np.sum(window)  # normalize window

  def track(self, sess, handle, logdir='/tmp'):
    """Runs tracking on a single image sequence.

    Args:
      sess: TensorFlow Session object.
      handle: a handle which generates image files and target pos in 1st frame,
        which mimic the interface of VOT.
    Returns:
      A list of Trajectories sorted by descending score.
    """
    # Get initial target bounding box and convert to center based
    bbox = handle.region()
    bbox = convert_bbox(bbox, 'center-based')

    # Feed in the first frame image to set initial state.
    # Note we use different padding values for each image while the original implementation uses only the average value
    # of the first image for all images.
    bbox_feed = [bbox.y, bbox.x, bbox.height, bbox.width]
    input_feed = [handle.frame(), bbox_feed]
    frame2crop_scale = self.model.initialize(sess, input_feed)

    # Storing target state
    original_target_height = bbox.height
    original_target_width = bbox.width
    search_center = np.array([get_center(self.x_image_size),
                              get_center(self.x_image_size)])
    current_target_state = TargetState(bbox=bbox,
                                       search_pos=search_center,
                                       scale_idx=int(get_center(self.num_scales)))

    # If track first frame
    include_first = get(self.track_config, 'include_first', False)
    logging.info('tracking include first -- {}'.format(include_first))

    # Run tracking loop
    i = -1 # Processing the i th frame in image sequence,
           # note that we will use the first image twice in total.
           # 1. It is used to initialize the tracker
           # 2. It is used as a test example for tracker, the detected result won't affect the final metrics though.
           #    this is needed because both OTB and VOT benchmark require a list of tracking results equal to the
           #    length of the test image sequences including the first image.
    while True:
      # Read new image
      filename = handle.frame()
      if not filename:
        if self.log_level > 0:
          np.save(osp.join(logdir, 'num_frames.npy'), [i + 1])
        break # All image files are processed, exiting while loop
      i += 1
      if i > 0 or include_first: # We don't really want to process the first image unless intended to do so.
        # Prepare input feed
        bbox_feed = [current_target_state.bbox.y, current_target_state.bbox.x,
                     current_target_state.bbox.height, current_target_state.bbox.width]
        input_feed = [filename, bbox_feed]

        # Feed in input
        outputs, metadata = self.model.inference_step(sess, input_feed)
        search_scale_list = outputs['scale_xs']
        response = outputs['response']

        # Choose the scale whole response map has the highest peak
        if self.num_scales > 1:
          current_scale_idx = int(get_center(self.num_scales))
          best_scale = current_scale_idx
          best_peak = - np.inf
          for s in range(self.num_scales):
            this_response = response[s]
            this_peak = np.max(this_response[:])

            # Penalize change of scale
            if s != current_scale_idx:
              this_peak *= self.track_config['scale_penalty']
            if this_peak > best_peak:
              best_peak = this_peak
              best_scale = s
        else:
          best_scale = 0

        response = response[best_scale]

        if self.log_level > 0:
          np.save(osp.join(logdir, 'best_scale{}.npy'.format(i)), [best_scale])
          np.save(osp.join(logdir, 'response{}.npy'.format(i)), response)

        # Normalize response
        with np.errstate(all='raise'):  # Raise error if something goes wrong
          logging.debug('mean response: {}'.format(np.mean(response)))
          response = response - np.min(response)
          response = response / np.sum(response)

        # Apply windowing
        window_influence = self.track_config['window_influence']
        response = (1 - window_influence) * response + window_influence * self.window
        if self.log_level > 0:
          np.save(osp.join(logdir, 'response_windowed{}.npy'.format(i)), response)

        # Find maximum response
        r_max, c_max = np.unravel_index(response.argmax(),
                                        response.shape)

        # Convert from crop-relative coordinates to frame coordinates
        p_coor = np.array([r_max, c_max])
        # displacement from the center in instance final representation ...
        disp_instance_final = p_coor - get_center(self.r_image_size)
        # ... in instance feature space ...
        upsampling_factor = self.r_image_size / self.r_embed_size
        disp_instance_feat = disp_instance_final / upsampling_factor
        # ... Avoid empty position ...
        r_radius = int(self.r_embed_size / 2)
        disp_instance_feat = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)
        # ... in instance input ...
        disp_instance_input = disp_instance_feat * self.model_config['stride']
        # ... in instance original crop (in frame coordinates)
        disp_instance_frame = disp_instance_input / search_scale_list[best_scale]
        # Position within frame in frame coordinates
        y = current_target_state.bbox.y
        x = current_target_state.bbox.x
        y += disp_instance_frame[0]
        x += disp_instance_frame[1]

        # Target scale damping and saturation
        target_scale = current_target_state.bbox.height / original_target_height
        search_factor = self.search_factors[best_scale]
        scale_damp = self.track_config['scale_damp']  # damping factor for scale update
        target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
        target_scale = np.maximum(0.2, np.minimum(5.0, target_scale))

        # Some book keeping
        height = original_target_height * target_scale
        width = original_target_width * target_scale
        current_target_state.bbox = Rectangle(x, y, width, height)
        current_target_state.scale_idx = best_scale
        current_target_state.search_pos = search_center + disp_instance_input

        assert 0 <= current_target_state.search_pos[0] < self.x_image_size, \
               'target position in feature space should be no larger than input image size'
        assert 0 <= current_target_state.search_pos[1] < self.x_image_size, \
               'target position in feature space should be no larger than input image size'
        logging.debug('search_position: {}'.format(current_target_state.search_pos))

      # I used to put this at the beginning of the loop, which makes the code visually looks better.
      # But it is also more demanding to really understand the logic behind that and prone to make
      # bugs. My opinion now is *easy is better than concise, if you can't have both.*
      # Record tracked target position
      reported_bbox = convert_bbox(current_target_state.bbox, 'top-left-based')
      handle.report(reported_bbox)