#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#

"""tracking model and training configurations

This configuration loads pretrained color model of [1] and save it using the file format of TensorFlow.

One Pass Evaluation on OTB-CVPR13 should get something like this:

	'ALL'	overlap : 68.19%	failures : 2.40	AUC : 0.606
	'BC'	overlap : 65.01%	failures : 3.06	AUC : 0.544
	'DEF'	overlap : 63.62%	failures : 3.22	AUC : 0.550
	'FM'	overlap : 62.38%	failures : 3.34	AUC : 0.540
	'IPR'	overlap : 66.18%	failures : 2.90	AUC : 0.576
	'IV'	overlap : 61.25%	failures : 3.70	AUC : 0.516
	'LR'	overlap : 56.40%	failures : 4.81	AUC : 0.417
	'MB'	overlap : 57.97%	failures : 4.11	AUC : 0.492
	'OCC'	overlap : 66.12%	failures : 2.99	AUC : 0.566
	'OPR'	overlap : 67.77%	failures : 2.43	AUC : 0.604
	'OV'	overlap : 69.48%	failures : 2.96	AUC : 0.586
	'SV'	overlap : 67.45%	failures : 2.57	AUC : 0.600

REFERENCE
  [1] Bertinetto, L., et al. (2016).
      "Fully-Convolutional Siamese Networks for Object Tracking."
      arXiv preprint arXiv:1606.09549.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import logging

DEBUG = False
BATCH_SIZE =8# 8#8  # training batch size

MODEL_CONFIG = {
  'seed': 0,

  ### Input config
  # input image database file path
  # Must be provided in training modes.
  'prefetch_config': {#'input_imdb': None,  #!need defined for different configuration
                      'preprocessing_name': 'siamese_fc_color',
                      'batch_size': BATCH_SIZE,
                      'values_per_shard': 10,
                      'max_frame_dist': 100,  # Maximum distance between any two random frame draw from videos.
                      'time_steps': 2, # It should always be 2, don't modify this!
                      },

  'preprocessing_name': 'basic',
  'fast_mode': False,  # fast data augmentation
  'normalize_image': False,  # if normalize image to the range of [-1, 1]
  'batch_size': BATCH_SIZE,

  # Examplar & Instance image size
  'z_image_size': 127,
  'x_image_size': 255,
  'u_image_size': 272,

  ### Embedding config
  'embed_config': {'embedding_name': 'siamese_fc',
                   'embedding_checkpoint_file': None, #need init for different config
                   'init_method': None,
                   'bn_momentum': 0.05,
                   'bn_epsilon': 1e-6,
                   'train_embedding': True,
                   'weight_decay': 5e-4},
  'stride': 8,

  ### Template config
  'template_name': 'identity',

  # L2 regularization weight
  'weight_decay': 5e-4,

  'adjust_response_config': {'train_bias': True,
                             'scale': 1e-3,
                             'upsample_method': 'bicubic',
                             'align_cornor': True},

  # Score map ground truth generation configs
  'gt_config': {'type': 'logistic',
                'rPos': 16,
                'rNeg': 0},

  # loss type one of 'L2', 'Cross_entropy'
  'loss_type': 'Cross_entropy',#'L2',#

  'z_embed_size': 6,
  'x_embed_size': 22,

  # It is often beneficial to track moving averages of the trained parameters.
  'moving_average': 0.0#0.9999#0.0,  # 0.0 will disable model moving average, default 0.9999
}
MODEL_CONFIG['r_embed_size'] = MODEL_CONFIG['x_embed_size'] - MODEL_CONFIG['z_embed_size'] + 1

assert MODEL_CONFIG['embed_config']['embedding_checkpoint_file'] or MODEL_CONFIG['embed_config']['train_embedding'], \
  'load embedding or train embedding, one has to be True'

# Sets the default training hyperparameters.
TRAIN_CONFIG = {
  'use_tfdbg': DEBUG,

  # Directory for saving and loading model checkpoints.
  # Need init at different station
  # 'train_dir': None,

  # Number of training epochs
  'epoch': 50,

  # Number of examples per epoch of training data
  'num_examples_per_epoch': 53200,

  # Train batch size should be equal to model_config.batch_size
  'batch_size': BATCH_SIZE,

  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'Momentum','momentum':0.9,'use_nesterov':False,},
#  'optimizer_config': {'optimizer': 'Adam'},
  # Learning rate configs
  'lr_config': {'policy': 'exponential',#,#
                'initial_lr': 0.00001,#0.00001 for 2 #0?001for no description # 0.01 for conv2 
        #        'initial_lr1': 0.5,
                'decay_steps':10000,
                #'num_epochs_per_decay': 25,
                'lr_decay_factor': 0.1,
                'staircase': True, },

  # If not None, clip gradients to this value.
  'clip_gradients':50.0,#None,#5.0,#5.0,

  # Frequency at which loss and global step are logged
  'log_every_n_steps': 10,

  # Frequency to save model
  'save_model_every_n_step': 10000,

  # sequence to be evaluated when model is saved
  # evaluation uses the parameters in track_config
  'eval_seqs': 'cvpr13',  # cvpr13/vid/otb50/otb100

  # How many model checkpoints to keep. No limit if None.
  'max_checkpoints_to_keep': None,

  # GPU percentage to use
  'gpu_fraction': 0.20,
}

TRACK_CONFIG = {
  # Directory for saving and loading model checkpoints.
  # Need init at different station
  # 'log_dir': None,

  # Logging level of inference, use 1 for detailed inspection. 0 for speed.
  'log_level': 0,

  "include_first": False, # If track target in the first frame. Normally, we don't track the first frame and use the ground
                         # truth instead. But this can be helpful when the initial ground truth is noisy.

  # Window influence of
  'window_influence': 0.176,  # The original paper use 0.168, but it seems 0.05 works better in this implementation.

  'num_scales': 3,  # number of scales to search, I am not sure if I have done scale search right...
  'scale_step': 1.0375,  # Scale changes between different scale search
  'scale_damp': 0.59,  # damping factor for scale update
  'scale_penalty': 0.9745,  # Score penalty for scale change

  # Gpu percentage to use
  'gpu_fraction': 0.20,
}

if DEBUG:
  from utils.misc import Experiment
  ex = Experiment()
  ex.config(lambda: [MODEL_CONFIG, TRAIN_CONFIG, TRACK_CONFIG])

else:
  from sacred import Experiment
  from sacred.observers import MongoObserver, FileStorageObserver

  ex = Experiment('Siam-FC')
  # using file logging storage


  @ex.config
  def configurations():
    log_dir = None
    run_name = None
    imdb_dir = None
    caffenet_dir = None
    gpu_fraction = None
    embedding_name = None
    model_config = MODEL_CONFIG
    train_config = TRAIN_CONFIG
    track_config = TRACK_CONFIG
    train_config['train_dir'] = osp.join(log_dir, 'track_model_checkpoints', run_name)
    train_config['caffenet_dir'] = caffenet_dir
    track_config['log_dir'] = osp.join(log_dir, 'track_model_inference', run_name)
    track_config['run_name'] = '' + run_name
    model_config['prefetch_config']['input_imdb'] = imdb_dir
    track_config['gpu_fraction'] = track_config['gpu_fraction'] = gpu_fraction
    model_config['embed_config']['embedding_name'] = embedding_name
    ex.observers.append(FileStorageObserver.create(osp.join(log_dir, 'sacred')))
    logging.getLogger().setLevel(logging.INFO)
  @ex.named_config
  def workstation():
    log_dir = '/home/v-chaoqw/MYSFC-ORI/'
    imdb_dir = "/home/v-chaoqw/MYSFC-ORI/workspace/train_imdb.pickle"
    caffenet_dir = '/home/v-chaoqw/MYSFC-ORI/caffenet.npy'
    gpu_fraction = 0.5
  @ex.named_config
  def gcr():
    log_dir = '/data/workspaces/O-SFC-V1'
    imdb_dir = '/data/datasets/ILSVRC15_VID/ILSVRC2015_INFO_BILI/train_imdb.pickle'
    caffenet_dir = '/data/codes/caffe-tensorflow/output/caffenet.npy'
    gpu_fraction = 0.4
  @ex.named_config  
  def sfc():
    run_name = 'SFC-2-momentum-lr0.00001-clip50-initconv1-4-bias-relu'
    embedding_name = 'siamese_fc'
#  @ex.named_config
#  def obj_sfc():
#    run_name = 'OBJ-SFC'
#    embedding_name = 'siamese_obj_fc'



