#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Train the track model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import json
import os
from pprint import pprint
import os.path as osp
import time
from datetime import datetime
import re
import numpy as np
import tensorflow as tf
from subprocess import Popen
from skimage import io
from PIL import Image  
import scipy
import configuration
from models import track_model as siamese_model
from models.model_utils import load_caffenet
from utils import crash_on_ipy
from utils.misc import auto_select_gpu,mkdir_p 
from configuration import ex
import logging
import random
import tensorflow.contrib.slim as slim
# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
tf.logging.set_verbosity(tf.logging.DEBUG)

#
#def _configure_learning_rate(train_config, global_step):
#  """Configures the learning rate.
#
#  Args:
#    num_examples_per_epoch: The number of samples in each epoch of training.
#    global_step: The global_step tensor.
#
#  Returns:
#    A `Tensor` representing the learning rate.
#
#  Raises:
#    ValueError: if
#  """
#  lr_config = train_config['lr_config']
#
#  num_batches_per_epoch = \
#    int(train_config['num_examples_per_epoch'] / train_config['batch_size'])
#
#  lr_policy = lr_config['policy']
#  if lr_policy == 'piecewise_constant':
#    lr_boundaries = \
#      [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
#    return tf.train.piecewise_constant(global_step,
#                                       lr_boundaries,
#                                       lr_config['lr_values'])
#  elif lr_policy == 'exponential':
#    decay_steps = lr_config['decay_steps']
#    return tf.train.exponential_decay(lr_config['initial_lr'],
#                                      global_step,
#                                      decay_steps=decay_steps,
#                                      decay_rate=lr_config['lr_decay_factor'],
#                                      staircase=lr_config['staircase'])
#  else:
#    raise NotImplementedError
##def _configure_learning_rate1(train_config, global_step):
##  """Configures the learning rate.
##
##  Args:
##    num_examples_per_epoch: The number of samples in each epoch of training.
##    global_step: The global_step tensor.
##
##  Returns:
##    A `Tensor` representing the learning rate.
##
##  Raises:
##    ValueError: if
##  """
##  lr_config = train_config['lr_config']
##
##  num_batches_per_epoch = \
##    int(train_config['num_examples_per_epoch'] / train_config['batch_size'])
##
##  lr_policy = lr_config['policy']
##  if lr_policy == 'piecewise_constant':
##    lr_boundaries = \
##      [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
##    return tf.train.piecewise_constant(global_step,
##                                       lr_boundaries,
##                                       lr_config['lr_values'])
##  elif lr_policy == 'exponential':
##    decay_steps = lr_config['decay_steps']
##    return tf.train.exponential_decay(lr_config['initial_lr1'],
##                                      global_step,
##                                      decay_steps=decay_steps,
##                                      decay_rate=lr_config['lr_decay_factor'],
##                                      staircase=lr_config['staircase'])
##  else:
##    raise NotImplementedError
#
#def _configure_optimizer(train_config, learning_rate):
#  """Configures the optimizer used for training.
#
#  Args:
#    learning_rate: A scalar or `Tensor` learning rate.
#
#  Returns:
#    An instance of an optimizer.
#
#  Raises:
#    ValueError: if train_config.optimizer is not recognized.
#  """
#  optimizer_config = train_config['optimizer_config']
#  if optimizer_config['optimizer'] == 'Momentum':
#    optimizer = tf.train.MomentumOptimizer(
#      learning_rate,
#      momentum=optimizer_config['momentum'],
#      use_nesterov=optimizer_config['use_nesterov'],
#      name='Momentum')
#  elif optimizer_config['optimizer'] == 'SGD':
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#  elif optimizer_config['optimizer'] == 'Adam':
#    optimizer = tf.train.AdamOptimizer(learning_rate)
#  else:
#    raise ValueError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
#  return optimizer
#
#
#def _save_cfgs(train_dir, model_config, train_config, track_config):
#  # Save all configurations in JSON for future reference.
#  with open(osp.join(train_dir, 'model_config.json'), 'w') as f:
#    json.dump(model_config, f, indent=2)
#  with open(osp.join(train_dir, 'train_config.json'), 'w') as f:
#    json.dump(train_config, f, indent=2)
#  with open(osp.join(train_dir, 'track_config.json'), 'w') as f:
#    json.dump(track_config, f, indent=2)
#
#
#
#
#
#
#
#@ex.main
#def main(model_config, train_config, track_config):
#  # Create training directory
#  train_dir = train_config['train_dir']
#  if not tf.gfile.IsDirectory(train_dir):
#    logging.info('Creating training directory: %s', train_dir)
#    tf.gfile.MakeDirs(train_dir)
#
#  # Build the Tensorflow graph
#  g = tf.Graph()
#  with g.as_default():
#    # Set fixed seed
#    np.random.seed(model_config['seed'])
#    tf.set_random_seed(model_config['seed'])
#
#    # Build the model
#    model = track_model.TrackModel(model_config, mode='train')
#    response=model.build()
#
#    model_va = track_model.TrackModel(model_config, mode='val')
#    response1=model_va.build(reuse=True)
#    
#    model_config = model.config  # model config will be updated by TrackModel
#
#    # Save configurations for future reference
#    _save_cfgs(train_dir, model_config, train_config, track_config) 
#
#    # Build optimizer
#    learning_rate = _configure_learning_rate(train_config, model.global_step)
#    optimizer = _configure_optimizer(train_config, learning_rate)
#    tf.summary.scalar('learning_rate', learning_rate)
#
####     Set up the training ops
#    opt_op = tf.contrib.layers.optimize_loss(
#      loss=model.total_loss,
#      global_step=model.global_step,
#      learning_rate=learning_rate,
#      optimizer=optimizer,
#      clip_gradients=train_config['clip_gradients'],
#      learning_rate_decay_fn=None,
#      summaries=['learning_rate', 'loss'])
#
#    variables_to_restore1 = tf.global_variables()
#    vname=[v.name for v in variables_to_restore1]
#    pprint(vname)
#    e1=re.compile(".*def")
#    var_0 = [v for v in variables_to_restore1 if e1.match(v.name)]#last
#    var_1 = [v for v in variables_to_restore1 if  not e1.match(v.name)] 
##    print('aiaiaiaiaiaiaiiaaiia',var_1)
##
##    learning_rate1 = _configure_learning_rate1(train_config, model.global_step)
#    learning_rate = _configure_learning_rate(train_config, model.global_step)
#    optimizer1 = _configure_optimizer(train_config, 0.0)
#    optimizer = _configure_optimizer(train_config, learning_rate)    
#    tf.summary.scalar('learning_rate', learning_rate)
#    opt_op0 = tf.contrib.layers.optimize_loss(
#      loss=model.total_loss,
#      global_step=model.global_step,
#      learning_rate=learning_rate,
#      optimizer=optimizer,
#      clip_gradients=train_config['clip_gradients'],
#      learning_rate_decay_fn=None,variables=var_0)
#    opt_op1 = tf.contrib.layers.optimize_loss(
#      loss=model.total_loss,
#      global_step=model.global_step,
#      learning_rate=0.0,
#      optimizer=optimizer1,
#      clip_gradients=train_config['clip_gradients'],
#      learning_rate_decay_fn=None,variables=var_1)    
##    opt_op_val = tf.contrib.layers.optimize_loss(
##      loss=model_va.total_loss,
##      global_step=model_va.global_step,
##      learning_rate=0.0,
##      optimizer=optimizer1,
##      clip_gradients=train_config['clip_gradients'],
##      learning_rate_decay_fn=None,)
#
#    # It is often beneficial to track moving averages of the trained parameters.
#    if model_config['moving_average'] > 0:
#      ema = tf.train.ExponentialMovingAverage(model_config['moving_average'], model.global_step)
#      maintain_averages_op = ema.apply(tf.trainable_variables())
#    else:
#      logging.info('Model moving average is disabled since decay factor is 0')
#      maintain_averages_op = tf.no_op()
#
#    # maintain_averages_op = tf.no_op(name='dumb')
#    with tf.control_dependencies([opt_op0,opt_op1, maintain_averages_op]):                                  #######################opt
#      train_op = tf.no_op(name='train')
##    with tf.control_dependencies([opt_op_val]):                                  #######################opt
##      train_op_val = tf.no_op(name='val')
##      
#      
#    saver = tf.train.Saver(tf.global_variables(),
#                           max_to_keep=train_config['max_checkpoints_to_keep'])
#
#    # Construct summary writer and store model graph
#    summary_writer = tf.summary.FileWriter(train_dir, g)
#
#    # Build the summary operation
#    summary_op = tf.summary.merge_all()
#
#    # Use ~7.5G GPU memory
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=train_config['gpu_fraction'])
#    sess_config = tf.ConfigProto(gpu_options=gpu_options)
#
#    sess = tf.Session(config=sess_config)
#    model_path = tf.train.latest_checkpoint(train_config['train_dir'])
#    
#    
#    
#
#
#
#
#
#
#  #  if not model_path:
#    if True:
#     # import re
#      # Initialize all variables
#      sess.run(tf.global_variables_initializer())
#      sess.run(tf.local_variables_initializer())
#      e1=re.compile(".*def.*") 
#      re1=re.compile(".*siamese_fc/conv1")
#      re2=re.compile(".*siamese_fc/conv2")
#      re3=re.compile(".*siamese_fc/conv3")
#      re4=re.compile(".*siamese_fc/conv4")
##      re4=re.compile(".*siamese_fc/conv4")
#      e3=re.compile(".*global_step.*") 
#      e4=re.compile(".*Momentum")      
#      e2=re.compile(".*OptimizeLoss.*")
#      variables_to_restore1 = tf.global_variables()
#      var_0 = [v for v in variables_to_restore1 if  not e1.match(v.name) and not e4.match(v.name) and not e3.match(v.name) and not e2.match(v.name)]
##      var_0 = [v for v in variables_to_restore1 if not e3.match(v.name) and not e4.match(v.name) and not e2.match(v.name)]
##      var_0 = [v for v in variables_to_restore1 if (re1.match(v.name) or re2.match(v.name) or re3.match(v.name) or re4.match(v.name)) and not e4.match(v.name) and not e2]
##      va=[v for v in variables_to_restore1 if not v in var_0 ]
##      start_step = 10001
##      var_0 = [v for v in variables_to_restore1  ]
#      
##      var_0 = [v for v in variables_to_restore1 if not e4.match(v.name)]
##      print(var_0)
##      print('kkkkkkkkkkkkkkkkkkkkkkk',va)
#      saver0 = tf.train.Saver(var_0)
#      
#      
##      saver0.restore(sess, tf.train.latest_checkpoint("/home/v-chaoqw/MYSFC-ORI/track_model_checkpoints/SFC-2-momentum-lr0.001-clip50-initconv1-4-fixqian/"))
##      
# 
#      
#      saver0.restore(sess, tf.train.latest_checkpoint("/home/v-chaoqw/MYSFC-ORI/track_model_checkpoints/SFC-only/"))
#      
##      saver0.restore(sess, tf.train.latest_checkpoint("/home/v-chaoqw/MYSFC-ORI/track_model_checkpoints/SFC-2-SGD-lr0.1-decay7000-nosplit-clip50-shortcut1//"))
#
#
#
#
# 
#
#
#
##      if model_config['embed_config']['embedding_name'] == 'siamese_obj_fc':
##        load_caffenet(train_config['caffenet_dir'])
#      start_step = 0
#
#      # Load pretrained embedding model if needed
##      if model_config['embed_config']['embedding_checkpoint_file']:
##        model.init_fn(sess)
#    else:
#      print("**************************************")
#      logging.info('Restor.offset1_inste from last checkpoint: {}'.format(model_path))
#      sess.run(tf.local_variables_initializer())
#      saver.restore(sess, model_path)
#      start_step = tf.train.global_step(sess, model.global_step.name) + 1
#
#    # Start queue runners
#    tf.train.start_queue_runners(sess=sess)
#    model.dataloader.start_threads(sess=sess)  # start customized queue runner
#    model_va.dataloader.start_threads(sess=sess)  # start customized queue runner
#
#    if train_config['use_tfdbg']:
#      from tensorflow.python import debug as tf_debug
#      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#
#    # Training loop
#    total_steps = int(train_config['epoch'] * train_config['num_examples_per_epoch'] / train_config['batch_size'])
#    logging.info('training for {} steps'.format(total_steps))
#    for step in range(start_step, total_steps):
#      start_time = time.time()
#      _, loss, batch_loss,offset1_inst,instances = sess.run([train_op, model.total_loss, model.batch_loss,model.offset1_inst,model.instances])
#      
#      duration = time.time() - start_time
#
#      if step % 10 == 0:
#        examples_per_sec = model_config['batch_size'] / float(duration)
#        time_remain = train_config['batch_size'] * (total_steps - step) / examples_per_sec
#        m, s = divmod(time_remain, 60)
#        h, m = divmod(m, 60)
#        format_str = ('%s: step %d, loss = %.2f, batch loss = %.2f (%.1f examples/sec; %.3f '
#                      'sec/batch; %dh:%02dm:%02ds remains)')
#        logging.info(format_str % (datetime.now(), step, loss, batch_loss,
#                                      examples_per_sec, duration, h, m, s))
#       
#
#
#      if step % 100 == 0:
#
#        summary_str = sess.run(summary_op)
#        summary_writer.add_summary(summary_str, step)
#        
#        sess.run(tf.Print(response1,[response1],summarize=15*15))
#
#      if step % 1000 == 0:
#        print(train_config['train_dir'])
#        
##        loss_val, batch_loss_val,offset1_inst_val,instances_val = sess.run([ model_va.total_loss, model_va.batch_loss,model_va.offset1_inst,model_va.instances])
##        for i in range(8):
##          np_save_name=str(step)+'.'+str(i)+'.npy'
##          im_save_name=str(step)+'.'+str(i)+'.jpg'
##          np_save_path=osp.join("/home/v-chaoqw/Deformable-ConvNets/noclip-fixqian-lr0.0001/deform_conv/np/",np_save_name)
##          im_save_path=osp.join("/home/v-chaoqw/Deformable-ConvNets/noclip-fixqian-lr0.0001/deform_conv/",im_save_name)
##          numpy.save(np_save_path,offset1_inst_val[i:(i+1),:,:,:])
##       
##          im=instances_val[i]#Image.fromarray(instances[0]) 
##
##          scipy.misc.imsave(im_save_path,im)
#        
#
#
#     
#        
#
#
#
#
#      # Save the model checkpoint periodically
#      if step % train_config['save_model_every_n_step'] == 0 or (step + 1) == total_steps:
#        checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
#        saver.save(sess, checkpoint_path, global_step=step)
#if __name__ == '__main__':
#  ex.run_commandline()








##########################################################################


def _configure_learning_rate(train_config, global_step):
  """Configures the learning rate.

  Args:
    num_examples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  lr_config = train_config['lr_config']

  num_batches_per_epoch = \
    int(train_config['num_examples_per_epoch'] / train_config['batch_size'])

  lr_policy = lr_config['policy']
  if lr_policy == 'piecewise_constant':
    lr_boundaries = \
      [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
    return tf.train.piecewise_constant(global_step,
                                       lr_boundaries,
                                       lr_config['lr_values'])
  elif lr_policy == 'exponential':
    decay_steps = int(num_batches_per_epoch) * lr_config['num_epochs_per_decay']
    return tf.train.exponential_decay(lr_config['initial_lr'],
                                      global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=lr_config['lr_decay_factor'],
                                      staircase=lr_config['staircase'])
  else:
    raise NotImplementedError

def _save_cfgs(train_dir, model_config, train_config, track_config):
  # Save all configurations in JSON for future reference.
  with open(osp.join(train_dir, 'model_config.json'), 'w') as f:
    json.dump(model_config, f, indent=2)
  with open(osp.join(train_dir, 'train_config.json'), 'w') as f:
    json.dump(train_config, f, indent=2)
  with open(osp.join(train_dir, 'track_config.json'), 'w') as f:
    json.dump(track_config, f, indent=2)



def _configure_optimizer(train_config, learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if train_config.optimizer is not recognized.
  """
  optimizer_config = train_config['optimizer_config']
  if optimizer_config['optimizer'] == 'Momentum':
    optimizer = tf.train.MomentumOptimizer(
      learning_rate,
      momentum=optimizer_config['momentum'],
      use_nesterov=optimizer_config['use_nesterov'],
      name='Momentum')
  elif optimizer_config['optimizer'] == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_config['optimizer'] == 'Adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
  return optimizer


@ex.automain
def main(model_config, train_config, track_config):


  # Create training directory which will be used to save: configurations, model files, TensorBoard logs
  train_dir = train_config['train_dir']
  if not osp.isdir(train_dir):
    logging.info('Creating training directory: %s', train_dir)
    mkdir_p(train_dir)

  g = tf.Graph()
  with g.as_default():
    # Set fixed seed for reproducible experiments
    random.seed(model_config['seed'])
    np.random.seed(model_config['seed'])
    tf.set_random_seed(model_config['seed'])

    # Build the training and validation model
    model = siamese_model.SiameseModel(model_config, train_config, mode='train')
    model.build()
    model_va = siamese_model.SiameseModel(model_config, train_config, mode='val')
    model_va.build(reuse=True)

    # Save configurations for future reference
    _save_cfgs(train_dir, model_config, train_config, track_config)

    learning_rate = _configure_learning_rate(train_config, model.global_step)
    optimizer = _configure_optimizer(train_config, learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    # Set up the training ops
    opt_op = tf.contrib.layers.optimize_loss(
      loss=model.total_loss,
      global_step=model.global_step,
      learning_rate=learning_rate,
      optimizer=optimizer,
      clip_gradients=train_config['clip_gradients'],
      learning_rate_decay_fn=None,
      summaries=['learning_rate'])
      
    

    with tf.control_dependencies([opt_op]):
      train_op = tf.no_op(name='train')



    summary_writer = tf.summary.FileWriter(train_dir, g)
    summary_op = tf.summary.merge_all()

    global_variables_init_op = tf.global_variables_initializer()
    local_variables_init_op = tf.local_variables_initializer()


    # Dynamically allocate GPU memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=sess_config)
    model_path = tf.train.latest_checkpoint(train_config['save_dir'])

    if not model_path:
      sess.run(global_variables_init_op)
      sess.run(local_variables_init_op)
      start_step = 0

      if model_config['embed_config']['embedding_checkpoint_file']:
        model.init_fn(sess)
    else:
      logging.info('Restore from last checkpoint: {}'.format(model_path))
      sess.run(local_variables_init_op)
      sess.run(global_variables_init_op)
      
      e1=re.compile(".*def.*") 
      e3=re.compile(".*global_step.*") 
      e4=re.compile(".*Momentum")      
      e2=re.compile(".*OptimizeLoss.*")
      variables_to_restore1 = tf.global_variables()
      var_0 = [v for v in variables_to_restore1 if  not e1.match(v.name) and not e4.match(v.name) and not e3.match(v.name) and not e2.match(v.name)]
#      var_0 = [v for v in variables_to_restore1 if not e3.match(v.name) and not e4.match(v.name) and not e2.match(v.name)]
#      var_0 = [v for v in variables_to_restore1 if (re1.match(v.name) or re2.match(v.name) or re3.match(v.name) or re4.match(v.name)) and not e4.match(v.name) and not e2]
#      va=[v for v in variables_to_restore1 if not v in var_0 ]
#      start_step = 10001
#      var_0 = [v for v in variables_to_restore1  ]
      
#      var_0 = [v for v in variables_to_restore1 if not e4.match(v.name)]
#      print(var_0)
#      print('kkkkkkkkkkkkkkkkkkkkkkk',va)
      saver0 = tf.train.Saver(var_0,max_to_keep=train_config['max_checkpoints_to_keep'])

      
      saver0.restore(sess, model_path)
      start_step = tf.train.global_step(sess, model.global_step.name) + 1
      
      
    tf.train.start_queue_runners(sess=sess)
    model.dataloader.start_threads(sess=sess)  # start customized queue runner
    model_va.dataloader.start_threads(sess=sess)  # start customized queue runner

    total_steps = int(train_config['epoch'] * train_config['num_examples_per_epoch'] / train_config['batch_size'])
    logging.info('training for {} steps'.format(total_steps))
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=train_config['max_checkpoints_to_keep'])
  #  g.finalize()  # Finalize graph to avoid adding ops by mistake      
    for step in range(0, total_steps):
      start_time = time.time()
      _, loss, batch_loss = sess.run([train_op, model.total_loss, model.batch_loss])


      duration = time.time() - start_time

      if step % 10 == 0:
        examples_per_sec = model_config['batch_size'] / float(duration)
        time_remain = train_config['batch_size'] * (total_steps - step) / examples_per_sec
        m, s = divmod(time_remain, 60)
        h, m = divmod(m, 60)
        format_str = ('%s: step %d, loss = %.2f, batch loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch; %dh:%02dm:%02ds remains)')
        logging.info(format_str % (datetime.now(), step, loss, batch_loss,
                                      examples_per_sec, duration, h, m, s))


#    # Training loop
#    data_config = train_config['train_data_config']
#    total_steps = int(data_config['epoch'] *
#                      data_config['num_examples_per_epoch'] /
#                      data_config['batch_size'])
#    logging.info('Train for {} steps'.format(total_steps))
#    for step in range(start_step, total_steps):
#      start_time = time.time()
#      _, loss, batch_loss = sess.run([train_op, model.total_loss, model.batch_loss])
#      duration = time.time() - start_time
#
#      if step % 10 == 0:
#        examples_per_sec = data_config['batch_size'] / float(duration)
#        time_remain = data_config['batch_size'] * (total_steps - step) / examples_per_sec
#        m, s = divmod(time_remain, 60)
#        h, m = divmod(m, 60)
#        format_str = ('%s: step %d, total loss = %.2f, batch loss = %.2f (%.1f examples/sec; %.3f '
#                      'sec/batch; %dh:%02dm:%02ds remains)')
#        logging.info(format_str % (datetime.now(), step, loss, batch_loss,
#                                   examples_per_sec, duration, h, m, s))

      if step % 100 == 0:
        sess.run(tf.Print(model.gt,[model.gt],summarize=15*15))
    
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      if step % train_config['save_model_every_n_step'] == 0 or (step + 1) == total_steps:
        checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
if __name__ == '__main__':
  ex.run_commandline()