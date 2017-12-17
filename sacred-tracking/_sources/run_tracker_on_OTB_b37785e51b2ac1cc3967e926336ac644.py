#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

r"""Support integration with OTB benchmark"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path as osp
import sys
import time

import tensorflow as tf
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
import logging
OTB_POSSIBLE_DIR = ['/home/v-chaoqw/MYSFC-ORI/tracker_benchmark/',
                    '/home/v-chaoqw/MYSFC-ORI/tracker_benchmark/']
#OTB_POSSIBLE_DIR = ['/data/anfeng/tracking/codes/OTB-py',
#                    '/home/v-anfhe/tracking/tracker_benchmark_py']
# LOG_DIR = '/workspace/bily/Logs/TPP/'

sys.path += OTB_POSSIBLE_DIR

from config import *
from scripts.model import *
from scripts import butil

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from utils.misc import auto_select_gpu, Struct, mkdir_p
from inference import inference_wrapper
from inference.tracker import Tracker
from inference.vot import Rectangle
from inference.inference_utils import Sequence
from utils import crash_on_ipy

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

tf.logging.set_verbosity(tf.logging.INFO)

ex = Experiment()

@ex.config
def configs():
  # Checkpoint for evaluation
  checkpoint = None
  eval_type = None
  eval_seqs = None
  log_dir = None
  gpu_fraction = None
  folder_ckpt = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
  # Read configurations from json
  with open(osp.join(folder_ckpt, 'model_config.json'), 'r') as f:
    model_config = json.load(f)
  with open(osp.join(folder_ckpt, 'track_config.json'), 'r') as f:
    track_config = json.load(f)
    track_config['log_level'] = 0  # Skip verbose logging for speed
  model_config['batch_size'] = track_config['num_scales']
  run_name = None
  benchmark_config = {
    'testname': run_name,
    'tracker': run_name,
    'evalTypes': [eval_type + ''], #['OPE']
    'loadSeqs': eval_seqs + '', # cvpr13
  }

  tracker_config = {
    'track_name': run_name,
    'log_dir': osp.join(log_dir, 'track_model_checkpoints', run_name),
    'checkpoint_path': checkpoint,
    'model_config': model_config,
    'track_config': track_config,
    'gpu_fraction': gpu_fraction
    
  }

  track_config['gpu_fraction'] = gpu_fraction
  ex.observers.append(FileStorageObserver.create(osp.join(log_dir, 'sacred-tracking')))
@ex.named_config
def gcr():
  log_dir = '/home/v-chaoqw/MYSFC-ORI/'
  gpu_fraction = 0.4
@ex.named_config
def workstation():
  log_dir = '/home/v-chaoqw/MYSFC-ORI/'
  gpu_fraction = 0.2
  checkpoint="/home/v-chaoqw/MYSFC-ORI/track_model_checkpoints/SFC-clip-10-10-test1/"
  eval_type='OPE'
  eval_seqs='cvpr13'
  run_name='SFC-clip-10-10-test111111'

def build_tracker(tracker_config):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
      model = inference_wrapper.InferenceWrapper()
      restore_fn = model.build_graph_from_config(tracker_config['model_config'], tracker_config['track_config'],
                                                 tracker_config['checkpoint_path'])
    g.finalize()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=tracker_config['gpu_fraction'])
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=g, config=sess_config)
    

    

    # Load the model from checkpoint.
    restore_fn(sess)
    tracker = Tracker(model,
                      model_config=tracker_config['model_config'],
                      track_config=tracker_config['track_config'])
    return sess, tracker


def run_SFC(seq, rp, bSaveImage, sess, tracker):
  tic = time.clock()
  # sorted_filenames = [osp.join(seq.path, f) for f in sorted(os.listdir(seq.path))]
  # sorted_filenames = sorted_filenames[seq.startFrame - 1: seq.endFrame]
  sorted_filenames = seq.s_frames
  raw_bb = seq.init_rect
  x, y, width, height = raw_bb  # OTB format
  init_bb = Rectangle(x - 1, y - 1, width, height) # x, y minus one since python start index with zero
  handle = Sequence(sorted_filenames, init_bb)

  video_name = sorted_filenames[0].split(osp.sep)[-3]
  video_log_dir = '/tmp/OTB/tmp'
  mkdir_p(video_log_dir)
  tracker.track(sess, handle, video_log_dir)
  trajectory_py = handle.quit()
  trajectory = [Rectangle(val.x + 1, val.y + 1, val.width, val.height) for val in trajectory_py] # x, y add one to match OTB format
  duration = time.clock() - tic

  result = dict()
  result['res'] = trajectory
  result['type'] = 'rect'
  result['fps'] = round(seq.len / duration, 3)
  return result


def run_trackers(seqs, evalType, shiftTypeSet, tracker_config):
  t = tracker_config['track_name']
  tmpRes_path = RESULT_SRC.format('tmp/{0}/'.format(evalType))
  if not os.path.exists(tmpRes_path):
    os.makedirs(tmpRes_path)

  sess, tracker = build_tracker(tracker_config)
  
  
  

  
  numSeq = len(seqs)
  trackerResult = []
  for idxSeq in range(numSeq):
    s = seqs[idxSeq]
    subSeqs, subAnno = butil.get_sub_seqs(s, 20.0, evalType)
    if not OVERWRITE_RESULT:
      trk_src = os.path.join(RESULT_SRC.format(evalType), t)
      result_src = os.path.join(trk_src, s.name + '.json')
      if os.path.exists(result_src):
        seqResults = butil.load_seq_result(evalType, t, s.name)
        trackerResult.append(seqResults)
        continue

    seqResults = []
    seqLen = len(subSeqs)
    for idx in range(seqLen):
      rp = tmpRes_path + '_' + t + '_' + str(idx + 1) + '/'
      if SAVE_IMAGE and not os.path.exists(rp):
        os.makedirs(rp)
      subS = subSeqs[idx]
      subS.name = s.name + '_' + str(idx)
      if not tf.gfile.IsDirectory(TRACKER_SRC + t):
        logging.info('Creating training directory: %s', TRACKER_SRC + t)
        tf.gfile.MakeDirs(TRACKER_SRC + t)
      os.chdir(TRACKER_SRC + t)
      try:
        logging.info('Processing {}...'.format(subS.name))
        res = run_SFC(subS, rp, SAVE_IMAGE, sess, tracker)
      except:
        print('failed to execute {0} : {1}'.format(t, sys.exc_info()))
        os.chdir(WORKDIR)
        break
      os.chdir(WORKDIR)

      if evalType == 'SRE':
        r = Result(t, s.name, subS.startFrame, subS.endFrame,
                   res['type'], evalType, res['res'], res['fps'], shiftTypeSet[idx])
      else:
        r = Result(t, s.name, subS.startFrame, subS.endFrame,
                   res['type'], evalType, res['res'], res['fps'], None)
      try:
        r.tmplsize = butil.d_to_f(res['tmplsize'][0])
      except:
        pass
      r.refresh_dict()
      seqResults.append(r)
    if SAVE_RESULT:
      butil.save_seq_result(seqResults)

    trackerResult.append(seqResults)
  return trackerResult

@ex.automain
def main(benchmark_config, tracker_config):
  if benchmark_config['loadSeqs'] not in ['All', 'all', 'tb50', 'tb100', 'cvpr13', 'vid', 'tc78']:
    loadSeqs = [x.strip() for x in benchmark_config['loadSeqs'].split(',')]
  else:
    loadSeqs = benchmark_config['loadSeqs']

  if SETUP_SEQ:
    print('Setup sequences ...')
    butil.setup_seqs(loadSeqs)

  tracker = benchmark_config['tracker']
  for evalType in benchmark_config['evalTypes']:
    seqNames = butil.get_seq_names(loadSeqs)
    seqs = butil.load_seq_configs(seqNames)
    results = run_trackers(seqs, evalType, shiftTypeSet, tracker_config)
    if len(results) > 0:
      evalResults, attrList = butil.calc_result(tracker,
                                                seqs, results, evalType)
      print("Result of Sequences\t -- '{0}'".format(tracker))
      for seq in seqs:
        try:
          print('\t\'{0}\'{1}'.format(seq.name, " " * (12 - len(seq.name))), end='')
          print("\taveCoverage : {0:.3f}%".format(sum(seq.aveCoverage) / len(seq.aveCoverage) * 100), end='')
          print("\taveErrCenter : {0:.3f}".format(sum(seq.aveErrCenter) / len(seq.aveErrCenter)))
        except:
          print('\t\'{0}\'  ERROR!!'.format(seq.name))

      print("Result of attributes\t -- '{0}'".format(tracker))
      for attr in attrList:
        print("\t\'{0}\'".format(attr.name), end='')
        print("\toverlap : {0:02.2f}%".format(attr.overlap), end='')
        print("\tfailures : {0:.2f}".format(attr.error), end='')
        print("\tAUC : {0:.3f}".format(sum(attr.successRateList) / len(attr.successRateList)))
        if attr.name == 'ALL':
          report_result = sum(attr.successRateList) / len(attr.successRateList)
      if SAVE_RESULT:
        butil.save_scores(attrList, benchmark_config['testname'])
    return report_result