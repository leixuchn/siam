import argparse
import os
import logging
import sys
import time
sys.path.append('./')


parser = argparse.ArgumentParser(description='evaluate checkpoints')
parser.add_argument('path_ckpt', type=str, help='path of ckpt')
parser.add_argument('--station', type=str, help='station of running, gcr or workstation',default='workstation')
parser.add_argument('--all', action='store_true', help='enable to evalute all ckpt',default=False)
parser.add_argument('--benchmark', type=str, help='select benchmark to evaluation', default='tb100')
parser.add_argument('--evaltype', type=str, help='select evaltype eg: OPE  or  OPE,TRE', default='OPE')
args = parser.parse_args()

def eval_ckpt(checkpoint_file,benchmark,evalType):
  if os.path.isdir(checkpoint_file):
    testname = os.path.split(checkpoint_file)[1]+'-latest'
  else:
    dirname, filename = os.path.split(checkpoint_file)
    testname = os.path.split(dirname)[1]
    testname += filename.split('model.ckpt')[-1]
  testname += ('-' + benchmark)
  print('Start evaluate: {}'.format(testname))
  print('Check point file: {}'.format(checkpoint_file))
  print('Benchmark: {}'.format(benchmark))
  print('EvalTypes: {}'.format(evalType))
  cmd = ('python benchmarks/run_tracker_on_OTB.py with checkpoint="{}" run_name="{}" ' \
      + 'eval_seqs="{}" eval_type="{}" {} --name="{}" --force')\
      .format(checkpoint_file, testname, benchmark, evalType, args.station, testname)
  print cmd
  os.system(cmd)
  print 'Sleep 5s'
  time.sleep(5)

args.path_ckpt = os.path.abspath(args.path_ckpt)

if not args.all:
  eval_ckpt(args.path_ckpt,args.benchmark,args.evaltype)
else:
  if not os.path.isdir(args.path_ckpt):
    args.path_ckpt, _ = os.path.split(args.path_ckpt)
  for f in os.listdir(args.path_ckpt):
    if f.find('.index') != -1:
      f = f.replace('.index','')
      eval_ckpt(os.path.join(args.path_ckpt, f),args.benchmark,args.evaltype)

