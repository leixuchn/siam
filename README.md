# Objectness Fully-Convolutional Siamese Networks

## Prerequisite
See `requirements.txt` for replicating my working environment. You can install all the libraries simply by
```bash
pip install -r requirements.txt
```

## Training
### Step 1: Data preparation
**1. Dataset curation**

Follow steps listed in [here](https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation).
Use `/workspace/common_datasets/ILSVRC2015_curated` if you are in a hurry.

**2. Build image database(imdb)**

Modify `dataset_dir` in `datasets/build_VID2015_imdb.py` accordingly.
```bash
# This will create two files in save_dir: train_imdb.pickle and val_imdb.pickle
python datasets/build_VID2015_imdb.py
```

### Step 2(optional): Download pretrained model
**Download Siamese-fc pretrained model**
```bash
wget http://www.robots.ox.ac.uk/~luca/stuff/siam-fc_nets/2016-08-17_gray025.net.mat
```

### Step 3: Start training
**1. modify configuration.py accordingly or use one of the example configurations in the cfgs directory**

The default configurations simply load the pre-trained color+gray model downloaded from step 2 and save it in the model format of TensorFlow.

**2. start training**
```bash
python train_track_model.py
```

**3. monitor training progress using tensorboard**
```bash
tensorboard --logdir your-log-directory/track_model_checkpoints
```

## Tracking
**1.Generate tracking trajectory** 

```bash
python run_tracking.py with checkpoint_path=your-trained-model \
                            input_files=videos-directory
```
Note that we use `with` to specify options, see [here](http://sacred.readthedocs.io/en/latest/command_line.html) for more details.

## Benchmarks
For evaluating tracker performance using common benchmarks, `OTB` and `VOT` are supported.

### OTB
We provide two ways for evaluation on OTB. Method 1 is recommended, since it will record all your experiment information using `sacred`.

#### Method 1
**1. Download OTB evaluation toolkit and dataset**
```bash
# Download toolkit
git clone https://github.com/jwlim/tracker_benchmark

# Download dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html.
# You can find it in /workspace/common_datasets/OTB which contains all 100 videos.
```

**2. Set up tracker for evaluation**
- Modify `BENCHMARK_TOOLKIT` in `benchmarks/OTB_config.py`
- Replace `config.py` in original toolkit by `benchmarks/OTB_config.py`

**3. Start evaluation**

Set parameters in `evaluate_tracker_on_OTB.py` accordingly. The default parameters evaluate on the OTB-cvpr13 dataset using One Pass Evaluation(OPE).
```bash
python evaluate_tracker_on_OTB.py
```

You should get something like:
```text
        'ALL'   overlap : 69.82%        failures : 2.10     AUC : 0.622
        'BC'    overlap : 66.21%        failures : 2.79     AUC : 0.557
        'DEF'   overlap : 67.27%        failures : 3.09     AUC : 0.543
        'FM'    overlap : 65.27%        failures : 2.59     AUC : 0.583
        'IPR'   overlap : 68.64%        failures : 2.55     AUC : 0.593
        'IV'    overlap : 64.86%        failures : 3.21     AUC : 0.534
        'LR'    overlap : 64.43%        failures : 3.35     AUC : 0.513
        'MB'    overlap : 62.36%        failures : 3.24     AUC : 0.538
        'OCC'   overlap : 69.01%        failures : 2.26     AUC : 0.610
        'OPR'   overlap : 70.04%        failures : 2.13     AUC : 0.616
        'OV'    overlap : 72.80%        failures : 1.96     AUC : 0.652
        'SV'    overlap : 69.99%        failures : 2.28     AUC : 0.609
```
#### Method 2
**1. Download OTB evaluation toolkit and dataset**
```bash
# Download toolkit
git clone https://github.com/jwlim/tracker_benchmark

# Download dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html.
# You can find it in /workspace/common_datasets/OTB which contains all 100 videos.
```

**2. Set up tracker for evaluation**

- Copy `benchmarks/run_TPP.py` to `tracker_benchmark/scripts/bscripts`
- Modify the `TPP_ROOT` and `CHECKPOINTS` variables in `run_TPP.py`
- Add `from run_TPP import *` in `tracker_benchmark/scripts/bscripts/__init__.py`
- Create directory `TPP` in `tracker_benchmark/trackers`

**3. Start evaluation**
```bash
# For example, run One Pass Evaluation in OTB50 dataset.
python run_trackers.py -t TPP -s tb50 -e OPE
```

### VOT
**1. Set up the workspace**

Follow steps [here](http://www.votchallenge.net/howto/workspace.html). Note that we have `vot 2015` and `vot 2016` data in `/workspace/common_datasets/VOT`

**2. Integrate tracker**

Follow steps [here](http://www.votchallenge.net/howto/integration.html), specifically, we use `Python trackers` and set our `tracker_TPP.m` file as:
 ```matlab
tracker_label = ['TPP'];
tracker_command = generate_python_command('benchmarks/vot_inference', {'Absolute-Path-To-TPP-codebase'};
tracker_interpreter = 'python';
```

**3. Evaluate**

Follow steps [here](http://www.votchallenge.net/howto/perfeval.html)
