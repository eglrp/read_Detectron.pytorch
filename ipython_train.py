""" Training script for steps_with_decay policy"""

import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import tools._init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from modeling.model_builder import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
from utils.collections import AttrDict

args = AttrDict()
args.iter_size = 1
args.batch_size = 2
args.num_workers = 4
args.optimizer = None
args.lr = None
args.lr_decay_gamma = None
args.set_cfgs = None
args.cfg_file = 'configs/baselines/e2e_mask_rcnn_R-50-C4_1x.yaml'
args.cuda = True
args.dataset = 'coco2017'
args.load_ckpt = 'models/e2e_mask_rcnn_R-50-C4_1x.pth'
args.load_detectron = None
args.resume = False
args.start_step = 0
args.disp_interval = 10
args.use_tfboard = False
args.no_save = True


print('Called with args:')
print(args)

if not torch.cuda.is_available():
    sys.exit("Need a CUDA device to run the code.")

if args.cuda or cfg.NUM_GPUS > 0:
    cfg.CUDA = True
else:
    raise ValueError("Need Cuda device to run !")

if args.dataset == "coco2017":
    cfg.TRAIN.DATASETS = ('coco_2017_train',)
    cfg.MODEL.NUM_CLASSES = 81
elif args.dataset == "keypoints_coco2017":
    cfg.TRAIN.DATASETS = ('keypoints_coco_2017_train',)
    cfg.MODEL.NUM_CLASSES = 2
else:
    raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

cfg_from_file(args.cfg_file)
if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
original_num_gpus = cfg.NUM_GPUS

if args.batch_size is None:
    args.batch_size = original_batch_size

cfg.NUM_GPUS = torch.cuda.device_count()
assert (args.batch_size % cfg.NUM_GPUS) == 0, \
    'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
effective_batch_size = args.iter_size * args.batch_size
print('effective_batch_size = batch_size * iter_size = %d * %d' % (args.batch_size, args.iter_size))

print('Adaptive config changes:')
print('    effective_batch_size: %d --> %d' % (original_batch_size, effective_batch_size))
print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))

### Adjust learning based on batch size change linearly
# For iter_size > 1, gradients are `accumulated`, so lr is scaled based
# on batch_size instead of effective_batch_size
old_base_lr = cfg.SOLVER.BASE_LR
cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
print('Adjust BASE_LR linearly according to batch_size change:\n'
      '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

### Adjust solver steps
step_scale = original_batch_size / effective_batch_size
old_solver_steps = cfg.SOLVER.STEPS
old_max_iter = cfg.SOLVER.MAX_ITER
cfg.SOLVER.STEPS = list(map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
      '    SOLVER.STEPS: {} --> {}\n'
      '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
                                              old_max_iter, cfg.SOLVER.MAX_ITER))

# Scale FPN rpn_proposals collect size (post_nms_topN) in `collect` function
# of `collect_and_distribute_fpn_rpn_proposals.py`
#
# post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
if cfg.FPN.FPN_ON and cfg.MODEL.FASTER_RCNN:
    cfg.FPN.RPN_COLLECT_SCALE = cfg.TRAIN.IMS_PER_BATCH / original_ims_per_batch
    print('Scale FPN rpn_proposals collect size directly propotional to the change of IMS_PER_BATCH:\n'
          '    cfg.FPN.RPN_COLLECT_SCALE: {}'.format(cfg.FPN.RPN_COLLECT_SCALE))


if args.num_workers is not None:
    cfg.DATA_LOADER.NUM_THREADS = args.num_workers
print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

### Overwrite some solver settings from command line arguments
if args.optimizer is not None:
    cfg.SOLVER.TYPE = args.optimizer
if args.lr is not None:
    cfg.SOLVER.BASE_LR = args.lr
if args.lr_decay_gamma is not None:
    cfg.SOLVER.GAMMA = args.lr_decay_gamma
assert_and_infer_cfg()

timers = defaultdict(Timer)

### Dataset ###
timers['roidb'].tic()
dataset_names = cfg.TRAIN.DATASETS
proposal_files = cfg.TRAIN.PROPOSAL_FILES
roidb, ratio_list, ratio_index = combined_roidb_for_training(dataset_names, proposal_files)

timers['roidb'].toc()
roidb_size = len(roidb)
print('{:d} roidb entries'.format(roidb_size))
print('Takes %.2f sec(s) to construct roidb' % timers['roidb'].average_time)

#######################################################################################################


# Effective training sample size for one epoch
train_size = roidb_size // args.batch_size * args.batch_size

batchSampler = BatchSampler(
    sampler=MinibatchSampler(ratio_list, ratio_index),
    batch_size=args.batch_size,
    drop_last=True
)
dataset = RoiDataLoader(
    roidb,
    cfg.MODEL.NUM_CLASSES,
    training=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=batchSampler,
    num_workers=cfg.DATA_LOADER.NUM_THREADS,
    collate_fn=collate_minibatch)
dataiterator = iter(dataloader)

### Model ###
maskRCNN = Generalized_RCNN()


if cfg.CUDA:
    maskRCNN.cuda()

### Optimizer ###
gn_params = []
bias_params = []
bias_param_names = []
nonbias_params = []
nonbias_param_names = []
for key, value in dict(maskRCNN.named_parameters()).items():
    if value.requires_grad:
        if 'gn' in key:
            gn_params.append(value)
        elif 'bias' in key:
            bias_params.append(value)
            bias_param_names.append(key)
        else:
            nonbias_params.append(value)
            nonbias_param_names.append(key)
# Learning rate of 0 is a dummy value to be set properly at the start of training
params = [
    {'params': nonbias_params,
     'lr': 0,
     'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
    {'params': bias_params,
     'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
     'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
    {'params': gn_params,
     'lr': 0,
     'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
]
# names of paramerters for each paramter
param_names = [nonbias_param_names, bias_param_names]

if cfg.SOLVER.TYPE == "SGD":
    optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
elif cfg.SOLVER.TYPE == "Adam":
    optimizer = torch.optim.Adam(params)

### Load checkpoint
if args.load_ckpt:
    load_name = args.load_ckpt
    logging.info("loading checkpoint %s", load_name)
    checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    net_utils.load_ckpt(maskRCNN, checkpoint['model'])
    if args.resume:
        args.start_step = checkpoint['step'] + 1
        if 'train_size' in checkpoint:  # For backward compatibility
            if checkpoint['train_size'] != train_size:
                print('train_size value: %d different from the one in checkpoint: %d'
                      % (train_size, checkpoint['train_size']))

        # reorder the params in optimizer checkpoint's params_groups if needed
        # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

        # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
        # However it's fixed on master.
        # optimizer.load_state_dict(checkpoint['optimizer'])
        misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
    del checkpoint
    torch.cuda.empty_cache()

if args.load_detectron:  # TODO resume for detectron weights (load sgd momentum values)
    logging.info("loading Detectron weights %s", args.load_detectron)
    load_detectron_weight(maskRCNN, args.load_detectron)

lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

# maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
#                              minibatch=True)

### Training Setups ###
args.run_name = misc_utils.get_run_name() + '_step'
output_dir = misc_utils.get_output_dir(args, args.run_name)
args.cfg_filename = os.path.basename(args.cfg_file)

if not args.no_save:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    blob = {'cfg': yaml.dump(cfg), 'args': args}
    with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
        pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        # Set the Tensorboard logger
        tblogger = SummaryWriter(output_dir)

### Training Loop ###
maskRCNN.train()

CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

# Set index for decay steps
decay_steps_ind = None
for i in range(1, len(cfg.SOLVER.STEPS)):
    if cfg.SOLVER.STEPS[i] >= args.start_step:
        decay_steps_ind = i
        break
if decay_steps_ind is None:
    decay_steps_ind = len(cfg.SOLVER.STEPS)

training_stats = TrainingStats(
    args,
    args.disp_interval,
    tblogger if args.use_tfboard and not args.no_save else None)

################################################################################################

input_data = next(dataiterator)
for key in input_data:
    if key != 'roidb':  # roidb is a list of ndarrays with inconsistent length
        input_data[key] = list(map(Variable, input_data[key]))

from modeling.model_builder import *

#####################
im_data = input_data['data'][0]
roidb = list(map(lambda x: blob_utils.deserialize(x)[0], input_data['roidb'][0]))
im_info = input_data['im_info'][0]

im_data = im_data.cuda()


return_dict = {}  # A dict to collect return variables

blob_conv = maskRCNN.Conv_Body(im_data)

rpn_ret = maskRCNN.RPN(blob_conv, im_info, roidb)
for k, v in rpn_ret.items():
    print(k + ' -> ' + str(v.shape))

######################################################################################################################
# box_feat, res5_feat = maskRCNN.Box_Head(blob_conv, rpn_ret)
# print('box_feat' + ' -> ' + str(box_feat.shape))
# print('res5_feat' + ' -> ' + str(res5_feat.shape))
x = blob_conv
x = maskRCNN.roi_feature_transform(
    x, rpn_ret,
    blob_rois='rois',
    method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
    resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
    spatial_scale=maskRCNN.Conv_Body.spatial_scale,
    sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
)
res5_feat = maskRCNN.Box_Head.res5(x)
box_feat = maskRCNN.Box_Head.avgpool(res5_feat)


cls_score, bbox_pred = maskRCNN.Box_Outs(box_feat)
print('cls_score' + ' -> ' + str(cls_score.shape))
print('bbox_pred' + ' -> ' + str(bbox_pred.shape))

######################################################################################################################
# mask_feat = maskRCNN.Mask_Head(res5_feat, rpn_ret, roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
# print('mask_feat' + ' -> ' + str(mask_feat.shape))

# x = maskRCNN.Mask_Head.roi_xform(
#     res5_feat, rpn_ret,
#     blob_rois='mask_rois',
#     method=cfg.MRCNN.ROI_XFORM_METHOD,
#     resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
#     spatial_scale=maskRCNN.Conv_Body.spatial_scale,
#     sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO
# )
#

x = res5_feat
roi_has_mask_int32=rpn_ret['roi_has_mask_int32']
inds = np.nonzero(roi_has_mask_int32 > 0)[0]
inds = Variable(torch.from_numpy(inds)).cuda(x.get_device())
x = x[inds]

x = maskRCNN.Mask_Head.upconv5(x)
mask_feat = F.relu(x, inplace=True)
print('mask_feat' + ' -> ' + str(mask_feat.shape))

# mask_pred = maskRCNN.Mask_Outs(mask_feat)

x = mask_feat
x = maskRCNN.Mask_Outs.classify(x)
if cfg.MRCNN.UPSAMPLE_RATIO > 1:
    x = maskRCNN.Mask_Outs.upsample(x)

# print('mask_pred' + ' -> ' + str(mask_pred.shape))






