from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import tools._init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
import glob
import subprocess

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# load config
dataset = datasets.get_coco_dataset()
cfg.MODEL.NUM_CLASSES = len(dataset.classes)
cfg_from_file('configs/baselines/e2e_mask_rcnn_R-50-C4_1x.yaml')
cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
assert_and_infer_cfg()

# load model
maskRCNN = Generalized_RCNN()
checkpoint = torch.load('/home/work/liupeng11/code/Detectron.pytorch/models/e2e_mask_rcnn_R-50-C4_1x.pth', map_location=lambda storage, loc: storage)
net_utils.load_ckpt(maskRCNN, checkpoint['model'])
maskRCNN.eval()
maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])

# load image
img_path = "/home/work/liupeng11/code/Detectron.pytorch/demo/sample_images/img1.jpg"
im = cv2.imread(img_path)

# detect bouding boxes and segments
from core.test import im_detect_bbox, im_detect_mask, box_results_with_nms_and_limit, segm_results
scores, boxes, im_scale, blob_conv = im_detect_bbox(maskRCNN, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, None)
scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
masks = im_detect_mask(maskRCNN, im_scale, boxes, blob_conv)
cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
cls_keyps = None

# save detected image
name = 'test'
output_dir = '/home/work/liupeng11/code/Detectron.pytorch/tmp'
vis_utils.vis_one_image(
    im[:, :, ::-1],  # BGR -> RGB for visualization
    name,
    output_dir,
    cls_boxes,
    cls_segms,
    cls_keyps,
    dataset=dataset,
    box_alpha=0.3,
    show_class=True,
    thresh=0.7,
    kp_thresh=2,
    ext='jpg',
)


################################################################################

# prepare input and model
pil = Image.open(img_path).convert('RGB')
trans = transforms.Compose([
    transforms.Resize((800, 600)),
    transforms.ToTensor()
    ])

x = Variable(torch.unsqueeze(trans(pil), 0))
x = x.cuda()
m = maskRCNN.module
m.eval()

# feature_map
blob_conv = m.Conv_Body(x)

# RPN network
rpn_conv = F.relu(m.RPN.RPN_conv(blob_conv), inplace=False)
rpn_cls_logits = m.RPN.RPN_cls_score(rpn_conv)
rpn_bbox_pred = m.RPN.RPN_bbox_pred(rpn_conv)
rpn_cls_prob = F.sigmoid(rpn_cls_logits)

# genrete proposals (rois)
im_info = Variable(torch.Tensor([[800, 600, 1]]))
rpn_rois, rpn_rois_prob = m.RPN.RPN_GenerateProposals(rpn_cls_prob, rpn_bbox_pred, im_info)

rpn_ret = {'rpn_cls_logits': rpn_cls_logits, 'rpn_bbox_pred': rpn_bbox_pred, 'rpn_rois': rpn_rois, 'rpn_roi_probs': rpn_rois_prob}
rpn_ret['rois'] = rpn_ret['rpn_rois']


# bouding box network
box_feat = m.Box_Head(blob_conv, rpn_ret)
cls_score, bbox_pred = m.Box_Outs(box_feat)

#################### mask

scores = cls_score
boxes = bbox_pred
masks = im_detect_mask(maskRCNN, im_scale, boxes, blob_conv)



