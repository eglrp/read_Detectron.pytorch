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

import _init_paths
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


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    parser.add_argument('--video', type=str)
    parser.add_argument('--video_save_dir', type=str)

    args = parser.parse_args()

    return args


def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    video_path = args.video
    video_save_dir = args.video_save_dir
    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])  # only support single GPU
    maskRCNN.eval()

    capture = cv2.VideoCapture(video_path)
    frame_count = 0
    batch_size = 1

    frame_list = []
    while True:
        ret, frame = capture.read()
        # Bail out when the video file ends
        if not ret:
            break
            # Save each frame of the video to a list
        frame_count += 1
        frame_list.append(frame)
        if len(frame_list) == batch_size:
            for i, frame in enumerate(frame_list):
                timers = defaultdict(Timer)
                cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, frame, timers=timers)

                name = '{0}'.format(frame_count + i - batch_size)
                print('\n' + name)

                vis_utils.vis_one_image(
                    frame[:, :, ::-1],  # BGR -> RGB for visualization
                    name,
                    video_save_dir,
                    cls_boxes,
                    cls_segms,
                    cls_keyps,
                    dataset=dataset,
                    box_alpha=0.3,
                    show_class=True,
                    thresh=0.8,
                    kp_thresh=2,
                    ext='jpg'
                )

            # Clear the frames array to start the next batch
            frame_list = []

    images = list(glob.iglob(os.path.join(video_save_dir, '*.jpg')))
    # Sort the images by integer index
    images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

    outvid = os.path.join(video_save_dir, "out.mp4")
    make_video(outvid, images, fps=30)

    extract_audio = 'ffmpeg -i %s -vn -acodec copy %s/out.aac' % (video_path, video_save_dir)
    subprocess.call(extract_audio, shell=True)

    merge_audio_video = "ffmpeg -i %s/out.aac -i %s/out.mp4 -codec copy -shortest %s/final.mp4" % (video_save_dir, video_save_dir, video_save_dir)
    subprocess.call(merge_audio_video, shell=True)


if __name__ == '__main__':
    main()
