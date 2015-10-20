#!/usr/bin/env python

import argparse
import numpy as np
import sys
sys.path.insert(1, '.')
from vdetlib.utils.common import iou
import scipy.io as sio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_boxfile')
    parser.add_argument('alexnet_det_file')
    parser.add_argument('save_file')
    args = parser.parse_args()

    print "Processing {}".format(args.gt_boxfile)
    gt_boxes = sio.loadmat(args.gt_boxfile)['boxes']
    alexnet_dets = sio.loadmat(args.alexnet_det_file)
    boxes = alexnet_dets['boxes']
    scores = alexnet_dets['zs']
    max_scores = np.max(scores, axis = 1)
    overlaps = iou(gt_boxes, boxes)

    # true positive masks
    tp = overlaps >= 0.5
    # maxinum score from true positives for each gt box
    thres = [np.max(max_scores[inds]) for inds in tp]
    sio.savemat(args.save_file,
        {'boxes': gt_boxes,
         'thresholds': thres}, do_compression=True)
