#!/usr/bin/env python

import argparse
import numpy as np
import sys
import h5py
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
    try:
        alexnet_dets = sio.loadmat(args.alexnet_det_file)['d']
        boxes = alexnet_dets['boxes']
        scores = alexnet_dets['zs']
    except NotImplementedError:
        alexnet_dets = h5py.File(args.alexnet_det_file)['d']
        boxes = alexnet_dets['boxes'][:].T
        scores = alexnet_dets['zs'][:].T
    max_scores = np.max(scores, axis = 1)
    overlaps = iou(gt_boxes, boxes)

    # true positive masks
    tp = overlaps >= 0.5
    # maxinum score from true positives for each gt box
    # yield -1000 if not any boxes covering the gt box
    thres = [np.max(max_scores[inds]) if np.any(inds) else -1000 for inds in tp]
    sio.savemat(args.save_file,
        {'boxes': gt_boxes,
         'thresholds': thres}, do_compression=True)
