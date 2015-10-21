#!/usr/bin/env python

import argparse
import numpy as np
import sys
import h5py
import scipy.io as sio
import os

def iou(boxes1, boxes2):
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    # intersection boundaries, widths and heights
    ix1 = np.maximum(boxes1[:,[0]], boxes2[:,[0]].T)
    ix2 = np.minimum(boxes1[:,[2]], boxes2[:,[2]].T)
    iy1 = np.maximum(boxes1[:,[1]], boxes2[:,[1]].T)
    iy2 = np.minimum(boxes1[:,[3]], boxes2[:,[3]].T)
    iw = np.maximum(0, ix2 - ix1 + 1)
    ih = np.maximum(0, iy2 - iy1 + 1)
    # areas
    areas1 = (boxes1[:, [2]] - boxes1[:, [0]] + 1) * \
             (boxes1[:, [3]] - boxes1[:, [1]] + 1)
    areas2 = (boxes2[:, [2]] - boxes2[:, [0]] + 1) * \
             (boxes2[:, [3]] - boxes2[:, [1]] + 1)
    inter = iw * ih
    overlaps = 1. * inter / (areas1 + areas2.T - inter)
    return overlaps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_boxfile')
    parser.add_argument('alexnet_det_file')
    parser.add_argument('save_file')
    args = parser.parse_args()

    if os.path.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit(0)

    print "Processing {}".format(args.gt_boxfile)
    gt_boxes = sio.loadmat(args.gt_boxfile)['boxes']
    if len(gt_boxes) == 0:
        print "No gt boxes in {}".format(args.gt_boxfile)
        sys.exit(0)
    try:
        alexnet_dets = sio.loadmat(args.alexnet_det_file)
        boxes = alexnet_dets['d']['boxes']
        scores = alexnet_dets['d']['zs']
    except NotImplementedError:
        alexnet_dets = h5py.File(args.alexnet_det_file)['d']
        boxes = alexnet_dets['boxes'][:].T
        scores = alexnet_dets['zs'][:].T
    except KeyError:
        boxes = alexnet_dets['boxes']
        scores = alexnet_dets['zs']
    max_scores = np.max(scores, axis = 1)
    overlaps = iou(gt_boxes, boxes)

    # true positive masks
    tp = overlaps >= 0.5
    # maxinum score from true positives for each gt box
    # yield -1000 if not any boxes covering the gt box
    thres = [np.max(max_scores[inds]) if np.any(inds) else -1000 for inds in tp]
    save_dir = os.path.dirname(args.save_file)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sio.savemat(args.save_file,
        {'boxes': gt_boxes,
         'thresholds': thres}, do_compression=True)


