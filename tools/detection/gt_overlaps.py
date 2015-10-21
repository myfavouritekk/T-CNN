#!/usr/bin/env python
import sys
from vdetlib.utils.common import iou, quick_args
import scipy.io as sio
import numpy as np

if __name__ == '__main__':
    args = quick_args(['gt_list', 'save_file'])

    with open(args.gt_list) as f:
        gt_list = [line.strip() for line in f.readlines()]
    tot_overlaps = []
    for ind, gt_file in enumerate(gt_list, start=1):
        gt_boxes = sio.loadmat(gt_file)['boxes']
        num_gt = len(gt_boxes)
        if ind % 1000 == 0:
            print "{:.2%}: Processed {} files.".format(1. * ind / len(gt_list), ind)
        if num_gt <= 1:
            continue
        o_ind = np.triu_indices(num_gt, 1)
        overlaps = iou(gt_boxes, gt_boxes)
        valid_overlaps = overlaps[o_ind]
        tot_overlaps += valid_overlaps.tolist()
    if ind % 1000 != 0:
        print "100 %: Processed {} files.".format(ind)
    sio.savemat(args.save_file,
        {'gt_overlaps': np.asarray(tot_overlaps)})
