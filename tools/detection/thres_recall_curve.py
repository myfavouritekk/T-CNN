#!/usr/bin/env python

import argparse
import numpy as np
import sys
sys.path.insert(1, '.')
from vdetlib.utils.common import iou
import scipy.io as sio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('thres_list')
    parser.add_argument('save_file')
    args = parser.parse_args()

    with open(args.thres_list) as f:
        thres_files = [line.strip() for line in f.readlines()]

    print "Loading gt thresholds..."
    thres = [sio.loadmat(f)['thresholds'] for f in thres_files]
    thres = np.asarray(sorted(thres, reverse=True))
    print "Accumulating..."
    uniq_thres = np.unique(thres)
    recalls = [np.count_nonzero(thres > cur_thres) / len(thres) \
                for cur_thres in uniq_thres]
    sio.savemat(args.save_file,
        {'thres': uniq_thres, 'recall': recalls},
        do_compression=True)