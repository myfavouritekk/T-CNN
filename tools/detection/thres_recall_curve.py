#!/usr/bin/env python

import argparse
import numpy as np
import scipy.io as sio
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('thres_list')
    parser.add_argument('save_file')
    args = parser.parse_args()

    with open(args.thres_list) as f:
        thres_files = [line.strip() for line in f.readlines()]

    print "Loading gt thresholds..."
    thres = []
    for thres_file in thres_files:
        thres += sio.loadmat(thres_file)['thresholds'].ravel().tolist()
    print "Accumulating..."
    uniq_thres = np.unique(thres)
    recalls = [1. * np.count_nonzero(thres > cur_thres) / len(thres) \
                for cur_thres in uniq_thres]
    assert len(uniq_thres) == len(recalls)
    save_dir = os.path.dirname(args.save_file)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sio.savemat(args.save_file,
        {'thres': uniq_thres, 'recall': recalls},
        do_compression=True)
    print "Saved to {}.".format(args.save_file)

