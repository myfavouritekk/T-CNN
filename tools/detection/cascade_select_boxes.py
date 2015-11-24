#!/usr/bin/env python

import argparse
import numpy as np
import scipy.io as sio
import h5py
import os.path
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile')
    parser.add_argument('varname')
    parser.add_argument('thres', type=float)
    parser.add_argument('save_file')
    args = parser.parse_args()

    if os.path.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit(0)
    if not os.path.isdir(os.path.dirname(args.save_file)):
        try:
            os.makedirs(os.path.dirname(args.save_file))
        except OSError, e:
            if e.errno != 17:
                raise e

    var = h5py.File(args.matfile)[args.varname]
    boxes = np.transpose(var['boxes'])
    scores = np.transpose(var['zs'])
    keep = np.max(scores, axis=1) > args.thres
    kept_boxes = boxes[keep, :]
    kept_scores = scores[keep, :]
    ratio = np.sum(keep) * 1. / keep.size
    sio.savemat(args.save_file, {'boxes': kept_boxes,
                                 'alexnet_scores': kept_scores})
    print "{}: {:.2f} %% boxes kept".format(args.matfile, ratio * 100)

