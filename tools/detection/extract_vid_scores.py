#!/usr/bin/env python

import argparse
import h5py
import scipy.io
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../..'))
from vdetlib.vdet.dataset import index_vdet_to_det

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('det_score_file')
    parser.add_argument('save_file')
    parser.add_argument('--bg_first', type=bool, default=True,
        required=False,
        help='Background class comes first in 201 classes. [True]')
    args = parser.parse_args()

    if os.path.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit(0)
    try:
        d = scipy.io.loadmat(args.det_score_file)['d']
        boxes = d['boxes']
        scores = d['zs']
    except NotImplementedError:
        d = h5py.File(args.det_score_file)['d']
        boxes = np.transpose(d['boxes'])
        scores = np.transpose(d['zs'])

    if scores.shape[1] == 201:
        if args.bg_first:
            print "Using 201 classes. Background comes first."
            ind = [index_vdet_to_det[i] for i in xrange(1,31)]
        else:
            print "Using 201 classes. Background comes last."
            ind = [index_vdet_to_det[i] - 1 for i in xrange(1,31)]
    elif scores.shape[1] == 200:
        print "Using 200 classes."
        ind = [index_vdet_to_det[i] - 1 for i in xrange(1,31)]
    else:
        raise ValueError('Dimensions of scores can only be 200 or 201.')
    vid_scores = scores[:,ind]

    save_dir = os.path.dirname(args.save_file)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    scipy.io.savemat(args.save_file,
        { 'boxes': boxes, 'zs': vid_scores})

