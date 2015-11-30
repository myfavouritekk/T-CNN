#!/usr/bin/env python

import argparse
import os
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load
import glob
import numpy as np

def max_det_score(det):
    return max([class_score['score'] for class_score in det['scores']])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculate ratio of detections given a certain ratio.')
    parser.add_argument('det_dir')
    args = parser.parse_args()

    det_files = glob.glob(os.path.join(args.det_dir, '*.det*'))
    max_scores = []
    for det_file in det_files:
        print "Processing {}...".format(det_file)
        det_proto = proto_load(det_file)
        max_scores.extend([max_det_score(det) for det in det_proto['detections']])

    sorted_scores = sorted(max_scores, reverse=True)
    for ratio in np.arange(0.01, 1, 0.01):
        num = int(ratio * len(sorted_scores))
        print "Thres {:.02f}: ratio {}".format(sorted_scores[num], ratio)
