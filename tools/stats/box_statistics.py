#!/usr/bin/env python

import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_list')
    args = parser.parse_args()

    tot_box = None
    for annot_file in [line.strip() for line in open(args.annot_list)]:
        print "Processing {}".format(annot_file)
        with open(annot_file) as f:
            annot = json.load(f)
        boxes = [frame['bbox'] for track in annot['annotations'] for frame in track['track']]
        if tot_box is None:
            tot_box = np.asarray(boxes)
        else:
            tot_box = np.r_[tot_box, boxes]
    sizes = np.amin(np.vstack((tot_box[:,2] - tot_box[:,0], tot_box[:,3] - tot_box[:,1])),axis=0)
    print "min: {} max: {} median: {} mean: {} std: {}".format(
        sizes.min(), sizes.max(), np.median(sizes), sizes.mean(), sizes.std())