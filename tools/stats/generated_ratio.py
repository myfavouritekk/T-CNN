#!/usr/bin/env python

import argparse
import json
import numpy as np
import sys
sys.path.insert(1, '.')
from vdetlib.vdet.dataset import imagenet_vdet_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_list')
    args = parser.parse_args()

    counts = [{"class": cls, "generated": 0, "manual": 0} for cls in imagenet_vdet_classes[1:]]
    for annot_file in [line.strip() for line in open(args.annot_list)]:
        print "Processing {}".format(annot_file)
        with open(annot_file) as f:
            annot = json.load(f)
        frames = [frame for track in annot['annotations'] for frame in track['track']]
        for frame in frames:
            cls_idx = frame['class_index']
            assert counts[cls_idx-1]['class'] == frame['class']
            if frame['generated'] == 1:
                counts[cls_idx-1]['generated'] += 1
            else:
                counts[cls_idx-1]['manual'] += 1
    print "class\tmanual\tgenerated"
    for count in counts:
        print "{}\t{}\t{}".format(
            count['class'], count['manual'], count['generated'])
