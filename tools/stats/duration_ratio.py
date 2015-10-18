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
    parser.add_argument('vid_list')
    args = parser.parse_args()

    durations = [{"class": cls, "ratio": []} for cls in imagenet_vdet_classes[1:]]
    annot_files = [line.strip() for line in open(args.annot_list)]
    vid_files = [line.strip() for line in open(args.vid_list)]
    try:
        assert len(annot_files) == len(vid_files)
    except:
        print len(annot_files)
        print len(vid_files)
    for annot_file, vid_file in zip(annot_files, vid_files):
        print "Processing {}".format(annot_file)
        with open(annot_file) as f:
            annot = json.load(f)
        with open(vid_file) as f:
            vid = json.load(f)
        assert vid['video'] == annot['video']
        num_tot = len(vid['frames'])
        tracks = annot['annotations']
        for track in tracks:
            class_idx = track['track'][0]['class_index']
            durations[class_idx - 1]['ratio'].append(float(len(track['track'])) / num_tot)

    print "class\t{}".format("\t".join(map(str, np.arange(0.1, 1.05, 0.1))))
    for duration in durations:
        bins = np.histogram(duration['ratio'], np.arange(0.0, 1.05, 0.1), density=False)
        print "{}\t{}".format(duration['class'], "\t".join(map(str,bins[0])))
