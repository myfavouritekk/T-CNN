#!/usr/bin/env python

import argparse
import json
import os
import scipy.io as sio
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('annot_file')
    parser.add_argument('save_dir')
    args = parser.parse_args()

    with open(args.vid_file) as f:
        vid_proto = json.load(f)
    with open(args.annot_file) as f:
        annot_proto = json.load(f)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    assert vid_proto['video'] == annot_proto['video']
    video = vid_proto['video']
    print "Processing {}".format(video)
    tracks = [annot['track'] for annot in annot_proto['annotations']]
    for frame in vid_proto['frames']:
        save_name = os.path.splitext(os.path.basename(frame['path']))[0]
        boxes_cls = [[box['bbox'], box['class_index']] for track in tracks for box in track if box['frame'] == frame['frame']]
        save_file = os.path.join(args.save_dir, save_name + '.mat')
        boxes = map(lambda x:x[0], boxes_cls)
        classes = map(lambda x:x[1], boxes_cls)
        sio.savemat(save_file, {'boxes': np.asarray(boxes, dtype='float64'),
                                'labels': np.asarray(classes, dtype='float64')},
            do_compression=True)
