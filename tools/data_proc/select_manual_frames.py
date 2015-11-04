#!/usr/bin/env python

import argparse
import json
import numpy as np
import os
import sys
sys.path.insert(1, '.')
from vdetlib.vdet.dataset import imagenet_vdet_classes
from vdetlib.utils.protocol import frame_path_at

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_list')
    parser.add_argument('--vid_dir')
    parser.add_argument('--save_dir')
    args = parser.parse_args()

    manual_boxes = [{"class": cls, "boxes": []} for cls in imagenet_vdet_classes[1:]]
    for annot_file in [line.strip() for line in open(args.annot_list)]:
        print "Processing {}".format(annot_file)
        with open(annot_file) as f:
            annot = json.load(f)
        vid_file = os.path.join(args.vid_dir, annot['video']+'.vid')
        assert os.path.isfile(vid_file)
        with open(vid_file) as f:
            vid_proto = json.load(f)
        assert vid_proto['video'] == annot['video']
        frames = [frame for track in annot['annotations'] for frame in track['track']]
        for frame in frames:
            cls_idx = frame['class_index']
            assert manual_boxes[cls_idx-1]['class'] == frame['class']
            if frame['generated'] == 1:
                # skip generated boxes
                continue
            # manually labeled boxes
            frame_path = frame_path_at(vid_proto, frame['frame'])
            manual_boxes[cls_idx-1]['boxes'].append(
                [frame_path] + frame['bbox'])

    for count in manual_boxes:
        save_file = os.path.join(args.save_dir, count['class']+'_manual_box_list.txt')
        with open(save_file, 'w') as f:
            print "Writing to {}".format(save_file)
            for box in count['boxes']:
                f.write('\t'.join(map(str,box))+'\n')
