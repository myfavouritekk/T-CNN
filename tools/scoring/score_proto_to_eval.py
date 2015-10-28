#!/usr/bin/env python

import argparse
import os
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load
from vdetlib.utils.cython_nms import nms

def image_name_at_fame(vid_proto, frame_idx):
    vid_name = vid_proto['video']
    for frame in vid_proto['frames']:
        if frame['frame'] == frame_idx:
            return os.path.join(vid_name, os.path.splitext(frame['path'])[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('score_file')
    parser.add_argument('image_set_file')
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    score_proto = proto_load(args.score_file)
    with open(args.image_set_file) as f:
        image_set = dict([line.strip().split() for line in f.readlines()])
    vid_name = vid_proto['video']
    assert vid_name == score_proto['video']

    for tubelet in score_proto['tubelets']:
        if tubelet['gt'] == 1:
            raise ValueError('Dangerous: Score file contains gt tracks!')
        class_index = tubelet['class_index']
        for box in tubelet['boxes']:
            frame_idx = box['frame']
            image_name = image_name_at_fame(vid_proto, frame_idx)
            frame_idx = image_set[image_name]
            bbox = box['bbox']
            score = box['det_score']
            print '{} {} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
                frame_idx, class_index, score,
                bbox[0], bbox[1], bbox[2], bbox[3])

