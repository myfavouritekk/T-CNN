#!/usr/bin/env python

import argparse
import os
import numpy as np
import sys
from scipy.misc import imread
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load, frame_path_at
from vdetlib.utils.cython_nms import vid_nms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('score_file')
    parser.add_argument('image_set_file')
    parser.add_argument('--varname')
    args = parser.parse_args()

    global_thres = -2.5

    vid_proto = proto_load(args.vid_file)
    score_proto = proto_load(args.score_file)
    with open(args.image_set_file) as f:
        image_set = dict([line.strip().split() for line in f.readlines()])
    vid_name = vid_proto['video']
    assert vid_name == score_proto['video']

    # build dict
    frame_to_image_name = {}
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        frame_to_image_name[frame_id] = os.path.join(
            vid_name, os.path.splitext(frame['path'])[0])

    # get image shape
    height, width = imread(frame_path_at(vid_proto, 1)).shape[:2]

    dets = []
    for tubelet in score_proto['tubelets']:
        if tubelet['gt'] == 1:
            raise ValueError('Dangerous: Score file contains gt tracks!')
        class_index = tubelet['class_index']
        for box in tubelet['boxes']:
            frame_idx = box['frame']
            image_name = frame_to_image_name[frame_idx]
            frame_idx = image_set[image_name]
            bbox = map(lambda x:max(x,0), box['bbox'])
            bbox[0] = min(width - 1, bbox[0])
            bbox[2] = min(width - 1, bbox[2])
            bbox[1] = min(height - 1, bbox[1])
            bbox[3] = min(height - 1, bbox[3])
            score = box[args.varname]
            # ignore boxes with very low confidence
            if score < global_thres:
                continue
            dets.append([int(frame_idx), class_index, score, bbox])

    nms_boxes = [[det[0],]+det[-1]+[det[2],] for det in dets]
    keep = vid_nms(np.asarray(nms_boxes).astype('float32'), 0.5)

    kept_dets = [dets[i] for i in keep]
    for frame_idx, class_index, score, bbox in kept_dets:
        print '{} {} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
            frame_idx, class_index, score,
            bbox[0], bbox[1], bbox[2], bbox[3])