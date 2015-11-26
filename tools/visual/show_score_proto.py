#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load, frame_path_at, tubelet_box_proto_at_frame
from vdetlib.utils.cython_nms import nms
from vdetlib.utils.visual import add_bbox
from vdetlib.utils.common import imread, imwrite

def image_name_at_fame(vid_proto, frame_idx):
    vid_name = vid_proto['video']
    for frame in vid_proto['frames']:
        if frame['frame'] == frame_idx:
            return os.path.join(vid_name, os.path.splitext(frame['path'])[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('score_file')
    parser.add_argument('--varname')
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--nms', dest='do_nms', action='store_true')
    parser.set_defaults(do_nms=False)
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    score_proto = proto_load(args.score_file)
    vid_name = vid_proto['video']
    assert vid_name == score_proto['video']

    if args.save_dir is None:
        cv2.namedWindow('tracks')
    for frame in vid_proto['frames']:
        if args.save_dir:
            if not os.path.isdir(args.save_dir):
                os.mkdir(args.save_dir)
            save_name = os.path.join(args.save_dir, "{:04d}.jpg".format(frame['frame']))
            if os.path.isfile(save_name):
                continue
        img = imread(frame_path_at(vid_proto, frame['frame']))
        frame_idx = frame['frame']
        tubelet_boxes = map(lambda x:tubelet_box_proto_at_frame(x, frame_idx), score_proto['tubelets'])
        if args.do_nms:
            valid_idx = [i for i, box in enumerate(tubelet_boxes) if box is not None]
            valid_boxes = [tubelet_boxes[i]['bbox'] for i in valid_idx]
            valid_scores = [tubelet_boxes[i][args.varname] for i in valid_idx]
            valid_box_score = [tubelet_boxes[i]['bbox']+[tubelet_boxes[i][args.varname],] for i in valid_idx]
            kept_index = nms(np.asarray(valid_box_score, dtype='float32'), 0.3)
            kept_orig_idx = [valid_idx[i] for i in kept_index]
            boxes = [tubelet_boxes[i]['bbox'] for i in kept_orig_idx]
        else:
            boxes = [x['bbox'] if x is not None else None for x in tubelet_boxes]
        tracked = add_bbox(img, boxes, None, 10)
        if args.save_dir:
            if not os.path.isdir(args.save_dir):
                os.mkdir(args.save_dir)
            imwrite(os.path.join(args.save_dir, "{:04d}.jpg".format(frame['frame'])),
                    tracked)
        else:
            cv2.imshow('tracks', tracked)
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)
    if args.save_dir is None:
        cv2.destroyAllWindows()
