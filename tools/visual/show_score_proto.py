#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load, frame_path_at
from vdetlib.utils.cython_nms import vid_nms
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
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    score_proto = proto_load(args.score_file)
    vid_name = vid_proto['video']
    assert vid_name == score_proto['video']

    dets = []
    for tubelet in score_proto['tubelets']:
        if tubelet['gt'] == 1:
            raise ValueError('Dangerous: Score file contains gt tracks!')
        class_index = tubelet['class_index']
        for box in tubelet['boxes']:
            frame_idx = box['frame']
            image_name = image_name_at_fame(vid_proto, frame_idx)
            bbox = map(lambda x:max(x,0), box['bbox'])
            score = box[args.varname]
            dets.append([int(frame_idx), class_index, score, bbox])

    nms_boxes = [[det[0],]+det[-1]+[det[2],] for det in dets]
    keep = vid_nms(np.asarray(nms_boxes).astype('float32'), 0.3)

    kept_dets = [dets[i] for i in keep]
    # for frame_idx, class_index, score, bbox in kept_dets:
    #     print '{} {} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
    #         frame_idx, class_index, score,
    #         bbox[0], bbox[1], bbox[2], bbox[3])

    for frame in vid_proto['frames']:
        if args.save_dir:
            if not os.path.isdir(args.save_dir):
                os.mkdir(args.save_dir)
            save_name = os.path.join(args.save_dir, "{:04d}.jpg".format(frame['frame']))
            if os.path.isfile(save_name):
                continue
        img = imread(frame_path_at(vid_proto, frame['frame']))
        frame_idx = frame['frame']
        boxes = [det[3] for det in kept_dets if det[0] == frame_idx]
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
            cv2.destroyAllWindows()
