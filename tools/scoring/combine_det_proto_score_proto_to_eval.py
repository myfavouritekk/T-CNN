#!/usr/bin/env python

import argparse
import os
import numpy as np
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load
from vdetlib.utils.cython_nms import vid_nms

def image_name_at_frame(vid_proto, frame_idx):
    vid_name = vid_proto['video']
    for frame in vid_proto['frames']:
        if frame['frame'] == frame_idx:
            return os.path.join(vid_name, os.path.splitext(frame['path'])[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('score_file')
    parser.add_argument('det_file')
    parser.add_argument('image_set_file')
    parser.add_argument('--score_var')
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    score_proto = proto_load(args.score_file)
    det_proto = proto_load(args.det_file)
    with open(args.image_set_file) as f:
        image_set = dict([line.strip().split() for line in f.readlines()])
    vid_name = vid_proto['video']
    assert vid_name == score_proto['video']
    assert vid_name == det_proto['video']
    assert len(score_proto['tubelets']) > 0

    dets = []
    class_index = score_proto['tubelets'][0]['class_index']

    # track detections
    for tubelet in score_proto['tubelets']:
        if tubelet['gt'] == 1:
            raise ValueError('Dangerous: Score file contains gt tracks!')
        assert class_index == tubelet['class_index']
        for box in tubelet['boxes']:
            frame_idx = box['frame']
            image_name = image_name_at_frame(vid_proto, frame_idx)
            frame_idx = image_set[image_name]
            bbox = map(lambda x:max(x,0), box['bbox'])
            score = box[args.score_var]
            dets.append([int(frame_idx), class_index, score, bbox])

    # standalone detections
    for det in det_proto['detections']:
        local_idx = det['frame']
        image_name = image_name_at_frame(vid_proto, local_idx)
        frame_idx = image_set[image_name]
        bbox = map(lambda x:max(x,0), det['bbox'])
        score = [score['score'] for score in det['scores'] if score['class_index'] == class_index][0]
        dets.append([int(frame_idx), class_index, score, bbox])

    nms_boxes = [[det[0],]+det[-1]+[det[2],] for det in dets]
    keep = vid_nms(np.asarray(nms_boxes).astype('float32'), 0.3)

    kept_dets = [dets[i] for i in keep]
    for frame_idx, class_index, score, bbox in kept_dets:
        print '{} {} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
            frame_idx, class_index, score,
            bbox[0], bbox[1], bbox[2], bbox[3])

