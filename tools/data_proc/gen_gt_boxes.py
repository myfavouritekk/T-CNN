#!/usr/bin/env python

import argparse
import json
import os
import scipy.io as sio
import numpy as np
import cv2

def frame_path_at(vid_proto, frame_id):
    frame = [frame for frame in vid_proto['frames'] if frame['frame'] == frame_id][0]
    return str(os.path.join(vid_proto['root_path'], frame['path']))

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
        boxes_cls_size = [[box['bbox'], box['class_index'], box['frame_size']] for track in tracks for box in track if box['frame'] == frame['frame']]
        save_file = os.path.join(args.save_dir, save_name + '.mat')
        boxes = map(lambda x:x[0], boxes_cls_size)
        classes = map(lambda x:x[1], boxes_cls_size)
        frame_size = map(lambda x:x[2], boxes_cls_size)
        try:
            height = frame_size[0][0]
            width = frame_size[0][1]
        except IndexError:
            frame_path = frame_path_at(vid_proto, frame['frame'])
            height, width = cv2.imread(frame_path).shape[:2]
        sio.savemat(save_file, {'boxes': np.asarray(boxes, dtype='float64'),
                                'labels': np.asarray(classes, dtype='float64'),
                                'channels': 3,
                                'height': height,
                                'width': width},
            do_compression=True)
