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
    parser.add_argument('save_dir')
    args = parser.parse_args()

    with open(args.vid_file) as f:
        vid_proto = json.load(f)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    video = vid_proto['video']
    print "Processing {}".format(video)
    for frame in vid_proto['frames']:
        save_name = os.path.splitext(os.path.basename(frame['path']))[0]
        save_file = os.path.join(args.save_dir, save_name + '.mat')
        frame_path = frame_path_at(vid_proto, frame['frame'])
        height, width = cv2.imread(frame_path).shape[:2]
        sio.savemat(save_file, {'boxes': [],
                                'labels': [],
                                'channels': 3,
                                'height': height,
                                'width': width},
            do_compression=True)
