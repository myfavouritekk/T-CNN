#!/usr/bin/env python

import argparse
import os
import sys
sys.path.insert(1, '.')
from vdetlib.vdet.dataset import imagenet_vdet_classes
from vdetlib.utils.visual import unique_colors, add_bbox
from vdetlib.utils.common import imread, imwrite
from vdetlib.utils.protocol import proto_dump, proto_load, top_detections, frame_path_at, track_box_at_frame
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('annot_file')
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    annot_proto = proto_load(args.annot_file)

    colors = unique_colors(len(annot_proto['annotations']))

    for frame in vid_proto['frames']:
        img = imread(frame_path_at(vid_proto, frame['frame']))
        boxes = [track_box_at_frame(tracklet, frame['frame']) \
                for tracklet in [annot['track'] for annot in annot_proto['annotations']]]
        tracked = add_bbox(img, boxes, None, 10)
        if args.save_dir:
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            imwrite(os.path.join(args.save_dir, "{:04d}.jpg".format(frame['frame'])),
                    tracked)
        else:
            cv2.imshow('tracks', tracked)
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)
            cv2.destroyAllWindows()
