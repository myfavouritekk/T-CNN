#!/usr/bin/env python

import matlab.engine
import argparse
import os
import sys
sys.path.insert(1, '.')
from vdetlib.vdet.track import greedily_track_from_raw_dets, fcn_tracker
from vdetlib.vdet.dataset import imagenet_vdet_class_idx
from vdetlib.utils.protocol import proto_dump, proto_load, det_score, load_det_info
from vdetlib.utils.common import options
from vdetlib.utils.log import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('det_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of detections to track. [10]')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum track length. [Inf]')
    parser.add_argument('--step', type=int, default=1,
                        help='Tracking frame step. [1]')
    parser.add_argument('--thres', type=float, default=0.,
                        help='Threshold to terminate tracking. [0.]')
    parser.add_argument('--nms_thres', type=float, default=0.3,
                        help='Overlap threshold to start new anchors. [0.3]')
    parser.add_argument('--job', type=int, default=1,
                        help='job id. [1]')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        try:
            os.makedirs(args.save_dir)
        except:
            pass

    vid_proto = proto_load(args.vid_file)
    vid_name = vid_proto['video']
    logging.info("Video: {}".format(vid_proto['video']))

    det_info = load_det_info(vid_proto, args.det_dir)

    eng = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
    opts = options({'engine': eng, 'max_tracks': args.num, 'thres': args.thres,
                   'gpu': args.job - 1, 'max_frames': args.max_frames,
                   'step': args.step, 'nms_thres': args.nms_thres})

    for cls_name in imagenet_vdet_class_idx:
        if cls_name == '__background__':
            continue
        save_name = os.path.join(args.save_dir, '.'.join([vid_name, cls_name, 'track.gz']))
        if os.path.isfile(save_name):
            logging.warning("{} already exists.".format(save_name))
            continue
        logging.info("Tracking {}...".format(cls_name))
        cls_idx = imagenet_vdet_class_idx[cls_name]
        track_proto = greedily_track_from_raw_dets(vid_proto, det_info, fcn_tracker,
            cls_idx, opts)
        if not track_proto['tracks']:
            logging.warning("{} has no tracks for class {}".format(vid_name, cls_name))
            continue
        proto_dump(track_proto, save_name)
