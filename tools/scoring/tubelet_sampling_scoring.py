#!/usr/bin/env python

import argparse
import sys
import os
sys.path.insert(1, '.')
sys.path.insert(1, './External/caffe-official/python')
from vdetlib.utils.protocol import proto_load, proto_dump, track_proto_from_annot_proto
from vdetlib.utils.common import caffe_net
from vdetlib.vdet.dataset import imagenet_vdet_class_idx, imagenet_det_200_class_idx
from vdetlib.vdet.tubelet_cls import scoring_tracks, rcnn_sampling_scoring

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('track_file')
    parser.add_argument('net_file')
    parser.add_argument('param_file')
    parser.add_argument('rcnn_model')
    parser.add_argument('save_file')
    parser.add_argument('--annot_file', default=None,
        help='Annotation file if available to calculate gt overlaps for training.')
    parser.add_argument('--cls')
    parser.add_argument('--job', type=int, default=1)
    parser.add_argument('--sampling_num', type=int)
    parser.add_argument('--sampling_ratio', type=float)
    parser.add_argument('--save_feat', dest='save_feat', action='store_true')
    parser.set_defaults(save_feat=False)
    parser.add_argument('--save_all_sc', dest='save_all_sc', action='store_true')
    parser.set_defaults(save_all_sc=False)
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    if args.annot_file is not None:
        annot_proto = proto_load(args.annot_file)
    else:
        annot_proto = None
    if os.path.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit(0)

    # classify ground truth tracks if no track_file provided
    if args.track_file == 'None':
        gt_classes = [annot['track'][0]['class'] \
            for annot in annot_proto['annotations']]
        if args.cls not in gt_classes:
            print "{} not in gt file {}".format(args.cls, args.annot_file)
            sys.exit(0)
        track_proto = track_proto_from_annot_proto(annot_proto)
    else:
        track_proto = proto_load(args.track_file)

    vid_name = vid_proto['video']
    if annot_proto is not None:
        assert vid_name == annot_proto['video']
    assert vid_name == track_proto['video']
    cls_index = imagenet_vdet_class_idx[args.cls]

    net = caffe_net(args.net_file, args.param_file, args.job-1)
    rcnn_sc = lambda vid_proto, track_proto, net, class_idx: \
        rcnn_sampling_scoring(vid_proto, track_proto, net, class_idx, args.rcnn_model,
            args.sampling_num, args.sampling_ratio,
            save_feat=args.save_feat, save_all_sc=args.save_all_sc)
    rcnn_sc.__name__ = "rcnn_sampling_{}".format(
        os.path.splitext(os.path.basename(args.param_file))[0])

    score_proto = scoring_tracks(vid_proto, track_proto, annot_proto,
        rcnn_sc, net, cls_index)
    # ground truth scores, only save gt class scores
    if args.track_file == 'None':
        for tubelet in score_proto['tubelets']:
            if tubelet['gt'] == 0:
                del tubelet
        if not score_proto['tubelets']:
            sys.exit(0)
    # save score proto or gt score proto
    save_dir = os.path.dirname(args.save_file)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    proto_dump(score_proto, args.save_file)
