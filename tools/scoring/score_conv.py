#!/usr/bin/env python

import argparse
import sys
import os
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load, proto_dump
from vdetlib.utils.common import caffe_net
from vdetlib.vdet.tubelet_cls import score_conv_cls
from vdetlib.utils.visual import plot_track_scores
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('score_file')
    parser.add_argument('net_file')
    parser.add_argument('param_file')
    parser.add_argument('save_file')
    parser.add_argument('--job', type=int, default=1)
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()

    score_proto = proto_load(args.score_file)
    net = caffe_net(args.net_file, args.param_file, args.job-1)
    new_score_proto = score_conv_cls(score_proto, net)
    save_root = os.path.dirname(args.save_file)
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    proto_dump(new_score_proto, args.save_file)

    if args.save_dir is not None:
        plots = plot_track_scores(new_score_proto)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        for ind, plot in enumerate(plots, start=1):
            plot.savefig(os.path.join(args.save_dir, "{}.png".format(ind)))

