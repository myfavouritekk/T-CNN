#!/usr/bin/env python

import argparse
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy import array
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load
from vdetlib.utils.cython_nms import vid_nms

def image_name_at_frame(vid_proto, frame_idx):
    vid_name = vid_proto['video']
    for frame in vid_proto['frames']:
        if frame['frame'] == frame_idx:
            return os.path.join(vid_name, os.path.splitext(frame['path'])[0])
    return None

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

def overall_idx(vid_proto, frame_idx, image_set):
    image_name = image_name_at_frame(vid_proto, frame_idx)
    if image_name is None:
        return None
    return int(image_set[image_name])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('score_file')
    parser.add_argument('image_set_file')
    parser.add_argument('--varname')
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--interpolate', dest='do_interpolate', action='store_true')
    parser.set_defaults(do_interpolate=False)
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    score_proto = proto_load(args.score_file)
    with open(args.image_set_file) as f:
        image_set = dict([line.strip().split() for line in f.readlines()])
    vid_name = vid_proto['video']
    assert vid_name == score_proto['video']

    dets = []
    for tubelet in score_proto['tubelets']:
        if tubelet['gt'] == 1:
            raise ValueError('Dangerous: Score file contains gt tracks!')
        class_index = tubelet['class_index']
        if args.do_interpolate:
            temp_dets = []
            all_frame_idx = []
            for box in tubelet['boxes']:
                frame_idx = box['frame']
                all_frame_idx.append(frame_idx)
                if args.interval == 1 or frame_idx % args.interval == 1:
                    image_name = image_name_at_frame(vid_proto, frame_idx)
                    if image_name is None:
                        break
                    bbox = map(lambda x:max(x,0), box['bbox'])
                    score = box[args.varname]
                    temp_dets.append([class_index, frame_idx, score, bbox])
            idx = map(lambda x:x[1], temp_dets)
            scores = map(lambda x:x[2], temp_dets)
            x1 = map(lambda x:x[3][0], temp_dets)
            y1 = map(lambda x:x[3][1], temp_dets)
            x2 = map(lambda x:x[3][2], temp_dets)
            y2 = map(lambda x:x[3][3], temp_dets)
            map_funs = []
            for y in [scores, x1, y1, x2, y2]:
                if len(y) >= 2:
                    interpolator = interp1d(idx, y)
                    extrapolator = extrap1d(interpolator)
                else:
                    # if not enough smaples, fall back to identity function
                    extrapolator = lambda x:map(lambda y:y, x)
                map_funs.append(extrapolator)
            o_indices = map(lambda x:overall_idx(vid_proto, x, image_set), all_frame_idx)
            all_scores = map_funs[0](all_frame_idx)
            all_x1 = map_funs[1](all_frame_idx)
            all_y1 = map_funs[2](all_frame_idx)
            all_x2 = map_funs[3](all_frame_idx)
            all_y2 = map_funs[4](all_frame_idx)
            for o_idx, score, bbox in zip(o_indices, all_scores, zip(all_x1, all_y1, all_x2, all_y2)):
                dets.append([int(o_idx), class_index, score, list(bbox)])
        else:
            for box in tubelet['boxes']:
                frame_idx = box['frame']
                if args.interval == 1 or frame_idx % args.interval == 1:
                    for step in xrange(args.interval):
                        image_name = image_name_at_frame(vid_proto, frame_idx+step)
                        if image_name is None:
                            break
                        overall_idx = image_set[image_name]
                        bbox = map(lambda x:max(x,0), box['bbox'])
                        score = box[args.varname]
                        dets.append([int(overall_idx), class_index, score, bbox])

    nms_boxes = [[det[0],]+det[-1]+[det[2],] for det in dets]
    keep = vid_nms(np.asarray(nms_boxes).astype('float32'), 0.3)

    kept_dets = [dets[i] for i in keep]
    for frame_idx, class_index, score, bbox in kept_dets:
        print '{} {} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
            frame_idx, class_index, score,
            bbox[0], bbox[1], bbox[2], bbox[3])

