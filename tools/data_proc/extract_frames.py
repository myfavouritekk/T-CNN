#!/usr/bin/env python

import argparse
import os
from util import os_command, stem

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
        help='Input video.')
    parser.add_argument('save_dir',
        help='Save directory.')
    parser.add_argument('--st_pos', default='00:00:10',
        help='Starting position in format hh:mm:ss[.xxx]. [00:00:10]')
    parser.add_argument('--fps', default='5',
        help='Frame per second. [5]')
    parser.add_argument('--max_frames', default=200, type=int,
        help='Maxium number of frames. [200]')
    args = parser.parse_args()


    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    vid_stem = stem(args.input)
    command = ['ffmpeg',
               '-i', args.input,
               '-r', args.fps,
               '-ss', args.st_pos,
               '-f', 'image2',
               '-vf', 'scale=-1:500',
               '-q:v', '2']
    if args.max_frames > 0:
        command += ['-vframes', args.max_frames]
    command += [os.path.join(args.save_dir, vid_stem + '_%04d.JPEG')]
    os_command(command)