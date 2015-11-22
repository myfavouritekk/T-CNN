#!/usr/bin/env python

import argparse
import subprocess as sp
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_pattern')
    parser.add_argument('save_file')
    parser.add_argument('--height', type=int, default=0)
    args = parser.parse_args()
    if os.path.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit(0)
    command = ['ffmpeg', '-f', 'image2', '-i', args.input_pattern, '-q', '1']
    if args.height > 0:
        command.extend(['-vf', 'scale=trunc(oh*a/2)*2:{}'.format(args.height)])
    command.append(args.save_file)
    print "Creating {}...".format(args.save_file)
    devnull = open(os.devnull, 'wb')
    sp.call(command, stderr=devnull, stdin=devnull, stdout=devnull)
