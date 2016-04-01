#!/usr/bin/env python

import argparse
import os
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import merge_score_protos, proto_dump, proto_load

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('score_file1')
    parser.add_argument('score_file2')
    parser.add_argument('save_file')
    parser.add_argument('--scheme', required=True, choices=['combine', 'max'])
    args = parser.parse_args()
    if os.path.isfile(args.save_file):
        print '{} already exists.'.format(args.save_file)
        sys.exit(0)
    score_proto1 = proto_load(args.score_file1)
    score_proto2 = proto_load(args.score_file2)
    new_proto = merge_score_protos(score_proto1, score_proto2, scheme=args.scheme)
    save_dir = os.path.dirname(args.save_file)
    if save_dir is not '' and not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    proto_dump(new_proto, args.save_file)
