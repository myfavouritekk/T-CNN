#!/usr/bin/env python

import argparse
import os
import sys
import h5py
import scipy.io as sio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile')
    parser.add_argument('save_file')
    parser.add_argument('--varname', default='boxes')
    args = parser.parse_args()

    if os.path.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit(0)
    if not os.path.isdir(os.path.dirname(args.save_file)):
        try:
            os.makedirs(os.path.dirname(args.save_file))
        except OSError, e:
            if e.errno != 17:
                raise e

    boxes = sio.loadmat(args.matfile)[args.varname]
    sio.savemat(args.save_file, {'boxes': boxes})
    print "Save boxes from {} to {}".format(args.matfile, args.save_file)
