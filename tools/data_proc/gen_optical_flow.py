#!/usr/bin/env python
import argparse
import cv2
import os
import glob
import sys
import numpy as np
import scipy.io as sio

def cvReadGrayImg(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--bound', type=float, required=False, default=15,
                        help='Optical flow bounding.')
    args = parser.parse_args()

    norm_width = 500.
    bound = args.bound

    images = glob.glob(os.path.join(args.vid_dir,'*'))
    print "Processing {}: {} files.".format(args.vid_dir, len(images))
    img2 = cvReadGrayImg(images[0])
    for ind, img_path in enumerate(images[:-1]):
        img1 = img2
        img2 = cvReadGrayImg(images[ind+1])
        h, w = img1.shape
        fxy = norm_width / w
        # normalize image size
        flow = cv2.calcOpticalFlowFarneback(
            cv2.resize(img1, None, fx=fxy, fy=fxy),
            cv2.resize(img2, None, fx=fxy, fy=fxy),
            0.5, 3, 15, 3, 7, 1.5, 0)
        # map optical flow back
        flow = flow / fxy
        # normalization
        flow = np.round((flow + bound) / (2 * bound) * 255)
        flow[flow < 0] = 0
        flow[flow > 255] = 255
        flow = cv2.resize(flow, (w, h))

        # save
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(args.save_dir, basename+'_x.JPEG'),
            flow[...,0])
        cv2.imwrite(os.path.join(args.save_dir, basename+'_y.JPEG'),
            flow[...,1])

    # duplicate last frame
    basename = os.path.splitext(os.path.basename(images[-1]))[0]
    cv2.imwrite(os.path.join(args.save_dir, basename+'_x.JPEG'),
        flow[...,0])
    cv2.imwrite(os.path.join(args.save_dir, basename+'_y.JPEG'),
        flow[...,1])


        # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # hsv = np.zeros_like(cv2.imread(img_path))
        # hsv[...,1] = 255
        # hsv[...,0] = ang*180/np.pi/2
        # # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # hsv[...,2] = mag * 5
        # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # cv2.imshow('frame2',bgr)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
