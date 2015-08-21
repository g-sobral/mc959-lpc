#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys

def sliding_window(image, step_size, window_size):
    # slide a window across the image
    for y in xrange(0, image.shape[0], step_size):
        for x in xrange(0, image.shape[1], step_size):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


try:
    image_path = sys.argv[1]
except IndexError:
    print 'Usage: %s IMAGE_PATH', argv[0]
    exit()

try:
    original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print 'reading image:', image_path
except:
    print 'image path isn\'t valid:', image_path
    exit()

win_w = 200
win_h = 150
step = 100
i = 0
for (x, y, window) in sliding_window(original_img, step, (win_w, win_h)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != win_h or window.shape[1] != win_w:
        continue

    path = str(image_path.rsplit('.', 1)[0].rsplit('/', 1)[-1]) + '_' + str(i) + '.jpg'
    cv2.imwrite(path, window)
    i = i+1

