#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys

def plate_from_annotation(annotation):
    """Return a dict containing license plate info extracted from
       annotation file.
    """
    try:
        f = open(txt_path, 'r')
        print 'reading annotation file:', txt_path
        line = f.readline()
    except IOError:
        print 'annotation file not found'
    f.close()

    line = line.rstrip('\n')
    print '>', line

    if line == 'None':
        return

    args = line.split(',')
    if len(args) != 9:
        print 'number of arguments in the line is incompatible:', args
        return

    plate = {
        'points': [],
        'string': ''
    }
    for i in range(0, 4):
        x = int(args.pop(0))
        y = int(args.pop(0))
        plate['points'].append( (x, y) )
    plate['string'] = args.pop(0)
    return plate

def crop_plate_image(img, plate):
    """Crop image region containing the license plate."""
    xmin = min(p[0] for p in plate['points'])
    xmax = max(p[0] for p in plate['points'])
    ymin = min(p[1] for p in plate['points'])
    ymax = max(p[1] for p in plate['points'])
    return img[ymin:ymax, xmin:xmax]


#
# main program
#

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

txt_path = image_path.rsplit('.', 1)[0] + '.txt'
plate = plate_from_annotation(txt_path)

cropped_img = crop_plate_image(original_img, plate)
path = str(image_path.rsplit('.', 1)[0].rsplit('/', 1)[-1]) + '_plate' + '.jpg'
cv2.imwrite(path, cropped_img)

