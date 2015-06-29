#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
ap.add_argument("-o", "--output", help = "path to output image")
ap.add_argument("-w", "--width", type = int, default = 800,
                help = "width for resized image")
args = ap.parse_args()

def plate_from_annotation(annotation, r=1):
    """Return a dict containing license plate info extracted from
       annotation file.
    """

    line = annotation.rstrip('\n')
    print '>', line

    if line == 'None':
        return None

    args = line.split(',')
    if len(args) != 9:
        print 'number of arguments in the line is incompatible:', args
        return

    plate = {
        'points': [],
        'string': ''
    }
    for i in range(0, 4):
        x = int(int(args.pop(0)) * r)
        y = int(int(args.pop(0)) * r)
        plate['points'].append( (x, y) )
    plate['string'] = args.pop(0)
    return plate

def save_resized(path, plate, res_image):
    cv2.imwrite(path, res_image)
    txt_path = path.rsplit('.', 1)[0] + '.txt'
    f = open(txt_path, 'w')

    line = ''
    if not plate == None:
        for (x, y) in plate['points']:
            line += str(x) + ',' + str(y) + ','
        line += plate['string'] + '\n'
    else:
        line = 'None'

    print 'writing resized annotaion:', line
    f.write(line)
    f.close


#
# main program
#

image_path = args.image
try:
    original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print 'reading image:', image_path
except:
    print 'image path isn\'t valid:', image_path
    exit()

resized, ratio = imutils.resize(original_img, width=args.width)

txt_path = image_path.rsplit('.', 1)[0] + '.txt'

try:
    f = open(txt_path, 'r')
    print 'reading annotation file:', txt_path
    line = f.readline()
except IOError:
    print 'annotation file not found'
    f.close()
else:
    plate = plate_from_annotation(line, ratio)


if args.output:
   save_resized(args.output, plate, resized)

