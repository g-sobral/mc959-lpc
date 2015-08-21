#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
    The License-Plate Challenge
    MC959 - Introduction to Computer Vision

    This program was developed for python 2.7 and openvc 2.4.1

    Author: Gabriel Sobral <gasan.sobral@gmail.com>
"""

import cv2
import sys
import numpy as np
import argparse

import ocr

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return (resized, r)

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('--cascade', default = 'cascade.xml',
                help = 'Path to cascade classifier xml')
ap.add_argument('--ocr_training_samples', default='trainingData.pkl',
                help = 'Path to OCR training samples data (pickle file)')
ap.add_argument('--ocr_training_classes', default='trainingClasses.data',
                help = 'Path to OCR training classes data (numpy generated txt)')
ap.add_argument('--show', action="store_true",
                help = 'Show image with detected plates')
ap.add_argument('image',
                help = "Path to the image to be processed")
ap.add_argument('--debug', action="store_true",
                help = "Enables debug mode (OCR stuff mostly)")
args = vars(ap.parse_args())

# Create the cascade classifier
classifier = cv2.CascadeClassifier(args["cascade"])

# Create OCR and loads trainng data
plateOCR = ocr.OCR(debug=args["debug"])
plateOCR.loadTraningData(args["ocr_training_samples"], args["ocr_training_classes"])

# Read, resize and convert image to grayscale
image = cv2.imread(args["image"])
image, ratio = resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect plates in the image
plates = classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    # minSize=(60, 20),
    # maxSize=(300, 100),
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

screenCnt = []

if len(plates) == 0:
    print 'None'
else:
    # Draw a rectangle around the plates
    widths = []
    for idx, (x, y, w, h) in enumerate(plates):
        widths.append(w)

    # if it detects more than one plate select the smallest
    (x, y, w, h) = plates[widths.index(min(widths))]
    plate = gray[y:y+h, x:x+w]

    # convert the image to grayscale, blur it, and find edges
    # in the image
    blurred = cv2.GaussianBlur(plate, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (cnts, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if len(screenCnt):
        # apply the four point transform to obtain a top-down
        # view of the original image
        plate = four_point_transform(plate, screenCnt.reshape(4, 2))
        xc,yc,w,h = cv2.boundingRect(screenCnt)
        x = x + xc
        y = y + yc

    points = [x, y, x+w, y, x+w, y+h, x, y+h]
    points = [int(p/ratio) for p in points]

    if args["show"]:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255), 2)
        cv2.imshow(args["image"], image)
        cv2.imshow("detected plate", plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    

    results, resultsPos, plateString = plateOCR.run(plate)

    print ','.join(map(str, points)) + ',' + plateString


