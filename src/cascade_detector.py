#!/usr/bin/python2.7

import cv2
import sys
import numpy as np
import argparse

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


# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('--cascade', default = 'cascade.xml',
                help = 'Path to cascade classifier xml')
ap.add_argument('--show', action="store_true",
                help = 'Show image with detected plates')
ap.add_argument('image',
                help = "Path to the image to be processed")
args = vars(ap.parse_args())

# Create the cascade classifier
classifier = cv2.CascadeClassifier(args["cascade"])

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
print "Found {0} plates!".format(len(plates))

if len(plates) == 0:
    print 'None'
else:
    # Draw a rectangle around the plates
    widths = []
    for idx, (x, y, w, h) in enumerate(plates):
        widths.append(w)
        plate = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # if it detects more than one plate select the smallest
    (x, y, w, h) = plates[widths.index(min(widths))]
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    points = [x, y, x+w, y, x+w, y+h, x, y+h]
    print ','.join(map(str, points)) + ',ABC1234'

if args["show"]:
    cv2.imshow("Plates found", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
