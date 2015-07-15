"""
    Usage: ocr_training.py (<path>) [options]
           ocr_training.py -h | --help

    Arguments:
      <path>          Path to traning data (jpegs)

    Options:
      -h --help         Displays help
      --debug           Turns debug mode on

    Example:
      ocr_training.py ./dataset/
      ocr_training.py ./traning_sets/dataset01/
"""

from docopt import docopt
import os
import sys
    
import numpy as np
import cv2

import ocr
import pickle

args = docopt(__doc__)

trainingDataf5 = np.empty((0,65))
trainingDataf10 = np.empty((0,140))
trainingDataf15 = np.empty((0,265))
trainingDataf20 = np.empty((0,440))

trainingDataBinary = np.empty((0,400))

classes = []
feature_ocr = ocr.OCR(debug=args['--debug'])

for directory in os.listdir(args['<path>']):
    if directory.endswith('.db'):
        continue
    if args['--debug']:
        print(directory)
    for fn in os.listdir(args['<path>'] + directory + '/'):
        if fn.endswith('.jpg'):
            #print('directory: ' + directory)    
            classes.append(ord(directory[0]))
            img_path = args['<path>'] + directory + '/' + fn

            img = cv2.imread(img_path)
        
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            imgThreshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,-10)

            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

            erode = cv2.erode(imgThreshold, kernel, iterations=1)

            img = feature_ocr.prepocessChar(erode)

            #print('img size: ' + str(img.size))

            # if args['--debug']:
            #     print('img shape: ' + str(img.shape))
            #     cv2.imshow('feature', img)
            #     cv2.waitKey(0)

            f5 = feature_ocr.features(img, 5)
            f10 = feature_ocr.features(img, 10)
            f15 = feature_ocr.features(img, 15)
            f20 = feature_ocr.features(img, 20)

            # print('f5 size: ' + str(f5.size))
            # print('f10 size: ' + str(f10.size))
            # print('f15 size: ' + str(f15.size))
            # print('f20 size: ' + str(f20.size))


            f5  = np.float32(np.reshape(f5, (1, f5.size)))
            f10  = np.float32(np.reshape(f10, (1, f10.size)))
            f15  = np.float32(np.reshape(f15, (1, f15.size)))
            f20  = np.float32(np.reshape(f20, (1, f20.size)))

            img = np.float32(np.reshape(img, (1, img.size)))

            trainingDataf5 = np.append(trainingDataf5, f5, 0)
            trainingDataf10 = np.append(trainingDataf10, f10, 0)
            trainingDataf15 = np.append(trainingDataf15, f15, 0)
            trainingDataf20 = np.append(trainingDataf20, f20, 0)

            trainingDataBinary = np.append(trainingDataBinary, img, 0)

classes = np.array(classes, dtype = np.float32)
classes = np.reshape(classes, (classes.size, 1))

# trainingDataf5 = np.array(trainingDataf5, dtype = np.float32)
# trainingDataf10 = np.array(trainingDataf10, dtype = np.float32)
# trainingDataf15 = np.array(trainingDataf15, dtype = np.float32)
# trainingDataf20 = np.array(trainingDataf20, dtype = np.float32)

np.savetxt('trainingClasses.data', classes)
output = open('trainingData.pkl', 'wb')
pickle.dump(trainingDataf5, output)
pickle.dump(trainingDataf10, output)
pickle.dump(trainingDataf15, output)
pickle.dump(trainingDataf20, output)
pickle.dump(trainingDataBinary, output)
output.close()
print "training complete"