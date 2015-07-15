"""
    Usage: demo_ocr.py (<trainingData>) (<trainingClasses>) (<imgPath>) [options]
           demo_ocr.py -h | --help

    Arguments:
      <trainingData>    Path to the traning data file (pickle file)
      <trainingClasses> Path to the traning classes file
      <imgPath>            Path to the image file (jpg)

    Options:
      --debug           Sets debug mode on
      -h --help         Displays help

    Example:
      ocr_training.py ./data/traningData.pkl ./data/traningClasses.data ./tests/plate_00001.jpg
"""
from docopt import docopt

import ocr
import cv2
import numpy

args = docopt(__doc__)

print(str(args['--debug']))

plateOCR = ocr.OCR(debug=args['--debug'])

plateOCR.loadTraningData(args['<trainingData>'], args['<trainingClasses>'])

plate = cv2.imread(args['<imgPath>'])
results, resultsPos, plateString = plateOCR.run(plate)

print(str(results))
print(plateString)
