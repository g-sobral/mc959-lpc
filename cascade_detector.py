import cv2
import sys
import imutils

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
classifier = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
image, ratio = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Detect plates in the image
plates = classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=20,
    minSize=(60, 120),
    maxSize=(300, 600),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} plates!".format(len(plates))

# Draw a rectangle around the plates
for (x, y, w, h) in plates:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.namedWindow("Plates found", cv2.WINDOW_NORMAL)
cv2.imshow("Plates found", image)
cv2.waitKey(0)
