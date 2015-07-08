import cv2
import numpy as np

class HistogramType:
    HORIZONTAL = 1
    VERTICAL = 2

class CharSegment:
    def __init__(self, img, rect):
        self.img = img
        self.pos = rect

class OCR:
    numChars = 30
    strChars = ['0','1','2','3','4','5','6','7','8','9', 'A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, debug = False):
        self.classes = [i for i in range(0,35)] # 20 letters + 10 numbers (excludes vocals and 'Q')
        self.charSize = 20
        self.knnClassifier = cv2.KNearest()
        self.K = 10

        self.debug = debug
        self.saveSegments = False
        self.filename = ''

    def prepocessChar(self, img):
        height = img.shape[0]
        width = img.shape[1]

        m = max(height, width)

        transformMat = np.eye(2, M=3, dtype=np.float32)
        transformMat[0][2] = m/2 - width/2
        transformMat[1][2] = m/2 - height/2

        warpImage = cv2.warpAffine(img, transformMat, (m,m), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        output = cv2.resize(warpImage, (self.charSize,self.charSize))

        if self.debug:
            cv2.imshow('prepocessed image', output)
            cv2.waitKey(0)

        return output

    def verifySizes(self, img):
        aspect = 33.0/60.0
        error = 0.25
        charAspect = float(img.shape[1])/float(img.shape[0])

        minHeight = 23.0
        maxHeight = 32.0

        minAspect = 0.15
        maxAspect = aspect + (aspect*error)

        if self.debug:
            print(str(cv2.countNonZero(img)))

        area = cv2.countNonZero(img)
        bbArea = img.shape[0]*img.shape[1]

        percPixels = float(area)/float(bbArea)

        if self.debug:
            print('verify sizes')
            print('aspect: ' + str(aspect))
            print('min/max aspect: ' + str(minAspect) + ',' + str(maxAspect))
            print('area: ' + str(percPixels))
            print('char aspect: ' + str(charAspect))
            print('char height: ' + str(img.shape[0]))
            print('\n')

        if (percPixels < 0.8) and (charAspect > minAspect) and (charAspect < maxAspect) and (img.shape[0] >= minHeight) and (img.shape[0] < maxHeight):
            if self.debug:
                print('True')
            return True
        else:
            if self.debug:
                print('False')
            return False

    def segment(self, plate):
        gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        retval, imgThreshold = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        if self.debug:
            cv2.imshow('threshold', imgThreshold)
            cv2.waitKey(0)

        imgContour = imgThreshold
        contours, hierarchy = cv2.findContours(imgContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        result = imgThreshold
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(result, contours, -1, (255,0,0), 1)

        output = []

        for contour in contours:
            [x,y,w,h] = cv2.boundingRect(contour)
            cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),1)
            auxRoi = imgThreshold[y:y+h,x:x+w]
            if(self.verifySizes(auxRoi)):
                auxRoi = self.prepocessChar(auxRoi)
                output.append(CharSegment(auxRoi, [x,y,w,h]))
                cv2.rectangle(result,(x,y),(x+w,y+h),(0,125,255),1)

        if self.debug:
            print('number of chars: ' + str(len(output)))
            print('output: ' + str(output))
            cv2.imshow('segment result', result)
            cv2.waitKey(0)

        return output

    def projectedHistogram(self, img, option):
        if option:
            size = img.shape[1]
        else:
            size = img.shape[0]

        mhist = np.zeros((1, size), dtype=np.float32)

        for i in range(0, size):
            if option:
                data = img[i]
            else:
                data = img.transpose()[i]
            mhist[0][i] = cv2.countNonZero(data)

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mhist)

        if maxVal > 0:
            mhist = mhist * (1/maxVal)

        return mhist

    def getVisualHistogram(self, hist, type):
        size = 100

        if type == HistogramType.HORIZONTAL:
            imHist = np.empty((size, len(hist), 3))
        else:
            imHist = np.empty((len(hist), size, 3))

        imHist.fill(55)

        for i in range (0, hist.shape[1]):
            value = hist[i]

    def features(self, img, dataSize):
        vhist = self.projectedHistogram(img, HistogramType.HORIZONTAL)
        hhist = self.projectedHistogram(img, HistogramType.VERTICAL)

        lowData = np.resize(img, (dataSize, dataSize))

        numCols = vhist.shape[1] + hhist.shape[1] + lowData.shape[1]**2

        out = np.zeros((1, numCols), dtype=np.float32)
        j = 0

        for i in range(0, vhist.shape[1]):
            out[0][j] = vhist[0][i]
            j += 1

        for i in range(0, hhist.shape[1]):
            out[0][j] = hhist[0][i]
            j += 1

        for x in range(0, lowData.shape[1]):
            for y in range(0, lowData.shape[0]):
                out[0][j] = lowData[y][x]
                j += 1

        if self.debug:
            print(str(out))
            print('=============================')

        return out

    def train(self, samples, classes, K = 10):
        self.K = k
        self.knnClassifier.train(samples, classes, maxK=self.K)

    def classify(self, img):
        respose = self.knnClassifier(img, self.K)

    def run(self, plate):
        segments = self.segment(plate)
        result = []
        resultsPos = []

        for i in range(0, len(segments)):
            charImg = self.prepocessChar(segments[i].img)
            if(self.saveSegments):
                stream = ''
                stream += 'tmpChars/' + self.filename + '_' + str(i) + '.jpg'
                cv2.imwrite(stream, charImg)

            features = self.features(charImg, 15)
            character = self.classify(charImg)
            result.append(strChars[character])
            resultsPos.append(segments[i].pos)

        return (results, resultsPos)





