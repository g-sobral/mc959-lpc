import cv2
import numpy as np
import pickle

class HistogramType:
    HORIZONTAL = 1
    VERTICAL = 2

class CharSegment:
    def __init__(self, img, rect):
        self.img = img
        self.pos = rect
        self.letter = 0

    def __repr__(self):
        return repr((self.img, self.pos, self.letter))

class OCR:
    numChars = 35
    strChars = ['1','2','3','4','5','6','7','8','9', 'A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, debug = False):
        self.classes = [i for i in range(0,35)] # 20 letters + 9 numbers (excludes vocals and 'Q')
        self.charSize = 40
        self.knnClassifier = cv2.KNearest()
        self.K = 10

        self.debug = debug
        self.saveSegments = True
        self.filename = 'ocr_char'

    def preprocessPlate(self, plate):
        """
        Plate preprocessing
        Attempts to find the innermost part of the plate containing the letters
        while removing the surrounding borders.

        Makes use of different filterings to try and obtain the result.

        If no good result is found, returns the original plate.

        @params
            plate: numpy array (image)

        @return
            output: numpy array (image)
        """
        imgArea = plate.shape[0] * plate.shape[1]

        blur = cv2.blur(plate, (3, 3))
        equalized = cv2.equalizeHist(blur)

        bilateral = cv2.bilateralFilter(plate, 9, 75, 75)
        equalized2 = cv2.equalizeHist(bilateral)

        imgThreshold = cv2.adaptiveThreshold(equalized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-10)
        testThreshold = cv2.adaptiveThreshold(equalized2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-10)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        
        dilate1 = cv2.dilate(imgThreshold, kernel, iterations=1)
        dilate2 = cv2.dilate(testThreshold, kernel, iterations=1)

        median = cv2.medianBlur(dilate1, 5)

        gaussian = cv2.GaussianBlur(dilate1,(5,5),0)
        gaussian2 = cv2.GaussianBlur(dilate2,(5,5),0)

        canny = cv2.Canny(gaussian, 150, 75)
        canny2 = cv2.Canny(median, 150, 75)
        canny3 = cv2.Canny(gaussian2, 200, 100)

        if(self.debug):
            cv2.imshow('canny', canny)
            cv2.waitKey(0)

            cv2.imshow('canny2', canny2)
            cv2.waitKey(0)

            cv2.imshow('canny3', canny3)
            cv2.waitKey(0)

        ratio = float(40.0)/float(13.0)
        error = 0.25

        maxRatio = ratio + ratio * error + (ratio * error / 2)
        minRatio = ratio - (ratio * error)

        rect = None

        if(rect == None):
            print('1')
            imgContour = canny
            contours, _ = cv2.findContours(imgContour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

            for contour in contours:
                [x,y,w,h] = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                if len(approx) == 4:
                    auxRoi = imgThreshold[y:y+h,x:x+w]
                    if(self.debug):
                        cv2.imshow('roi', auxRoi)
                        cv2.waitKey(0)
                        print('float(w)/float(h): ' + str(float(w)/float(h)))
                    if (float(w)/float(h)) >= minRatio and (float(w)/float(h)) <= maxRatio and (plate.shape[1] - w) < (plate.shape[1]*0.31):
                        rect = [x,y,w,h]
                        cv2.rectangle(plate,(x,y),(x+w,y+h),(0,0,255),2)
                        break


        if(rect == None):
            print('2')
            imgContour = canny2
            contours, _ = cv2.findContours(imgContour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

            for contour in contours:
                [x,y,w,h] = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                if len(approx) == 4:
                    auxRoi = imgThreshold[y:y+h,x:x+w]
                    if(self.debug):
                        cv2.imshow('roi', auxRoi)
                        cv2.waitKey(0)
                        print('float(w)/float(h): ' + str(float(w)/float(h)))
                    if (float(w)/float(h)) >= minRatio and (float(w)/float(h)) <= maxRatio and (plate.shape[1] - w) < (plate.shape[1]*0.31):
                        rect = [x,y,w,h]
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        break



        if(rect == None):
            print('3')
            imgContour = canny3
            contours, _ = cv2.findContours(imgContour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

            for contour in contours:
                [x,y,w,h] = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                if len(approx) == 4:
                    auxRoi = imgThreshold[y:y+h,x:x+w]
                    if(self.debug):
                        cv2.imshow('roi', auxRoi)
                        cv2.waitKey(0)
                        print('float(w)/float(h): ' + str(float(w)/float(h)))
                    if (float(w)/float(h)) >= minRatio and (float(w)/float(h)) <= maxRatio and (plate.shape[1] - w) < (plate.shape[1]*0.31):
                        rect = [x,y,w,h]
                        cv2.rectangle(plate,(x,y),(x+w,y+h),(0,0,255),2)
                        break  

        if rect != None:
            return plate[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        else:
            return plate



    def prepocessChar(self, img):
        """
        Segmented character preprocessing
        After segmentation, the character is normalized through warping
        and resizing operations.

        @params
            img: numpy array (image)

        @return
            output: numpy array (image)

        """

        height = img.shape[0]
        width = img.shape[1]

        m = max(height, width)

        transformMat = np.eye(2, M=3, dtype=np.float32)
        transformMat[0][2] = m/2 - width/2
        transformMat[1][2] = m/2 - height/2

        warpImage = cv2.warpAffine(img, transformMat, (m,m), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        output = cv2.resize(warpImage, (20,20))

        if self.debug:
            print('output shape: '  + str(output.shape))
            cv2.imshow('prepocessed image', output)
            cv2.waitKey(0)

        return output



    def verifySizes(self, img):
        """
        Verify character size
        Called during segmentation on order to check if the given segment
        follows certain propreties common to characters. If it does, returns
        true and the input is considered a character. Otherwise returns false.

        @params
            img: numpy array (image)

        @return
            boolean
        """

        aspect = 33.0/60.0
        error = 0.25
        charAspect = float(img.shape[1])/float(img.shape[0])

        minHeight = 18.0
        maxHeight = 45.0

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
            print('min/max aspect: ' + str(minAspect) + ',' + str(maxAspect) + ' state: ' + str((charAspect > minAspect)) + '/' + str((charAspect < maxAspect)))
            print('area: ' + str(percPixels) + ' state: ' + str((percPixels < 0.8) or (percPixels > 0.95)))
            print('char aspect: ' + str(charAspect))
            print('char height: ' + str(img.shape[0]) + ' state: min = ' + str((img.shape[0] >= minHeight)) + '/max = ' + str((img.shape[0] < maxHeight)))

        if ((percPixels < 0.8) or (percPixels > 0.95)) and (charAspect > minAspect) and (charAspect < maxAspect) and (img.shape[0] >= minHeight) and (img.shape[0] < maxHeight):
            if self.debug:
                print('True')
                print('----------------')
                print('\n')
            return True
        else:
            if self.debug:
                print('False')
                print('----------------')
                print('\n')
            return False



    def segment(self, plate):
        """
        Segment plate
        Given a plate imege, tries to segment it in seven characters.

        @params
            plate: numpy array (image)

        @return
            output: array of CharSegments
        """
        im = cv2.resize(plate, (150, 50), interpolation = cv2.INTER_LINEAR)

        #gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(im, 3, 40, 30)
        #equalized = cv2.equalizeHist(bilateral)
        imgThreshold = cv2.adaptiveThreshold(bilateral,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,-10)

        if self.debug:
            cv2.imshow('threshold', imgThreshold)
            cv2.waitKey(0)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        
        erode = cv2.erode(imgThreshold, kernel, iterations=1)


        morph = cv2.morphologyEx(imgThreshold, cv2.MORPH_OPEN, kernel)

        if self.debug:
            cv2.imshow('erode', morph)
            cv2.waitKey(0)        

        imgContour = np.array(morph, copy=True)
        contours, hierarchy = cv2.findContours(imgContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        result = erode
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)       

        cv2.drawContours(result, contours, -1, (0,255,0), 1)

        output = []

        for contour in contours:
            [x,y,w,h] = cv2.boundingRect(contour)
            cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),1)
            auxRoi = imgThreshold[y:y+h,x:x+w]
            if(self.verifySizes(auxRoi)):
                auxRoi = self.prepocessChar(auxRoi)
                output.append(CharSegment(auxRoi, [x,y,w,h]))
                cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,255),1)

        if len(output) < 7 or len(output) > 7:
            imgContour = np.array(erode, copy=True)
            contours, hierarchy = cv2.findContours(imgContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            result = erode
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            cv2.drawContours(result, contours, -1, (0,255,0), 1)            

            output = []

            for contour in contours:
                [x,y,w,h] = cv2.boundingRect(contour)
                cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),1)
                auxRoi = imgThreshold[y:y+h,x:x+w]
                if(self.verifySizes(auxRoi)):
                    auxRoi = self.prepocessChar(auxRoi)
                    output.append(CharSegment(auxRoi, [x,y,w,h]))
                    cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,255),1)

        if len(output) < 7 or len(output) > 7:
            imgContour = np.array(imgThreshold, copy=True)
            contours, hierarchy = cv2.findContours(imgContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            result = imgThreshold
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            cv2.drawContours(result, contours, -1, (0,255,0), 1)            

            output = []

            for contour in contours:
                [x,y,w,h] = cv2.boundingRect(contour)
                cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),1)
                auxRoi = imgThreshold[y:y+h,x:x+w]
                if(self.verifySizes(auxRoi)):
                    auxRoi = self.prepocessChar(auxRoi)
                    output.append(CharSegment(auxRoi, [x,y,w,h]))
                    cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,255),1)

        if len(output) > 7:
            minHeight = min(output, key=lambda element: element.pos[3])
            output.remove(minHeight)

        if self.debug:
            print('number of chars: ' + str(len(output)))
            #print('output: ' + str(output))
            cv2.imshow('segment result', result)
            cv2.waitKey(0)

        return output



    def train(self, samples, classes, K = 10):
        """
        Train K-Nearest
        Trains the machine learning algorithm with the training data.

        @params
            samples: numpy array of numpy arrays, each array of the same size and dimension
            classes: numpy array of floats
            K: int (number of neighbors to look)

        @return
            None
        """
        self.K = k
        self.knnClassifier.train(samples, classes, maxK=self.K)



    def classify(self, img):
        """
        Classify image
        Uses the trained K-Nearest algorithm to classify a given character

        @params
            img: numpy array (image)

        @return
            retval: int (class of the given image)
        """
        tempImg = np.float32(np.reshape(img, (1, img.size)))

        print('tmp data shape: ' + str(tempImg.shape))
        print('tmp data type: ' + str(tempImg.dtype))

        retval, results, neigh_resp, dists = self.knnClassifier.find_nearest(tempImg, self.K)

        if self.debug:
            print('retval: ' + str(retval))
            print('results: '  + str(results))
            print('neigh_resp: ' + str(neigh_resp))
            #print('dists: ' + str(dists))

        return int(retval)



    def processSortedSegments(self, sortedSegments):
        """
        Sorted Segments Post-processing
        After obtaining a string of characters out of the image and sorting them
        according to their x position, iterates over the proposed letters and numbers.
        
        If something is in the position of a letter and is a number, exchange it for
        the most probable letter, given that number, and if there is a letter in the
        place of a number, change it for the most probable number given that letter.

        @params
            sortedSegments: array of CharSegments (see definition on the top)

        @return
            plate: string (final answer)
        """

        letters = sortedSegments[:3]
        numbers = sortedSegments[3:]

        plate = ''

        for letter in letters:
            if self.debug: print('letter: ' + str(letter.letter))
            if letter.letter < 65:
                if letter.letter == 48:   # 0
                    letter.letter = 79           # O
                elif letter.letter == 49: # 1
                    letter.letter = 73    # I
                elif letter.letter == 52: # 4
                    letter.letter = 65    # A
                elif letter.letter == 55: # 7
                    letter.letter = 90    # Z
                elif letter.letter == 56: # 8
                    letter.letter = 66    # B

            plate += chr(letter.letter)


        for number in numbers:
            if self.debug: print('number: ' + str(number.letter))
            if number.letter > 57:
                if number.letter == 65:    # A
                    number.letter = 52
                elif number.letter == 66:  # B
                    number.letter = 56
                elif number.letter == 73:  # I      
                    number.letter = 49
                elif number.letter == 79:  # O
                    number.letter = 48
                elif number.letter == 90:  # Z
                    number.letter = 55

            plate += chr(number.letter)

        return plate

    def loadTraningData(self, data, classes):
        """
        Load Training Data
        Loads the data from the files and calls the training routine.

        @params:
            data: path to the dataset (string)
            classes: path to the classes file (string) 
        """
        dataFile = open(data, 'rb')

        # Legacy code - not removed for magical reasons (beware of gnomes)
        trainingDataf5 = pickle.load(dataFile)
        trainingDataf10 = pickle.load(dataFile)
        trainingDataf15 = pickle.load(dataFile)
        trainingDataf20 = pickle.load(dataFile)
        trainingDataBinary = pickle.load(dataFile)
        # end of legacy code

        classes = np.loadtxt(classes, dtype=np.float32)

        trainingDataf15 = np.float32(trainingDataf15)
        classes = np.float32(classes)

        if(self.debug):
            print('train data shape: ' + str(trainingDataBinary.shape))
            print('train data type: ' + str(trainingDataBinary.dtype))
            print('class data shape: ' + str(classes.shape))
            print('class data type: ' + str(classes.dtype))

        self.knnClassifier.train(np.float32(trainingDataBinary), classes, maxK=10)



    def run(self, plate):
        """
        @params
            plate: numpy array (image)

        @return
            result: array of charactes (in the order it was obtained)
            resultPos: array of tuples with the position of each character
            plateString: string with the recognized plate string
        """
        preprocessedPlate = self.preprocessPlate(plate)
        segments = self.segment(preprocessedPlate)
        result = []
        resultsPos = []

        for i in range(0, len(segments)):
            #charImg = self.prepocessChar(segments[i].img)
            charImg = segments[i].img
            if(self.saveSegments):
                stream = ''
                stream += './tmpChars/' + self.filename + '_' + str(i) + '.jpg'
                cv2.imwrite(stream, charImg)

            if(self.debug):
                cv2.imshow('char', charImg)
                cv2.waitKey(0)

            character = self.classify(charImg)
            result.append(chr(character))
            resultsPos.append(segments[i].pos)
            segments[i].letter = character

        sortedSegments = sorted(segments, key=lambda segment: segment.pos[0])

        plateString = self.processSortedSegments(sortedSegments)

        return (result, resultsPos, plateString)
