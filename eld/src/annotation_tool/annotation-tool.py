#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""The License-Plate Challenge Annotation Tool.

This program is an annotation tool to help on data collection tasks for The
License-Plate Challenge on "MC959 - Introduction to Computer Vision" course. It
allows defining, moving, editing, and deleting quadrilateral objects on a
selected image. The program also allows the assignment of a license plate string
to each quadrilateral.

Usage:
    The program expects an image path as argument:

        $ ./annotation-tool.py IMAGE_PATH

    It will load the indicated image and associated annotation file
    informations, if there is any.

    To create a new quadrilateral:
        - press 'n'
        - left click on the four vertices that define your quadrilateral
        - the defined quadrilateral should be displayed
        - type the license-plate string and press 'Enter'

    To move a quadrilateral:
        - just click inside the quadrilateral and move it, 'drag and drop'

    To move a vextex:
        - just click on the vertex and move it, 'drag and drop'

    To delete a quadrilateral:
        - double click inside the quadrilateral
        - if correctly selected it should be displayed in red
        - press 'd'

    To save the annotation file:
        - press 's'
        It will save a file at the same path and same name of the image file,
        with extension '.txt'.

    To exit:
        - press 'Esc'

Annotation file format:
    This is a file with same name of image file but extension txt, it contains
    one string per line in the format "x1,y1,x2,y2,x3,y3,x4,y4,ABC1234", where
    ABC1234 is the license plate, and x,y are the coordinates of each vertex
    in clockwise orientation starting on the top left coordinate (the user
    is suposed to create the quadrilateral in that order). Assuming that the
    coordinates of the image are on the superior left corner, and that grow to
    the right and bottow respectively. Images without license plate will
    contain a string "None".

NOTES:
    The size of displayed objects (lines, vextex, and text) is optimized to
    work with image sizes around 1600 x 1200 pixels. If your image size is far
    different from that you may want to edit the variables VERTEX_SIZE,
    LINE_THICKNESS and TEXT_SIZE.

    This program was developed for python 2.7 and openvc 2.4.1


Author: Gabriel Sobral <gasan.sobral@gmail.com>

"""

import numpy as np
import cv2
import sys

VERTEX_SIZE = 20
LINE_THICKNESS = 2
TEXT_SIZE = 2

window = 'MC959 Annotation Tool'
quadrilaterals = []
selected_vert = None
moving_quad = None
selected_quad = None
registering_quad = 0
new_quad = {'x': [], 'y': []}
help_str = ''


class Quadrilateral:
    def __init__(self, x, y, plate):
        self.x = x
        self.y = y
        self.plate = plate

    def contains(self, x, y):
        if x > min(self.x) and x < max(self.x) and \
           y > min(self.y) and y < max(self.y):
            return True
        return False

    def has_vertex(self, x, y):
        for i in range(0, 4):
            if x >= (self.x[i] - VERTEX_SIZE/2) and \
               x <= (self.x[i] + VERTEX_SIZE/2) and \
               y >= (self.y[i] - VERTEX_SIZE/2) and \
               y <= (self.y[i] + VERTEX_SIZE/2):
                return i
        return -1

    def get_vertex(self, index):
        return (self.x[index], self.y[index])

    def set_vertex(self, index, x, y):
        self.x[index] = x
        self.y[index] = y

    def move(self, dx, dy):
        for i in range(0, 4):
            self.x[i] += dx
            self.y[i] += dy

    def draw(self, img, selected):
        if selected:
            line_color = (0, 0, 255)
            vertex_color = (0, 0, 255)
            plate_color = (0, 0, 255)
        else:
            line_color = (0, 255, 0)
            vertex_color = (255, 0, 255)
            plate_color = (255, 0, 255)

        # draw lines
        for i in range(-1, 3):
            cv2.line(img, (self.x[i], self.y[i]),
                     (self.x[i + 1], self.y[i + 1]),
                     line_color, LINE_THICKNESS)
        # draw vertex
        for i in range(0, 4):
            cv2.rectangle(img,
                          (self.x[i] - VERTEX_SIZE/2, self.y[i] - VERTEX_SIZE/2),
                          (self.x[i] + VERTEX_SIZE/2, self.y[i] + VERTEX_SIZE/2),
                          vertex_color,
                          -1)
        # draw license plate string
        cv2.putText(img, self.plate, (self.x[0], self.y[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, plate_color, 2*TEXT_SIZE)

    def get_string(self):
        quad_str = ''
        for i in range(0, 4):
            quad_str += str(self.x[i])
            quad_str += ','
            quad_str += str(self.y[i])
            quad_str += ','
        quad_str += self.plate
        return quad_str

    def set_from_string(self, str):
        args = str.split(',')
        if len(args) != 9:
            print 'number of arguments in string is incompatible:', args
            return
        for i in range(0, 4):
            self.x.append(int(args.pop(0)))
            self.y.append(int(args.pop(0)))
        self.plate = args.pop(0)

    def set_plate(self, plate):
        self.plate = plate


def on_mouse_event(event, x, y, flags, param):
    global selected_vert
    global moving_quad
    global selected_quad
    global registering_quad
    global new_quad

    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print 'event: double click:', x, y
        selected_quad = None
        for quad in quadrilaterals:
            if quad.contains(x, y):
                selected_quad = quad
                print 'select quadrilateral:', selected_quad.get_string()

    elif event == cv2.EVENT_LBUTTONDOWN:
        # print 'event: left button down:', x, y
        if registering_quad:
            new_quad['x'].append(x)
            new_quad['y'].append(y)
            registering_quad -= 1
            if registering_quad == 0:
                new_quadrilateral(new_quad['x'][:], new_quad['y'][:])
                del new_quad['x'][:]
                del new_quad['y'][:]
        else:
            for quad in quadrilaterals:
                i = quad.has_vertex(x, y)
                if i >= 0:
                    selected_vert = {'quad': quad, 'vert': i}
                    print 'selected vertex:', i, quad.get_vertex(i)
                    break
                elif quad.contains(x, y):
                    moving_quad = {'quad': quad, 'x0': x, 'y0': y}
                    print 'moving quadrilateral:', quad.get_string()
                    break
    elif event == cv2.EVENT_LBUTTONUP:
        # print 'event: left button up:', x, y
        if selected_vert:
            quad = selected_vert['quad']
            quad.set_vertex(selected_vert['vert'], x, y)
        elif moving_quad:
            quad = moving_quad['quad']
            quad.move(x - moving_quad['x0'], y - moving_quad['y0'])
        selected_vert = None
        moving_quad = None
        update_image()


def new_quadrilateral(x, y):
    global quadrilaterals
    global help_str
    print 'new quadrilateral created:', x, y
    plate_str = ''
    q = Quadrilateral(x, y, plate_str)
    quadrilaterals.append(q)

    help_str = 'type license plate and press \'Enter\''
    while True:
        key = cv2.waitKey(10)
        if key >= 0:
            key = key % 256
            if key != 10:  # Enter key
                plate_str += chr(key)
            else:
                break
    quadrilaterals[-1].set_plate(plate_str)
    print 'license plate registered:', plate_str
    help_str = ''
    update_image()


def update_image():
    img = original_img.copy()
    for quad in quadrilaterals:
        if quad == selected_quad:
            quad.draw(img, True)
        else:
            quad.draw(img, False)
    cv2.putText(img, help_str, (10, height-10), cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SIZE, (255, 0, 255), 2*TEXT_SIZE)
    cv2.imshow(window, img)


def white_file():
    f = open(txt_path, 'w')
    print 'writing annotation file:', txt_path
    if len(quadrilaterals):
        for quad in quadrilaterals:
            f.write(quad.get_string())
            f.write('\n')
    else:
        f.write('None\n')
    f.close()


def read_file():
    global quadrilaterals
    try:
        f = open(txt_path, 'r')
        print 'reading annotation file:', txt_path
        for line in f:
            line = line.rstrip('\n')
            print '>', line
            if line != 'None':
                q = Quadrilateral([], [], '')
                q.set_from_string(line)
                quadrilaterals.append(q)
        f.close()
    except IOError:
        print 'annotation file not found'

#
# main program
#

try:
    image_path = sys.argv[1]
except IndexError:
    print 'Usage: problem_set_01 IMAGE_PATH'
    exit()

try:
    original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width, depth = original_img.shape
    print 'reading image:', image_path
except:
    print 'image path isn\'t valid:', image_path
    exit()

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window, on_mouse_event)

txt_path = image_path.rsplit('.', 1)[0] + '.txt'
read_file()

while True:
    update_image()
    key = cv2.waitKey(10)
    if key >= 0:
        key = key % 256
        if key == ord('n'):
            print 'pressed \'n\': create new quadrilateral'
            help_str = 'click the four vertices'
            registering_quad = 4
        elif key == ord('s'):
            print 'pressed \'s\': saving data to file'
            white_file()
        elif key == ord('d'):
            if selected_quad:
                print 'pressed \'d\': delete quadrilateral',
                selected_quad.get_string()
                quadrilaterals.remove(selected_quad)
                selected_quad = None
        # Esc key: exit
        elif key == 27:
            print 'pressed \'Esc\': exiting'
            break

cv2.destroyAllWindows()
