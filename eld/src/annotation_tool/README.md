#The License-Plate Challenge Annotation Tool

This program is an annotation tool to help on data collection tasks for The
License-Plate Challenge on "MC959 - Introduction to Computer Vision" course. It
allows defining, moving, editing, and deleting quadrilateral objects on a
selected image. The program also allows the assignment of a license plate string
to each quadrilateral.

##Usage:
The program expects an image path as argument

    $ ./annotation-tool.py IMAGE_PATH

It will load the indicated image and associated annotation file
informations, if there is any.

###To create a new quadrilateral
- press 'n'
- left click on the four vertices that define your quadrilateral
- the defined quadrilateral should be displayed
- type the license-plate string and press 'Enter'

###To move a quadrilateral
- just click inside the quadrilateral and move it, 'drag and drop'

###To move a vextex
- just click on the vertex and move it, 'drag and drop'

###To delete a quadrilateral
- double click inside the quadrilateral
- if correctly selected it should be displayed in red
- press 'd'

###To save the annotation file
- press 's'
It will save a file at the same path and same name of the image file,
with extension '.txt'.

###To exit
- press 'Esc'

##Annotation file format
This is a file with same name of image file but extension txt, it contains
one string per line in the format "x1,y1,x2,y2,x3,y3,x4,y4,ABC1234", where
ABC1234 is the license plate, and x,y are the coordinates of each vertex
in clockwise orientation starting on the top left coordinate (the user
is suposed to create the quadrilateral in that order). Assuming that the
coordinates of the image are on the superior left corner, and that grow to
the right and bottow respectively. Images without license plate will
contain a string "None".

##NOTES
The size of displayed objects (lines, vextex, and text) is optimized to
work with image sizes around 1600 x 1200 pixels. If your image size is far
different from that you may want to edit the variables VERTEX_SIZE,
LINE_THICKNESS and TEXT_SIZE.

This program was developed for python 2.7 and openvc 2.4.1

##Author
Gabriel Sobral <gasan.sobral@gmail.com>
