#begin image recognition and import of file for first test
# initial information and some learning sourced from here: https://datacarpentry.org/image-processing/08-edge-detection/


#below code is absed on this https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
#The initial intent is to perform edge recognition through sobel convolution filtering via Laplacian with opencv, cv2 package
# also this for cannyt edgehttps: // www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Canny_Edge_Detection.php

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from stl import mesh
#from PIL import Image

im_width = 1500 #appears as if min res of 600x450 required for 
im_height = 0.75*im_width #DO NOT CHANGE VALUE... pictures taken in 4x3 as rationale here

im01 = cv.imread('img/side_checker.jpg')

# decrease image res for processing speed, effective denoise
im01 = cv.resize(im01, (int(im_width), int(im_height)))


#image modification... --> comment out code if not desired in final image
im01 = cv.cvtColor(im01, cv.COLOR_BGR2GRAY) # convert to greyscale for analysis

#im01 = cv.GaussianBlur(im01, (3,3), 0) # Gaussian filter to smooth noise

#im01 = cv.Laplacian(im01, cv.CV_64F)
#im01 = cv.Canny(im01,100,200)

#im01 = cv.GaussianBlur(im01, (3,3), 0) #try to refilter to eliminate non-adjacent arm edge points???
#^^ May have to do this later once in stl mode to set a radius that is acceptable and if less than threshold points within radius, delete

################ test checkerboard

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((13*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:13, 0:9].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

ret, corners = cv.findChessboardCorners(im01, (13, 9), None)
corners2 = cv.cornerSubPix(im01, corners, (13, 9), (-1, -1), criteria)
imgpoints.append(corners)
#cv.drawChessboardCorners(im01, (13, 9), corners[:2, :, :], ret)



##testing length calibration of image, sensor calibration not done in this case
im01 = cv.circle(im01, (corners[0,0,0], corners[0, 0, 1]), radius=8, color=(0, 0, 255), thickness=-1)
im01 = cv.circle(im01, (corners[1, 0, 0], corners[1, 0, 1]), radius=8, color=(0, 0, 255), thickness=-1)

def calibrate(corners):
    pixels = np.sqrt((corners[1, 0, 0]-corners[0, 0, 0])
                     ** 2+(corners[1, 0, 1]-corners[0, 0, 1])**2)
    true_distance = 30 #30mm

    ppmm = pixels/true_distance

    return ppmm

ppmm = calibrate(corners)
################End checkerboard

######## // ---> Likely need some code to take in all images and poses to calibrate, webpage said needed >10 images

################ Begin orientation detection


#def draw(img, corners, imgpts): #pre-coded on opencv site... draws a 3d axis based on chessboard pose
    #corner = tuple(corners[0].ravel())
    #img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    #img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    #img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    #return img

#im01 = draw(im01, corners, imgpoints)


#now plot to show the image
plt.subplot(1,1,1),plt.imshow(im01,cmap='gray')

print(f'Calibrated distance is {ppmm} pixels per mm true distance')

plt.show()


#NOTES:
# check out this to do some resizing? May borrow code for now, ideally automate https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
# this seems good on image orientation and size calibration: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# for information on stl conversion and import/export: https://pypi.org/project/numpy-stl/
# --> or use the open3d package...? https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba 
