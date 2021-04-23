import cv2 as cv
import imutils
import argparse
import sys
import numpy as np
import copy

#from a_img_section import img_section


#python-specific information and commands from here: https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
#https://www.pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/
#General opencv contributor information here: https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html

#cv.aruco
#draw with cv.aruco.drawMarker

#generated markers with https://chev.me/arucogen/
#using information DICT:  4x4_50, 15mm square
'''
def markerGen():

    arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

    print(arucoDict)


    a_parse = argparse.ArgumentParser()

    a_parse.add_argument('-o','--output', required=True,
        help='path to output image with aruco tags')
    a_parse.add_argument('-i','--id', type=int, required=True,
        help='id of aruco tag requested to gen')
    a_parse.add_argument('-t', '--type', required=True,
        default='DICT_ARUCO_ORIGINAL',
        help='type of aruco tag to generate')

    args = vars(a_parse.parse_args())
'''


def readArucoMarkers(image_in):

    image = image_in
    

    arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
    arucoParams = cv.aruco.DetectorParameters_create()

    #could probably use as a function to gather corner data and overwrite images to show a visual
    corners, ids, rejected = cv.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    markerParams = np.zeros((6, 8), dtype=float)

    im_copy = copy.copy(image)

    if len(corners)>0: #check if any corners detected
        ids = ids.flatten() #flatten ids list, not sure why necessart but do it

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            cv.line(im_copy, topLeft, topRight, (0, 255, 0), 2)
            cv.line(im_copy, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(im_copy, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(im_copy, bottomLeft, topLeft, (0, 255, 0), 2)
        
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv.circle(im_copy, (cX,cY), 4, (0, 0, 255), -1)

            
            cv.putText(im_copy, str(markerID),
                (topLeft[0], topLeft[1] -15), cv.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            
            #print(f'[INFO] aruco marker ID: {markerID}')

            #store aruco marker params
            markerParams[0, 2*markerID-2] = markerID
            markerParams[1, 2*markerID-2] = topLeft[0]
            markerParams[1, 2*markerID-1] = topLeft[1]
            markerParams[2, 2*markerID-2] = topRight[0]
            markerParams[2, 2*markerID-1] = topRight[1]
            markerParams[3, 2*markerID-2] = bottomRight[0]
            markerParams[3, 2*markerID-1] = bottomRight[1]
            markerParams[4, 2*markerID-2] = bottomLeft[0]
            markerParams[4, 2*markerID-1] = bottomLeft[1]
            markerParams[5, 2*markerID-2] = cX
            markerParams[5, 2*markerID-1] = cY

            #print(f'Markers @ {cX, cY}')

        cv.imshow("MarkersPlotted", im_copy)
        #cv.waitKey(0)

    #print(markerParams)

    #calculate actual distance in the aruco markers
    #for ii in range()


    return markerParams#, ppmm

#these aruco markers can be used to take the distances and calibrate images instead of the chessboard,
#will work far better

if __name__ == '__main__':

    image_in = 'dev1//1.jpg'
    image = cv.imread(image_in)
    image = imutils.resize(image, width=1200)
    
    
    readArucoMarkers(image)
    pass
