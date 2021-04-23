#Image processing derived from the experimental_learning file aloimport cv2 as cv

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

from stl import mesh

### Construct Point Cloud ###
def im2points(im_canny):
    edge_array = np.asarray(im_canny)  # convert canny img to matrix x,y
    # convert the edge array to only non zeroes (edges)
    edge_points = edge_array.nonzero()

    count_points = np.size(edge_points)

    return edge_points, count_points

##define matrix points
def rotate_x(theta, matrix_in):
    r1 = np.array([[1, 0, 0], [0, np.cos(theta), np.sin(theta)],
                [0, -1*np.sin(theta), np.cos(theta)]])

    mat_out = np.dot(r1, matrix_in)
    return mat_out

def rotate_y(theta, matrix_in):
    r1 = np.array([[np.cos(theta), np.sin(theta), 0], [-1*np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

    mat_out = np.dot(r1, matrix_in)
    return mat_out

def rotate_z(theta, matrix_in):
    r1 = np.array([[np.cos(theta), 0, -1*np.sin(theta)], [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)]])

    mat_out = np.dot(r1, matrix_in)
    return mat_out

def position_align(matrix_in, dx, dy, dz=0):

    sub_arry = np.array([[dx],[dy],[dz]])
    centered_matrix = np.subtract(matrix_in, sub_arry)

    return centered_matrix

# define image manipulation params
drawing = False #becomes true when mouse L pressed
mode = True # if true, will draw rectangle, press 'm' to toggle curve
first_point_flag = False
ix, iy = -1, -1

### Image parameters
im_width = int(600)
im_height = int(0.75*im_width)


### Image import
color_img = cv.imread('img/side_watch.jpg')

### Image processing
color_img = cv.resize(color_img, (int(im_width), int(im_height)))

bw_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
bw_img = cv.GaussianBlur(bw_img, (5, 5), 0)

im_laplace = cv.Laplacian(bw_img, cv.CV_64F)

im_canny = cv.Canny(bw_img, 150, 200)

edge_points, count_points = im2points(im_canny)

edge_points = np.array(edge_points, dtype=int)
edge_verticals = np.array(np.zeros((1, int(np.size(edge_points[0])))))

edge_points = np.vstack((edge_points, edge_verticals))
img_copy = copy.copy(color_img)
######################################################
def select_crop(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, mode, img_copy
    if event == cv.EVENT_FLAG_LBUTTON:
        drawing = True

        ix, iy = x, y

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

        if mode == True:
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 1)
            fx, fy = x, y

cv.namedWindow('Select_Crop_Bounds')
cv.setMouseCallback('Select_Crop_Bounds', select_crop)

while(1):
    cv.imshow('Select_Crop_Bounds', img_copy)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode

    elif k == 27:
        break

img_crop = im_canny[iy:fy, ix:fx]

def define_axis(event, x, y, flags, param):
    global aix, aiy, afx, afy, drawing, mode, first_point_flag
    if event == cv.EVENT_FLAG_LBUTTON and first_point_flag == False:
        drawing = True
        first_point_flag = True
        cv.circle(img_crop, (x, y), 6, (0, 255, 0), -1)
        aix, aiy = x, y

    elif event == cv.EVENT_FLAG_LBUTTON and first_point_flag == True:
        drawing = False
        if mode == True:
            cv.circle(img_crop, (x, y), 6, (0, 255, 0), -1)
            afx, afy = x, y

cv.namedWindow('Cropped_Select_Axis')
cv.imshow('Cropped_Select_Axis', img_crop)
cv.setMouseCallback('Cropped_Select_Axis', define_axis)
cv.waitKey(0)
cv.destroyAllWindows()


## Manipulate points based on user selection


edge_points, no_points = im2points(img_crop)
edge_points = np.array(edge_points, dtype=int)
edge_verticals = np.array(np.zeros((1, int(np.size(edge_points[0])))))

edge_points = np.vstack((edge_points, edge_verticals))

#align the beginning point to center of arm axis
edge_points = position_align(edge_points, aiy, aix)

offset_angle = np.arctan((afx-aix)/(afy-aiy))

#rotate the arm so that it lays along y axis
edge_points = rotate_y(offset_angle, edge_points)

stored_points = copy.copy(edge_points)


for point_iterate in range(np.size(edge_points[1])):
    edge_points[:, point_iterate] = rotate_x(
        np.pi/2, edge_points[:, point_iterate])

edge_points = np.hstack((stored_points, edge_points))


print(f'ix: {ix}, iy: {iy}')
print(f'fx: {fx}, fy: {fy}')
print(f'aix: {aix}, aiy: {aiy}')
print(f'afx: {afx}, afy: {afy}')

fig_cloud = plt.figure(figsize=plt.figaspect(1))
ax3 = fig_cloud.add_subplot(111, projection='3d')
ax3.scatter(edge_points[0], edge_points[1], edge_points[2])

plt.show()



#Run main thread of function when this program is used
if __name__ == '__main__':
    pass
