#Image processing derived from the experimental_learning file aloimport cv2 as cv

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D
from numpy.testing._private.utils import print_assert_equal



class  img_section(object):
    def __init__(self, image_in):

        #pre-definition of image_interaction params
        self.drawing = False
        self.mode = True
        self.cropped = False
        self.first_point_flag = False
        self.ix, self.iy = -1, -1

        ### Image parameters
        self.im_width = int(1500)
        self.im_height = int(0.75*self.im_width) #assuming 4x3 aspect ratio

        self.image_import(image_in)


    def im2points(self, im_canny):
        edge_array = np.asarray(im_canny)  # convert canny img to matrix x,y
        # convert the edge array to only non zeroes (edges)
        edge_points = edge_array.nonzero()

        count_points = np.size(edge_points)

        return edge_points, count_points

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

        sub_arry = np.array([[dx], [dy], [dz]])
        centered_matrix = np.subtract(matrix_in, sub_arry)

        return centered_matrix

    def image_import(self, image_input):
        self.color_img = cv.imread(image_input)
        '''cv.namedWindow('testWindow')
        cv.imshow('testWindow', self.color_img)'''
        self.color_img = cv.resize(self.color_img, (int(self.im_width), int(self.im_height)))

        self.bw_img = cv.cvtColor(self.color_img, cv.COLOR_BGR2GRAY)
        self.bw_img = cv.GaussianBlur(self.bw_img, (3, 3), 0)
        self.im_canny = cv.Canny(self.bw_img, 1, 150)

        self.img_copy = copy.copy(self.color_img)

    def final_crop(self):
        pass

    def select_crop(self, event, x, y, flags, param):
        if event == cv.EVENT_FLAG_LBUTTON and self.cropped == False:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv.EVENT_LBUTTONUP and self.cropped == False:
            self.drawing = False
            self.cropped = True
            if self.mode == True:
                cv.rectangle(self.img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 1)
                self.fx, self.fy = x, y
        
            self.img_crop = self.im_canny[self.iy:self.fy, self.ix:self.fx]

    def define_axis(self, event, x, y, flags, param):
        if event == cv.EVENT_FLAG_LBUTTON and self.first_point_flag == False:
            self.drawing = True
            self.first_point_flag = True
            cv.circle(self.img_crop, (x, y), 6, (0, 255, 0), -1)
            self.aix, self.aiy = x, y
            print(f'First point chosen at ({self.aix},{self.aiy})')

        elif event == cv.EVENT_FLAG_LBUTTON and self.first_point_flag == True:
            self.drawing = False
            if self.mode == True:
                cv.circle(self.img_crop, (x, y), 6, (0, 255, 0), -1)
                self.afx, self.afy = x, y
                print(f'Second point chosen at ({self.afx},{self.afy})')




#Run main thread of function when this program is used
if __name__ == '__main__':
    pass
