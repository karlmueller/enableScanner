# Main Script, Capstone MECH9000
# Karl W. Mueller

#_________________________________ooo__________________________________#

#General package import
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import imutils

#Custom Script(s)
from aruco_detection import readArucoMarkers

#_________________________________ooo__________________________________#

#Create initial parameters
# --> Image Processing Parameters:
im_width = int(1500)
im_height = int(0.75*im_width)

crop_width_buffer = 0.01  # addutional buffer for the crop past the aruco points, to scale with the input image size, proportion of image length
crop_height_buffer = 0.025

# --> Image Location Informtion:
image_directory = 'dev3'


beginning_angle = -90
end_angle = 90
angle_increment = 20


# --> Development Parameters:
canny_low_thresh = 100 # for thresholding the canny edge deteciton gradient parameters
canny_hi_thresh = 130

crop_buffer_x = int(crop_width_buffer * im_width)   #addutional buffer for the crop past the aruco points, to scale with the input image size, percentage of 
crop_buffer_y = int(crop_height_buffer * im_width)

test_bw_thresh_lo = 0 #for thresholding white values to set non-background to 1
test_bw_thresh_hi = 0

gaussian_size = 7


# --> Pre-allocation of variables:
image_it = []
points_3d = np.array([[],[],[], []])
angle_vector = np.arange(beginning_angle, end_angle+angle_increment, angle_increment)

pixel_correction = []

#_________________________________ooo__________________________________#

#Pre-defintiion of commonly-used functions

def rotate_x(theta, matrix_in):
    theta = theta/180*np.pi
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

#ensures that aruco points are positioned the same way each time, 1.2.3.4 clockwise
def correct_orientation(image, arucoParams):
    if arucoParams[5, 0] > arucoParams[5, 4]:
        image = cv.rotate(image, cv.cv2.ROTATE_180)
        arucoParams= readArucoMarkers(image)
    else:
        pass

    return image, arucoParams


def im2points(image_canny):
    edge_array = np.asarray(image_canny)
    edge_points = edge_array.nonzero()
    count_points = np.size(edge_points)

    return edge_points, count_points


def rect_align(matrix_in, dx, dy, dz=0):
    sub_array = np.array([[dx],[dy],[dz]])
    centered_matrix = np.subtract(matrix_in, sub_array)

    return centered_matrix


def debug_images(image_in, arucoParams):
    #for plotting markers on copy of the image, maybe use the internal function in aruco_detection instead
    pass
#_________________________________ooo__________________________________#

#Main loop for image processing and point generation

for image_name in os.listdir(image_directory):
    if image_name.endswith('.jpg'): #exclude non-image files
        
        #image import and pre-definition
        image = cv.imread(f'{image_directory}/{image_name}')                # read in current image
        image = imutils.resize(image, width=int(im_width))

        image_it.append(cv.imread(image_name))      #collect image names
        c_index = np.size(image_it)-1               #defines index of current image

        c_angle = angle_vector[c_index]*np.pi/180             #angle at which current image was taken    

        #image data capture, aruco information
        markerParams = readArucoMarkers(image)

        image, markerParams = correct_orientation(image, markerParams)

        #print(markerParams)

        #gather croppinng UL(1) and LR (3) points, crop with Y, X order
        crop1 = np.array([markerParams[3, 0], markerParams[3, 1]],dtype=int)
        crop3 = np.array([markerParams[1, 4], markerParams[1, 5]], dtype=int)

        #get information on mm/px distance
        mmpx = 15 / np.linalg.norm(markerParams[3, :2] - markerParams[4, :2]) #take distance between lower right and lower left corners of aruco 1 for distance cal... 15 mm true length
        pixels_standard = np.linalg.norm(markerParams[3, :2] - markerParams[4, :2])
        pixel_correction.append(mmpx)
        print(f'image{c_index} --> {mmpx} mm per pixel')
        print(f'standard _pixel length: {pixels_standard}')

        #BEFORE CROP, define a centerline relative to the BARREL, not the arm

        #initial x, y, proximal - for aligment
        vec_1_4 = np.array(markerParams[5, 6:]) - np.array(markerParams[5, :2])
        norm14 = int(np.linalg.norm(vec_1_4))
        to_ctr_14 = 0.5*vec_1_4
        ctr_vector_14 = np.array(markerParams[5, : 2]) + to_ctr_14
        ix = ctr_vector_14[0]
        iy = ctr_vector_14[1]
        
        #end x, y distal - for alignment
        vec_2_3 = np.array(markerParams[5, 4:6]) - np.array(markerParams[5, 2:4])
        norm23 = int(np.linalg.norm(vec_2_3))
        to_ctr_23 = 0.5*vec_2_3
        ctr_vector_23 = np.array(markerParams[5, 2:4]) + to_ctr_23
        fx = ctr_vector_23[0]
        fy = ctr_vector_23[1]



        

        #CROP
        image_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
        bw_img = cv.GaussianBlur(image_bw, (gaussian_size, gaussian_size), 0)  # may need to tune blur
        image_canny = cv.Canny(image_bw, canny_low_thresh, canny_hi_thresh)

        # note this is BW
        image_canny_crop = image_canny[crop1[1] +
                                       crop_buffer_y:crop3[1]-crop_buffer_y, crop1[0]+crop_buffer_x:crop3[0]-crop_buffer_x]

        #correct ix, iy, fx, fy for the cropped image, is this necessary? causes skew about Z and Y axis
        #ix = ix - crop1[0]
        #iy = iy - crop1[1]

        # correct the position of edge
        offset_angle = np.arctan2((fy-iy), (fx-ix))

        print(f"off_angle: {offset_angle} \n ix: {ix} \n iy: {iy}")

        #cv.imshow("pre-crop", image)
        #cv.imshow("canny, precrop", image_canny)
        #cv.imshow("pre-canny", image_bw[crop1[1] + crop_buffer_y:crop3[1]-crop_buffer_y, crop1[0]:crop3[0]])

        #cv.imshow("Cropped", image_canny_crop)
        #cv.waitKey(0)

        #generate edge points from the cropped and thresholded image
        edge_points, count_points = im2points(image_canny_crop)
        edge_points = np.array(edge_points, dtype=int)
        edge_verticals = np.array(np.zeros((1, int(np.size(edge_points[0])))))
        num_tags = c_index * np.array(np.ones((1, int(np.size(edge_points[0])))))

        edge_points = np.vstack((edge_points, edge_verticals))
        

        #align beginning of edge points to the origin

        edge_points = rect_align(edge_points[:3, :], ix/2, iy/2)
        edge_points = rotate_y(offset_angle, edge_points )

        if c_angle != 0:  # rotate about central axis
            for point_iterate in range(np.size(edge_points[1])):
                edge_points[0:3, point_iterate] = rotate_z(
                    c_angle, edge_points[:, point_iterate])
        else:
            pass

        #scale the edge points to true scale with the mm per pixel calculated
        edge_points = mmpx*edge_points

        #plotting the 3d point surface/cloud like object
        edge_points = np.vstack((edge_points, num_tags))
        points_3d = np.hstack((points_3d, edge_points)) #append 3d points, these are persistent b/w images



        #plt.show()

#np.savetxt('edge_point_cloud.csv', points_3d, delimiter=',')
fig_cloud = plt.figure(figsize=plt.figaspect(1))
ax3 = fig_cloud.add_subplot(111, projection='3d')
ax3.scatter(points_3d[0], points_3d[1], points_3d[2])


ax3.set_xlim3d(-300*mmpx, 300*mmpx)
ax3.set_ylim3d(-300*mmpx, 1.5*300*mmpx)
ax3.set_zlim3d(-300*mmpx, 300*mmpx)


ax3.set_xlabel('X Axis - mm')
ax3.set_ylabel('Z Axis - mm')
ax3.set_zlabel('Y Axis - mm')

#tool to 
for 


forearm_set = 

print(f'Total length of scan: ... {max(points_3d[1, :])-min(points_3d[1, :])} ')
print(
    f'Largest width in X direction: ... {max(points_3d[0, :])-min(points_3d[0, :])}')
print(
    f'Largest width in Y direction: ... {max(points_3d[2, :])-min(points_3d[2, :])}')



plt.show()
np.savetxt('toscale_3d_point_cloud.csv', points_3d, delimiter=',')
