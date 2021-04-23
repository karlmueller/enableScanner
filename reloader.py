import numpy as np
import matplotlib.pyplot as plt


try_mesh = True


points_3d = np.genfromtxt(
    'toscale_3d_point_cloud.csv', delimiter=',', encoding=None)


fig_cloud = plt.figure(figsize=plt.figaspect(1))
ax3 = fig_cloud.add_subplot(111, projection='3d')
ax3.scatter(points_3d[0], points_3d[1], points_3d[2])

ax3.set_xlim3d(-300, 300)
ax3.set_ylim3d(-300, 300)
ax3.set_zlim3d(-300, 300)

ax3.set_xlabel('X Axis - pixels')
ax3.set_ylabel('Z Axis - pixels')
ax3.set_zlabel('Y Axis - pixels')

plt.show()


'''
#mesh gen OOS for the capstone but found a version for testing, attribute this person on stack overflow
#https: // stackoverflow.com/questions/60066405/create-a-stl-file-from-a-collection-of-points
if try_mesh==True:
    #these are to be imported into mesh_lab
    edge_surf_points = points_3d[:3, :]
    np.savetxt('edge_mesh_pts.csv', edge_surf_points, delimiter=',')
    np.savetxt('edge_mesh_pts_transp.csv', edge_surf_points.transpose(), delimiter=',')
'''