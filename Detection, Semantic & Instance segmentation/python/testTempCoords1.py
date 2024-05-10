# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import scipy.io as sio
# from PIL import Image
# from eightpoint import eightpoint
# from epipolarCorrespondence import epipolarCorrespondence
# from essentialMatrix import essentialMatrix
# from camera2 import camera2
# from triangulate import triangulate
# from displayEpipolarF import displayEpipolarF
# from epipolarMatchGUI import epipolarMatchGUI

# # Load images and points
# img1 = cv2.imread('../data/im1.png')
# img2 = cv2.imread('../data/im2.png')
# pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
# pts1 = pts['pts1']
# pts2 = pts['pts2']
# M = pts['M']

# # write your code here
# R1, t1 = np.eye(3), np.zeros((3, 1))
# R2, t2 = np.eye(3), np.zeros((3, 1))

# # save extrinsic parameters for dense reconstruction
# np.save('../results/extrinsics', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF

# Implement your other functions as needed
def project_points(pts3d, P, K):
    # Homogeneous coordinates for 3D points
    #print(pts3d)
    #### homogenous_pts = np.column_stack((pts3d, np.ones(pts3d.shape[0])))
    homogenous_pts = np.column_stack((pts3d[:, :3], np.ones(pts3d.shape[0])))
    #print(homogenous_pts.shape)
    # Project 3D points to 2D using camera projection matrix
    #pts2d_projected = np.dot(P, homogenous_pts.T).T
    pts2d_projected = np.dot(homogenous_pts, P.T)
    #print(pts2d_projected.shape)
    # Normalize by the third coordinate
    pts2d_projected /= pts2d_projected[:, 2, None]
    #print(pts2d_projected.shape)
    #print(pts2d_projected[:, :2].shape)
    # Apply intrinsic matrix to obtain final 2D points
    homogenous_pts_final = np.dot(pts2d_projected, K.T)
    #print(homogenous_pts_final.shape)
    pts2d_final = homogenous_pts_final[:, :2] / homogenous_pts_final[:, 2, None]
    #print(pts2d_final.shape)
    return pts2d_final

def compute_reprojection_error(pts_true, pts_reprojected):
    # Compute Euclidean distance between true and reprojected points
    error = np.linalg.norm(pts_true - pts_reprojected, axis=1)
    # Compute mean error
    mean_error = np.mean(error)
    return mean_error

def plot_3d_points(pts3d):


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Load images and points
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = pts['pts2']

M = pts['M']

# Compute the fundamental matrix
F = eightpoint(pts1, pts2, M)

temple_coords = np.load('../data/templeCoords.npy', allow_pickle=True).tolist()
pts1_temple = temple_coords['pts1']  # what should be the pts_temple

# Compute corresponding points in image 2 using epipolarC
pts2_temple = epipolarCorrespondence(img1, img2, F, pts1_temple)

# Load intrinsic matrices
intrinsics = np.load('../data/intrinsics.npy', allow_pickle=True).item()
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# Compute the essential matrix
E = essentialMatrix(F, K1, K2)

# Obtain possible extrinsic matrices for P2 using essential matrix
extrinsics = camera2(E)

# Initialize variables
best_pts3d = None
best_error = float('inf')

# Iterate over possible extrinsic matrices
for P2_extrinsic in extrinsics:
    # Triangulate 3D points using current extrinsic matrix
    # Use pts1_temple and pts2_temple for triangulation
    pts3d_current = triangulate(np.eye(3, 4), pts1_temple, P2_extrinsic, pts2_temple)

    # Project 3D points back to image 1 and compute re-projection error
    # Use pts1_temple for the reprojection and error calculation
    pts1_reprojected = project_points(pts3d_current, np.eye(3, 4), K1)
    error = compute_reprojection_error(pts1_temple, pts1_reprojected)

    # Update if the current extrinsic matrix gives a lower error
    if error < best_error:
        best_pts3d = pts3d_current
        best_error = error

# get the error
print(best_error)

# Save the best extrinsic parameters
R1, t1 = np.eye(3), np.zeros((3, 1))
R2, t2 = extrinsics[:, :, 0][:, :3], extrinsics[:, :, 0][:, 3].reshape(3, 1)

# save files
results_dir = '../results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save extrinsics
np.save(os.path.join(results_dir, 'extrinsics.npy'), {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})

# Plot and save 3D points
plot_3d_points(best_pts3d)
np.save(os.path.join(results_dir, '3d_points.npy'), best_pts3d)
