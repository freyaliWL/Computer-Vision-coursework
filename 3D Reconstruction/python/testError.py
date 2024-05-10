import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate

def project_points(pts3d, P, K):
    # Homogeneous coordinates for 3D points
    homogenous_pts = np.column_stack((pts3d, np.ones(pts3d.shape[0])))
    # Project 3D points to 2D using camera projection matrix
    pts2d_projected = np.dot(homogenous_pts, P.T)
    # Normalize by the third coordinate
    pts2d_projected /= pts2d_projected[:, 2, None]
    # Apply intrinsic matrix to obtain final 2D points
    homogenous_pts_final = np.dot(pts2d_projected, K.T)
    pts2d_final = homogenous_pts_final[:, :2] / homogenous_pts_final[:, 2, None]
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

# Load points from templeCoords.npy
temple_coords = np.load('../data/templeCoords.npy', allow_pickle=True).tolist()
pts1_temple = temple_coords['pts1']

# Compute corresponding points in image 2 using epipolarCorrespondence
pts2_temple = epipolarCorrespondence(img1, img2, F, pts1_temple)

# Load intrinsic matrices
intrinsics = np.load('../data/intrinsics.npy', allow_pickle=True).item()
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# Compute the essential matrix
E = essentialMatrix(F, K1, K2)

# Obtain possible extrinsic matrices for P2 using essential matrix
extrinsics = camera2(E)

# Initialize variables to keep track of the best configuration
best_pts3d = None
best_error = float('inf')
best_P2 = None

# Iterate over possible extrinsic matrices
for P2_extrinsic in extrinsics:
        # Extract the rotation matrix and translation vector from P2_extrinsic
    R = P2_extrinsic[:3, :3]
    t = P2_extrinsic[:3, 3]

    # Form the camera matrix for the second camera by combining R, t, and K2
    P2 = np.dot(K2, np.hstack((R, t.reshape(-1, 1))))
    # Form the camera matrix for the second camera
    # P2 = np.dot(K2, P2_extrinsic)

    # Triangulate points using the current P2 candidate
    pts3d_current = triangulate(np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1))))), pts1_temple, P2, pts2_temple)

    # Check the number of points in front of both cameras
    num_positive_depths = np.sum(pts3d_current[:, 2] > 0)

    # Project 3D points back to the first camera and compute reprojection error
    pts1_reprojected = project_points(pts3d_current[:, :3], np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1))))), K1)
    error = compute_reprojection_error(pts1_temple, pts1_reprojected)

    # Update best configuration if current one has more points in front of both cameras and lower error
    if num_positive_depths > len(pts3d_current) / 2 and error < best_error:
        best_error = error
        best_P2 = P2_extrinsic

R = best_P2[:3, :3]  # Extract the rotation matrix
t = best_P2[:3, 3]   # Extract the translation vector

# Combine R and t to form a 3x4 extrinsic matrix
best_P2_full = np.hstack((R, t.reshape(-1, 1)))

# Use the best extrinsic matrix to triangulate points again
best_P2_full = np.dot(K2, best_P2_full)
best_pts3d = triangulate(np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1))))), pts1_temple, best_P2_full, pts2_temple)

if best_pts3d.shape[1] == 4:
    best_pts3d = best_pts3d[:, :3]

# Project the 3D points back to the first camera to calculate reprojection error
P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))  # Camera matrix for the first camera
pts1_reprojected = project_points(best_pts3d, P1, K1)
final_error = compute_reprojection_error(pts1_temple, pts1_reprojected)
print(f"Final reprojection error: {final_error}")

# # Save the best extrinsic parameters
# R1, t1 = np.eye(3), np.zeros((3, 1))
# R2, t2 = best_P2[:3, :3], best_P2[:3, 3]

# # save files
# results_dir = '../results'
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

# # Save extrinsics
# np.save(os.path.join(results_dir, 'extrinsics.npy'), {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})

# # Plot and save 3D points
# plot_3d_points(best_pts3d)
# np.save(os.path.join(results_dir, '3d_points.npy'), best_pts3d)
