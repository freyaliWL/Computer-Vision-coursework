import numpy as np
import cv2
import os
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_points(pts3d):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# Load data
data = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = data['pts1']
pts2 = data['pts2']
M = data['M']

intrinsics = np.load('../data/intrinsics.npy', allow_pickle=True).tolist()
K1 = intrinsics['K1']
K2 = intrinsics['K2']

I1 = cv2.imread('../data/im1.png')
I2 = cv2.imread('../data/im2.png')

# Compute the fundamental matrix
F = eightpoint(pts1, pts2, M)

# Load points from templeCoords.npy

temple_coords = np.load('../data/templeCoords.npy', allow_pickle=True).tolist()
pts1_temple = temple_coords['pts1']
#num = pts1_temple.shape[0]

# Find epipolar correspondences
pts2_temple = np.zeros_like(pts1_temple)

# Compute corresponding points in image 2 using epipolarC
pts2_temple = epipolarCorrespondence(I1, I2, F, pts1_temple)

# Compute the essential matrix
E = essentialMatrix(F, K1, K2)

# Camera matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
candidates = camera2(E)

# Select the correct camera matrix and triangulate
final_P = None
for i in range(4):
    P2 = K2 @ candidates[:, :, i]
    P, error1, error2 = triangulate(P1, pts1_temple, P2, pts2_temple)
    #print(error1, error2)
    if np.all(P[:, 2] > 0):  # Check if all z-coordinates are positive
        final_P = P
        M2 = candidates[:, :, i]
        break

if final_P is not None:
    # Extract R1, t1, R2, t2
    R1, t1 = np.eye(3), np.zeros(3)
    R2, t2 = M2[:, :3], M2[:, 3]

    # Save extrinsics
    results_dir = '../results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.save(os.path.join(results_dir, 'extrinsics.npy'), {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})
    # np.save('../data/extrinsics.npy', R1=R1, t1=t1, R2=R2, t2=t2)

    # Plot 3D points
    plot_3d_points(final_P)

# final_P is a (N, 3) array of 3D points
# Example: final_P = final_P[::2, :]  # This will take every other point

    # fig = plt.figure(figsize=(18, 6))

    # # Front side view
    # ax1 = fig.add_subplot(131, projection='3d')
    # ax1.scatter(final_P[:, 0], final_P[:, 1], final_P[:, 2], marker='.', s=10)  # Adjust size with 's'
    # ax1.view_init(elev=20, azim=45)
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Z')

    # # Side view
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax2.scatter(final_P[:, 0], final_P[:, 1], final_P[:, 2], marker='.', s=10)
    # ax2.view_init(elev=20, azim=135)
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')

    # # Top-bottom view
    # ax3 = fig.add_subplot(133, projection='3d')
    # ax3.scatter(final_P[:, 0], final_P[:, 1], final_P[:, 2], marker='.', s=10)
    # ax3.view_init(elev=90, azim=45)
    # ax3.set_xlabel('X')
    # ax3.set_ylabel('Y')
    # ax3.set_zlabel('Z')

    # plt.tight_layout()
    # plt.show()
  
else:
    print("No valid camera configuration found.")

