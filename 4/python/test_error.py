import numpy as np
import cv2
import os
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
import matplotlib.pyplot as plt

best_error1 = float('inf')
best_error2 = float('inf')
best_P2 = None
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

# Find epipolar correspondences

# Compute corresponding points in image 2 using epipolarC
pts2 = epipolarCorrespondence(I1, I2, F, pts1)

# Compute the essential matrix
E = essentialMatrix(F, K1, K2)

# Camera matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
candidates = camera2(E)

# Select the correct camera matrix and triangulate
final_P = None
for i in range(4):
    P2_candidate = K2 @ candidates[:, :, i]
    P, error1, error2 = triangulate(P1, pts1, P2_candidate, pts2)
    
    # Check if the current errors are the best ones
    if error1 < best_error1 and error2 < best_error2:
        best_error1 = error1
        best_error2 = error2
        best_P2 = P2_candidate

# After the loop, print the best errors
print(f"Best error for points in image 1: {best_error1}")
print(f"Best error for points in image 2: {best_error2}")