import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    # Transpose x and X for convenience
    x1 = x.T
    X1 = X.T
    
    # Prepare arrays for the X and Y coordinates of the points
    X2 = x1[:, 0]
    Y2 = x1[:, 1]
    X3D = X1[:, 0]
    Y3D = X1[:, 1]
    Z3D = X1[:, 2]
    
    # Initialize the matrix that will be used for SVD
    num_points = X3D.shape[0]
    mat = np.zeros((2 * num_points, 12))
    
    # Construct the matrix using the correspondences
    for i in range(num_points):
        mat[2 * i] = [-X3D[i], -Y3D[i], -Z3D[i], -1, 0, 0, 0, 0, X3D[i] * X2[i], Y3D[i] * X2[i], Z3D[i] * X2[i], X2[i]]
        mat[2 * i + 1] = [0, 0, 0, 0, -X3D[i], -Y3D[i], -Z3D[i], -1, X3D[i] * Y2[i], Y3D[i] * Y2[i], Z3D[i] * Y2[i], Y2[i]]
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(mat)
    
    # Reshape the last column of V into a 3x4 matrix and transpose it
    P = Vt[-1].reshape((3, 4))
    
    return P

# Example use:
# x would be a 2xN array of 2D image points
# X would be a 3xN array of 3D world points

# x = np.array([[x1, y1], [x2, y2], ..., [xN, yN]]).T
# X = np.array([[X1, Y1, Z1], [X2, Y2, Z2], ..., [XN, YN, ZN]]).T

# P = estimate_pose(x, X)
