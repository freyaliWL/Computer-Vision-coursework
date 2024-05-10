import numpy as np
from scipy.linalg import svd, qr
import numpy as np

def estimate_params(P):
    # Compute the SVD of the camera matrix P
    _, _, V = np.linalg.svd(P)
    v = V[:, -1]  # Last column of V
    c = v[:3] / v[3]  # Camera center in homogeneous coordinates

    # Extract the rotation and translation from P
    M = P[:3, :3]  # Rotation-translation matrix
    R, K = np.linalg.qr(np.linalg.inv(M))  # QR decomposition of the inverse of M
    K = np.linalg.inv(K)  # Intrinsic parameter matrix
    K /= K[2, 2]  # Normalize K

    # Ensure the rotation matrix has positive determinant
    if np.linalg.det(R) < 0:
        R = -R
        K = -K

    t = -np.dot(R, c)  # Translation vector

    # Return the intrinsic matrix, rotation matrix, and translation vector
    return K, R, t

# Example usage:
# P is the camera matrix estimated previously
# K, R, t = estimate_params(P)
