# import numpy as np

# def rectify_pair(K1, K2, R1, R2, t1, t2):
#     """
#     takes left and right camera paramters (K, R, T) and returns left
#     and right rectification matrices (M1, M2) and updated camera parameters. You
#     can test your function using the provided script testRectify.py
#     """
#     # YOUR CODE HERE

#     M1, M2, K1n, K2n, R1n, R2n, t1n, t2n = None, None, None, None, None, None, None, None

#     return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n

import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    # Compute the optical centers of the cameras
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)

    # Compute the new rotation matrix (R)
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r2 = np.cross(R1[2, :], r1.T).T
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r2.T, r1.T)
    r3 = r3 / np.linalg.norm(r3)

    R = np.vstack((r1.T, r2.T, r3.T)).T

    # New rotation matrices are the same (R)
    R1n = R
    R2n = R

    # New intrinsic parameters (K)
    K = K2
    K1n = K
    K2n = K

    # New translation vectors (t)
    t1n = -R @ c1
    t2n = -R @ c2

    # Rectification matrices (M)
    M1 = K @ R @ np.linalg.inv(K1 @ R1)
    M2 = K @ R @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n

# Example usage:
# K1, K2, R1, R2, t1, t2 are the camera parameters obtained from calibration.
# M1, M2, K1n, K2n, R1n, R2n, t1n, t2n = rectify_pair(K1, K2, R1, R2, t1, t2)
