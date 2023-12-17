import numpy as np

def triangulate(P1, pts1, P2, pts2):
    num_points = pts1.shape[0]
    pts3d_homog = np.zeros((4, num_points))  # Homogeneous coordinates of 3D points

    for i in range(num_points):
        # Construct matrix A for each point
        A = np.vstack([
            pts1[i, 1] * P1[2, :] - P1[1, :],
            P1[0, :] - pts1[i, 0] * P1[2, :],
            pts2[i, 1] * P2[2, :] - P2[1, :],
            P2[0, :] - pts2[i, 0] * P2[2, :]
        ])

        # SVD of A
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]  # Solution is the last row of V
        pts3d_homog[:, i] = X / X[3]  # Normalize to make the last element 1

    # Convert to non-homogeneous coordinates
    pts3d = pts3d_homog[:3, :].T

    # Calculate reprojection error
    reproj1 = (P1 @ pts3d_homog).T
    reproj2 = (P2 @ pts3d_homog).T
    reproj1 /= reproj1[:, 2, None]
    reproj2 /= reproj2[:, 2, None]

    error1 = np.mean(np.linalg.norm(reproj1[:, :2] - pts1, axis=1))
    error2 = np.mean(np.linalg.norm(reproj2[:, :2] - pts2, axis=1))

    return pts3d, error1, error2
