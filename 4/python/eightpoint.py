import numpy as np
from numpy.linalg import svd
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x, y) coordinates
        pts2 - Nx2 matrix of (x, y) coordinates
        M    - max(imwidth, imheight)
    """
    # Normalize points
    normalized1 = pts1 / M
    normalized2 = pts2 / M

    num_points = pts1.shape[0]
    A = np.ones((num_points, 9))

    for i in range(num_points):
        x1, y1 = normalized1[i, 0], normalized1[i, 1]
        x2, y2 = normalized2[i, 0], normalized2[i, 1]

        A[i, :] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Solve linear system using SVD
    _, _, V = svd(A)
    F = V[-1, :].reshape((3, 3))

    # Enforce rank-2 constraint
    U, S, V = svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V

    # Refine F using nonlinear optimization (if available)
    F = refineF(F, normalized1, normalized2)

    # Denormalize F
    denorm = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = np.dot(np.dot(np.transpose(denorm), F), denorm)

    return F

# # #Load data from someCorresp.npy for testing results
# data = np.load('../data/someCorresp.npy', allow_pickle=True).item()

# # Extract points
# pts1 = data['pts1']
# pts2 = data['pts2']

# # Extract M
# M = data['M']

# # Call eightpoint function
# F = eightpoint(pts1, pts2, M)

# #Print the estimated fundamental matrix
# print("Estimated Fundamental Matrix (F):")
# print(F)
