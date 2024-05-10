# import numpy as np

# def get_depth(dispM, K1, K2, R1, R2, t1, t2):
#     """
#     creates a depth map from a disparity map (DISPM).
#     """
#     depthM = np.zeros_like(dispM, dtype=float)

#     return depthM

import numpy as np

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # Calculate the baseline (distance between the two camera centers)
    c1 = -np.linalg.inv(R1) @ t1
    c2 = -np.linalg.inv(R2) @ t2
    b = np.linalg.norm(c1 - c2)
    
    # Focal length from the K matrix
    f = K1[0, 0]
    
    # Initialize the depth map with ones and same size as the disparity map
    depthM = np.ones(dispM.shape)
    
    # Replace non-zero disparity values with the inverse to calculate depth
    with np.errstate(divide='ignore', invalid='ignore'):
        depthM[dispM != 0] = 1.0 / dispM[dispM != 0]
    
    # Multiply by the baseline and focal length to get the depth in meters
    depthM *= b * f
    
    # Set infinite values to zero if disparity was zero (to avoid division by zero)
    depthM[np.isinf(depthM)] = 0
    
    return depthM

# Usage:
# dispM = ... # 
# depth_map = get_depth(dispM, K1, K2, R1, R2, t1, t2)
