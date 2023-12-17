import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
camera_params = {
    'templeR0013.png': {
        'k_values': [1520.4, 0, 302.32, 0, 1525.9, 246.87, 0, 0, 1],
        'r_values': [0.115411678, 0.991389001, 0.061870781, -0.684052897, 0.034160817, 0.728632056, 0.720244249, -0.126415535, 0.682105075],
        't_values': [-0.019347492, 0.043210508, 0.589790752]
    },
    'templeR0014.png': {
        'k_values': [1520.4, 0.0, 302.32, 0.0, 1525.9, 246.87, 0.0, 0.0, 1.0],
        'r_values': [0.127771962, 0.990677009, 0.047258737, -0.580737067, 0.036103437, 0.813290230, 0.804001731, -0.131360589, 0.579935868],
        't_values': [-0.020311130, 0.045204446, 0.583107596]
    },
    'templeR0016.png': {
        'k_values': [1520.4, 0.0, 302.32, 0.0, 1525.9, 246.87, 0.0, 0.0, 1.0],
        'r_values': [0.146128927, 0.989169691, 0.013771642, -0.345045694, 0.037916486, 0.937819710, 0.927140661, -0.141794433, 0.346849438],
        't_values': [-0.022314990, 0.046419784, 0.569254845]
    },
    'templeR0043.png': {
        'k_values': [1520.4, 0.0, 302.32, 0.0, 1525.9, 246.87, 0.0, 0.0, 1.0],
        'r_values': [-0.085135479, -0.993998697, -0.068691626, 0.806313330, -0.028231285, -0.590814529, 0.585329619, -0.105686252, 0.803877884],
        't_values': [0.027855751, -0.051647186, 0.598117193]
    },
    'templeR0045.png': {
        'k_values': [1520.4, 0.0, 302.32, 0.0, 1525.9, 246.87, 0.0, 0.0, 1.0],
        'r_values': [-0.054399951, -0.994970632, -0.084107587, 0.933915538, -0.020893501, -0.356882654, 0.353330458, -0.097963781, 0.930355138],
        't_values': [0.026476030, -0.042897984, 0.609023113]
    }
}

def extract_intrinsic_matrices(camera_params):
    intrinsic_matrices = {}
    for img_name, params in camera_params.items():
        k_values = params['k_values']
        K = np.array(k_values).reshape(3, 3)  # Reshape to a 3x3 matrix
        intrinsic_matrices[img_name] = K
    return intrinsic_matrices

# Usage
intrinsic_matrices = extract_intrinsic_matrices(camera_params)

# To get the intrinsic matrix for a specific image, e.g., 'templeR0013.png':
K_templeR0013 = intrinsic_matrices['templeR0013.png']
print("Intrinsic matrix for 'templeR0013.png':\n", K_templeR0013)
