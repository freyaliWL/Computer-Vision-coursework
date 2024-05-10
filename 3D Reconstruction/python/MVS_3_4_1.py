# read_camera_parameters(file_path): Reads camera parameters from a *_par.txt file and returns a dictionary with parameters for each image.

# camera_params: Use the function to populate the camera_params dictionary with parameters for each image.

# construct_projection_matrix(k_values, r_values, t_values): Constructs a 3x4 projection matrix from intrinsic and extrinsic camera parameters.

# load_images(image_file_names): Loads images given a list of image filenames and returns a dictionary of image data.

# project_point(point_3d, p_matrix): Projects a 3D point to 2D space using a projection matrix.

# compute_3d_point(x, y, depth, p_matrix): Computes the 3D coordinates for a point at a given depth and pixel coordinates in the reference image.

# within_bounds(point, shape): Checks if a point is within the bounds of an image shape.

# compute_consistency(i0, i1, x, p_matrix_i0, p_matrix_i1): Computes the consistency score between two images for a set of 3D points.

# compute_depth_map(ref_image, other_images, p_matrices, bbox_corners, depth_range): Main function that orchestrates the computation of the depth map for the reference image.

# NCC.compute(c0, c1): Computes the normalized cross-correlation between two sets of pixel color data.

import numpy as np
import cv2
import os

# Given camera parameters for I0 and I1 (example for one image, repeat for all images)
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


def load_images_from_folder(folder_path, camera_params):
    images = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') and filename in camera_params:
            img_path = os.path.join(folder_path, filename)
            images[filename] = cv2.imread(img_path)
            if images[filename] is None:
                print(f"Failed to load image at {img_path}")
    return images

# Usage
folder_path = '../data'
images = load_images_from_folder(folder_path, camera_params)

ref_image_name = 'templeR0013.png'  # No need for relative path here, as keys are just filenames

# When accessing images, use the filename as the key
if ref_image_name in images:
    ref_image_data = images[ref_image_name]
    print(f"Loaded reference image data for {ref_image_name}")
else:
    print(f"Reference image {ref_image_name} not found in loaded images.")

# Function definitions

def construct_projection_matrix(k_values, r_values, t_values):
    K = np.array(k_values).reshape(3, 3)
    R = np.array(r_values).reshape(3, 3)
    t = np.array(t_values).reshape(3, 1)
    return K @ np.hstack((R, t))

p_matrices = {img_name: construct_projection_matrix(params['k_values'], params['r_values'], params['t_values']) for img_name, params in camera_params.items()}

# for debugging parameters image and matrices loading
# for img_name, p_matrix in p_matrices.items():
#     print(f"Image Name: {img_name}, Type: {type(img_name)}")
#     print(f"Projection Matrix:\n{p_matrix}\n")

# # Load images
# image_names = ['../data/templeR0013.png', '../data/templeR0014.png', '../data/templeR0016.png', '../data/templeR0043.png', '../data/templeR0045.png']
# images = {img_name: cv2.imread(img_name) for img_name in image_names}

# # Print the keys of the images dictionary to verify
# print("Loaded image keys:", images.keys())



def compute(c0, c1):
    c0_mean = np.mean(c0, axis=0)
    c1_mean = np.mean(c1, axis=0)
    c0_centered = c0 - c0_mean
    c1_centered = c1 - c1_mean
    c0_norm = np.linalg.norm(c0_centered)
    c1_norm = np.linalg.norm(c1_centered)
    c0_normalized = c0_centered / c0_norm if c0_norm != 0 else c0_centered
    c1_normalized = c1_centered / c1_norm if c1_norm != 0 else c1_centered
    ncc_value = np.dot(c0_normalized.flatten(), c1_normalized.flatten())
    return ncc_value

def project_point(point_3d, p_matrix):
    point_homog = np.append(point_3d, 1)
    projected_point = p_matrix @ point_homog
    projected_point = projected_point[:2] / projected_point[2]
    return projected_point


def compute_3d_point(x, y, depth, p_matrix):
    q_homog = np.array([x, y, 1])
    p_matrix_3x3 = p_matrix[:, :3]
    p_matrix_4th_col = p_matrix[:, 3]
    p_inv = np.linalg.inv(p_matrix_3x3)
    x_3d_homog = p_inv @ (q_homog * depth - p_matrix_4th_col)
    x_3d = x_3d_homog / x_3d_homog[2]
    return x_3d

def compute_consistency(i0, i1, x, p_matrix_i0, p_matrix_i1):
    c0 = []
    c1 = []
    for point_3d in x:
        q_i0 = project_point(point_3d, p_matrix_i0)
        q_i1 = project_point(point_3d, p_matrix_i1)
        if within_bounds(q_i0, i0.shape) and within_bounds(q_i1, i1.shape):
            c0.append(i0[int(q_i0[1]), int(q_i0[0])])
            c1.append(i1[int(q_i1[1]), int(q_i1[0])])
    c0 = np.array(c0)
    c1 = np.array(c1)
    return compute(c0, c1) if len(c0) and len(c1) else 0

def within_bounds(point, shape):
    x, y = point
    h, w = shape[:2]
    return 0 <= x < w and 0 <= y < h


def compute_bounding_box_depths(bbox_min, bbox_max, p_matrix):
    # Compute the 8 corners of the bounding box
    bbox_corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2], 1],
        [bbox_min[0], bbox_min[1], bbox_max[2], 1],
        [bbox_min[0], bbox_max[1], bbox_min[2], 1],
        [bbox_min[0], bbox_max[1], bbox_max[2], 1],
        [bbox_max[0], bbox_min[1], bbox_min[2], 1],
        [bbox_max[0], bbox_min[1], bbox_max[2], 1],
        [bbox_max[0], bbox_max[1], bbox_min[2], 1],
        [bbox_max[0], bbox_max[1], bbox_max[2], 1]
    ])

    # No need to project the points, directly use the z-values
    depths = bbox_corners[:, 2]  # Extract the z-values

    return min(depths), max(depths)

def compute_depth_map(ref_image, other_images, p_matrices, bbox_min, bbox_max, S, threshold):
    ref_image_name = 'templeR0013.png'
    p_matrix = p_matrices[ref_image_name]  # This should be a string key
    min_depth, max_depth = compute_bounding_box_depths(bbox_min, bbox_max, p_matrix)
   
    depth_map = np.zeros(ref_image.shape[:2])

    for y in range(S // 2, ref_image.shape[0] - S // 2):
        for x in range(S // 2, ref_image.shape[1] - S // 2):
            best_depth = None
            best_score = -np.inf

            for d in np.arange(min_depth, max_depth, 0.01):  # Adjust depth_step as needed
                X = []
                scores = []

                for dy in range(-S // 2, S // 2 + 1):
                    for dx in range(-S // 2, S // 2 + 1):
                        q = (x + dx, y + dy)
                        X.append(compute_3d_point(q[0], q[1], d, p_matrices[ref_image_name]))
                #print(f"Processing pixel: ({x}, {y}), Depth Points: {len(X)}")

                for other_img_name, other_p_matrix in p_matrices.items():
                    if other_img_name == ref_image_name:
                        continue
                    score = compute_consistency(ref_image, other_images[other_img_name], X, p_matrices[ref_image_name], other_p_matrix)
                    scores.append(score)

                average_score = np.mean(scores)
                if average_score > best_score and average_score > threshold:
                    best_score = average_score
                    best_depth = d
            #print(f"Pixel: ({x}, {y}), Best Depth: {best_depth}, Best Score: {best_score}")
            depth_map[y, x] = best_depth if best_depth is not None else 0

    return depth_map

# Main execution
S = 5  # Window size for depth computation
threshold = 0.3  # Threshold for consistency scoring
bbox_min = np.array([-0.023121, -0.038009, -0.091940])
bbox_max = np.array([0.078626, 0.121636, -0.017395])


## HERE is for 3.4.1
def draw_projected_box_corners(image, bbox_corners, p_matrix):
    # Convert image to color if it's not already
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for corner in bbox_corners:
        # Project each corner
        projected_2d = project_point(corner[:3], p_matrix)
        # Draw a circle at the projected point
        cv2.circle(image, (int(projected_2d[0]), int(projected_2d[1])), 5, (0, 255, 0), -1)

    return image


# Compute the 8 corners of the bounding box
bbox_corners = np.array([
    [bbox_min[0], bbox_min[1], bbox_min[2], 1],
    [bbox_min[0], bbox_min[1], bbox_max[2], 1],
    [bbox_min[0], bbox_max[1], bbox_min[2], 1],
    [bbox_min[0], bbox_max[1], bbox_max[2], 1],
    [bbox_max[0], bbox_min[1], bbox_min[2], 1],
    [bbox_max[0], bbox_min[1], bbox_max[2], 1],
    [bbox_max[0], bbox_max[1], bbox_min[2], 1],
    [bbox_max[0], bbox_max[1], bbox_max[2], 1]
])

# Visualize the projected corners on all images
for img_name, img_data in images.items():
    print(img_name)
    if img_name in p_matrices:
        image_with_corners = draw_projected_box_corners(img_data.copy(), bbox_corners, p_matrices[img_name])
        cv2.imwrite(f"image_with_corners_{img_name}.png", image_with_corners)
        cv2.imshow(f"Image with projected corners: {img_name}", image_with_corners)
    else:
        print(f"Projection matrix for {img_name} not found.")


# #Compute the depth map
# depth_map = compute_depth_map(images[ref_image_name], images, p_matrices, bbox_min, bbox_max, S, threshold)

# #Normalize the depth map for visualization
# depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
# depth_map_visualized = cv2.applyColorMap(np.uint8(depth_map_normalized), cv2.COLORMAP_JET)

# #Save the depth map
# cv2.imwrite('depth_map_013.png', depth_map_visualized)

# # Display the depth map
# cv2.imshow('Depth Map_013', depth_map_visualized)

# #Save or display the depth map
# cv2.imwrite('depth_map.png', depth_map)
# cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
