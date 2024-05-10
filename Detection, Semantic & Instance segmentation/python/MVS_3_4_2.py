import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

camera_centers = []
for img_name, params in camera_params.items():
    R = np.array(params['r_values']).reshape(3, 3)
    t = np.array(params['t_values']).reshape(3, 1)
    # Camera center in world coordinates is -R.T @ t
    camera_centers.append(-R.T @ t)

# Compute pairwise distances between camera centers
pairwise_distances = []
for i in range(len(camera_centers)):
    for j in range(i + 1, len(camera_centers)):
        distance = np.linalg.norm(camera_centers[i] - camera_centers[j])
        pairwise_distances.append(distance)

# Calculate the average distance as an approximate baseline
approx_baseline = np.mean(pairwise_distances)
print(f"Approximate baseline: {approx_baseline}")

# Use the approximate baseline to define the search range for depth
# This is just an example and the actual implementation may vary
# min_depth = approx_baseline / 10  # for example
# max_depth = approx_baseline * 2   # for example

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

# When accessing images, use the filename as the key
# if ref_image_name in images:
#     ref_image_data = images[ref_image_name]
#     print(f"Loaded reference image data for {ref_image_name}")
# else:
#     print(f"Reference image {ref_image_name} not found in loaded images.")

# Function definitions

def construct_projection_matrix(k_values, r_values, t_values):
    K = np.array(k_values).reshape(3, 3)
    R = np.array(r_values).reshape(3, 3)
    t = np.array(t_values).reshape(3, 1)
    return K @ np.hstack((R, t))

p_matrices = {img_name: construct_projection_matrix(params['k_values'], params['r_values'], params['t_values']) for img_name, params in camera_params.items()}
print(p_matrices)
# for debugging parameters image and matrices loading
# for img_name, p_matrix in p_matrices.items():
#     print(f"Image Name: {img_name}, Type: {type(img_name)}")
#     print(f"Projection Matrix:\n{p_matrix}\n")

# # Load images
# image_names = ['../data/templeR0013.png', '../data/templeR0014.png', '../data/templeR0016.png', '../data/templeR0043.png', '../data/templeR0045.png']
# images = {img_name: cv2.imread(img_name) for img_name in image_names}

# # Print the keys of the images dictionary to verify
# print("Loaded image keys:", images.keys())

# # Focal length from the intrinsic matrix
# focal_length = camera_params['templeR0013.png']['k_values'][0]  # This assumes k_values[0] is the focal length

# # Baseline calculation (example between templeR0013.png and templeR0014.png)
# # This assumes you have the extrinsic parameters (rotation R and translation t)
# R1 = np.array(camera_params['templeR0013.png']['r_values']).reshape(3, 3)
# t1 = np.array(camera_params['templeR0013.png']['t_values']).reshape(3, 1)
# R2 = np.array(camera_params['templeR0014.png']['r_values']).reshape(3, 3)
# t2 = np.array(camera_params['templeR0014.png']['t_values']).reshape(3, 1)

# # Convert rotation matrices to rotation vectors
# rvec1, _ = cv2.Rodrigues(R1)
# rvec2, _ = cv2.Rodrigues(R2)

# # Assuming you have the rotation and translation between the two camera views
# # Calculate the position of each camera in the world coordinates
# cam1_pos = -R1.T @ t1
# cam2_pos = -R2.T @ t2

# # Calculate the baseline as the distance between camera positions
# baseline = np.linalg.norm(cam2_pos - cam1_pos)

# # Disparity would come from your disparity map, which you have not calculated yet
# # It requires matching points between images, which is a separate problem


def compute(c0, c1):
    c0_mean = np.mean(c0, axis=0)
    c1_mean = np.mean(c1, axis=0)
    #print(c0_mean, c1_mean)
    c0_centered = c0 - c0_mean
    c1_centered = c1 - c1_mean
    c0_norm = np.linalg.norm(c0_centered)
    c1_norm = np.linalg.norm(c1_centered)
    #print(c0_norm,c1_norm)
    c0_normalized = c0_centered / c0_norm if c0_norm != 0 else c0_centered
    c1_normalized = c1_centered / c1_norm if c1_norm != 0 else c1_centered
    ncc_value = np.dot(c0_normalized.flatten(), c1_normalized.flatten())
    #print(ncc_value)
    return ncc_value

def compute_3d_point(x, y, depth, p_matrix):
    # Scale the homogeneous coordinate of the pixel q by the depth d
    q_homog = np.array([x * depth, y * depth, depth])
    # print(q_homog)
    # Extract the 3x3 part and the last column of the projection matrix
    p_matrix_3x3 = p_matrix[:, :3]
    p_matrix_4th_col = p_matrix[:, 3]
    # print(p_matrix_3x3)
    # print(p_matrix_4th_col)
    # Compute the inverse of the 3x3 part of the projection matrix
    p_inv = np.linalg.inv(p_matrix_3x3)
    # Apply the inverse to (dx - P14, dy - P24, d - P34)
    x_3d_homog = p_inv @ (q_homog - p_matrix_4th_col)
    # print(x_3d_homog)
    # print("----------")
    return x_3d_homog

def compute_consistency(i0, i1, x, y, d, S, p_matrix_i0, p_matrix_i1):
    c0 = []
    c1 = []
    for dy in range(-S//2+1, S // 2 + 1):
        for dx in range(-S//2+1, S // 2 + 1):
            q = (x + dx, y + dy)
            c0.append(compute_3d_point(q[0], q[1], d, p_matrix_i0))
    
    for dy in range(-S//2+1, S // 2 + 1):
        for dx in range(-S//2+1, S // 2 + 1):
            q = (x + dx, y + dy)
            c1.append(compute_3d_point(q[0], q[1], d, p_matrix_i1))
    c0 = np.array(c0)
    c1 = np.array(c1)
    #print(c0, c1)
    return compute(c0, c1) if len(c0) and len(c1) else 0


# def project_3d_to_2d(point_3d, p_matrix):
#     # Convert point_3d to homogeneous coordinates
#     point_3d_homog = np.append(point_3d, 1)

#     # Project the point using the projection matrix
#     point_2d_homog = p_matrix @ point_3d_homog

#     # Convert from homogeneous coordinates to 2D
#     point_2d = point_2d_homog[:2] / point_2d_homog[2]

#     # Return the 2D coordinates (x, y)
#     return point_2d.astype(int)

# def get_pixel_value(image, x, y):
#     height, width = image.shape[:2]
#     if 0 <= x < width and 0 <= y < height:
#         return image[y, x]
#     return 0  # Or handle this case as needed

# def compute_consistency(i0, i1, x, y, d, S, p_matrix_i0, p_matrix_i1):
#     c0 = []
#     c1 = []
#     for dy in range(-S//2+1, S // 2 + 1):
#         for dx in range(-S//2+1, S // 2 + 1):
#             q = (x + dx, y + dy)
#             point_3d = compute_3d_point(q[0], q[1], d, p_matrix_i0)
#             point_2d_i0 = project_3d_to_2d(point_3d, p_matrix_i0)
#             pixel_value_i0 = get_pixel_value(i0, *point_2d_i0)
#             c0.append(pixel_value_i0)

#             point_3d = compute_3d_point(q[0], q[1], d, p_matrix_i1)
#             point_2d_i1 = project_3d_to_2d(point_3d, p_matrix_i1)
#             pixel_value_i1 = get_pixel_value(i1, *point_2d_i1)
#             c1.append(pixel_value_i1)

#     c0 = np.array(c0)
#     c1 = np.array(c1)
#     return compute(c0, c1) if len(c0) and len(c1) else 0


def within_bounds(point, shape):
    x, y = point
    h, w = shape[:2]
    x, y = int(round(x)), int(round(y))
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    return 0 <= x < w and 0 <= y < h


def compute_bounding_box_depths(bbox_min, bbox_max):
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


ref_image_name = "templeR0013.png"
print(p_matrices[ref_image_name])

def compute_depth_map(ref_image, other_images, p_matrices, S, threshold, max_iterations):
    # Ensure that the reference image data is correctly fetched
    ref_image = images[ref_image_name]
   
    #we don't project it, project first
    #print(ref_image,ref_p_matrix)
    min_depth, max_depth = compute_bounding_box_depths(bbox_min, bbox_max)
   
    depth_map = np.zeros(ref_image.shape[:2])

    max_iterations = 3  # Example value
    initial_depth_step = 1.0  # Start with a larger depth step
    depth_step_reduction_factor = 0.5  # Reduce the depth step in each iteration
    last_depth_map = None
    convergence_threshold = 1E-6  # Define a suitable threshold
    for iteration in range(max_iterations):
        for y in range(S // 2, ref_image.shape[0] - S // 2):
            for x in range(S // 2, ref_image.shape[1] - S // 2):
                depth_step = initial_depth_step * (depth_step_reduction_factor ** iteration)
                best_depth = None
                best_score = -np.inf

                for d in np.arange(min_depth, max_depth, depth_step):  # Adjust depth_step
                    scores = []
                    for other_img_name, other_p_matrix in p_matrices.items():
                        # print(other_img_name)
                        # print(other_p_matrix)
                        # print("------------------")
                        if other_img_name == ref_image_name:
                            #print(other_img_name)
                            continue
                        other_images = images[other_img_name]
                        
                        score = compute_consistency(ref_image, other_images, x, y, d, S, p_matrices[ref_image_name], p_matrices[other_img_name])
                        scores.append(score)
                        #print(len(scores))
                        average_score = np.mean(scores)
                        #print(average_score)
                        if average_score > best_score and average_score > threshold:
                            best_score = average_score
                            best_depth = d
                print(best_depth, best_score)
                print(x,y)
                depth_map[y, x] = best_depth if best_depth is not None else 0
                # Check for convergence if not the first iteration

        last_depth_map = depth_map.copy()
        print(last_depth_map)
    return depth_map


# Main loop to compute depth maps for all images
image_names = ['templeR0013.png', 'templeR0014.png', 'templeR0016.png', 'templeR0043.png', 'templeR0045.png']

# Main execution
S = 5  # Window size for depth computation
threshold = 0.95 # Threshold for consistency scoring
max_iterations  = 100
bbox_min = np.array([-0.023121, -0.038009, -0.091940])
bbox_max = np.array([0.078626, 0.121636, -0.017395])

for img_name in image_names:
    print(f"Processing {img_name}")
    depth_map = compute_depth_map(img_name, images, p_matrices, S, threshold, max_iterations)
    #print(depth_map)

# Assuming p_matrix_i0 is your reference image's projection matrix
# min_depth, max_depth = compute_bounding_box_depths(bbox_min, bbox_max, p_matrix)
# print(f"min_depth: {min_depth}, max_depth: {max_depth}")


# Sample pixels and depths
# sample_pixels = [(100, 150), (200, 250), (300, 350)]
# sample_depths = [0.5, 1.0, 1.5]

# Compute 3D points
# computed_3d_points = []
# for pixel, depth in zip(sample_pixels, sample_depths):
#     x, y = pixel
#     point_3d = compute_3d_point(x, y, depth, p_matrix)
#     computed_3d_points.append(point_3d)
#     print(f"Pixel: {pixel}, Depth: {depth}, 3D Point: {point_3d}")

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for point in computed_3d_points:
#     ax.scatter(point[0], point[1], point[2], marker='o')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# plt.show()


# depth_map = compute_depth_map(images[ref_image_name], images, p_matrices, bbox_min, bbox_max, S, threshold)

#Normalize the depth map for visualization
# depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
# depth_map_visualized = cv2.applyColorMap(np.uint8(depth_map_normalized), cv2.COLORMAP_JET)

# #Save the depth map
# cv2.imwrite('depth_map_0.png', depth_map_visualized)

# # Display the depth map
# cv2.imshow('Depth Map_0', depth_map_visualized)

# # #Save or display the depth map
# # cv2.imwrite('depth_map.png', depth_map)
# # cv2.imshow('Depth Map', depth_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()