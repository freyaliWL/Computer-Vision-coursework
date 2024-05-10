import numpy as np
from scipy.signal import convolve2d

def get_disparity(im1, im2, max_disp, window_size):
    # Ensure the images are floats to prevent overflow/underflow
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    
    # Define the mask for the convolution
    mask = np.ones((window_size, window_size))
    
    # Initialize the disparity map
    dispM = np.zeros_like(im1)
    
    # Iterate over each pixel in the image
    for y in range(window_size // 2, im1.shape[0] - window_size // 2):
        for x in range(window_size // 2, im1.shape[1] - window_size // 2):
            # Initialize the minimum SSD to a large number
            min_ssd = float('inf')
            best_disp = 0
            
            # Iterate over all possible disparities
            for d in range(0, min(max_disp, x - window_size // 2) + 1):
                # Define the window in the first image
                window_im1 = im1[y - window_size // 2:y + window_size // 2 + 1,
                                 x - window_size // 2:x + window_size // 2 + 1]
                
                # Define the corresponding window in the second image
                window_im2 = im2[y - window_size // 2:y + window_size // 2 + 1,
                                 x - window_size // 2 - d:x + window_size // 2 + 1 - d]
                
                # Calculate the SSD for this disparity
                ssd = np.sum((window_im1 - window_im2) ** 2)
                
                # Update the disparity if the SSD is lower than the current minimum
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disp = d
            
            # Store the best disparity for this pixel
            dispM[y, x] = best_disp
    
    # Normalize the disparity map for display
    dispM = (dispM - dispM.min()) / (dispM.max() - dispM.min()) * 255
    return dispM.astype(np.uint8)

# Usage example:
# im1 = cv2.imread('im1.png', 0)  # Read as grayscale
# im2 = cv2.imread('im2.png', 0)
# disparity_map = get_disparity(im1, im2, max_disp=64, window_size=9)
# cv2.imshow('Disparity Map', disparity_map)
# cv2.waitKey(0)
