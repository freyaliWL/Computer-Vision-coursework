import numpy as np
import cv2
#from scipy.ndimage import convolve

# def epipolarCorrespondence(im1, im2, F, pts1):
#     """
#     Args:
#         im1:    Image 1
#         im2:    Image 2
#         F:      Fundamental Matrix from im1 to im2
#         pts1:   coordinates of points in image 1
#     Returns:
#         pts2:   coordinates of points in image 2
#     """

#     pts2 = np.zeros_like(pts1)
#     return pts2

def epipolarCorrespondence(im1, im2, F, pts1):
    pts2 = np.zeros_like(pts1)

    window = 3
    window_im2 = 10

    for i in range(pts1.shape[0]):
        x, y = pts1[i]

        p1 = np.array([x, y, 1])
        epipolar_line = np.dot(F, p1)

        window1 = im1[int(y - window):int(y + window + 1), int(x - window):int(x + window + 1), :]

        dist = np.inf
        epipolar_line /= -epipolar_line[1]

        # Convert x to an integer
        for j in range(int(x - window_im2), int(x + window_im2 + 1)):
            x1 = j
            y1 = int(epipolar_line[0] * j + epipolar_line[2])

            window2 = im2[y1 - window:y1 + window + 1, x1 - window:x1 + window + 1, :]
            difference = (window1 - window2) ** 2

            distance = np.sqrt(np.sum(difference, axis=(0, 1)))

            if (distance < dist).any():
                dist = distance
                pts2[i, 0] = x1
                pts2[i, 1] = y1

    return pts2
