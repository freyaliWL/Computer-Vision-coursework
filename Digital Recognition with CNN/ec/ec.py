import os
import sys
sys.path.append("../python")
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from utils import get_lenet
from scipy.io import loadmat
from conv_net import convnet_forward

true_labels = {
    'image1':[1,2,3,4,5,6,7,8,9,0],
    'image2':[1,2,3,4,5,6,7,8,9,0],
    'image3':[6,0,6,2,4],
    'image4':[7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 4, 5, 4, 7, 4,
              0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 9, 2, 3, 5, 1, 2, 4, 4]
}

total_correct = 0
total_digits = 0


# Load the model architecture
layers = get_lenet()
layers[0]['batch_size'] = 1

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

params = []
for param_idx in range(params_raw.shape[1]):
    w = params_raw[0, param_idx][0, 0][0]
    b = params_raw[0, param_idx][0, 0][1]
    params.append({'w': w, 'b': b})

kernel_size = 3
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

# Read images and recognize handwritten numbers
files = sorted(glob.glob("../images/image*"))

for idx, file in enumerate(files):
    predicted_labels = []
    #print(idx)
    filename = os.path.splitext(os.path.basename(file))[0]
    #print(filename)
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)  # Invert colors

    # # Manual thresholding based on image index
    # if idx == 3:
    #     threshold = 40
    # elif idx == 2:
    #     threshold = 100
    # else:
    #     threshold = 150

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # plt.imshow(img, cmap='gray')
    # plt.title("Original Image")
    # plt.show()

    #Use adaptive thresholding for the first two images, manual for the last two
    if filename == 'image1':
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif filename == 'image2':
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif filename == 'image3':
         _, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    elif filename == 'image4':
        _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
    
    # plt.imshow(thresh, cmap='gray')
    # plt.title("Thresholded Image")
    # plt.show()

    #_, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

    for i in range(1, num_labels):
        # Extract each character
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        char_image = thresh[y:y+h, x:x+w]
        cropped = cv2.dilate(char_image, kernel, iterations = 4)
        # Keep aspect ratio intact
        d = max(w, h)
        padded = cv2.copyMakeBorder(cropped, (d-h)//2, (d-h)-(d-h)//2, (d-w)//2, (d-w)-(d-w)//2, cv2.BORDER_CONSTANT, value=0)
        resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize
        resized = resized / 255.0
        resized = resized.reshape(28*28, 1)
        # Pass through the network
        *_, P = convnet_forward(params, layers, resized, test= True)
        pred = np.argmax(P)
        # collect the predicted single digit into a list
        predicted_labels.append(pred)
        # Draw bounding box around the digit and annotate with prediction
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, str(pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    current_true_labels = true_labels[filename] 
    correct_predictions = sum([1 for pred, true in zip(predicted_labels, current_true_labels) if pred == true])
    
    total_correct += correct_predictions
    total_digits += len(current_true_labels)

    print(f"Accuracy for {filename}: {correct_predictions / len(current_true_labels) * 100:.2f}%")

    plt.imshow(img, cmap='gray')
    plt.show()

print(f"Overall Accuracy: {total_correct / total_digits * 100:.2f}%")


