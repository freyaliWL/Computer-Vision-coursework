import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import get_lenet
from scipy.io import loadmat
from conv_net import convnet_forward


# Load the trained weights
data = loadmat('../results/lenet.mat')
params_raw = data['params'] 

params = []
for param_idx in range(params_raw.shape[1]):
    w = params_raw[0, param_idx][0, 0][0]
    b = params_raw[0, param_idx][0, 0][1]
    params.append({'w': w, 'b': b})

# Mapping filenames to labels
label_mapping = {
    "im1.jpeg": 7,
    "im2.jpeg": 0,
    "im3.jpeg": 1,
    "im4.jpeg": 4,
    "im5.jpeg": 6,
}

num_images = len(label_mapping)
imgs = np.zeros((784, num_images))

# Read images and preprocess
for idx, (filename, label) in enumerate(label_mapping.items()):
    
    img_path = f'../real_images/{filename}'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0  # Convert to range [0, 1]
    img = cv2.resize(img, (28, 28))
    img = np.transpose(img)
    imgs[:, idx] = img.flatten()

layers = get_lenet()
layers[0]['batch_size'] = num_images

# Run through the network
*_, P = convnet_forward(params, layers, imgs, test= True)
out_label = np.argmax(P, axis=0)
print(f'out_labels:',list(out_label))
# Extract true labels from the dictionary
true_labels = list(label_mapping.values())
print(f'true_labels:',true_labels)

# Compare predictions to actual labels
correctly_classified = np.sum(out_label == true_labels)
print(f'Correctly classified images: {correctly_classified} out of {num_images}')
