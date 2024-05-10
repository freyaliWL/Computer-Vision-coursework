import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)

#print(ytest.shape)

# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []
all_actual = []
for i in range(0, xtest.shape[1], 100):
    _, P = convnet_forward(params, layers, xtest[:, i:i+100], test=True)
    #print(P.shape)
    preds = np.argmax(P, axis=0)
    actual = ytest[:, i:i+100].flatten()
    all_preds.extend(preds)
    all_actual.extend(actual)

# Compute the confusion matrix
confusion = confusion_matrix(all_actual, all_preds)
print(confusion)

# Normalize the confusion matrix
row_sums = confusion.sum(axis=1, keepdims=True)
normalized_confusion = confusion / row_sums

# Identify the top confused pairs (excluding the diagonal)
np.fill_diagonal(normalized_confusion, 0)
top_two_confused_indices = normalized_confusion.argsort(axis=None)[-2:]
top_two_confused_pairs = [(i//10, i%10) for i in top_two_confused_indices]

# Print the results
print("Normalized Confusion Matrix:")
print(normalized_confusion)
print(f"\nThe most confused pair is {top_two_confused_pairs[1]} with a normalized total confusion of {normalized_confusion[top_two_confused_pairs[1]]:.4f}.")
print(f"The second most confused pair is {top_two_confused_pairs[0]} with a normalized total confusion of {normalized_confusion[top_two_confused_pairs[0]]:.4f}.")

# Visualization
# fig, ax = plt.subplots(figsize=(10, 10))
# cax = ax.matshow(normalized_confusion, cmap='viridis')
# plt.title("Normalized Confusion Matrix", pad=20)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.colorbar(cax)
# plt.savefig("../results/normalized_confusion.png")
# plt.show()

# Visualization
# fig, ax = plt.subplots(figsize=(10, 10))
# cax = ax.matshow(confusion, cmap='viridis')
# plt.title("Confusion Matrix", pad=20)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.colorbar(cax)
# plt.savefig("../results/confusion.png")
# plt.show()

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(confusion, cmap='viridis')
plt.title("Confusion Matrix", pad=20)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.colorbar(cax)

# Add numeric annotations to each cell in the matrix
num_classes = confusion.shape[0]
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(confusion[i, j]), va='center', ha='center', color='red' if confusion[i,j] != 0 else 'black')

# Set tick marks and labels for the axes
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(range(num_classes))
ax.set_yticklabels(range(num_classes))

plt.savefig("../results/confusion.png")
plt.show()


# for i in range(0, xtest.shape[1], 100):
#     cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)

# # hint: 
# #     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
# #     to compute the confusion matrix. Or you can write your own code :)

