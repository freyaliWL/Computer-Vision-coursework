import numpy as np
import matplotlib.pyplot as plt
from eightpoint import eightpoint
from displayEpipolarF import displayEpipolarF
from epipolarCorrespondence import epipolarCorrespondence
from epipolarMatchGUI import epipolarMatchGUI
from essentialMatrix import essentialMatrix

# Load data from someCorresp.npy
data = np.load('../data/someCorresp.npy', allow_pickle=True).item()
pts1 = data['pts1']
pts2 = data['pts2']
M = data['M']

# Call eightpoint function
F = eightpoint(pts1, pts2, M)
print("Fundamental Matrix F:\n", F)

I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')

#Call displayEpipolarF function to visualize epipolar lines
# displayEpipolarF(I1, I2, F)
# initF = F.ravel()
# print("Shape of initF:", initF.shape)
pts1_selected, pts2_selected = epipolarMatchGUI(I1, I2, F)

# # Print the selected points
print("Selected Points in Image 1:", pts1_selected)
print("Corresponding Points in Image 2:", pts2_selected)


# Load data from intrinsics.npy
intrinsics_data = np.load('../data/intrinsics.npy', allow_pickle=True).item()
K1 = intrinsics_data['K1']
K2 = intrinsics_data['K2']
# F = eightpoint(pts1, pts2, M)

# Call essentialMatrix function
E = essentialMatrix(F, K1, K2)

# Print the computed essential matrix
print("Computed Essential Matrix (E):")
print(E)
