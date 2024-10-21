from skimage.io import imread
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import regionprops_table
from pandas import DataFrame
import cv2
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt
rectangle = imread('c.jpg',0)

rectangle = cv2.cvtColor(rectangle, cv2.COLOR_RGB2GRAY)
print(rectangle.shape)

binary_mask = rectangle > 0.5
contours = find_contours(binary_mask, 0.5)
print(len(contours))
new_img = np.zeros((rectangle.shape[0], rectangle.shape[1]))
for contour in contours:
    contour = np.array(contour, dtype=int)
    new_img[contour[:, 0], contour[:, 1]] = 1



cv2.imshow('image', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the image with contours using matplotlib
plt.show()
# new_img = np.zeros((rectangle.shape[0], rectangle.shape[1]))
#
# # Plot each contour
# for contour in contours:
#     contour = np.array(contour, dtype=int)
#     new_img[contour[:, 0], contour[:, 1]] = 1
#     plt.plot(contour[:, 1], contour[:, 0], color='red')


