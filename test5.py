import numpy as np
import cv2

image_paths = ['./data/winning_closed_hand_cropped.png', './data/test_hand_1.png']
images = []
for image_path in image_paths:
    images.append(cv2.imread(image_path))

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = images[0].reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)
# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 5
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(images[0].shape)

cv2.imshow('image', segmented_image)
cv2.waitKey(0)
