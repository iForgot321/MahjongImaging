import numpy as np
import cv2


image_paths = ['./data/winning_closed_hand_cropped.png', './data/test_hand_1.png']
images = []
for image_path in image_paths:
    images.append(cv2.imread(image_path))

images[1] = images[1][1000:-1000, :]

gray1 = images[0]
gray2 = images[1]

thresh1 = 50
thresh2 = 150

canny = cv2.Canny(gray1, thresh1, thresh2)
canny2 = cv2.Canny(gray1, thresh1, thresh2, L2gradient=True)

canny3 = cv2.Canny(gray2, thresh1, thresh2)
canny4 = cv2.Canny(gray2, thresh1, thresh2, L2gradient=True)

hori1 = np.concatenate((canny, canny2), axis=0)
hori2 = np.concatenate((canny3, canny4), axis=0)

cv2.imshow('image', hori1)
cv2.waitKey(0)
cv2.imshow('image', hori2)
cv2.waitKey(0)