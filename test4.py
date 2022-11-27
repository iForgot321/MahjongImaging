import numpy as np
import cv2


image_paths = ['./data/winning_closed_hand_cropped.png', './data/test_hand_1.png']
images = []
for image_path in image_paths:
    images.append(cv2.imread(image_path))

images[1] = images[1][1000:-1000, :]


gray1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

bi_filter1 = cv2.bilateralFilter(gray1, 9, 75, 75)
bi_filter2 = cv2.bilateralFilter(gray2, 9, 75, 75)

cv2.imshow('image', bi_filter1)
cv2.waitKey(0)
cv2.imshow('image', bi_filter2)
cv2.waitKey(0)


thresh1 = 25
thresh2 = 150

canny = cv2.Canny(bi_filter1, thresh1, thresh2)
canny2 = cv2.Canny(bi_filter1, thresh1, thresh2, L2gradient=True)

canny3 = cv2.Canny(bi_filter2, thresh1, thresh2)
canny4 = cv2.Canny(bi_filter2, thresh1, thresh2, L2gradient=True)

# _, thresh1 = cv2.threshold(bi_filter1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, thresh2 = cv2.threshold(bi_filter2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

hori1 = np.concatenate((canny, canny2), axis=0)
hori2 = np.concatenate((canny3, canny4), axis=0)

cv2.imshow('image', hori1)
cv2.waitKey(0)
cv2.imshow('image', hori2)
cv2.waitKey(0)
# blur = cv2.GaussianBlur(thresh, (5, 5), 0)
# cv2.imshow('image', blur)
# cv2.waitKey(0)

erosion_size = 3
kernel = cv2.getStructuringElement(2, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))

eroded1 = cv2.dilate(canny, kernel)
eroded2 = cv2.dilate(canny3, kernel)

cv2.imshow('image', eroded1)
cv2.waitKey(0)
cv2.imshow('image', eroded2)
cv2.waitKey(0)
