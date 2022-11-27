import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread('./data/sou_cropped.png',0)
# img = cv.medianBlur(img,3)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,14)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,10)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

kernel = np.ones((3, 3), np.uint8)
th3 = cv.erode(th3, kernel)


cv.imwrite('./out/sou_mean_gray.png', th2)
cv.imwrite('./out/sou_gaussian_gray.png', th3)
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()