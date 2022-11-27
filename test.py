import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened


img = cv.imread('./data/honour_cropped.png',0)
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#sharpen_kernel = 1/3 * sharpen_kernel
#img = cv.filter2D(img, -1, sharpen_kernel)
img = unsharp_mask(img, (5, 5), 2.0, 8.0)

# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

erosion_size = 1
kernel = cv.getStructuringElement(0, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))

th1 = cv.erode(th1, kernel)
th2 = cv.erode(th2, kernel)
th3 = cv.erode(th3, kernel)

cv.imwrite('./out/sou_global.png', th1)
cv.imwrite('./out/sou_otsu.png', th2)
cv.imwrite('./out/sou_otsu_gaussian_gray.png', th3)


# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# for i in range(3):
#     plt.subplot(1,3,i+1),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()