import math

import numpy as np
import cv2
from scipy import signal

# Constants
CARD_MAX_AREA = 150000
CARD_MIN_AREA = 30000


def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 100, 200)
    canny2 = cv2.Canny(gray, 100, 200, L2gradient=True)

    hori = np.concatenate((canny, canny2), axis=0)

    cv2.imshow('image', hori)
    cv2.waitKey(0)

    _, thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('image', thresh)
    cv2.waitKey(0)

    # blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    # cv2.imshow('image', blur)
    # cv2.waitKey(0)

    erosion_size = 4
    kernel = cv2.getStructuringElement(2, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))

    eroded = cv2.erode(thresh, kernel)

    cv2.imshow('image', eroded)
    cv2.waitKey(0)

    return eroded


def preprocess_image2(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpen = unsharp_mask(gray, (5, 5), 1.0, 12.0)

    cv2.imshow('image', sharpen)
    cv2.waitKey(0)

    _, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('image', thresh)
    cv2.waitKey(0)

    # blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    # cv2.imshow('image', blur)
    # cv2.waitKey(0)

    erosion_size = 4
    kernel = cv2.getStructuringElement(2, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))

    eroded = cv2.erode(thresh, kernel)

    cv2.imshow('image', eroded)
    cv2.waitKey(0)

    return eroded


def find_cards(thresh_image):
    contours, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_result = []
    for i in range(len(contours)):
        size = cv2.contourArea(contours[i])
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.05 * peri, True)

        if (size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and len(approx) == 4 and hier[0][i][3] == -1:
            contours_result.append(contours[i])

    return contours_result


def preprocess_card(contour, image, border_ratio=0.05):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = cv2.minAreaRect(contour)

    crop = crop_image(rect, gray)
    height, width = crop.shape
    if height < width:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    whiten_border(thresh, border_ratio)

    symbol = isolate_symbol(thresh)
    return symbol


def match_symbols(symbols, reference_symbols, ratio_bound):
    best_similarity = -2
    index = -1
    for symbol in symbols:
        height, width = symbol.shape
        ratio = height / width

        for j in range(len(reference_symbols)):
            source_height, source_width = reference_symbols[j].shape
            source_ratio = source_height / source_width

            if (1 - ratio_bound) * source_ratio <= ratio <= (1 + ratio_bound) * source_ratio:
                resized = cv2.resize(symbol, (source_width, source_height))
                rotated_180 = cv2.rotate(resized, cv2.ROTATE_180)

                similarity = normxcorr2D(resized, reference_symbols[j])[0, 0]
                similarity = max(similarity, normxcorr2D(rotated_180, reference_symbols[j])[0, 0])
                if similarity > best_similarity:
                    best_similarity = similarity
                    index = j

    return best_similarity, index


def isolate_symbol(image):
    inverse = 255 - image.copy()  # taking inverse of the input image
    contours, _ = cv2.findContours(inverse, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        left, right, top, bottom = get_boundest_box(contours)
        return image[top:bottom, left:right]
    else:
        return image


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def crop_image(rect, src):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(src, M, (width, height))
    return warped


def whiten_border(image, percent):
    height, width = image.shape
    l_height = int(percent * height)
    u_height = int((1-percent) * height)
    l_width = int(percent * width)
    u_width = int((1-percent) * width)
    for i in range(height):
        for j in range(width):
            if i <= l_height or i >= u_height or j <= l_width or j >= u_width:
                image[i, j] = 255


def get_boundest_box(contours):
    left = math.inf
    right = -1
    top = math.inf
    bottom = -1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x < left:
            left = x
        if x + w > right:
            right = x + w
        if y < top:
            top = y
        if y + h > bottom:
            bottom = y + h
    return left, right, top, bottom


def normxcorr2D(image1, image2):
    assert image1.shape == image2.shape

    matrix1 = image1 - np.mean(image1)
    norm1 = math.sqrt(np.sum(np.square(matrix1)))
    if norm1 == 0:
        norm1 = 1
    matrix1 = matrix1 / norm1

    matrix2 = image2 - np.mean(image2)
    norm2 = math.sqrt(np.sum(np.square(matrix2)))
    if norm2 == 0:
        norm2 = 1
    matrix2 = matrix2 / norm2

    result = signal.correlate2d(matrix1, matrix2, mode='valid')
    return result
