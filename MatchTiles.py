import numpy as np
import cv2
import Tiles


def index_to_tile_string(tile_index):
    tile_type = tile_index // 9
    tile_value = tile_index % 9
    if tile_type == 3:
        return honour_names[tile_value]
    else:
        return str(tile_value + 1) + images_dirs[tile_type]


similarity_threshold = 0.5
ratio_bound = 0.1
border_ratios = [0.05, 0.075, 0.1, 0.15]  # TODO: implement 4 way border variables !!!!!!!  and percentage red for dora
honour_names = ['south', 'west', 'north', 'green', 'red', 'white', 'east']

reference_dir = './data/tiles/'
images_dirs = ['man', 'pin', 'sou', 'honour']
image_counts = [9, 9, 9, 7]

reference_symbols = []
for i in range(len(images_dirs)):
    for j in range(image_counts[i]):
        input_path = reference_dir + images_dirs[i] + '/' + str(j+1) + '.png'
        tile_image = cv2.imread(input_path, 0)
        reference_symbols.append(tile_image)

# image_paths = ['./data/winning_closed_hand.png', './data/test_hand_1.png', './data/test_hand_2.png', './data/test_hand_3.png', './data/test_hand_4.png']
image_paths = ['./data/winning_closed_hand_cropped.png']

for image_path in image_paths:
    image = cv2.imread(image_path)
    preprocessed_image = Tiles.preprocess_image(image)
    contours = Tiles.find_cards(preprocessed_image)
    boxes = [np.int0(cv2.boxPoints(cv2.minAreaRect(x))) for x in contours]

    backup = image.copy()
    cv2.drawContours(backup, contours, -1, (0, 255, 0), 3)
    cv2.imshow('image', backup)
    cv2.waitKey(0)

    backup2 = image.copy()
    cv2.drawContours(backup2, boxes, -1, (0, 255, 0), 3)
    cv2.imshow('image', backup2)
    cv2.waitKey(0)

    hand = []
    for i in range(len(contours)):
        test_symbols = []
        for border_ratio in border_ratios:
            test_symbols.append(Tiles.preprocess_card(contours[i], image, border_ratio))

        similarity, index = Tiles.match_symbols(test_symbols, reference_symbols, ratio_bound)

        if similarity > similarity_threshold:
            hand.append(index_to_tile_string(index))

        # print(similarity)
        #for test_symbol in test_symbols:
        #cv2.imshow('image', test_symbol)
        #cv2.waitKey(0)
        cv2.imshow('image', reference_symbols[index])
        cv2.waitKey(0)

    print(hand)
