import cv2
import Tiles

root_dir = './data/'
images_files = ['man_cropped.png', 'pin_cropped.png', 'sou_cropped.png', 'honour_cropped.png']
image_paths = [root_dir + x for x in images_files]

out_paths = ['./data/tiles/man/', './data/tiles/pin/', './data/tiles/sou/', './data/tiles/honour/']
image_counts = [9, 9, 9, 7]

for i in range(len(image_paths)):
    image = cv2.imread(image_paths[i])
    preprocessed_image = Tiles.preprocess_image(image)

    card_contours = Tiles.find_cards(preprocessed_image)

    for j in range(len(card_contours)):
        symbol = Tiles.preprocess_card(card_contours[j], image)
        # result_path = out_paths[i] + str(j) + 'b.png'
        # cv2.imwrite(result_path, symbol)
