# Python-OpenCV SET Card Game Card Detector and SET Solver
#
# Author: Peter Jakubowski
# Date: 8/2/2023
# Description: Functions for set_solver_vision.py that perform
# various steps of the card detection algorithm.
#

# Import necessary packages
import os
import cv2 as cv
import numpy as np
from imutils import paths

# dictionary of card attributes converting integer values from card vectors to strings
SET_MAP = {"shape": {-1: "none", 0: "diamond", 1: "oval", 2: "squiggle"},
           "color": {-1: "none", 0: "red", 1: "green", 2: "purple"},
           "shading": {-1: "none", 0: "solid", 1: "open", 2: "stripe"},
           "number": {-1: "none", 0: "one", 1: "two", 2: "three"},
           }

LABELS_MAP = {"shape": {"none": -1, "diamond": 0, "oval": 1, "squiggle": 2},
              "color": {"none": -1, "red": 0, "green": 1, "purple": 2},
              "shading": {"none": -1, "solid": 0, "open": 1, "stripe": 2},
              "number": {"none": -1, "one": 0, "two": 1, "three": 2}
              }


class SetCard:
    # structure to store information about SET cards

    def __init__(self):
        self.image = []  # image of individual card
        self.card_contour = []  # contour of card
        self.bbox = []  # bounding rectangle of card  on image (x, y, w, h)
        self.box = []  # coordinates of four corners for minimum area bounding box of card
        self.shapes_contours = []  # contours of shapes on card
        self.shapes_boxes = []  # bounding rectangle (box) of each shape on card (x, y, w, h)

        # card labels
        self.shape = -1  # 0: "diamond", 1: "oval", 2: "squiggle"
        self.color = -1  # 0: "red", 1: "green", 2: "purple"
        self.shading = -1  # 0: "solid", 1: "outline", 2: "stripe"
        self.number = -1  # 0: "one", 1: "two", 2: "three"

    def matrix(self):
        # returns the card labels in an array
        return [self.shape, self.color, self.shading, self.number]

    def __str__(self):
        # format a string with card's label values
        shape = SET_MAP['shape'][self.shape]
        color = SET_MAP['color'][self.color]
        shading = SET_MAP['shading'][self.shading]
        number = SET_MAP['number'][self.number]

        return f"Card with {number} {color} {shape} shape{'s' if self.number > 0 else ''} in {shading} shading."


def is_set(set_cards):
    # decision function that tells whether three given cards make a 'set',
    # boolean function for judging a combination of three cards

    # make sure there are exactly three cards
    if len(set_cards) != 3:
        print(f"Incorrect number of cards, there must be three, {len(set_cards)} were provided.")
        return False

    cards_make_set = True
    # initialize empty set collections for each of the four labels
    shapes = set()
    colors = set()
    shadings = set()
    numbers = set()
    # add card labels to set collections from each of the three cards
    for c in set_cards:
        shapes.add(c.shape)
        colors.add(c.color)
        shadings.add(c.shading)
        numbers.add(c.number)
    # check the length of all set collections, we do not have a 'set'
    # if any set collection is of length two, the length of
    # each set collection must be 1 or 3, meaning all labels must be
    # the same (length 1) or unique (length 3).
    if len(shapes) == 2:
        cards_make_set = False
    elif len(colors) == 2:
        cards_make_set = False
    elif len(shadings) == 2:
        cards_make_set = False
    elif len(numbers) == 2:
        cards_make_set = False

    return cards_make_set


# Helper functions

def read_image(image_path):
    # Function for reading an image,
    # takes a path to input image and reads the image

    # check if the path exists
    if os.path.exists(image_path):
        # open the image
        image = cv.imread(image_path)
        if image is None:
            print(f"{image_path} image can't be read")
    else:
        print(f"{image_path} Invalid file path")
        image = None

    return image


def resize_image(image, size=300):
    # Function for rescaling the width and height
    # of an image to keep aspect ratio. Size is
    # the desired length of the longest edge in pixels.

    # get image width
    width = image.shape[1]
    # get image height
    height = image.shape[0]

    # check if the image is vertical,
    # height is the longest edge
    if height > width:
        # set height to size
        h = size
        # determine the ratio for resizing
        ratio = height / size
        # calculate new width by dividing by ratio
        w = int(width / ratio)
    # check if the image is horizontal,
    # width is the longest edge
    elif height < width:
        # set width to size
        w = size
        # determine the ratio for resizing
        ratio = width / size
        # calculate new height by dividing by ratio
        h = int(height / ratio)
    # if image is not vertical or horizontal,
    # image must be square
    else:
        # set width and height to size
        w = h = size
    # return the new width and height
    return w, h


def rotate_card(image, rectangle):
    # crops a rotated rectangle from a minimum area bounding box
    # https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv.boxPoints(rectangle)
    box = np.intp(box)

    # get width and height of the detected rectangle
    width = int(rectangle[1][0])
    height = int(rectangle[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    transformation_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    rotated_card = cv.warpPerspective(image, transformation_matrix, (width, height))

    # rotate the image 90 degrees to be horizontal if orientation is vertical
    if rotated_card.shape[0] > rotated_card.shape[1]:
        rotated_card = cv.rotate(rotated_card, cv.ROTATE_90_CLOCKWISE)

    return rotated_card


def pad_box(box, img_shape):
    # adds 5 pixels to each box dimension
    # if coordinates fit in the image

    # unpack the box tuple
    x, y, w, h = box
    # set x1 and y1 to the minimum bounding value
    x1 = y1 = 0
    # set x2 and y2 to the maximum bounding value
    x2, y2 = img_shape[1], img_shape[0]
    # check if y - 5 is in bounds
    if y - 5 > 0:
        # set y1 to y - 5 if > 0, otherwise y1 = 0
        y1 = y - 5
    # check if y + h + 5 is in bounds
    if y + h + 5 < img_shape[0]:
        # set y2 to y + h + 5 if < img_shape[0], otherwise y2 = img_shape[0]
        y2 = y + h + 5
    # check if x - 5 is in bounds
    if x - 5 > 0:
        # set x1 to x - 5 if > 0, otherwise x1 = 0
        x1 = x - 5
    # check if x + w + 5 is in bounds
    if x + w + 5 < img_shape[1]:
        # set x2 to x + w + 5 if < img_shape[1], otherwise x2 = img_shape[1]
        x2 = x + w + 5

    return x1, x2, y1, y2

# Image processing
# Find outer contours of cards


def find_card_contours(image):
    # Function for finding the contours of cards.
    # Reads an image, then resizes, blurs, thresholds,
    # and finds the outer contours of all cards in the image.
    # Returns a list of SET card objects and the open image of all card.

    # get the width and height for image to be resized
    w, h = resize_image(image, size=2000)
    # resize the image
    image = cv.resize(image, (w, h), cv.INTER_AREA)
    # make sure the image is horizontal, rotate 90 deg if vertical
    if w < h:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    # convert the image color from BGR to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # convert the image color from BGR to GRAY
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # blur the image
    image_blur = cv.GaussianBlur(image_gray, (3, 3), 0)
    # threshold the image
    ret, thresh = cv.threshold(image_blur, 180, 255, 0)
    # get the contours of the image
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # create card objects from contours, get bounding boxes and crop cards
    cards = []  # list of card objects
    for cnt in contours:
        # get the area of the contour
        area = cv.contourArea(cnt)
        # check if the area is large enough to be a card
        if area > 60000:
            # create a new card object of class SetCard
            card = SetCard()
            # save the contour of the card
            card.card_contour = cnt
            # get the bounding rectangle (box) of contour
            card.bbox = [cv.boundingRect(cnt)]
            # get minimum area rectangle from contour
            rect = cv.minAreaRect(cnt)
            # crop and rotate card from image
            rotated_card = rotate_card(image, rect)
            # resize card to 300 x 200 pixels
            card.image = cv.resize(rotated_card, (300, 200), cv.INTER_LINEAR)
            # get corners of min area bounding box from rectangle
            box = cv.boxPoints(rect)
            # save bounding box of entire card
            card.box = np.intp(box)
            # save card to list of cards
            cards.append(card)

    # print the number of cards found
    print(f"Found {len(cards)} cards")

    return cards, image


# Find contours of shapes on cards

def get_shapes_on_cards(card, area_thresh=4000):
    # Takes a card object of class SetCard,
    # finds all shapes on a card.
    # Modify the card object by updating the
    # shape contours and shape bounding boxes lists,
    # as well as the card label number.

    # copy the image of the card from object
    card_img = card.image.copy()
    # convert the image color to gray
    gray_img = cv.cvtColor(card_img, cv.COLOR_RGB2GRAY)
    # blur the image
    blur_img = cv.GaussianBlur(gray_img, (5, 5), cv.BORDER_DEFAULT)
    # find the optimal threshold value with Otsu's thresholding
    ret, otsu = cv.threshold(blur_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # threshold the image using Otsu's threshold value
    ret, thresh_img = cv.threshold(blur_img, ret + 10, 255, cv.THRESH_BINARY_INV)
    # find the contours of shapes in the image (on the card)
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # set card shapes (shape contours) and shape boxes to empty list
    card.shapes_contours = []
    card.shapes_boxes = []
    # loop through all the contours and append to list if area meets threshold
    for cnt in contours:
        # get the area of the contour
        area = cv.contourArea(cnt)
        if area > area_thresh:
            x, y, w, h = cv.boundingRect(cnt)
            card.shapes_contours.append(cnt)
            card.shapes_boxes.append((x, y, w, h))
    # set card label number to the number of shapes found - 1
    if len(card.shapes_contours) < 4:
        card.number = len(card.shapes_contours) - 1

    return


# Shape Detection
# load images of shapes from disk.
# images represent the combinations of
# shape (diamond, oval, squiggle) and shading (solid, open, stripe).

# path to the directory with images of shapes for comparing
SHAPES_DIR = "shapes"

# list of paths to all images of shapes
shape_paths = list(paths.list_images(SHAPES_DIR))


def load_shapes_train(shapes_train):
    # function that reads the images of shapes from disk
    # and returns lists of shape labels, images of shapes,
    # and outer contours of shapes. Images loaded from disk
    # are binary and have had thresholding performed.

    labels = []  # list of labels (shape and shading)
    shape_images = []  # list of images (shapes)
    shape_contours = []  # list of outer contours of shapes

    for path in shapes_train:
        # extract the shape and shading label from path and filename
        shape, shading = path.split("/")[-1].strip(".jpg").split("_")
        # save the shape and shading labels to list of labels
        labels.append((shape, shading))
        # read the image from disk
        img = cv.imread(path)
        # convert the color from BGR to GRAY
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # finds the outer contours in the image
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # loops through the contours and saves contours with an area greater than 4000 to list of contours
        for cnt in contours:
            # get the area of the contour
            area = cv.contourArea(cnt)
            if area > 4000:
                shape_contours.append(cnt)
        # save the image to the list of images
        shape_images.append(img)

    return labels, shape_images, shape_contours


shapesTrainLabels, shapesTrain, contoursTrain = load_shapes_train(shape_paths)


def compare_shapes(labels, shape_images, shape_contours, cur_contour):
    # function uses cv.matchShapes and compares
    # the current shape contour on the card
    # to all shape contours in the training set.
    # returns the best match for shape "diamond", "oval", "squiggle"

    # number of shapes in the training set
    n = len(shape_images)
    # array to keep results of contour comparisons
    contours_results = [0] * n
    # loop through all the shapes in the training set
    # and compare them to the current shape contour
    for i in range(n):
        # select ith contour from training set
        cnt = shape_contours[i]
        # compare training contour to current contour
        contour_res = cv.matchShapes(cnt, cur_contour, 1, 0.0)
        # save result of contour comparison
        contours_results[i] = contour_res
    # choose the index with the minimum score
    contour_idx = np.argmin(contours_results)
    # choose the best contour to predict the shape label
    shape_prediction = labels[contour_idx][0]

    return shape_prediction


def get_shape_labels(card):
    # function takes a SetCard object and
    # sets its shape label

    # get the contour of the first shape on the card
    if len(card.shapes_contours) > 0:
        contour = card.shapes_contours[0]
        # get the shape label as a string
        shape_prediction = compare_shapes(shapesTrainLabels,
                                          shapesTrain,
                                          contoursTrain,
                                          contour)
        # set the shape label 0:"diamond", 1:"oval", 2:"squiggle"
        card.shape = LABELS_MAP["shape"][shape_prediction]

    return


# Shading Detection
# Detect the shading of shapes on Set cards.

def check_shading(thresh_img):
    # takes a binary image (threshold) of a shape from a card
    # returns the shading label as a string: "solid", "open", "stripe".

    # find the coordinates for the middle of the image
    x, y = thresh_img.shape[1] // 2, thresh_img.shape[0] // 2
    # slice the image, get a patch from the middle of the image of size 20x20
    patch = thresh_img[y - 10:y + 10, x - 10:x + 10]
    # average value of pixels in the patch
    patch_val = int(np.mean(patch))
    # check for the shading using the average of pixel values in the patch
    # a solid shading is completely white, all pixels are 255
    if patch_val == 255:
        shading_label = "solid"
    # an open shading is completely black, all pixels are 0
    elif patch_val == 0:
        shading_label = "open"
    # a stripe shading is both white and black,
    # 0 < average pixel values in patch < 255
    else:
        shading_label = "stripe"

    return shading_label, patch_val


def get_shading_labels(card):
    # function takes a SetCard object and
    # sets its shading label

    # read the image of the card
    img = card.image
    # get the bounding box for the first
    # shape on the card (x, y, w, h)
    if len(card.shapes_boxes) > 0:
        box = card.shapes_boxes[0]
        # unpack the box tuple
        # x, y, w, h = box
        # slice the image, crop to first shape on the car
        x1, x2, y1, y2 = pad_box(box, img.shape)
        # shape_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
        shape_img = img[y1:y2, x1:x2]
        # blur the image of the shape
        blur = cv.GaussianBlur(shape_img, (3, 3), 0)
        # convert the image color from RGB to GRAY
        gray_img = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
        # threshold the image
        ret, thresh_img = cv.threshold(gray_img, 215, 255, cv.THRESH_BINARY_INV, 0)
        # get the shading label as a string
        shading_label, patch_val = check_shading(thresh_img)
        # set the shading label 0:"solid", 1:"open", 2:"stripe"
        card.shading = LABELS_MAP["shading"][shading_label]

    return


# Color Detection
# Detect the color of shapes on SET cards.

def mask_score(mask):
    # Function calculates the proportion of
    # pixels in the mask representing a color
    # and returns the float value used for scoring

    # count the number of pixels with value 255
    count = 0
    # number of rows
    n = len(mask)
    # number of columns
    m = len(mask[0])
    # loop through all pixels in the mask
    for i in range(n):
        for j in range(m):
            # check if pixel value equals 255
            if mask[i][j] == 255:
                count += 1
    # total number of pixels in mask
    pixels = n * m
    # calculate the score by dividing the count of pixels
    # equal to 255 by the total number of pixels in the maks
    score = count / pixels

    # print report
    # print(f"found {count} {color} pixels out of {pixels} total pixels")
    # print(f"{np.round((count / pixels) * 100, 4)}% of pixels are {color}")

    return score


def check_colors(hsv):
    # function takes an image of a shape in hsv color,
    # detects the color of the shape using cv.inRange
    # for three colors: red, green, and purple.
    # returns the integer label 0: "red", 1: "green", 2: "purple"

    # ordered list of colors to detect
    # colors = ["red", "green", "purple"]
    # list of inRange color masks
    masks = []
    # list of color scores
    scores = [0.0] * 3
    # boundaries of red, green, and purple is hsv color
    color_boundaries = [([0, 50, 64], [30, 255, 255]),
                        ([50, 10, 64], [80, 255, 255]),
                        ([130, 10, 64], [160, 255, 255])]
    # loop over the boundaries
    for i, boundary in enumerate(color_boundaries):
        # create NumPy arrays from the boundaries
        lower = np.array(boundary[0], dtype="uint8")
        upper = np.array(boundary[1], dtype="uint8")
        # find the colors within the specified boundaries and apply the mask
        mask = cv.inRange(hsv, lower, upper)
        # append the resulting mask to the list of masks
        masks.append(mask)
        # calculate the score of the mask
        scores[i] = mask_score(mask)

    # result is the index of the maximum score
    res = np.argmax(scores)

    return res


def get_color_labels(card):
    # function takes a SetCard object and
    # sets its color label

    # read the image of the card
    img = card.image
    # get the bounding box for the first
    # shape on the card (x, y, w, h)
    if len(card.shapes_boxes) > 0:
        box = card.shapes_boxes[0]
        # slice the image, crop to first shape on the card
        x1, x2, y1, y2 = pad_box(box, img.shape)
        # shape_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
        shape_img = img[y1:y2, x1:x2]
        # shape_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
        # blur the image of the shape
        blur = cv.GaussianBlur(shape_img, (3, 3), 0)
        # convert the image color from RGB to HSV
        hsv = cv.cvtColor(blur, cv.COLOR_RGB2HSV)
        # detect the color of the shape from HSV image and
        # set the color label 0: "red", 1: "green", 2: "purple"
        card.color = check_colors(hsv)

    return


def get_labels(card):
    # function takes a SetCard object and
    # sets its shape, shading, and color labels

    if len(card.shapes_contours) > 0:
        # read the image of the card
        img = card.image
        # get the contour for the first
        # shape on the card
        contour = card.shapes_contours[0]
        # get the shape label as a string
        shape_prediction = compare_shapes(shapesTrainLabels,
                                          shapesTrain,
                                          contoursTrain,
                                          contour)
        # set the shape label 0:"diamond", 1:"oval", 2:"squiggle"
        card.shape = LABELS_MAP["shape"][shape_prediction]
        # get the bounding box for the first
        # shape on the card (x, y, w, h)
        box = card.shapes_boxes[0]
        # unpack and pad the box
        x1, x2, y1, y2 = pad_box(box, img.shape)
        # slice the image, crop to first shape on the card
        shape_img = img[y1:y2, x1:x2]
        # blur the image of the shape
        blur = cv.GaussianBlur(shape_img, (3, 3), 0)
        # convert the image color from RGB to GRAY
        gray_img = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
        # threshold the image
        ret, thresh_img = cv.threshold(gray_img, 215, 255, cv.THRESH_BINARY_INV, 0)

        # get the shading label as a string
        shading_label, patch_val = check_shading(thresh_img)
        # set the shading label 0:"solid", 1:"open", 2:"stripe"
        card.shading = LABELS_MAP["shading"][shading_label]

        # convert the image color from RGB to HSV
        hsv = cv.cvtColor(blur, cv.COLOR_RGB2HSV)
        # detect the color of the shape from HSV image and
        # set the color label 0: "red", 1: "green", 2: "purple"
        card.color = check_colors(hsv)

    return

# Annotate the original image


def annotate_cards_image(set_cards, annotate_img, save, name):
    # convert the color of the image from RGB to BGR
    annotate_img = cv.cvtColor(annotate_img, cv.COLOR_RGB2BGR)
    # loop through all card objects in list
    for card in set_cards:
        # draw boxes around cards
        cv.drawContours(annotate_img, [card.box], -1, (0, 255, 0), 5)
        # build the string for putting text on the image
        text = ""
        for i, key in enumerate(SET_MAP.keys()):
            text += SET_MAP[key][card.matrix()[i]].title()
            if i < len(SET_MAP.keys()) - 1:
                text += " "
        # get x, y coordinates for text
        x, y, w, h = card.bbox[0]
        # get text size
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 1.3, 2)
        dim = text_size[0]
        baseline = text_size[1]
        # Use text size to create a black rectangle
        cv.rectangle(annotate_img, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv.FILLED)
        # put text labels on the image
        cv.putText(annotate_img, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 255), 2)

    if save:
        # write the solved image to disk
        cv.imwrite(f"images/solved/{name}.jpg", annotate_img)

    return annotate_img


def show_image_opencv(frame_title, annotate_img):
    # show the image with opencv
    cv.imshow(frame_title, annotate_img)
    # wait for any key to be pressed
    cv.waitKey(0)
    # closing all open windows
    cv.destroyAllWindows()
    return
