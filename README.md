# SET-Card-Game-Solver

This is a Python program that solves the SET card game using OpenCV. The program opens an image of a game of SET (usually 12 cards), and displays the solution by drawing boxes around the three cards that make a SET.

What is SET? Set (stylized as SET or SET!) is a real-time card game designed by Marsha Falco in 1974 and published by Set Enterprises in 1991. The deck consists of 81 unique cards that vary in four features across three possibilities for each kind of feature: number of shapes (one, two, or three), shape (diamond, squiggle, oval), shading (solid, striped, or open), and color (red, green, or purple). Each possible combination of features (e.g. a card with three striped green diamonds) appears as a card precisely once in the deck. In the game, certain combinations of three cards are said to make up a "set". For each one of the four categories of features — color, number, shape, and shading — the three cards must display that feature as either a) all the same, or b) all different. Read more about the game on [Wikipedia](https://en.wikipedia.org/wiki/Set_(card_game)).

![Solved SET game](images/solved/solved.jpg)

***

## Algorithm Overview

The SET card game solver utilizes a computer vision pipeline to identify and decode each card from an input image, then iterates through all possible combinations of three cards to find valid sets. The process can be broken down into the following key stages:

### 1. Card Detection and Extraction

The initial step is to locate and isolate each individual card within the source image. This is accomplished through the following sequence of image processing operations:

* **Grayscale Conversion and Blurring:** The input image is first converted to grayscale to simplify the feature space. A Gaussian blur is then applied to reduce noise and smooth out minor details, which aids in the subsequent contour detection process.
* **Contour Detection:** The `cv2.findContours()` function is used to identify the outlines of all shapes in the processed image. These contours represent potential cards.
* **Card Isolation:** Each detected contour is then processed to extract the corresponding card. The program filters out contours that are too small to be cards and then performs a perspective transform (`cv2.warpPerspective()`) on the remaining contours. This corrects for any angular distortion and produces a straightened, top-down view of each card.

### 2. Shape Identification and Feature Extraction

Once the cards have been extracted, the program analyzes the contents of each card to determine its four defining attributes: number, shape, color, and shading.

* **Number:** The number of shapes on the card is determined by finding and counting the contours within the boundaries of the isolated card image.
* **Shape:** The shape of the symbols (diamond, oval, or squiggle) is identified using template matching. The program compares the detected shape contours to a set of pre-defined template images for each possible shape and selects the one with the highest correlation score.
* **Color:** The color of the shapes (red, green, or purple) is determined by applying color masks to the original, full-color card image. The program creates a separate mask for each of the three possible colors and then calculates which mask reveals the most non-zero pixels within the shape's contour. The color corresponding to the most effective mask is then assigned to the card.
* **Shading:** The shading of the shapes (solid, striped, or open) is determined by analyzing the pixel density within the shape's contour. A small region of interest (ROI) is sampled from the center of the shape, and the percentage of non-white pixels is calculated. This percentage is then compared against a set of predefined thresholds to classify the shading as solid, striped, or open.

### 3. Set Identification

With a complete list of all cards and their decoded attributes, the final step is to identify all valid sets. The program iterates through every possible combination of three cards and, for each combination, checks if it satisfies the two conditions for a valid SET:

1.  For each of the four attributes (number, shape, color, and shading), the three cards must be either **all the same** or **all different**.
2.  Both of these conditions must hold true for all four attributes simultaneously.

Any combination of three cards that meets these criteria is flagged as a valid set. The program then returns the original image with the identified sets highlighted.

***

## Usage

Provide a path to an image of SET cards.

```
python set_solver_vision.py -i images/sample/IMG_5254.JPG
```

```
usage: set_solver_vision.py [-h] -i IMAGE [-a] [-s]

solve the 'sets' in the image

optional arguments:
  -h, --help                show this help message and exit
  -i IMAGE, --image IMAGE   path to input image
  -a, --annotate            skip solving and annotate all SET cards in image
  -s, --save                save a copy of the solved or annotated
                            output image to images/solved/

```

## Files

* `images/` - Image data, including sample game images, solved and annotated card images.
* `shapes/` - Images of each variation of a shape that is on a SET card. Used for matching shapes.
* `notebooks/set_card_detection.ipynb` - Jupyter notebook that shows each step of the card detection algorithm. 
* `set_solver_vision.py` - Contains the main python script to solve the SET card game from an image of SET cards.
* `set_card_detection.py` - Class and Functions for set_solver_vision.py that perform various steps of the card detection algorithm.

## Dependencies

```
Python 3.9+
OpenCV 4.7.0
numpy 1.25.1
imutils 0.5.4
```

## Documentation

* OpenCV [documentation](https://docs.opencv.org/master/)
* opencv-python [PyPi](https://pypi.org/project/opencv-python/)
