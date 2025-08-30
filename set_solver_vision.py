# Python-OpenCV SET Card Game Card Detector and SET Solver
#
# Author: Peter Jakubowski
# Date: 8/2/2023
# Description: Python script to solve the SET card game
# by detecting and classifying SET playing cards
# from an image using OpenCV.
#

# Import necessary packages
import sys
import argparse
from argparse import Namespace
from itertools import combinations
import set_card_detection
from typing import List
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(levelname)s: %(message)s')


def parse_args(args: List[str]) -> Namespace:
    # create an argument parser using argparse
    parser = argparse.ArgumentParser(description="solve the 'sets' in the image")
    # add an argument to the parser for the image path
    parser.add_argument("-i", "--image",
                        required=True,
                        help="path to input image",
                        type=str)
    # add an argument to the parser for annotating cards in the image
    parser.add_argument("-a", "--annotate",
                        help="skip solving and annotate all SET cards in image",
                        action="store_true")
    # add an argument to the parser for saving a copy of the output image,
    # either solved or annotated
    parser.add_argument("-s", "--save",
                        help="save a copy of the solved or annotated output image to images/solved/",
                        action="store_true")
    # parse the arguments in args
    args = parser.parse_args(args)

    return args


def main(args: List[str]):
    """
    Takes a path to an image of SET cards,
    detects the cards in the image and solves
    for all 'set' combinations. Displays the original
    image with rectangles drawn around cards that
    form a 'set'. Displays one solution at a time.
    """

    # parse arguments
    args = parse_args(args)
    # get image path from arguments
    img_path = args.image
    # get filename from image path
    filename = img_path.split('/')[-1]
    # open/read the image
    img = set_card_detection.read_image(img_path)
    # check if the image can be read
    if img is not None:
        # process image of all cards on the table.
        # return the open image and a list of card objects
        # of class SetCard, one for each of the cards in the image

        logging.info(f"Processing image: {filename}...")
        set_cards, img = set_card_detection.find_card_contours(img)

        logging.info("Predicting labels for cards...")
        # loop through all SET card objects in the list of cards
        # and get the labels for shapes on each card
        for card in set_cards[:]:
            # get the shapes on the cards and count them
            set_card_detection.get_shapes_on_cards(card)
            # get the labels for shape, pattern and color
            if card.number > -1:
                set_card_detection.get_labels(card)

        # filter cards and exclude any that have a label value equal to 'None'
        set_cards = [card for card in set_cards if -1 not in card.matrix()]

        # if argument to annotate an image is False, then solve the 'set'
        if not args.annotate:
            # list of all possible 'sets' solved with cards in play
            solved_sets = []

            # find all combinations of 3 cards
            # iterate over all combinations of cards and check
            # whether a combination is a 'set', add to the
            # list of solved 'sets' if a 'set' is found
            for combo in list(combinations(set_cards, 3)):
                if set_card_detection.is_set(combo):
                    solved_sets.append(combo)

            logging.info(f"Solved {len(solved_sets)} 'set{'s' if len(solved_sets) != 1 else ''}'")

            # print and show all the solved 'sets' that were found if any
            if len(solved_sets) > 0:
                #
                count = 1
                # loop through all 'set' solutions
                for set_solution in solved_sets:
                    frame_title = f"{count} of {len(solved_sets)} 'sets' in {filename}"
                    title = f"{count} of {len(solved_sets)} 'sets'\n\n"
                    logging.info(f"{count}) These three cards make a 'set':")
                    for i in range(len(set_solution)):
                        title += f"{i + 1}) {set_solution[i]}\n"
                        logging.info(f"{set_solution[i]}")
                    # get the annotated image with 'set' solution
                    annotate_img = set_card_detection.annotate_cards_image(set_solution, img, args.save,
                                                                           f"{filename.split('.')[0]}_Set{count}")
                    # show the annotated image
                    set_card_detection.show_image_opencv(frame_title, annotate_img)
                    count += 1

        # annotate and show all SET cards in the image
        else:
            # get the annotated image with all SET cards labeled
            annotate_img = set_card_detection.annotate_cards_image(set_cards, img, args.save,
                                                                   f"{filename.split('.')[0]}_annotated")
            # show the annotated image
            set_card_detection.show_image_opencv(filename, annotate_img)

    return


if __name__ == '__main__':
    main(sys.argv[1:])