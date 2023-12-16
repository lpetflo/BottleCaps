from typing import List

import cv2
import numpy as np

from ScriptsMain.HTC import hough_transform_circle
from ScriptsMain.Blobs import get_avg_size_all_blobs

MAX_WIDTH_IMAGE = 1000
MAX_HEIGHT_IMAGE = 1000


# -- Resizing --
def resize_image(src, factor):
    height, width = src.shape[:2]
    new_size = (int(width * factor), int(height * factor))
    return cv2.resize(src, new_size)


def crop_image_into_rectangles(photo_image, rectangles):
    """
    Crop the image based on the rectangles, if the position is negative put it to zero

    :param np.ndarray photo_image: the original photo
    :param list[tuple[int, int, int, int]] rectangles: a list of tuples with the x,y and width and height position
    :return: list[np.ndarray, tuple[int, int, int, int]] Returns a list of list which contains the cropped image and the
     position on where it was cropped
    """
    cropped_images = []
    for x, y, w, h in rectangles:
        # Sometimes we have to guarantee that rectangle size is greater than 0
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        cropped_image = photo_image[y:y + h, x:x + w]
        if len(cropped_image) > 0:
            cropped_images.append((cropped_image, (x, y, w, h)))
    return cropped_images


def get_rectangles(circles: List) -> List:
    """
    Based in the center of the circle and the ratio, transform it into a rectangle so the image can be cropped

    :param list[tuple[nt,int,int]] circles: A list with tuples of the circles, x,y (center) and radius
    :return: Returns the list of rectangles transforming into width and height
    """
    rectangles = []
    for x, y, r in circles:
        x1 = x - r
        y1 = y - r
        width = r * 2
        height = r * 2
        rectangles.append((x1, y1, width, height))
    return rectangles


def preprocess_image_size(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for SIFT currently it resizes it

    :param np.ndarray img: Original image, preprocess it for SIFT
    :return: np.ndarray The image preprocessed for SIFT
    """
    height, width = img.shape[:2]
    size = height * width
    max_size_img = MAX_WIDTH_IMAGE * MAX_HEIGHT_IMAGE
    resized = img
    while size > max_size_img:
        resized = resize_image(resized, 0.66)
        height, width = resized.shape[:2]
        size = height * width
    return resized


def detect_caps(img) -> List:
    # Preprocess image
    processed_img = preprocess_image_size(img.copy())

    _, avg_size = get_avg_size_all_blobs(processed_img)
    cropped_images = []
    if avg_size != 0:
        _, circles = hough_transform_circle(processed_img, avg_size)
        # Get the positions of the rectangles
        rectangles = get_rectangles(circles)
        # Crop the images from the rectangles
        cropped_images = crop_image_into_rectangles(processed_img, rectangles)

        # Draw the detected caps on the image
        for rect in rectangles:
            x, y, w, h = rect  # Unpacking x, y, width, and height
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)  # Green rectangle with thickness 3

    return cropped_images, img  # Return the cropped images and the marked image

