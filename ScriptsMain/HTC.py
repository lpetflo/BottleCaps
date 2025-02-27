
import cv2
import numpy as np

DEBUG_HOUGH_TRANSFORM = 0
multiplier_left_max_radius = 0.8
multiplier_right_max_radius = 1


def combine_overlapping_circles(circles):
    circles = np.round(circles[0, :]).astype("int")
    combined_circles = []
    for (x, y, r) in circles:
        found_overlap = False
        for (cx, cy, cr) in combined_circles:
            if (x - cx) ** 2 + (y - cy) ** 2 < (r + cr) ** 2:
                found_overlap = True
                break
        if not found_overlap:
            combined_circles.append((x, y, r))
    return combined_circles


def hough_transform_circle(img: np.ndarray, max_radius: int) -> (np.ndarray, int):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=18,
                               minRadius=int(max_radius * multiplier_left_max_radius),
                               maxRadius=int(max_radius * multiplier_right_max_radius))
    circles = np.uint16(np.around(circles))

    circles = combine_overlapping_circles(circles)

    if DEBUG_HOUGH_TRANSFORM:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Draw combined circles on image
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.imshow("Hough-Transform-Debug", img)
        cv2.waitKey(0)

    return img, circles
