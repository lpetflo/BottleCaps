import os
from pathlib import Path
import cv2
import numpy as np


def get_name_from_path(path: str) -> str:
    return path.split("\\")[-1]


def resize_img_pix_with_name(cap_path, path_output, pix):
    cap_name = get_name_from_path(cap_path)
    lst_name_cap = cap_name.split(".")
    cap_name = lst_name_cap[0] + "_{}".format(str(pix)) + "." + lst_name_cap[-1]
    output = resize_image_and_save(path_to_image=cap_path, width=pix, height=pix, where_save=path_output, name_output=cap_name)
    return output


def read_img(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


# -- Resizing --
def resize_image(src, factor):
    height, width = src.shape[:2]
    new_size = (int(width * factor), int(height * factor))
    return cv2.resize(src, new_size)


def resize_image_and_save(path_to_image, width, height, where_save, name_output):
    src = read_img(path_to_image)
    resized = cv2.resize(src, (width, height))
    output = os.path.join(where_save, name_output)

    cv2.imwrite(output, resized)
    print(f'Resizing:{output}')
    return output


def resize_all_images(path, output, size):
    files = os.listdir(path)
    for file in files:
        resize_img_pix_with_name(os.path.join(path, file), output, size)


# -- Finish resize


def rgb_to_bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    """
    Given a tuple of colors it returns the same tuple but changing the order, this is because OpenCV uses BGR instead of RGB

    :param int r: value from 0 to 255 to represent red
    :param int g: int r: value from 0 to 255 to represent green
    :param int b: int r: value from 0 to 255 to represent blu
    :return: The tuple with the three colors
    """
    return tuple((b, g, r))


def main():
    cwd = Path(os.getcwd())
    path = os.path.join(cwd.parent.absolute(), r'database\1000-caps-s3-images')
    output = os.path.join(cwd.parent.absolute(), r'database\caps-resized')
    resize_all_images(path=path, output=output, size=125)


if __name__ == '__main__':
    main()
