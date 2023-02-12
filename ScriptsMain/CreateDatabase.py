import json
import os

import cv2
import numpy as np
from pathlib import Path

DEBUG_BLOB = False
MY_CAPS_IMGS_FOLDER = r"database/caps-s3"
DATABASE_FODLER = r"database/caps_db-s3"
MY_CAPS_IMGS_FOLDER_RGB = "../database/RGB-caps-s3"


def read_img(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path), 1)


def rgb_to_bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    """
    Given a tuple of colors it returns the same tuple but changing the order, this is because OpenCV uses BGR instead of RGB

    :param int r: value from 0 to 255 to represent red
    :param int g: int r: value from 0 to 255 to represent green
    :param int b: int r: value from 0 to 255 to represent blu
    :return: The tuple with the three colors
    """
    return tuple((b, g, r))


def transform_bgr_image_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Transforms the image to numpy rgb from bgr

    :param np.ndarray img: The original image
    :return: The image transformed to rgb
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_name_from_path(path: str) -> str:
    return path.split("/")[-1]


def resize_img_pix_with_name(cap_path, path_output, pix):
    cap_name = get_name_from_path(cap_path)
    lst_name_cap = cap_name.split(".")
    cap_name = lst_name_cap[0] + "_{}".format(str(pix)) + "." + lst_name_cap[-1]
    output = resize_image_and_save(cap_path, pix, pix, path_output, cap_name)
    return output


def resize_image(src, factor):
    height, width = src.shape[:2]
    return cv2.resize(src, (int(src * factor), int(height * factor)))


def resize_image_and_save(path_to_image, width, height, where_save, name_output):
    src = read_img(path_to_image)
    resized = cv2.resize(src, (width, height))
    output = where_save + name_output
    cv2.imwrite(output, resized)
    return output


def resize_all_images(path, output, size):
    files = os.listdir(path)
    for file in files:
        resize_img_pix_with_name(path + file, output, size)


def crate_db_for_cap(cap_name, folder: str):
    cap_path = os.path.join(folder, cap_name)

    cap_img = cv2.imread(cap_path)
    cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kps, dcps = sift.detectAndCompute(cap_img, None)

    keypoints_list = [[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in kps]

    dcps = dcps.tolist()[:200]

    entry = {
        "name": cap_name,
        "path": cap_path,
        "kps": keypoints_list,
        "dcps": dcps
    }
    cap_name = cap_name.split(".")[0]
    cap_result = os.path.join(DATABASE_FODLER, cap_name)

    with open('../' + cap_result + ".json", "w") as outfile:
        print("Writing:{}".format(cap_result))
        json.dump(entry, outfile)


def create_json_for_all_caps():
    path = Path(os.getcwd())
    path_caps = os.path.join(path.parent.absolute(), MY_CAPS_IMGS_FOLDER)

    entries = os.listdir(path_caps)
    for name_img in entries:
        crate_db_for_cap(name_img, path_caps)


# RGB Section for database

def get_all_rgb_images():
    entries = os.listdir(MY_CAPS_IMGS_FOLDER_RGB)
    dict_rgb = {}
    # Variable para comprobar la mascara aplicada a las imagenes. Se podría afinar más en algunas.
    DEBUG = True
    for name_img in entries:
        cap_str = os.path.join(MY_CAPS_IMGS_FOLDER_RGB, name_img)
        imagen = read_img(cap_str)
        imagen_rgb = transform_bgr_image_to_rgb(imagen)
        # Create a circular mask
        mask, center, radio = create_circular_mask(imagen_rgb)
        cv2.circle(mask, center, radio, (255, 255, 255), -1)
        # Apply the mask to the image
        image_mask = cv2.bitwise_and(imagen_rgb, imagen_rgb, mask=mask)

        r, g, b = cv2.split(image_mask)
        # Format for k-means,
        rgb_cap = (r, g, b)

        dict_rgb[cap_str] = rgb_cap

        if DEBUG:
            cv2.imshow("prueba", image_mask)
            cv2.waitKey(0)

    return dict_rgb


def create_clustering_kmeans():
    """
    Create k cluster using kmeans based on the components RGB. Returns a dictionary with the clusters and their corresponding images
    :return:
    :rtype:
    """

    dict_caps = get_all_rgb_images()
    # rgb_values = list(dict_caps.values())
    # rgb_matrix = np.array([list(img) for img in rgb_values])
    # kmeans = KMeans(n_clusters=5)
    # kmeans.fit(rgb_matrix)
    #
    # cluster_dict = {}
    #
    # for i, label in enumerate(kmeans.labels_):
    #     if label not in cluster_dict:
    #         cluster_dict[label] = []
    #     for key in dict_caps:
    #         if rgb_values[i] == dict_caps[key]:
    #             cluster_dict[label].append(key)
    #
    # print(cluster_dict)
    # return cluster_dict
    return dict_caps


# def predict_cluster_cap(kmean, imagen):
#     prediction =

def create_circular_mask(imagen):
    high, width, _ = imagen.shape
    center = (width // 2, high // 2)
    radio = min(high, width) // 2
    mask = np.zeros((high, width), np.uint8)
    return mask, center, radio


def main_rgb():
    get_all_rgb_images()


if __name__ == '__main__':
    create_json_for_all_caps()
