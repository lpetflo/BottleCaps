import json
import math
import os
import shutil
from typing import Dict, List

import keras
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.layers import Dense, Flatten
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from PIL import Image

from Pinecone import PineconeContainer, image_to_vector
from ScriptsMain.utilsFun import read_img_from_path

PROJECT_PATH = os.getcwd()


def create_training_folder():
    path_all_images = os.path.join(PROJECT_PATH, r'database\caps-resized')
    folder_create = os.path.join(PROJECT_PATH, 'training')

    names_images = os.listdir(path=path_all_images)
    for name in names_images:
        folder_name = name.split('.')[0]
        folder_result = os.path.join(folder_create, folder_name)

        if not os.path.exists(folder_result):
            os.mkdir(folder_result)
            shutil.copy(os.path.join(path_all_images, name), os.path.join(folder_result))


def create_model():
    img_size = 224
    num_classes = 110
    embedding_size = 64

    model = Sequential()

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3), pooling='max')
    model.add(base_model)

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))  # Add fully connected layers
    model.add(Dense(embedding_size, activation='linear'))  # Embedding layer with desired size

    return model


def generate_vector_database(pinecone_container, model: keras.Sequential):
    root_dir = os.path.join(PROJECT_PATH, 'training')
    batch_size = 64
    img_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,  # Random rotation between -20 to +20 degrees
        zoom_range=0.2,  # Random zoom between 0.8 to 1.2
        horizontal_flip=True  # Random horizontal flipping
    )
    datagen = img_gen.flow_from_directory(root_dir,
                                          target_size=(224, 224),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False)

    num_images = len(datagen.filepaths)
    num_epochs = int(math.ceil(num_images / batch_size))

    feature_list = model.predict(datagen, steps=num_epochs)

    json_path = os.path.join(PROJECT_PATH, 'vector_database_pinecone.json')

    json_object = {
        "vectors": [],
        "namespace": "bottle_caps"
    }

    for i in range(0, len(feature_list)):
        cap_info = {
            'id': datagen.filenames[i],
            'values': feature_list[i].tolist()
        }
        pinecone_container.upsert_to_pinecone(vector=cap_info)
        json_object['vectors'].append(cap_info)

    with open(json_path, 'w') as json_file:
        json.dump(json_object, json_file)


def save_model(model, path):
    model.save(path)


def get_model() -> keras.Sequential:
    path = os.path.join(PROJECT_PATH, 'model')
    return load_model(path)


def generate_all(pinecone_container: PineconeContainer):
    create_model()
    model = get_model()
    generate_vector_database(pinecone_container=pinecone_container, model=model)


def show_similar_images(org_img: str, match_result: Dict):
    images = [os.path.join(PROJECT_PATH, 'training', match['id']) for match in match_result['matches']]
    values = ["{:.3f}".format(match['score']) for match in match_result['matches']]
    show_images(images, org_img, values)


# Auxiliary function so i can know what is the similarity
def show_images(images, specific_image_path, values):
    num_images = len(images) + 1  # Add 1 for the specific image
    rows = int(np.ceil(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))

    fig, axes = plt.subplots(rows, cols)

    # Display the specific image first
    specific_image = Image.open(specific_image_path)
    axes[0, 0].imshow(specific_image)
    axes[0, 0].axis('off')

    for i, ax in enumerate(axes.flatten()[1:]):
        if i < num_images - 1:
            image_path = images[i]
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')
            ax.text(0.5, -0.2, values[i], ha='center', transform=ax.transAxes)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    pinecone_container = PineconeContainer()
    model: keras.Sequential = get_model()

    path = os.path.join(PROJECT_PATH, r'database/test-images/one-image/7.png')
    img = read_img_from_path(path)

    vector = image_to_vector(img=img, model=model)
    result = pinecone_container.query_database(vector=vector)
    print(result)
    show_similar_images(path, result)


if __name__ == '__main__':
    # create_training_folder()
    # pinecone_container = PineconeContainer()
    # generate_all(pinecone_container)
    main()
