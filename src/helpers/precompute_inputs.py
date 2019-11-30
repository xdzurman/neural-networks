import os
import logging

import numpy as np
import pandas as pd
from skimage import color
from skimage.transform import resize

import src.consts as c
from src.helpers.dataset_helpers import load_image
import tensorflow.keras.applications.inception_resnet_v2 as inception_resnet_v2
import tensorflow.keras.applications.vgg19 as vgg19


def calc_inception_embedding(img, inception):
    rgb_img = color.gray2rgb(color.rgb2gray(img))
    rgb_img_resize = resize(rgb_img, (299, 299, 3), mode="constant")
    rgb_img_resize = np.array([rgb_img_resize])
    rgb_img_resize = inception_resnet_v2.preprocess_input(rgb_img_resize)
    embed = inception.predict(rgb_img_resize)
    return embed[0]


def calc_vgg_embedding(img, vgg):
    rgb_img = color.gray2rgb(color.rgb2gray(img))
    rgb_img_resize = resize(rgb_img, (224, 224, 3), mode="constant")
    rgb_img_resize = np.array([rgb_img_resize])
    rgb_img_resize = vgg19.preprocess_input(rgb_img_resize)
    embed = vgg.predict(rgb_img_resize)
    return embed[0]


def precompute_inception_resnet(img_paths):
    logging.info('precomputing inception resnet v2 inputs')
    inception = inception_resnet_v2.InceptionResNetV2(weights="imagenet", include_top=True)

    embeddings = []
    for i, (img_path, image_name) in enumerate(img_paths):
        img = load_image(img_path) / 255
        embedding = calc_inception_embedding(img, inception)
        embeddings.append({"file_name": image_name, "embedding": embedding})
        if (i+1) % 100 == 0:
            print(i+1)

    df = pd.DataFrame(embeddings)
    df.to_csv("./precomputed_inputs/inception_res_net_v2.csv")


def precomputed_vgg19(img_paths):
    logging.info('precomputing vgg19 inputs')
    vgg = vgg19.VGG19(weights="imagenet", include_top=True)

    embeddings = []
    for i, (img_path, image_name) in enumerate(img_paths):
        img = load_image(img_path) / 255
        embedding = calc_vgg_embedding(img, vgg)
        embeddings.append({"file_name": image_name, "embedding": embedding})
        if (i+1) % 100 == 0:
            print(i+1)

    df = pd.DataFrame(embeddings)
    df.to_csv("./precomputed_inputs/vgg19.csv")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    path = c.DATASET_PATH
    file_paths = os.listdir(path)
    img_paths = [(path + filepath, filepath) for filepath in file_paths]
    # precompute_inception_resnet(img_paths)
    # precomputed_vgg19(img_paths)
