import os

import numpy as np
import pandas as pd
from skimage import color
from skimage.transform import resize
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2, preprocess_input)

import src.consts as c
from src.helpers.dataset_helpers import load_image


def calc_inception_embedding(img):
    rgb_img = color.gray2rgb(color.rgb2gray(img))
    rgb_img_resize = resize(rgb_img, (299, 299, 3), mode="constant")
    rgb_img_resize = np.array([rgb_img_resize])
    rgb_img_resize = preprocess_input(rgb_img_resize)
    embed = inception.predict(rgb_img_resize)
    return embed[0]


if __name__ == "__main__":
    path = c.DATASET_PATH
    file_paths = os.listdir(path)
    img_paths = [(path + filepath, filepath) for filepath in file_paths]
    inception = InceptionResNetV2(weights="imagenet", include_top=True)

    embeddings = []
    for i, (img_path, image_name) in enumerate(img_paths):
        img = load_image(img_path) / 255
        embedding = calc_inception_embedding(img)
        dic = {"file_name": image_name, "embedding": embedding}
        embeddings.append(dic)
        if (i+1) % 100 == 0:
            print(i+1)
    df = pd.DataFrame(embeddings)
    df.to_csv("./precomputed_inputs/inception_res_net_v2.csv")
