import os
import numpy as np
import tensorflow as tf

from skimage import color
from tensorflow.keras.preprocessing.image import load_img
import src.consts as c
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

L_RANGE = 100
AB_RANGE = 128
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()


def get_train_valid_test(path):
    file_paths = os.listdir(path)
    file_paths = [path+filepath for filepath in file_paths]

    valid_test_ratio = 0.05
    split = int(valid_test_ratio * len(file_paths))
    double_split = 2 * split

    valid_paths = file_paths[:split]
    test_paths = file_paths[split:double_split]
    train_paths = file_paths[double_split:]

    return train_paths, valid_paths, test_paths


def load_image(img_path):
    img = np.array(load_img(img_path), dtype=float)
    return img


def split_img_to_l_ab(img):
    lab = color.rgb2lab(img)
    img_l = lab[:, :, 0] / L_RANGE
    img_ab = lab[:, :, 1:] / AB_RANGE

    return img_l, img_ab


def join_l_ab(l, ab):
    img = np.zeros((c.IMG_HEIGHT, c.IMG_WIDTH, 3))
    img[:,:,0] = l[:,:,0] * L_RANGE
    img[:,:,1:] = ab * AB_RANGE
    return color.lab2rgb(img)


def inception_embedding(img):
    rgb_img = color.gray2rgb(color.rgb2gray(img))
    rgb_img_resize = []
    for i in rgb_img:
        i = resize(i, (299, 299, 3), mode='constant')
        rgb_img_resize.append(i)
    rgb_img_resize = np.array(rgb_img_resize)
    rgb_img_resize = preprocess_input(rgb_img_resize)
    with inception.graph.as_default():
        embed = inception.predict(rgb_img_resize)
    return embed


def image_generator(img_paths):
    for img_path in img_paths:
        img = load_image(img_path) / 255
        img_l, img_ab = split_img_to_l_ab(img)
        embedding = inception_embedding(img)

        yield [img_l, embedding], img_ab


def create_tf_dataset(img_paths, batch_size=1):
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(img_paths),
        output_shapes=(
            tf.TensorShape([c.IMG_HEIGHT, c.IMG_WIDTH, 1]),
            tf.TensorShape([c.IMG_HEIGHT, c.IMG_WIDTH, 2])
        ),
        output_types=(tf.float32, tf.float32)
    )

    tf_dataset = tf_dataset.batch(batch_size=batch_size)
    tf_dataset = tf_dataset.repeat()

    return tf_dataset
