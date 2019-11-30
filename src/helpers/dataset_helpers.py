import os
import warnings
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from skimage import color
from tensorflow.keras.preprocessing.image import load_img
from skimage import img_as_ubyte

import src.consts as c

L_RANGE = 100
AB_RANGE = 128

df_inception_resnet2 = pd.read_csv("./precomputed_inputs/inception_res_net_v2.csv").drop(
    columns=['Unnamed: 0']).set_index('file_name')
df_inception_resnet2['embedding'] = df_inception_resnet2['embedding'].apply(
    lambda x: np.array(eval(x.replace(' ', ','))))

# df_vgg = pd.read_csv("./precomputed_inputs/vgg19.csv").drop(columns=['Unnamed: 0']).set_index('file_name')
# df_vgg['embedding'] = df_vgg['embedding'].apply(lambda x: np.array(eval(x.replace(' ', ','))))

embedding_df = df_inception_resnet2

"""
Generating and preprocessing is 
inspired by: https://github.com/emilwallner/Coloring-greyscale-images/blob/master/Full-version/full_version.ipynb
Main difference is parsing batches in different manner, which should be more ram optimized, for parsing more images.
Also we use precomputed inputs from inception resnet.
"""


def get_train_valid_test(path):
    file_paths = os.listdir(path)
    file_paths = sorted([path + filepath for filepath in file_paths])

    train_ratio = 0.80
    split = int(train_ratio * len(file_paths))
    double_split = int((len(file_paths) - split) / 2 + split)

    train_paths = file_paths[:split]
    valid_paths = file_paths[split:double_split]
    test_paths = file_paths[double_split:]

    logging.info(f'dataset sizes: train {len(train_paths)}, valid {len(valid_paths)}, test {len(test_paths)}')
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
    img[:, :, 0] = l[:, :, 0] * L_RANGE
    img[:, :, 1:] = ab * AB_RANGE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return color.lab2rgb(img)


def image_generator(img_paths):
    for img_path in img_paths:
        img = load_image(img_path) / 255
        img_l, img_ab = split_img_to_l_ab(img)
        img_l.reshape(img_l.shape + (1,))
        embedding = embedding_df.loc[img_path.split('/')[-1]]['embedding']
        yield (img_l.reshape(img_l.shape + (1,)), embedding), (img_ab)


def create_tf_dataset(img_paths, batch_size=1):
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(img_paths),
        output_shapes=(
            (tf.TensorShape([c.IMG_HEIGHT, c.IMG_WIDTH, 1]), tf.TensorShape((1000,))),
            (tf.TensorShape([c.IMG_HEIGHT, c.IMG_WIDTH, 2])),
        ),
        output_types=((tf.float32, tf.float32), (tf.float32)),
    )

    tf_dataset = tf_dataset.batch(batch_size=batch_size)
    tf_dataset = tf_dataset.repeat()

    return tf_dataset


def get_imgs_from_model_and_dataset(model, inputs, ab):
    l = inputs[0]
    bw_img = l[:, :, 0]
    original_img = join_l_ab(l, ab)
    predict_input = [np.expand_dims(l, axis=0), np.expand_dims(inputs[1], axis=0)]
    predict_output = model.predict(predict_input)
    predict_img = join_l_ab(l, predict_output)

    bw_img = img_as_ubyte(bw_img)
    original_img = img_as_ubyte(original_img)
    predict_img = img_as_ubyte(predict_img)

    return bw_img, original_img, predict_img
