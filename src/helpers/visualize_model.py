import os
import sys
import logging
import tensorflow as tf
from skimage import io

import src.consts as c
from src.helpers.dataset_helpers import (get_imgs_from_model_and_dataset, image_generator, get_train_valid_test,
                                         create_tf_dataset)
from src.model.create_model import create_model


def save_models_images(model, train_paths, test_paths, save_path, sizes=5):
    img_path = f"{save_path}/img"
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    train_gen = image_generator(train_paths)
    for i in range(sizes):
        inputs, ab = next(train_gen)
        bw_img, original_img, predict_img = get_imgs_from_model_and_dataset(
            model, inputs, ab
        )
        io.imsave(f"{img_path}/train_{i}_bw.jpg", bw_img)
        io.imsave(f"{img_path}/train_{i}_original.jpg", original_img)
        io.imsave(f"{img_path}/train_{i}_predict.jpg", predict_img)

    test_gen = image_generator(test_paths)
    for i in range(sizes):
        inputs, ab = next(test_gen)
        bw_img, original_img, predict_img = get_imgs_from_model_and_dataset(
            model, inputs, ab
        )
        io.imsave(f"{img_path}/test_{i}_bw.jpg", bw_img)
        io.imsave(f"{img_path}/test_{i}_original.jpg", original_img)
        io.imsave(f"{img_path}/test_{i}_predict.jpg", predict_img)


if __name__ == "__main__":
    SAVE_PATH = sys.argv[1]
    train_paths, valid_paths, test_paths = get_train_valid_test(c.DATASET_PATH)
    model = create_model()

    latest = tf.train.latest_checkpoint(SAVE_PATH)

    test_data = create_tf_dataset(test_paths)
    logging.info(model.evaluate(test_data, steps=5))
    logging.info('visualising', latest, SAVE_PATH)

    model.load_weights(latest)

    sizes = 5
    if len(sys.argv) > 2:
        sizes = int(sys.argv[2])
    save_models_images(model, train_paths, test_paths, SAVE_PATH, sizes)
