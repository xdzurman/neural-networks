import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from skimage import io
import datetime
import os
import logging

from src.model import model
import src.consts as c
from src.helpers.dataset_helpers import get_imgs_from_model_and_dataset, get_train_valid_test, image_generator, create_tf_dataset


def create_model():
    embed_input = Input(shape=(c.EMBED_SIZE,))
    input = Input(shape=(c.IMG_HEIGHT, c.IMG_WIDTH, 1), dtype=tf.dtypes.float32)
    colorizing_model = Model(inputs=[input, embed_input], outputs=[model.model_layers(input, embed_input)])
    colorizing_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    return colorizing_model


def train_model(train_paths, valid_paths):
    batch_size = c.BATCH_SIZE

    logging.info(f'train images size {len(train_paths)}, valid images size {len(valid_paths)}, batch_size {batch_size}')
    train_data = create_tf_dataset(train_paths, batch_size)
    valid_data = create_tf_dataset(valid_paths, batch_size)
    
    log_dir="logs/train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model = create_model()
    model.fit(
        train_data,
        epochs=c.EPOCHS,
        steps_per_epoch=c.STEPS_PER_EPOCH,
        # validation_data=valid_data,
        # validation_steps=1,
        callbacks=[tensorboard_callback]
    )

    return model


if __name__ == '__main__':
    train_paths, valid_paths, test_paths = get_train_valid_test(c.DATASET_PATH)
    test_data = create_tf_dataset(test_paths)

    MODEL_PATH = f'models'
    MODEL_NAME = f'{len(train_paths)}_images_koalarization'
    SAVE_MODEL_PATH = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{MODEL_NAME}'

    if not os.path.exists(f'{MODEL_PATH}/{SAVE_MODEL_PATH}'):
        os.makedirs(f'{MODEL_PATH}/{SAVE_MODEL_PATH}')

    model = train_model(train_paths, valid_paths)
    model.save(f'{MODEL_PATH}/{SAVE_MODEL_PATH}/model.h5')

    logging.info(model.evaluate(test_data, steps=5))
    # saving images
    img_path = f'{MODEL_PATH}/{SAVE_MODEL_PATH}/img'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    train_gen = image_generator(train_paths)
    for i in range(5):
        inputs, ab = next(train_gen)
        bw_img, original_img, predict_img = get_imgs_from_model_and_dataset(model, inputs, ab)
        io.imsave(f'{img_path}/train_{i}_bw.jpg', bw_img)
        io.imsave(f'{img_path}/train_{i}_original.jpg', original_img)
        io.imsave(f'{img_path}/train_{i}_predict.jpg', predict_img)

    test_gen = image_generator(test_paths)
    for i in range(5):
        inputs, ab = next(test_gen)
        bw_img, original_img, predict_img = get_imgs_from_model_and_dataset(model, inputs, ab)
        io.imsave(f'{img_path}/test_{i}_bw.jpg', bw_img)
        io.imsave(f'{img_path}/test_{i}_original.jpg', original_img)
        io.imsave(f'{img_path}/test_{i}_predict.jpg', predict_img)
