import datetime
import logging
import os

import tensorflow as tf

import src.consts as c
from src.helpers.dataset_helpers import (create_tf_dataset, get_train_valid_test)
from src.model.visualize_model import (save_models_images)
from src.model.create_model import create_model


def train_model(train_paths, valid_paths, SAVE_PATH):
    batch_size = c.BATCH_SIZE

    logging.info(
        f"train images size {len(train_paths)}, valid images size {len(valid_paths)}, batch_size {batch_size}"
    )
    train_data = create_tf_dataset(train_paths, batch_size)
    valid_data = create_tf_dataset(valid_paths, batch_size)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint_name = 'weights.{epoch:02d}-{val_loss:.2f}'
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=f'{SAVE_PATH}/{model_checkpoint_name}.hdf5,', save_weights_only=True)

    valid_steps = len(valid_paths) // batch_size
    valid_steps = valid_steps if valid_steps >= 1 else 1

    model = create_model()
    model.fit(
        train_data,        epochs=c.EPOCHS,
        steps_per_epoch=len(train_paths) // batch_size,
        validation_data=valid_data,
        validation_steps=valid_steps,
        callbacks=[tensorboard_callback, model_checkpoint_cb]
    )

    return model


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    train_paths, valid_paths, test_paths = get_train_valid_test(c.DATASET_PATH)
    test_data = create_tf_dataset(test_paths)

    MODEL_PATH = f"models"
    MODEL_NAME = f"{len(train_paths)}_images_koalarization"
    SAVE_MODEL_PATH = (
        f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{MODEL_NAME}'
    )
    SAVE_PATH = f"{MODEL_PATH}/{SAVE_MODEL_PATH}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    model = train_model(train_paths, valid_paths, SAVE_PATH)
    model.save_weights(f"{SAVE_PATH}/weights.after_training.hdf5")

    eval_steps = len(test_paths) // c.BATCH_SIZE
    eval_steps = eval_steps if eval_steps >= 1 else 1
    model_eval = model.evaluate(test_data, steps=eval_steps)
    logging.info(f'Test dataset value {model_eval}')

    save_models_images(model, train_paths, test_paths, SAVE_PATH)
