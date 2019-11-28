import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Model
import datetime

from src.model import model
import src.consts as c
from src.helpers.dataset_helpers import create_tf_dataset, get_train_valid_test


def create_model():
    embed_input = Input(shape=(c.EMBED_SIZE,))
    input = Input(shape=(c.IMG_HEIGHT, c.IMG_WIDTH, 1), dtype=tf.dtypes.float32)
    colorizing_model = Model(inputs=[input, embed_input], outputs=[model.model_layers(input, embed_input)])
    colorizing_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    return colorizing_model


def train_model():
    train_paths, valid_paths, _ = get_train_valid_test(c.DATASET_PATH)
    batch_size = c.BATCH_SIZE

    print(f'train images size {len(train_paths)}, valid images size {len(valid_paths)}, batch_size {batch_size}')
    train_data = create_tf_dataset(train_paths, batch_size)
    valid_data = create_tf_dataset(valid_paths, batch_size)
    
    log_dir="logs/train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model = create_model()
    model.fit(
        train_data,
        epochs=c.EPOCHS,
        steps_per_epoch=len(train_paths)//batch_size,
        # validation_data=valid_data,
        # validation_steps=1,
        callbacks=[tensorboard_callback]
    )

    return model
