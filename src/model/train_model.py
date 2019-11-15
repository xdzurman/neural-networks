import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Model

from src.model import model
import src.consts as c
from src.helpers.dataset_helpers import create_tf_dataset, get_train_valid_test


def create_model():
    input = Input(shape=(c.IMG_HEIGHT, c.IMG_WIDTH, 1), dtype=tf.dtypes.float32)
    colorizing_model = Model(inputs=[input], outputs=[model.model_layers(input)])
    colorizing_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return colorizing_model


def train_model():
    train_paths, valid_paths, _ = get_train_valid_test(c.DATASET_PATH)
    batch_size = 1

    print(f'train images size {len(train_paths)}, valid images size {len(valid_paths)}, batch_size {batch_size}')
    train_data = create_tf_dataset(train_paths, batch_size)
    valid_data = create_tf_dataset(valid_paths, batch_size)

    model = create_model()
    model.fit(
        train_data,
        epochs=30,
        steps_per_epoch=5,
        validation_data=valid_data,
        validation_steps=5
    )

    return model
