import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Model

from src.model import model
import src.consts as c
from src.helpers.dataset_helpers import create_tf_dataset, get_train_valid_test


def create_model():
    embed_input = Input(shape=(c.EMBED_SIZE,))
    input = Input(shape=(c.IMG_HEIGHT, c.IMG_WIDTH, 1), dtype=tf.dtypes.float32)
    colorizing_model = Model(inputs=[input, embed_input], outputs=[model.model_layers(input, embed_input)])
    colorizing_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return colorizing_model


def train_model():
    train_paths, valid_paths, _ = get_train_valid_test(c.DATASET_PATH)
    batch_size = 16

    print(f'train images size {len(train_paths)}, valid images size {len(valid_paths)}, batch_size {batch_size}')
    train_data = create_tf_dataset(train_paths[:10], batch_size)
    valid_data = create_tf_dataset(valid_paths, batch_size)

    model = create_model()
    model.fit(
        train_data,
        epochs=500,
        steps_per_epoch=5,
        validation_data=valid_data,
        validation_steps=5
    )

    return model
