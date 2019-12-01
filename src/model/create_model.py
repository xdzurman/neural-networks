import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import src.consts as c
from src.model.model import (model_layers)


def create_model():
    embed_input = Input(shape=(c.EMBED_SIZE,))
    input = Input(shape=(c.IMG_HEIGHT, c.IMG_WIDTH, 1), dtype=tf.dtypes.float32)
    colorizing_model = Model(
        inputs=[input, embed_input], outputs=[model_layers(input, embed_input)]
    )

    optimizer = Adam()
    colorizing_model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])

    return colorizing_model
