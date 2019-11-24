from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, InputLayer, BatchNormalization, concatenate, RepeatVector, Reshape
import src.consts as c

def model_layers(input, embed_input):
    # Going Down
    x = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(input)
#     x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(x)
#     x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(x)
#     x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
#     x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
#     x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    
    # Embedding
    f = RepeatVector(32 * 32)(embed_input) 
    f = Reshape(([32, 32, c.EMBED_SIZE]))(f)
    x = concatenate([x, f], axis=3) 
    x = Conv2D(256, (1, 1), activation='relu', padding='same')(x) 
    
    # Going Up
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
#     x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
#     x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    return x
