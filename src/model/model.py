from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, InputLayer, BatchNormalization, concatenate
import src.consts as c

def model_layers(input, embed_input):
    # Going Down
    x = Conv2D(16, 3, padding='same', activation='relu')(input)
    x = down_1 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = down_2 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = down_3 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = down_4 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    
    # Embedding
    f = RepeatVector(32 * 32)(embed_input) 
    f = Reshape(([32, 32, c.EMBED_SIZE]))(f)
    x = concatenate([x, f], axis=3) 
    x = Conv2D(256, (1, 1), activation='relu', padding='same')(x) 
    
    # Going Up
    x = concatenate([x, down_4])
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)

    x = concatenate([x, down_3])
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)

    x = concatenate([x, down_2])
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)

    x = concatenate([x, down_1])
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(2, 3, padding='same', activation='tanh')(x)

    return x
