import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

def alexnet(width, height, lr):
    input_layer = Input(shape=(width, height, 1))

    # Layer 1
    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = BatchNormalization()(x)

    # Layer 2
    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = BatchNormalization()(x)

    # Layer 3
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = BatchNormalization()(x)

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(4096, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='tanh')(x)
    x = Dropout(0.5)(x)

    # Output Layer
    output_layer = Dense(3, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    optimizer = SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model