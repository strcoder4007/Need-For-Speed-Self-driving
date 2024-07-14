import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define constants
WIDTH = 240
HEIGHT = 180
CHANNELS = 3
LR = 1e-3
EPOCHS = 5
BATCH_SIZE = 32  # Reduced batch size
MODEL_NAME = 'nfsmwai-{}-{}-{}-epochs.h5'.format(LR, 'InceptionResNetV2', EPOCHS)

# Load InceptionResNetV2 with pre-trained ImageNet weights, excluding the top layer
base_model = InceptionResNetV2(weights="imagenet", 
                               include_top=False, 
                               input_shape=(WIDTH, HEIGHT, CHANNELS))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax', dtype='float32')(x)
# Define the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LR), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the dataset
train_data = np.load('../datasets/checkpoint_balanced.npy', allow_pickle=True)

train = train_data[:-500]
test = train_data[-500:]

# Prepare the training data
X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, CHANNELS)
Y = np.array([i[1] for i in train])

# Prepare the test data
test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, CHANNELS)
test_y = np.array([i[1] for i in test])

# Train the model
model.fit(X, Y,
          epochs=EPOCHS,
          validation_data=(test_x, test_y),
          batch_size=BATCH_SIZE,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='log')])

# Save the model
model.save('../models/' + MODEL_NAME)