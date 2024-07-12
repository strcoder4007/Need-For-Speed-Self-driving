import numpy as np
import tensorflow as tf
from alexnet import alexnet


WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'nfsmwai-{}-{}-{}-epochs.h5'.format(LR, 'alexnetv2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)


train_data = np.load('../datasets/checkpoint_balanced.npy', allow_pickle=True)

train = train_data[:-500]/255.0
test = train_data[-500:]/255.0

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = np.array([i[1] for i in test])

# Train the model
model.fit(X, Y,
          epochs=EPOCHS,
          validation_data=(test_x, test_y),
          batch_size=32,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='log')])

# Save the model
model.save(MODEL_NAME)
 
# tensorboard --logdir='log'