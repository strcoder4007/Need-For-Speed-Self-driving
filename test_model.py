import cv2
import os
import numpy as np
import time
import pydirectinput
# from alexnet import alexnet
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


from grabscreen import grab_screen
from getkeys import key_check


# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')


WIDTH = 240
HEIGHT = 180
CHANNELS = 3
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'nfsmwai-{}-{}-{}-epochs.h5'.format(LR, 'InceptionResNetV2', EPOCHS)


def straight():
    pydirectinput.keyDown('w')
    pydirectinput.keyUp('a')
    pydirectinput.keyUp('d')

def left():
    pydirectinput.keyDown('a')
    pydirectinput.keyDown('w')
    pydirectinput.keyUp('d')
    time.sleep(0.01)
    pydirectinput.keyUp('a')

def right():
    pydirectinput.keyDown('d')
    pydirectinput.keyDown('w')
    pydirectinput.keyUp('a')
    time.sleep(0.01)
    pydirectinput.keyUp('d')

def reset():
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('a')
    pydirectinput.keyUp('d')


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


model.load_weights('../models/' + MODEL_NAME)


def main():
    for i in list(range(7))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    while True:
        if not paused:
            screen = grab_screen(region=(0, 30, 800, 630))
            # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            # print('Frame took {} seconds'.format(time.time()-last_time))
            # last_time = time.time()

            preds = model.predict([screen.reshape(-1, WIDTH, HEIGHT, CHANNELS)])[0]
            moves = list(np.around(preds))

            if moves == [1, 0, 0]:
                left()
                print(f'Turning LEFT.\nPredictions: {preds}')
            elif moves == [0, 1, 0]:
                straight()
                print(f'Going STRAIGHT.\nPredictions: {preds}')
            elif moves == [0, 0, 1]:
                right()
                print(f'Turning RIGHT.\nPredictions: {preds}')
            else:
                print('Doing nothing')


        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('d')
                time.sleep(1)

                
main()