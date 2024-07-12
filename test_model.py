import cv2
import os
import numpy as np
import time
import pydirectinput
from alexnet import alexnet


from grabscreen import grab_screen
from getkeys import key_check


WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'nfsmwai-{}-{}-{}-epochs.h5'.format(LR, 'alexnetv2', EPOCHS)


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

model = alexnet(WIDTH, HEIGHT, LR)
model.load_weights(MODEL_NAME)


def main():
    for i in list(range(2))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    while True:
        if not paused:
            screen = grab_screen(region=(0, 30, 800, 630))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            # print('Frame took {} seconds'.format(time.time()-last_time))
            # last_time = time.time()

            preds = model.predict([screen.reshape(-1, WIDTH, HEIGHT, 1)])[0]
            moves = list(np.around(preds))
            print(moves, preds)

            if moves == [1, 0, 0]:
                left()
            elif moves == [0, 1, 0]:
                straight()
            elif moves == [0, 0, 1]:
                right()


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