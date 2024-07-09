import cv2
import numpy as np
from PIL import ImageGrab
import time
import pyautogui
import pydirectinput
from directKeys import PressKey, ReleaseKey, W, A, S, D


for i in range(2, 0, -1):
    print(i)
    time.sleep(1)

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD
X = 0x2D

def reset():
    ReleaseKey(0xCD) # right
    ReleaseKey(0xCB) # left
    ReleaseKey(0xC8) # up
    ReleaseKey(0xD0) # down
    ReleaseKey(X)



# pydirectinput.press('w')

print('starting....')


# source activate base && conda activate nfs


def process_img(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=150, threshold2=200)
    return processed_image


last_time = time.time()

while True:
    screen = np.array(ImageGrab.grab(bbox=(0, 30, 800, 630)))
    new_screen = process_img(screen)

    print('Pressing Up')
    PressKey(up)
    time.sleep(0.1)
    print('Releasing Up')
    ReleaseKey(up)
    

    if np.random.randint(0, 10) > 8:
        print('Pressing X')
        PressKey(X)
    
    if np.random.randint(0, 10) > 7:
        print('Releasing X')
        ReleaseKey(X)
    


    print(f"Loop took {time.time() - last_time} seconds")

    last_time = time.time()

    cv2.imshow('window', new_screen)
    #cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))


    if cv2.waitKey(25) & 0xFF == ord('q'):
        reset()
        cv2.destroyAllWindows()
        break