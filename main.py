import cv2
import numpy as np
from PIL import ImageGrab
import time


def process_img(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=150, threshold2=200)
    return processed_image


last_time = time.time()

while True:
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    new_screen = process_img(screen)

    print(f"Loop took {time.time() - last_time} seconds")

    last_time = time.time()

    cv2.imshow('window', new_screen)
    #cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))


    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break