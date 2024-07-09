import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api



while True:
    printscreen_pil = ImageGrab.grab(bbox=(0, 40, 800, 640))
    printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8')\
        .reshape((printscreen_pil.size[1], printscreen_pil.size[0], 3))
    cv2.imshow('window', printscreen_numpy)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break