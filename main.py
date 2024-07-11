import cv2
import os
import numpy as np
import time
import pyautogui
import pydirectinput


from grabscreen import grab_screen
from drawlanes import draw_lanes
from getkeys import key_check


# source activate base && conda activate nfs
# from directkeys import PressKey, ReleaseKey, W, A, S, D
up = 0xC8
L = 0x1E
R = 0x20
X = 0x2D


def keys_to_output(keys):
    # [A, W, D]
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1
    elif 'W' in keys:
        output[1] = 1
    else: # D
        output[2] = 1
    return output

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


def displayOnlyWhite(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('White Objects Only', result)

def removeHorizontalLines(lines):
    if lines is None:
        return lines

    filtered_lines = []

    # Function to calculate the slope of a line
    def calculate_slope(x1, y1, x2, y2):
        if x2 - x1 == 0:  # To avoid division by zero
            return float('inf')
        else:
            return (y2 - y1) / (x2 - x1)

    # Iterate over the lines and filter based on the slope
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = calculate_slope(x1, y1, x2, y2)
            # print(f"Slope: {slope}")
            if abs(slope) > 0.1:  # Consider absolute value to handle negative slopes
                filtered_lines.append([[x1, y1, x2, y2]])
    return filtered_lines

def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 150, threshold2=250)
    
    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    
    vertices = np.array([[220,550], [220, 380], [10, 380], [10,350],[300, 280], [500, 280],[800,350],[800,550], [400, 500]], np.int32)

    processed_img = roi(processed_img, [vertices])
     
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 200, np.array([]), 150, 5)

    lines = removeHorizontalLines(lines)

    m1 = 0
    m2 = 0
    try:
        l1, l2, m1, m2 = draw_lanes(original_image,lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 20)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 20)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_img,original_image, m1, m2

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked



def main():
    for i in list(range(2))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    while True:
        screen = grab_screen(region=(0, 30, 800, 630))
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        new_screen, original_image, m1, m2 = process_img(screen)
        cv2.imshow('window', new_screen)
        cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        # Car Controls
        if m1 < 0 and m2 < 0:
            print('RIGHT')
            right()
        elif m1 > 0 and m2 > 0:
            print('LEFT')
            left()
        else:
            print('UP')
            straight()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()