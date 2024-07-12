import cv2
import os
import numpy as np
import time
import pydirectinput


from grabscreen import grab_screen
# from drawlanes import draw_lanes
from getkeys import key_check


# source activate base && conda activate nfs


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



training_data = []
file_name = '../datasets/training_data.npy'
checkpoint_data_file_name = '../datasets/checkpoint.npy'


# Check if there is checkpoint data present, 
# if there is then assign it to training_data
if os.path.isfile(checkpoint_data_file_name):
    print('Checkpoint File exists, loading checkpoint data...')
    training_data = list(np.load(checkpoint_data_file_name, allow_pickle=True))
elif os.path.isfile(file_name):
    print('Training Data exists, loading training data...')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh...')
    training_data = []


def main():
    for i in list(range(2))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    while len(training_data) < 150000:
        screen = grab_screen(region=(0, 30, 800, 630))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (160, 120))

        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen, output])

        # print('Frame took {} seconds'.format(time.time()-last_time))
        # last_time = time.time()

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, np.array(training_data, dtype=object))


        if len(training_data) % 10000 == 0:
            print(f'Saving checkpoint at {len(training_data)}')
            np.save(checkpoint_data_file_name, np.array(training_data, dtype=object))


main()