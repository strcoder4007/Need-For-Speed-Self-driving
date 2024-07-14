import numpy as np 
import pandas as pd 
from collections import Counter
from random import shuffle
import cv2


training_data_file_name = '../datasets/training_data.npy'
checkpoint_data_file_name = '../datasets/checkpoint_balanced.npy'
train_data = np.load(checkpoint_data_file_name, allow_pickle=True)

for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test', img)

    print(choice)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
