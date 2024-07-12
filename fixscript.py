import numpy as np 
import pandas as pd 
from collections import Counter
from random import shuffle
import cv2

checkpoint_data_file_name = '../datasets/checkpoint.npy'
train_data = np.load(checkpoint_data_file_name, allow_pickle=True)
print(train_data.shape)
# np.save(checkpoint_data_file_name, train_data[500 :])


# new_data = np.load(checkpoint_data_file_name, allow_pickle=True)
# print(train_data.shape)