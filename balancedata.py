import numpy as np 
import pandas as pd 
from collections import Counter
from random import shuffle
import cv2

checkpoint_data_file_name = '../datasets/checkpoint.npy'
train_data = np.load(checkpoint_data_file_name, allow_pickle=True)


df = pd.DataFrame(train_data)
print(df.head())

print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0, 0]:
        lefts.append([img, choice])
    elif choice == [0, 1, 0]:
        forwards.append([img, choice])
    elif choice == [0, 0, 1]:
        rights.append([img, choice])
    else:
        print('no matches')

forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

final_data = forwards + lefts + rights

shuffle(final_data)
print(len(final_data))
np.save('../datasets/checkpoint_balanced.npy', np.array(final_data, dtype=object))
