import pandas as pd
import h5py
import numpy as np
import sys
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
# hdf = pd.HDFStore('E:/OPEN_1/data/H-N.hdf5',mode='r')
# print(hdf.s)

f = h5py.File('E:/OPEN_1/data/H-N.hdf5', 'r')
class_arr = f['class'][:]
labels_arr = f['img_labels'][:]
image_arr = f['img_dataset'][:]
f.close()

for i in class_arr:
        print(i)



# with h5py.File('E:/OPEN_1/data/H-N.hdf5','r') as f:
#     ls = list(f.keys())
#     print(ls)
#     classes = f.get('class')
#     imgdata = f.get('img_dataset')
#     imglabel = f.get('img_labels')
    # print(np.array(classes))
    # print(np.array(imgdata[37]))
    # print(imgdata.shape)

# np.set_printoptions(threshold=sys.maxsize)
# print(image_arr[:36])
# arr = np.array(imglabel)
# ls = arr.tolist()
# print(ls)
# print(arr)
# print(labels_arr)
# print(image_arr)
# x_train, x_test, y_train, y_test = train_test_split(class_arr, image_arr, test_size=0.3, random_state=21)
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# print(x_train.shape())


