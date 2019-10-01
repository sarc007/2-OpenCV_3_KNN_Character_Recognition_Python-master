import cv2
import tensorflow as tf
import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, Flatten, MaxPooling2D


import matplotlib.pyplot as plt

# minst = tf.keras.datasets.mnist
# data = minst.load_data()
# check_data = data[1][0][0]
# print(check_data)
# print(len(check_data))
# check_data_np = np.asarray(data)
# print(check_data_np.shape)
# print(check_data_np[0])
src_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\img_resized"
dst_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\data"
model_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\model"
chk_pnt_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\training_all_caps\\cp.ckpt"
data_file = dst_dir + "\\" + "alpha_caps.hdf5"

if not os.path.exists(model_dir):
	os.makedirs(model_dir)
if not os.path.exists(chk_pnt_dir):
	os.makedirs(chk_pnt_dir)
cp_callback = tf.keras.callbacks.ModelCheckpoint(chk_pnt_dir, save_best_only=True, save_weights_only=True, verbose=1)

f = h5py.File(data_file, 'r')
class_arr = f['class'][:]
labels_arr = f['img_labels'][:]
image_arr = f['img_dataset'][:]
f.close()

print("Loaded HDF5 File")
x_train, x_test, y_train, y_test = train_test_split(image_arr, labels_arr, test_size=0.2, random_state=21)
print("Loaded Training Data")
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])



y_train[y_train < 58] -= 48
y_train[y_train > 64] -= 55

y_test[y_test < 58] -= 48
y_test[y_test > 64] -= 55
# Creating a Sequential Model and adding the layers
print("Creating Model")
def create_model():
	model = Sequential()
	model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, kernel_size=(3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
	model.add(Dense(256, activation=tf.nn.relu))
	model.add(Dropout(0.3))
	model.add(Dense(256, activation=tf.nn.relu))
	model.add(Dropout(0.4))
	model.add(Dense(26, activation=tf.nn.softmax))

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model
model = create_model()
model.summary()

model.fit(x_train, y_train,  validation_data=(x_test, y_test), epochs=25,
          validation_split=0.2,  verbose=1, shuffle=True, callbacks=[cp_callback])
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss = {0}' , Accuracy = {1} ", val_loss, val_acc)
model.save(model_dir + '\\reader_allcaps.h5', overwrite=True)
new_model = tf.keras.models.load_model(model_dir + '\\reader_allcaps.h5')
predictions = new_model.predict([x_test])
print(np.argmax(predictions[0]))
pred_ascii = np.argmax(predictions[0])
if pred_ascii < 10:
	pred_ascii += 48
elif pred_ascii > 9:
	pred_ascii += 55
print(chr(pred_ascii))
cv2.imshow('x test 0 ', (x_test[0] * 255))
cv2.waitKey(0)
new_model_weights = create_model()
new_model_weights.load_weights(chk_pnt_dir)
val_loss, val_acc = model.evaluate(x_test, y_test)


# model.add()
