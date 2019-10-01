import cv2
from PIL import Image, ImageFilter, ImageOps
import tensorflow as tf
import numpy as np
import operator
import h5py
from sklearn.model_selection import train_test_split
import imageio
import matplotlib.pyplot as plt
import os
import PIL
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential


def main():
	# minst = tf.keras.datasets.mnist
	# data = minst.load_data()
	# check_data = data[1][0][0]
	# print(check_data)
	# print(len(check_data))
	# check_data_np = np.asarray(data)
	# print(check_data_np.shape)
	# print(check_data_np[0])
	from tensorflow.python.keras import layers

	src_dir = "E:\\OPEN\\IMG"
	dst_dir = "E:\\OPEN\\data"
	model_dir = "E:\\OPEN\\model"
	chk_pnt_dir = "E:\\OPEN\\training_all_nums\\cp.ckpt"
	data_file = dst_dir + "\\" + "alphanumeric.hdf5"
	input_shape = (28, 28, 1)
	MIN_CONTOUR_AREA = 2000
	MAX_CONTOUR_AREA = 20000
	RESIZED_IMAGE_WIDTH = 28
	RESIZED_IMAGE_HEIGHT = 28
	allContoursWithData = []  # declare empty lists,
	validContoursWithData = []  # we will fill these shortly
	try:
		npaClassifications = np.loadtxt("classifications.txt", np.float32)  # read in training classifications
	except:
		print("error, unable to open classifications.txt, exiting program\n")
		os.system("pause")
		return
	# end try

	try:
		npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
	except:
		print("error, unable to open flattened_images.txt, exiting program\n")
		os.system("pause")
		return
	# end try

	npaClassifications = npaClassifications.reshape(
			(npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train

	kNearest = cv2.ml.KNearest_create()  # instantiate KNN object

	kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
	def rotateImage(img, angle):
		image_center = tuple(np.array(img.shape[1::-1]) / 2)
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
		return result
	#
	# f = h5py.File(data_file, 'r')
	# class_arr = f['class'][:]
	# labels_arr = f['img_labels'][:]
	# image_arr = f['img_dataset'][:]
	# f.close()
	# x_train, x_test, y_train, y_test = train_test_split(image_arr, labels_arr, test_size=0.2, random_state=42)
	# print(chr(y_train[200000]))
	# cv2.imshow('x train 0 ', x_train[200000])
	# cv2.waitKey(0)
	# im = cv2.imread('21.jpg')
	# h, w = im.shape[:2]
	# scale_factor = 5
	# im = cv2.resize(im, (w * scale_factor, h * scale_factor), interpolation = cv2.INTER_CUBIC)
	class ContourWithData():

		# member variables ############################################################################
		npaContour = None  # contour
		boundingRect = None  # bounding rect for contour
		intRectX = 0  # bounding rect top left corner x location
		intRectY = 0  # bounding rect top left corner y location
		intRectWidth = 0  # bounding rect width
		intRectHeight = 0  # bounding rect height
		fltArea = 0.0  # area of contour
		midpoints = None

		def calculateRectTopLeftPointAndWidthAndHeight(self):  # calculate bounding rect info
			[intX, intY, intWidth, intHeight] = self.boundingRect
			self.intRectX = intX
			self.intRectY = intY
			self.intRectWidth = intWidth
			self.intRectHeight = intHeight

		# def calculateMidpoints(self):


		def checkIfContourIsValid(self):  # this is oversimplified, for a production grade program
			if MIN_CONTOUR_AREA > self.fltArea or self.fltArea > MAX_CONTOUR_AREA: return False  # much better validity checking would be necessary
			if self.intRectWidth > self.intRectHeight: return False
			return True


	print("Creating Model")
	# def create_model():
	# 	model = Sequential()
	# 	model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
	# 	model.add(Activation('relu'))
	# 	model.add(MaxPooling2D(pool_size=(2, 2)))
	#
	# 	model.add(Conv2D(32, kernel_size=(3, 3)))
	# 	model.add(Activation('relu'))
	# 	model.add(MaxPooling2D(pool_size=(2, 2)))
	#
	# 	model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
	# 	model.add(Dense(128, activation=tf.nn.relu))
	# 	model.add(Dropout(0.3))
	# 	model.add(Dense(128, activation=tf.nn.relu))
	# 	model.add(Dropout(0.4))
	# 	model.add(Dense(36, activation=tf.nn.softmax))
	#
	# 	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	# 	return model

	new_model_weights = tf.keras.models.load_model(model_dir + '\\reader_allnums.hdf5')
	# new_model_weights.load_weights(chk_pnt_dir, by_name=True)
	# val_loss, val_acc = new_model_weights.evaluate(x_test, y_test)
	# new_model_weights.predict()

	img = Image.open('21.jpg')
	if img is None:  # if image was not read successfully
		print("error: image not read from file \n\n")  # print error message to std out
		os.system("pause")  # pause so user can see error message
		return  # and exit function (which exits program)
	# end if
	# img = ImageOps.autocontrast(img)
	img_sharp = img.filter(ImageFilter.SHARPEN)
	img = np.array(img)

	img_sharp = np.array(img_sharp)
	# cv2.imshow('License Plate Auto contrast', img)
	# cv2.waitKey(0)
	#
	# cv2.imshow('License Plate Auto contrast', img_sharp)
	# cv2.waitKey(0)

	h, w = img_sharp.shape[:2]
	img_sharp = rotateImage(img_sharp, 0)
	scale_factor = 10
	img_sharp = cv2.resize(img_sharp, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
	img_sharp = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)
	img_sharp = Image.fromarray(img_sharp)
	img_sharp = img_sharp.filter(ImageFilter.SHARPEN)
	img_sharp = np.asanyarray(img_sharp)

	imgray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
	# imgGray = cv2.resize(imgray, (int(w / scale_factor), int(h / scale_factor)), interpolation=cv2.INTER_CUBIC)
	imgGray = imgray
	ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
	npaContours, npaHierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	img_sharp_copy = img_sharp.copy()
	img_sharp = cv2.drawContours(img_sharp, npaContours, -1, (0, 255, 0), 3)
	cv2.imshow('License Plate thresh', img_sharp)
	cv2.waitKey(0)
	print(cv2.contourArea(npaContours[1]))
	print(cv2.arcLength(npaContours[1],True))
	# img_exp = img_sharp
	# for expContour in npaContours:
	# 	expContour = np.array(expContour)
	# 	not_min_max_cont  = expContour[expContour]

	img_sharp = img_sharp_copy

	imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur
	# cv2.imshow('License Plate blurred', imgBlurred)
	# cv2.waitKey(0)
	# filter image from grayscale to black and white
	imgThresh = cv2.adaptiveThreshold(imgGray,  # input image
	                                  255,  # make pixels that pass the threshold full white
	                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	                                  # use gaussian rather than mean, seems to give better results
	                                  cv2.THRESH_BINARY_INV,
	                                  # invert so foreground will be white, background will be black

	                                  11,  # size of a pixel neighborhood used to calculate threshold value
	                                  2)  # constant subtracted from the mean or weighted mean
	cv2.imshow('License Plate threshold', imgThresh)
	cv2.waitKey(0)
	imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

	# npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
	#                                               cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
	#                                               cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

	for npaContour in npaContours:  # for each contour
		contourWithData = ContourWithData()  # instantiate a contour with data object
		contourWithData.npaContour = npaContour  # assign contour to contour with data
		contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # get the bounding rect
		contourWithData.calculateRectTopLeftPointAndWidthAndHeight()  # get bounding rect info
		contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # calculate the contour area
		allContoursWithData.append(contourWithData)  # add contour with data object to list of all contours with data
	# end for
	# for i in allContoursWithData:  # for all contours
	# 	if not npaHierarchy[0,i,3]== -1:
	# 		return False




	for contourWithData in allContoursWithData:  # for all contours
				if contourWithData.checkIfContourIsValid():  # check if valid
					validContoursWithData.append(contourWithData)  # if so, append to valid contour list
					print(contourWithData.fltArea, contourWithData.intRectWidth, contourWithData.intRectHeight)

	# end if
	# end for

	validContoursWithData.sort(key=operator.attrgetter("intRectX"))  # sort contours from left to right

	strFinalString = ""  # declare final string, this will have the final number sequence by the end of the program
	count = 0
	for contourWithData in validContoursWithData:  # for each contour
		# draw a green rect around the current char
		img_sharp = cv2.rectangle(img_sharp,  # draw rectangle on original testing image
		                          (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
		                          (contourWithData.intRectX + contourWithData.intRectWidth,
		                           contourWithData.intRectY + contourWithData.intRectHeight),  # lower right corner
		                          (0, 255, 0),  # green
		                          1)  # thickness

		imgROI = thresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
		         # crop char out of threshold image
		         contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

		def recalibrate_width(im, top_add, bottom_add):
			im = Image.fromarray(im)
			w_im, h_im = im.size
			if w_im >= h_im:
				return np.asanyarray(im)
			else:
				diff_w_h = h_im - w_im
				if diff_w_h % 2 ==0:
					diff_w_h = diff_w_h /2
				else:
					diff_w_h = (diff_w_h +1) /2

				left = 0 - diff_w_h
				top = 0 - top_add
				right = w_im + diff_w_h
				bottom = h_im + bottom_add
				im = im.crop((left, top, right, bottom))
			return np.asanyarray(im)
		imgROI = recalibrate_width(imgROI, 10, 10)
		cv2.imshow("image recalibrated " + str(count), imgROI)
		cv2.waitKey(0)
		count += 1

		# resize image, this will be more consistent for recognition and storage
		imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
		cv2.imwrite("E:\\OPEN\\img small\\img_small_" + str(count) + ".png", imgROIResized)
		imgROIResized = Image.fromarray(imgROIResized)
		imgROIResized = imgROIResized.filter(ImageFilter.SHARPEN)
		imgROIResized = np.asanyarray(imgROIResized)

		imgROIResized = imgROIResized.reshape(28, 28, 1)
		img_test = np.array([imgROIResized])
		cv2.imshow('for 7 ', imgROIResized)
		cv2.waitKey(0)
		img_test = img_test.reshape(img_test.shape[0], 28, 28, 1)
		img_test = tf.keras.utils.normalize(img_test, axis=1)
		predictions = new_model_weights.predict([img_test])
		print(np.argmax(predictions[0]))
		print(predictions[0])
		pred_ascii = np.argmax(predictions[0])
		if pred_ascii < 10:
			pred_ascii += 48
		elif pred_ascii > 9:
			pred_ascii += 55
		print(chr(pred_ascii))
		# imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
		#                                     RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage
		#
		# npaROIResized = imgROIResized.reshape(
		# 		(1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image into 1d numpy array
		#
		# npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats
		#
		# retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest
		# #
		# strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results
		strCurrentChar = str(chr(pred_ascii))  # get character from results
		#
		strFinalString = strFinalString + strCurrentChar  # append current char to full string
	# # end for
	#
	print("\n" + strFinalString + "\n")  # show the full string

	cv2.imshow("imgTestingNumbers", img_sharp)  # show input image with green boxes drawn around found digits
	cv2.waitKey(0)  # wait for user key press

	cv2.destroyAllWindows()  # remove windows from memory


if __name__ == "__main__":
	main()
# end if
