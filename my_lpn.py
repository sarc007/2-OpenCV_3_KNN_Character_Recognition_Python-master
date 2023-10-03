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
# from tensorflow.keras.models import Sequential
import is_inner_contour_true


tuples = []

def main():
	from tensorflow.python.keras import layers

	src_dir = "E:\\OPEN\\IMG"
	dst_dir = "E:\\OPEN\\data"
	model_dir = "E:\\OPEN\\model"
	chk_pnt_dir = "E:\\OPEN\\training_all_nums\\cp.ckpt"
	data_file = dst_dir + "\\" + "alphanumericontourWithData.hdf5"
	input_shape = (28, 28, 1)
	MIN_CONTOUR_AREA = 4000
	MAX_CONTOUR_AREA = 15000
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

	class ContourWithData():

		# member variables ############################################################################
		npaContour = None  # contour
		boundingRect = None  # bounding rect for contour
		# innerboundingRect = None
		intRectX = 0  # bounding rect top left corner x location
		# innermin_X = 0  # inner bounding rect top left corner x location
		# innermin_Y = 0  # inner bounding rect top left corner y location
		intRectY = 0  # bounding rect top left corner y location
		intRectWidth = 0  # bounding rect width
		intRectHeight = 0  # bounding rect height
		min_x = 0
		min_y = 0
		max_x = 0
		max_y = 0
		# innerWidth = 0
		# innerHeight = 0
		# innermax_X = 0
		# innermax_Y = 0



		def calc_min_max(self):
			self.min_x = self.intRectX
			self.min_y = self.intRectY
			self.max_x = self.min_x + self.intRectWidth
			self.max_y = self.min_y + self.intRectHeight
			# self.innermax_X = self.innermin_X + self.innerWidth
			# self.innermax_Y = self.innermin_Y + self.innerHeight


		fltArea = 0.0  # area of contour
		midpoints = None
		innercontour = False

		def calculateRectTopLeftPointAndWidthAndHeight(self):  # calculate bounding rect info
			[intX, intY, intWidth, intHeight,] = self.boundingRect
			self.intRectX = intX
			self.intRectY = intY
			self.intRectWidth = intWidth
			self.intRectHeight = intHeight

		# def calculateMidpoints(self):

		def checkIfContourIsValid(self):  # this is oversimplified, for a production grade program
			if MIN_CONTOUR_AREA > self.fltArea or self.fltArea > MAX_CONTOUR_AREA: return False  # much better validity checking would be necessary
			if self.intRectWidth > self.intRectHeight: return False
			if self.innercontour:
				return False
			return True

	print("Creating Model")

	new_model_weights = tf.keras.models.load_model(model_dir + '\\reader_allnums.hdf5')

	img = Image.open('21.jpg')
	if img is None:  # if image was not read successfully
		print("error: image not read from file \n\n")  # print error message to std out
		os.system("pause")  # pause so user can see error message
		return  # and exit function (which exits program)
	# end if
	img = ImageOps.autocontrast(img)
	img_sharp = img.filter(ImageFilter.SHARPEN)
	img = np.array(img)

	img_sharp = np.array(img_sharp)

	h, w = img_sharp.shape[:2]

	img_sharp = rotateImage(img_sharp, 0)
	scale_factor = 10
	img_sharp = cv2.resize(img_sharp, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
	img_sharp = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)
	img_sharp = Image.fromarray(img_sharp)
	img_sharp = img_sharp.filter(ImageFilter.SHARPEN)
	img_sharp = np.asanyarray(img_sharp)
	h,w = img_sharp.shape[:2]
	img_inner_contours = np.zeros(shape=[h,w, 3], dtype=np.uint8)

	def plot_rectangles(img, tuples, color):
		for rectangle in tuples:
			if len(rectangle) > 5:
				if rectangle[5] == False:
					cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color)
			else:
				cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color)

	imgray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
	imgGray = imgray
	ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
	npaContours, npaHierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	img_sharp_copy = img_sharp.copy()
	img_sharp = cv2.drawContours(img_sharp, npaContours, -1, (0, 255, 0), 3)
	cv2.imshow('License Plate thresh', img_sharp)
	cv2.waitKey(0)
	print(cv2.contourArea(npaContours[1]))
	print(cv2.arcLength(npaContours[1], True))

	img_sharp = img_sharp_copy

	imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

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

	for npaContour in npaContours:  # for each contour
		contourWithData = ContourWithData()  # instantiate a contour with data object
		contourWithData.npaContour = npaContour  # assign contour to contour with data
		contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # get the bounding rect
		# contourWithData.innerboundingRect = cv2.boundingRect(contourWithData.npaContour)
		contourWithData.calculateRectTopLeftPointAndWidthAndHeight()  # get bounding rect info
		# contourWithData.calc_min_max(contourWithData)
		contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # calculate the contour area
		allContoursWithData.append(contourWithData)  # add contour with data object to list of all contours with data


	for contourWithData in allContoursWithData:
		tuples.append((allContoursWithData.index(contourWithData),contourWithData.min_x, contourWithData.min_y, contourWithData.max_x, contourWithData.max_y))




	for contourWithData in allContoursWithData:  # for all contours
		# i = allContoursWithData.index(contourWithData)
		# contourWithData.is_inner = list_of_inner_contour[i][5]
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

		# def inner_contour_removal():
		# 	for rectangle in tuples:
		# 		index_of_rectangle = tuples.index(rectangle)
		# 		i = 0
		# 		while i < len(tuples):
		# 			if i != index_of_rectangle:
		# 				if rectangle[1] > tuples[i][1] and rectangle[2] > tuples[i][2] \
		# 						and rectangle[3] < tuples[i][3] and rectangle[4] < tuples[i][4]:
		# 					tuples.pop(index_of_rectangle)
		# 			i += 1
		# 	return True


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
				if diff_w_h % 2 == 0:
					diff_w_h = diff_w_h / 2
				else:
					diff_w_h = (diff_w_h + 1) / 2

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