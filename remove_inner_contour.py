import numpy as np
import cv2
import pandas as pd

list_of_contour_rect_coordinates = pd.read_csv("list.csv",header=None,delimiter=',')
tuples = [tuple(x) for x in list_of_contour_rect_coordinates.values] #list of all rectangles,each rectangle is a tuple
print(tuples)
index_of_inner_contour_coordinates = []
index_of_outer_contour_coordinates = []

img = np.zeros(shape =[4000,4000,3], dtype=np.uint8)
img_outer_contours = np.zeros(shape =[4000,4000,3], dtype=np.uint8)
img_inner_contours = np.zeros(shape =[4000,4000,3], dtype=np.uint8)
def plot_rectangles(img,tuples,color):
	for rectangle in tuples:
		if len(rectangle) > 5:
			if rectangle[5] == False:
				cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color)
		else:
			cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color)

# plot_rectangles(img,tuples,(0,255,0))

def list_inner_contour():
	for rectangle in tuples:
		index_of_rectangle = tuples.index(rectangle)
		i = 0
		while i < len(tuples):
			if i != index_of_rectangle:
				if rectangle[1] > tuples[i][1] and rectangle[2] > tuples[i][2] \
					and	rectangle[3] < tuples[i][3] and rectangle[4] < tuples[i][4]:
					index_of_inner_contour_coordinates.append(index_of_rectangle)

			i+=1
	return True

def slice_contour_indices(tuples,index_list):
	sliced_tuple = []
	for index in index_list:
		sliced_tuple.append(tuples[index])
	return sliced_tuple

def list_outer_contours_index():
	for results in tuples:
		if results[0] not in index_of_inner_contour_coordinates:
			index_of_outer_contour_coordinates.append(results[0])
	return True

list_inner_contour()
list_outer_contours_index()
plot_rectangles(img,tuples,(0,255,0))
print()
tuples_of_inner_contours = slice_contour_indices(tuples,index_of_inner_contour_coordinates)
tuples_of_outer_contours = slice_contour_indices(tuples,index_of_outer_contour_coordinates)
plot_rectangles(img,tuples_of_inner_contours,(255,145,200))
plot_rectangles(img_outer_contours,tuples_of_outer_contours,(125,255,60))
plot_rectangles(img_inner_contours,tuples_of_inner_contours,(125,255,60))
print(tuples_of_outer_contours)
print(tuples_of_inner_contours)

cv2.imshow('rectangles',img)
cv2.waitKey(0)
cv2.imshow('outer contour rectangles',img_outer_contours)
cv2.waitKey(0)
cv2.imshow('inner contour rectangles',img_inner_contours)
cv2.waitKey(0)