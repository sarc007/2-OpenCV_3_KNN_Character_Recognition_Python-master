import numpy as np
import cv2
import pandas as pd

list_of_contour_rect_coordinates = pd.read_csv("list.csv", header=None, delimiter=',')
tuples = [tuple(x) for x in list_of_contour_rect_coordinates.values]
dict = {}

img = np.zeros(shape =[2000,2000,3], dtype=np.uint8)
def plot_rectangles(img,tuples,color):
	for rectangle in tuples:
		cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color, 3 )
plot_rectangles(img,tuples,(0,255,0),)
cv2.imshow('rectangles',img)
cv2.waitKey(0)