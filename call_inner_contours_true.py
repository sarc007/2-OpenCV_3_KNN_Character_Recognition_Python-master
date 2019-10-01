import numpy as np
import cv2
import pandas as pd
import is_inner_contour_true

img_inner_contours = np.zeros(shape=[4000,4000, 3], dtype=np.uint8)
def plot_rectangles(img, tuples, color):
    for rectangle in tuples:
        if len(rectangle) > 5:
            if rectangle[5] == False:
                cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color)
        else:
            cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color)

list_of_contour_rect_coordinates = pd.read_csv("list.csv",header=None,delimiter=',')
tuples = [tuple(x) for x in list_of_contour_rect_coordinates.values] #list of all rectangles,each rectangle is a tuple
list_of_contours = is_inner_contour_true.is_inner_contour(tuples)
print(list_of_contours)

plot_rectangles(img_inner_contours,tuples,(0,255,0))
cv2.imshow('rectangles',img_inner_contours)
cv2.waitKey(0)
