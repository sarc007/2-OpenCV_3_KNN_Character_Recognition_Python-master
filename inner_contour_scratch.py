import pandas as pd
import numpy as np
import cv2

list_of_rectangular_coordinates = pd.read_csv('list.csv',header=None)
print(list_of_rectangular_coordinates)
tuples = [tuple(x) for x in list_of_rectangular_coordinates.values]
# print(tuples)


img = np.zeros([4000,4000,3],dtype=np.uint8)
def plot_all_rectangles(img,tuples,color):
    for rectangle in tuples:
        cv2.rectangle(img,(rectangle[1],rectangle[2]),(rectangle[3],rectangle[4]),color)


def inner_contour():
    for rectangle in tuples:
        index_of_rectangles = tuples.index(rectangle)


plot_all_rectangles(img,tuples,(0,255,0))


cv2.imshow('rectangle',img)
cv2.waitKey(0)


