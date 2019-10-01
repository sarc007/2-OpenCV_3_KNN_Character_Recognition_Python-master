import os
import numpy as np
from skimage import io
import h5py
import pandas as pd

list_data = []
# list_of_numbers = [90, 26, 67, 47, 8]
list_of_contour_rect_coordinates = pd.read_csv('list.csv', header=None, delimiter=',')
tuples = [tuple(x) for x in list_of_contour_rect_coordinates.values]
index_of_inner_contours = []
# print(tuples)
img = np.zeros(shape=[2000, 2000, 3], dtype=np.uint8)
def generate_and_save_JSON(tuples):
	for rectangle in tuples:
		list_data.append([int(rectangle[0]), int(rectangle[1]), int(rectangle[2]), int(rectangle[3]), int(rectangle[4]),
			tuples.index(rectangle) in index_of_inner_contours])

def list_inner_contour():
	for rectangle in tuples:
		index_of_rectangle = tuples.index(rectangle)
		i = 0
		while i < len(tuples):
			if i != index_of_rectangle:
				if rectangle[1] > tuples[i][1] and rectangle[2] > tuples[i][2] \
						and rectangle[3] < tuples[i][3] and rectangle[4] < tuples[i][4]:
					index_of_inner_contours.append(index_of_rectangle)
			i += 1

	return True

def slice_on_custom_list_of_indices(tuples, list_of_indices):
	sliced_tuple = []
	for index in list_of_indices:
		sliced_tuple.append(tuples[index])
	return sliced_tuple


list_inner_contour()
# plot_rectangles(img,tuples,(0,255,0))
tuple_of_inner_contours = slice_on_custom_list_of_indices(tuples, index_of_inner_contours)
# plot_rectangles(img,tuple_of_inner_contours,(0,0,255))
# generate_and_save_JSON(tuples)

# with h5py.File('E:\\OPEN\\data\\rectangular_coordinates.hdf5','w') as f:
# 	f.create_dataset('coordinates',data=np.array(list_data))

# h = h5py.File('myfile.hdf5')
# for k, v in list_data.items():
# 	h.create_dataset(k.strftime('%Y-%m-%dT%H:%M:%SZ'), data=np.array(v, dtype=np.int8))

# with h5py.File('E:\\OPEN\\data\\rectangular_coordinates.hdf5','r') as f:r
# 	data = f.get('coordinates')
# 	print(data)
#
