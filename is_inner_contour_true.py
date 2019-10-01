import numpy as np
import cv2
import pandas as pd

def only_inner_contour(tuples):

    index_of_inner_contour_coordinates = []

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

    def slice_contour_indices(tuples,index_list,is_inner):
        sliced_tuple = []
        for index in index_list:
             # tuples[index]:
            sliced_tuple.append((tuples[index][0],tuples[index][1],tuples[index][2],tuples[index][3],tuples[index][4],is_inner))

        return sliced_tuple

    # def list_outer_contours_index():
    #     for results in tuples:
    #         if results[0] not in index_of_inner_contour_coordinates:
    #             index_of_outer_contour_coordinates.append(results[0])
    #     return True


    list_inner_contour()
    # list_outer_contours_index()
    tuples_of_inner_contours = slice_contour_indices(tuples,index_of_inner_contour_coordinates,True)

    return tuples_of_inner_contours

def only_outer_contour(tuples):

    index_of_inner_contour_coordinates = []
    index_of_outer_contour_coordinates = []

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

    def slice_contour_indices(tuples,index_list,is_inner):
        sliced_tuple = []
        for index in index_list:
             # tuples[index]:
            sliced_tuple.append((tuples[index][0],tuples[index][1],tuples[index][2],tuples[index][3],tuples[index][4],is_inner))

        return sliced_tuple

    def list_outer_contours_index():
        for results in tuples:
            if results[0] not in index_of_inner_contour_coordinates:
                index_of_outer_contour_coordinates.append(results[0])
        return True

    list_inner_contour()
    list_outer_contours_index()
    # plot_rectangles(img,tuples,(0,255,0))
    tuples_of_outer_contours = slice_contour_indices(tuples,index_of_outer_contour_coordinates,False)
    return tuples_of_outer_contours


def  is_inner_contour(tuples):

# list_of_contour_rect_coordinates = pd.read_csv("list.csv",header=None,delimiter=',')
# tuples = [tuple(x) for x in list_of_contour_rect_coordinates.values] #list of all rectangles,each rectangle is a tuple
#     print(tuples)
    index_of_inner_contour_coordinates = []
    index_of_outer_contour_coordinates = []
#
# img = np.zeros(shape =[4000,4000,3], dtype=np.uint8)
# img_outer_contours = np.zeros(shape =[4000,4000,3], dtype=np.uint8)
# img_inner_contours = np.zeros(shape =[4000,4000,3], dtype=np.uint8)
# def plot_rectangles(img,tuples,color):
# 	for rectangle in tuples:
# 		if len(rectangle) > 5:
# 			if rectangle[5] == False:
# 				cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color)
# 		else:
# 			cv2.rectangle(img, (rectangle[1], rectangle[2]), (rectangle[3], rectangle[4]), color)

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

    def slice_contour_indices(tuples,index_list,is_inner):
        sliced_tuple = []
        for index in index_list:
             # tuples[index]:
            sliced_tuple.append((tuples[index][0],tuples[index][1],tuples[index][2],tuples[index][3],tuples[index][4],is_inner))

        return sliced_tuple

    def list_outer_contours_index():
        for results in tuples:
            if results[0] not in index_of_inner_contour_coordinates:
                index_of_outer_contour_coordinates.append(results[0])
        return True

    def add_lists(list1,list2):
        i = 0
        j = 0
        k = 0
        list = []
        while i < len(list1)+len(list2):
            for each_element in list2:
                if i == each_element[0]:
                    list.append(each_element)

            for each_element in list1:
                if i == each_element[0]:
                    list.append(each_element)
            i+=1
        return list


    list_inner_contour()
    list_outer_contours_index()
    # plot_rectangles(img,tuples,(0,255,0))
    tuples_of_inner_contours = slice_contour_indices(tuples,index_of_inner_contour_coordinates,True)
    tuples_of_outer_contours = slice_contour_indices(tuples,index_of_outer_contour_coordinates,False)
    # plot_rectangles(img,tuples_of_inner_contours,(255,145,200))
    # plot_rectangles(img_outer_contours,tuples_of_outer_contours,(125,255,60))
    # plot_rectangles(img_inner_contours,tuples_of_inner_contours,(125,255,60))
    # print(tuples_of_outer_contours)
    # print(tuples_of_inner_contours)
    # tuple_to_return =(tuples_of_inner_contours + tuples_of_outer_contours )
    return add_lists(tuples_of_inner_contours,tuples_of_outer_contours)

    # cv2.imshow('rectangles',img)
    # cv2.waitKey(0)
    # cv2.imshow('outer contour rectangles',img_outer_contours)
    # cv2.waitKey(0)
    # cv2.imshow('inner contour rectangles',img_inner_contours)
    # cv2.waitKey(0)