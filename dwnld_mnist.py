# Pythono3 code to rename multiple 
# files in a directory or folder 
# print(data_label[0])
	# check_data = data[0]
	# print('check_data 0 0 0')
	# print(check_data)
	# print('len of check_data')
	# print(len(check_data))
	# check_data_np = np.array(check_data)
	# print('check_data shape')
	# print(check_data_np.shape)
	# print('check_data np 1')
	# print(check_data_np[1])
	# cv2.imshow('Iframe', check_data_np)
	# cv2.waitKey(0)
	# x_list = np.concatenate(data,data_test)
	# y_list = np.concatenate(data_label,data_label_test)
	# print(x_list[0])
	#print(y_list[0])
# importing os module 
import os
import tensorflow as tf
import cv2
import numpy as np
# Function to rename multiple files 
def main():
	dst_dir = "E:\\OPEN\\IMG"
	minst = tf.keras.datasets.mnist
	(data, data_label),(data_test,data_label_test) = minst.load_data()


	i=7777
	print(data_label[i])
	cv2.imshow('window', data[i])
	print(data_label_test[i])
	cv2.imshow('window', data_test[i])
	cv2.waitKey(0)
	i=0
	for y in data_label:
		if not os.path.exists(dst_dir + "\\" + str(y)):
			os.makedirs(dst_dir + "\\" + str(y))
		file_num = len(os.listdir(dst_dir + "\\" + str(y))) + 1
		x_list_np = np.array(data[i])
		cv2.imwrite(dst_dir + "\\" + str(y) + "\\img_" + f'{file_num:08}.png',x_list_np)
		print(str(i)+'/ '+ str(len(data_label)) + ' Writitng File for train imgs ' + dst_dir + "\\" + str(y) + "\\img_" + f'{file_num:08}.png')
		i +=1
	i = 0
	for y in data_label_test:

		if not os.path.exists(dst_dir + "\\" + str(y)):
			os.makedirs(dst_dir + "\\" + str(y))
		file_num = len(os.listdir(dst_dir + "\\" + str(y)))+1
		x_list_np = np.array(data_test[i])
		cv2.imwrite(dst_dir + "\\" + str(y) + "\\img_" + f'{file_num:08}.png',x_list_np)
		print(str(i)+'/ '+ str(len(data_label_test)) +' Writitng File for test imgs '+dst_dir + "\\" + str(y) + "\\img_" + f'{file_num:08}.png')
		i +=1

# Driver Code
if __name__ == '__main__':

	# Calling main() function
	main()
