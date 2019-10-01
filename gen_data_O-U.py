# Pythono3 code to rename multiple 
# files in a directory or folder 

# importing os module 
import os
import cv2
import numpy as np
from skimage import io
import h5py


# Function to generate mnist data from images


def main():
    src_dir = "E:\\OPEN_1\\O-U"
    dst_dir = "E:\\OPEN_1\\data"
    data_file = dst_dir + "\\" + "O-U.hdf5"
    np_intValidChars = np.asarray(intValidChars)
    with h5py.File(data_file, "w") as f:
        dset = f.create_dataset("class", data=np_intValidChars)

    j = 1
    all_images = []
    all_labels = []

    for dirname in os.listdir(src_dir):
        i = 1
        # ord(dirname) :

        for image_path in os.listdir(src_dir + "\\" + dirname):
            img = cv2.imread(src_dir + "\\" + dirname + "\\" + image_path, 0)
            all_images.append(img)
            all_labels.append(ord(dirname))
            print("Directory " + dirname + " Adding file  : " + image_path + " out of " + str(
                len(os.listdir(src_dir + "\\" + dirname))) + " To Array Creating HD5 File ")
            # rename() function will
            # rename all the files

            i += 1
        j += 1
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    x_data = np.array(all_images)
    y_data = np.array(all_labels)
    #	tuple_xy = [x_data, y_data]
    # tuple_xy = np.array(tuple_xy)
    with h5py.File(data_file, "a") as f:
        dset_lbl = f.create_dataset("img_labels", data=y_data)
        dset_img = f.create_dataset("img_dataset", data=x_data)
    #	dset_img_lbl = f.create_dataset("img_lbl_ds", data=tuple_xy)

    with h5py.File(data_file, 'r') as f:
        class_arr = f['class'][:]
        labels_arr = f['img_labels'][:]
        image_arr = f['img_dataset'][:]
        print(class_arr.shape)
        print(labels_arr.shape)
        print(image_arr.shape)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]
    WIDTH = 28
    HEIGHT = 28
    main()
