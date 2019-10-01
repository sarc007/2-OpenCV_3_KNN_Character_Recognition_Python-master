# Pythono3 code to rename multiple 
# files in a directory or folder 

# importing os module 
import os
import cv2
from PIL import Image, ImageFilter, ImageOps
import numpy as np


# Function to rename multiple files
def main():
	src_dir = "F:\\PycharmProjects\\ai_Lessons\\img"
	dst_dir = "F:\\PycharmProjects\\ai_Lessons\\img_resized"
	j = 0
	for dirname in os.listdir(src_dir):
		i = 1
		if not os.path.exists(dst_dir + "\\" + dirname):
			os.makedirs(dst_dir + "\\" + dirname)
		for filename in os.listdir(src_dir + "\\" + dirname):
			src = src_dir + "\\" + dirname + "\\" + filename
			dst = filename
			img = cv2.imread(src)
			img_sharp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img_sharp = Image.fromarray(img_sharp)
			img_sharp = img_sharp.filter(ImageFilter.SHARPEN)
			img = np.asanyarray(img_sharp)
			width = 28
			height = 28
			dim = (width, height)
			# resize image
			img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

			dst = dst_dir + "\\" + dirname + "\\" + dst
			cv2.imwrite(dst, img)
			print("Resizing file : " + dst)
			# rename() function will 
			# rename all the files 

			i += 1
		j += 1


# Driver Code
if __name__ == '__main__':
	# Calling main() function 
	main()
